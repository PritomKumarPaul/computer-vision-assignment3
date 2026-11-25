"""Fit and evaluate rank-4 SigLIP image adapters on the fixed one-shot split."""

import argparse
import csv
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from run_checkpoint1 import check_support_overlap, load_encoder, load_protocol, prompt, write_json
from run_checkpoint2 import read_baseline, read_rgb


RANK = 4
ALPHA = 4
STEPS = 50
LEARNING_RATE = 1e-4
SEED = 0


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, RANK, bias=False)
        self.up = nn.Linear(RANK, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.up(self.down(inputs)) * (ALPHA / RANK)


def attach_adapters(model):
    model.requires_grad_(False)
    layers = model.vision_model.encoder.layers
    for layer in layers[-2:]:
        attention = layer.self_attn
        attention.q_proj = LoRALinear(attention.q_proj)
        attention.v_proj = LoRALinear(attention.v_proj)
    return [f"vision_model.encoder.layers.{i}.self_attn.{projection}"
            for i in range(len(layers) - 2, len(layers))
            for projection in ("q_proj", "v_proj")]


def encode_images(model, pixels):
    return F.normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1)


def image_tensor(processor, paths, device):
    images = [read_rgb(path) for path in paths]
    return processor(images=images, return_tensors="pt").pixel_values.to(device)


def fit(model, parameters, support_pixels, text_vectors):
    optimizer = torch.optim.AdamW(parameters.values(), lr=LEARNING_RATE, weight_decay=0.0)
    signs = 2 * torch.eye(len(text_vectors), device=text_vectors.device) - 1
    losses = []
    with tqdm(range(STEPS), desc="SigLIP LoRA", unit="step") as progress:
        for _ in progress:
            optimizer.zero_grad(set_to_none=True)
            similarities = encode_images(model, support_pixels) @ text_vectors.T
            logits = similarities * model.logit_scale.exp() + model.logit_bias
            # One positive on the diagonal and 15 negatives per image.
            loss = -F.logsigmoid(signs * logits).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
    return losses


@torch.inference_mode()
def evaluate(args, model, processor, device, text_vectors, classes, queries):
    correct = 0
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "label", "siglip_lora"])
        with tqdm(total=len(queries), desc="SigLIP LoRA test inference", unit="img") as progress:
            for start in range(0, len(queries), args.batch_size):
                batch = queries[start:start + args.batch_size]
                pixels = image_tensor(processor, [path for path, _ in batch], device)
                predictions = (encode_images(model, pixels) @ text_vectors.T).argmax(dim=1).tolist()
                for (path, label), predicted in zip(batch, predictions):
                    correct += predicted == label
                    writer.writerow([path.relative_to(args.data_root).as_posix(),
                                     classes[label], classes[predicted]])
                progress.update(len(batch))
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/checkpoint1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoint3"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Test inference batch size.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--adapter", type=Path, help="Evaluate a saved adapter and skip training.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu or launch on a GPU node.")

    classes, supports, queries = load_protocol(args.data_root)
    check_support_overlap(supports, queries)
    references = [(label, path.relative_to(args.data_root).as_posix())
                  for label, path in zip(classes, supports)]
    query_labels = [(path.relative_to(args.data_root).as_posix(), classes[label])
                    for path, label in queries]
    baseline = read_baseline(args.baseline_dir, "siglip", references, query_labels)
    prompts = [prompt(label) for label in classes]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    print(f"Loading SigLIP on {device}", flush=True)
    model, processor, tokenizer, metadata = load_encoder("siglip", args.offline)
    if any(baseline[key] != value for key, value in metadata.items()):
        raise ValueError("The active SigLIP encoder differs from Checkpoint 1.")
    targets = attach_adapters(model)
    model.eval().to(device)
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if len(parameters) != 8:
        raise RuntimeError("Expected exactly two trainable matrices in each of four projections.")
    trainable = sum(param.numel() for param in parameters.values())
    print(f"LoRA trainable parameters: {trainable:,}", flush=True)

    with torch.no_grad():
        tokens = tokenizer(
            prompts, padding="max_length", max_length=model.config.text_config.max_position_embeddings,
            truncation=True, return_tensors="pt",
        ).to(device)
        text_vectors = F.normalize(model.get_text_features(**tokens).float(), dim=-1)

    losses = None
    fit_seconds = None
    if args.adapter is None:
        start = time.perf_counter()
        support_pixels = image_tensor(processor, supports, device)
        losses = fit(model, parameters, support_pixels, text_vectors)
        fit_seconds = time.perf_counter() - start
        save_file({name: param.detach().cpu() for name, param in parameters.items()},
                  args.output_dir / "adapter.safetensors", metadata=metadata)
    else:
        weights = load_file(args.adapter)
        if weights.keys() != parameters.keys() or any(
            weights[name].shape != param.shape for name, param in parameters.items()
        ):
            raise ValueError("Adapter tensors do not match the four SigLIP LoRA projections.")
        with torch.no_grad():
            for name, param in parameters.items():
                param.copy_(weights[name])

    start = time.perf_counter()
    correct = evaluate(args, model, processor, device, text_vectors, classes, queries)
    accuracy = correct / len(queries)
    metrics = {
        **metadata, "targets": targets, "rank": RANK, "alpha": ALPHA,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "seed": SEED, "steps": STEPS,
        "device": device, "trainable_parameters": trainable, "fit_seconds": fit_seconds,
        "test_seconds": time.perf_counter() - start, "step_losses": losses,
        "test_images": len(queries), "correct": correct, "accuracy": accuracy,
        "checkpoint1_zero_shot_accuracy": baseline["zero_shot_accuracy"],
        "checkpoint1_one_shot_accuracy": baseline["one_shot_accuracy"],
        "change_vs_zero_shot_pp": 100 * (accuracy - baseline["zero_shot_accuracy"]),
        "change_vs_one_shot_pp": 100 * (accuracy - baseline["one_shot_accuracy"]),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "support_manifest.json", {
        "classes": [{"label": label, "image": image, "prompt": text}
                    for (label, image), text in zip(references, prompts)],
    })
    print(f"SigLIP | zero-shot {baseline['zero_shot_accuracy']:.2%} | "
          f"one-shot {baseline['one_shot_accuracy']:.2%} | LoRA {accuracy:.2%} "
          f"({correct}/{len(queries)})")


if __name__ == "__main__":
    main()
