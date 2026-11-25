"""Compare frozen CLIP and SigLIP zero-shot and one-shot classifiers."""

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path


MODEL = "openai/clip-vit-base-patch32"
PROCESSOR_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WEIGHTS_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_TEXT = {"InsideCity": "city street between buildings", "Suburb": "suburban neighborhood"}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def image_folders(split):
    return {
        folder.name: sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for folder in sorted(split.iterdir())
        if folder.is_dir() and not folder.name.startswith(".")
    }


def load_protocol(root):
    train = image_folders(root / "train")
    test = image_folders(root / "test")
    if len(train) != 16 or train.keys() != test.keys():
        raise ValueError("Expected 16 matching class folders in train/ and test/.")
    if any(not images for images in (*train.values(), *test.values())):
        raise ValueError("Every class must have at least one image in train/ and test/.")
    supports = [images[0] for images in train.values()]
    queries = [(path, label) for label, images in enumerate(test.values()) for path in images]
    return list(train), supports, queries


def check_support_overlap(supports, queries):
    hashes = {hashlib.sha256(path.read_bytes()).digest(): path for path in supports}
    for path, _ in queries:
        match = hashes.get(hashlib.sha256(path.read_bytes()).digest())
        if match is not None:
            raise ValueError(f"Identical train reference and test image: {match} and {path}")


def prompt(class_name):
    readable = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", class_name)
    readable = readable.replace("_", " ").replace("-", " ").lower()
    return f"a photo of a scene showing {CLASS_TEXT.get(class_name, readable)}."


def load_siglip_tokenizer(settings):
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    # Load the publisher's serialized tokenizer without converting a SentencePiece model.
    tokenizer_path = hf_hub_download(SIGLIP_MODEL, "tokenizer.json", **settings)
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, eos_token="</s>", pad_token="</s>",
        unk_token="<unk>", model_max_length=64, model_input_names=["input_ids"],
    )


def load_encoder(name, offline):
    from huggingface_hub import hf_hub_download

    if name == "clip":
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

        model_id, processor_revision, weights_revision = MODEL, PROCESSOR_REVISION, WEIGHTS_REVISION
        settings = {"revision": processor_revision, "local_files_only": offline}
        processor = CLIPImageProcessor.from_pretrained(model_id, **settings)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id, **settings)
        config = CLIPConfig.from_pretrained(model_id, **settings)
        model_class = CLIPModel
    else:
        from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel

        model_id, processor_revision, weights_revision = SIGLIP_MODEL, SIGLIP_REVISION, SIGLIP_REVISION
        settings = {"revision": processor_revision, "local_files_only": offline}
        processor = SiglipImageProcessor.from_pretrained(model_id, **settings)
        tokenizer = load_siglip_tokenizer(settings)
        config = SiglipConfig.from_pretrained(model_id, **settings)
        model_class = SiglipModel
    # CLIP's config and safetensors are pinned to separate snapshots. Resolve
    # the weight file explicitly so the config's commit cannot redirect lookup.
    weights_file = hf_hub_download(
        model_id, "model.safetensors", revision=weights_revision, local_files_only=offline,
    )
    model = model_class.from_pretrained(
        Path(weights_file).parent, config=config,
        use_safetensors=True, local_files_only=offline,
    )
    metadata = {
        "model": model_id, "processor_revision": processor_revision,
        "weights_revision": weights_revision,
    }
    return model, processor, tokenizer, metadata


def evaluate(args, name, classes, supports, queries, prompts):
    import torch
    from PIL import Image
    from torch.nn.functional import normalize
    from tqdm import tqdm

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or launch on a GPU node.")
    output_dir = args.output_dir / name
    output_dir.mkdir(exist_ok=True)
    print(f"Loading {name.upper()} on {device}", flush=True)
    start = time.perf_counter()
    model, processor, tokenizer, metadata = load_encoder(name, args.offline)
    model.eval().to(device)

    def embed_images(paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
        return normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1)

    def image_batches(paths):
        for i in range(0, len(paths), args.batch_size):
            yield embed_images(paths[i:i + args.batch_size])

    correct_zero = correct_one = 0
    start_inference = time.perf_counter()
    with torch.inference_mode():
        padding = {"padding": True} if name == "clip" else {
            "padding": "max_length", "max_length": model.config.text_config.max_position_embeddings,
            "truncation": True,
        }
        tokens = tokenizer(prompts, return_tensors="pt", **padding).to(device)
        text_vectors = normalize(model.get_text_features(**tokens).float(), dim=-1)
        image_vectors = torch.cat(list(image_batches(supports)))
        write_json(output_dir / "classifier.json", {
            **metadata, "classes": classes, "prompts": prompts,
            "text_prototypes": text_vectors.cpu().tolist(),
            "image_prototypes": image_vectors.cpu().tolist(),
        })
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "label", "zero_shot", "one_shot"])
            paths = [path for path, _ in queries]
            with tqdm(total=len(queries), desc=f"{name.upper()} test inference", unit="img") as progress:
                for batch_index, features in enumerate(image_batches(paths)):
                    zero = (features @ text_vectors.T).argmax(dim=1).tolist()
                    one = (features @ image_vectors.T).argmax(dim=1).tolist()
                    offset = batch_index * args.batch_size
                    for (path, label), z, o in zip(queries[offset:offset + len(zero)], zero, one):
                        correct_zero += z == label
                        correct_one += o == label
                        writer.writerow([path.relative_to(args.data_root).as_posix(),
                                         classes[label], classes[z], classes[o]])
                    progress.update(len(zero))
    count = len(queries)
    metrics = {
        **metadata,
        "test_images": count,
        "zero_shot_correct": correct_zero, "one_shot_correct": correct_one,
        "zero_shot_accuracy": correct_zero / count, "one_shot_accuracy": correct_one / count,
        "device": str(device), "model_load_seconds": start_inference - start,
        "inference_seconds": time.perf_counter() - start_inference,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"{name.upper()} test accuracy | zero-shot: {correct_zero / count:.2%} "
          f"| one-shot: {correct_one / count:.2%}")
    print(f"Outputs: {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoint1"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", choices=("both", "clip", "siglip"), default="both")
    parser.add_argument("--offline", action="store_true", help="Use locally cached model files.")
    parser.add_argument("--prepare-only", action="store_true", help="Audit data and write the manifest.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        classes, supports, queries = load_protocol(args.data_root)
        check_support_overlap(supports, queries)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Dataset error: {error}\n")
    prompts = [prompt(name) for name in classes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "support_manifest.json", {
        "selection": "First image by lexicographic filename within each training class",
        "classes": [
            {"label": name, "image": path.relative_to(args.data_root).as_posix(), "prompt": text}
            for name, path, text in zip(classes, supports, prompts)
        ],
        "test_images": len(queries),
    })
    print(f"Selected {len(supports)} training images; labeled test set: {len(queries)} images.")
    if not args.prepare_only:
        names = ("clip", "siglip") if args.model == "both" else (args.model,)
        results = {name: evaluate(args, name, classes, supports, queries, prompts) for name in names}
        if args.model == "both":
            write_json(args.output_dir / "comparison.json", results)


if __name__ == "__main__":
    main()
