"""Evaluate one-shot prototypes averaged across 32 views of each reference."""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from run_checkpoint1 import check_support_overlap, load_encoder, load_protocol, write_json


BRIGHTNESS = (0.90, 0.96, 1.04, 1.10)
CONTRAST = (0.90, 0.96, 1.04, 1.10)
VIEW_GRID = tuple(itertools.product((False, True), BRIGHTNESS, CONTRAST))


def read_rgb(path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def augmented_views(image):
    from PIL import ImageEnhance, ImageOps

    mirrored = ImageOps.mirror(image)
    for flip, brightness, contrast in VIEW_GRID:
        view = ImageEnhance.Brightness(mirrored if flip else image).enhance(brightness)
        yield ImageEnhance.Contrast(view).enhance(contrast)


def read_baseline(baseline_dir, name, references, queries):
    with (baseline_dir / "support_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    expected_references = [(row["label"], row["image"]) for row in manifest["classes"]]
    if expected_references != references:
        raise ValueError("Checkpoint 1 and Checkpoint 2 reference images differ.")
    with (baseline_dir / name / "predictions.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if [(row["image"], row["label"]) for row in rows] != queries:
        raise ValueError(f"Checkpoint 1 and Checkpoint 2 test images or labels differ for {name}.")
    with (baseline_dir / name / "metrics.json").open(encoding="utf-8") as file:
        return json.load(file)


def evaluate(args, name, classes, supports, queries, references, query_labels, baseline):
    import torch
    from torch.nn.functional import normalize
    from tqdm import tqdm

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or launch on a GPU node.")

    print(f"Loading {name.upper()} on {device}", flush=True)
    model, processor, _, metadata = load_encoder(name, args.offline)
    if any(baseline[key] != value for key, value in metadata.items()):
        raise ValueError(f"The active {name} encoder differs from Checkpoint 1.")
    model.eval().to(device)
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    def embed(images):
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
        return normalize(model.get_image_features(pixel_values=pixels).float(), dim=-1)

    start = time.perf_counter()
    with torch.inference_mode():
        prototypes = []
        for path in tqdm(supports, desc=f"{name.upper()} prototypes", unit="class"):
            views = list(augmented_views(read_rgb(path)))
            embeddings = [
                embed(views[start:start + args.batch_size])
                for start in range(0, len(views), args.batch_size)
            ]
            mean_embedding = torch.cat(embeddings).mean(dim=0)
            prototypes.append(normalize(mean_embedding, dim=0))
        prototypes = torch.stack(prototypes)
        vectors = prototypes.cpu().tolist()
        prototype_seconds = time.perf_counter() - start

        start = time.perf_counter()
        correct = 0
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "label", "augmented_one_shot"])
            with tqdm(total=len(queries), desc=f"{name.upper()} test inference", unit="img") as progress:
                for offset in range(0, len(queries), args.batch_size):
                    batch = queries[offset:offset + args.batch_size]
                    features = embed([read_rgb(path) for path, _ in batch])
                    predictions = (features @ prototypes.T).argmax(dim=1).tolist()
                    for (image, label), pred in zip(query_labels[offset:offset + len(batch)], predictions):
                        prediction = classes[pred]
                        correct += prediction == label
                        writer.writerow([image, label, prediction])
                    progress.update(len(batch))
        test_seconds = time.perf_counter() - start

    write_json(output_dir / "classifier.json", {
        **metadata,
        "classes": classes,
        "references": dict(references),
        "view_grid": [
            {"horizontal_flip": flip, "brightness": brightness, "contrast": contrast}
            for flip, brightness, contrast in VIEW_GRID
        ],
        "image_prototypes": vectors,
    })
    accuracy = correct / len(queries)
    metrics = {
        **metadata,
        "device": device,
        "views_per_reference": len(VIEW_GRID),
        "test_images": len(queries),
        "correct": correct,
        "augmented_one_shot_accuracy": accuracy,
        "checkpoint1_zero_shot_accuracy": baseline["zero_shot_accuracy"],
        "checkpoint1_one_shot_accuracy": baseline["one_shot_accuracy"],
        "change_vs_one_shot_pp": (accuracy - baseline["one_shot_accuracy"]) * 100,
        "prototype_seconds": prototype_seconds,
        "test_seconds": test_seconds,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"{name.upper()} | zero-shot {baseline['zero_shot_accuracy']:.2%} | "
          f"one-shot {baseline['one_shot_accuracy']:.2%} | augmented one-shot {accuracy:.2%} "
          f"({metrics['change_vs_one_shot_pp']:+.2f} pp vs one-shot)")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/checkpoint1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoint2"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--model", choices=("both", "clip", "siglip"), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    names = ("clip", "siglip") if args.model == "both" else (args.model,)
    try:
        classes, supports, queries = load_protocol(args.data_root)
        check_support_overlap(supports, queries)
        references = [(label, path.relative_to(args.data_root).as_posix())
                      for label, path in zip(classes, supports)]
        query_labels = [(path.relative_to(args.data_root).as_posix(), classes[label])
                        for path, label in queries]
        baselines = {
            name: read_baseline(args.baseline_dir, name, references, query_labels)
            for name in names
        }
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Dataset or Checkpoint 1 output error: {error}\n")

    results = {
        name: evaluate(args, name, classes, supports, queries, references, query_labels, baselines[name])
        for name in names
    }
    if args.model == "both":
        write_json(args.output_dir / "comparison.json", results)


if __name__ == "__main__":
    main()
