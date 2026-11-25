# Assignment 3 — One-Shot Image Classification

Checkpoint 1 compares frozen CLIP ViT-B/32 and SigLIP ViT-B/16 (224 pixels).
Each encoder uses the same text prompts for zero-shot and the same 16 labeled
reference images for one-shot. Both methods rank classes by cosine similarity
and share test image embeddings within each encoder. The one-shot reference
is the first image by lexicographic filename inside each training class.

Place the dataset from the assignment handout in `data/`, with
`data/train/<class>/` and `data/test/<class>/`. Use `--data-root` if the
dataset is elsewhere. The runner checks the 16 reference images against the
labeled test set for exact byte duplicates before loading a model.

```bash
conda activate vlm_clean
python run_checkpoint1.py --device auto
```

The first run downloads missing model files from Hugging Face. Use
`--offline` once both pinned snapshots are cached. Use `--model clip` or
`--model siglip` to run one encoder. For a fresh environment, install
`requirements.txt` with a PyTorch build appropriate for the host.
Use `--prepare-only` to check the split and write the support manifest
without loading either encoder.

The default output directory is `outputs/checkpoint1/`. It contains the
shared 16-image manifest, `comparison.json` for the two-model run, and a
per-model directory with predictions, accuracies, timings, and both prototype
sets. For GitHub, commit the source and documentation. The dataset and raw
outputs are excluded by `.gitignore`.

## Checkpoint 3 — SigLIP contrastive LoRA

Run Checkpoint 1 first to generate the SigLIP baseline and split manifest.
Then run the following in the same environment:

```bash
python run_checkpoint3.py --device auto --offline
```

This fits rank-4 adapters on the query and value projections in the last two
image blocks. Each of 50 optimizer steps uses the 16 fixed reference images
and 16 class prompts with the pairwise SigLIP objective. The learning rate is
`1e-4`. The text encoder and original image weights are frozen. The test set
is scored once after fitting. Remove `--offline` if the model is not cached.

The saved adapter can be evaluated again with:

```bash
python run_checkpoint3.py --device auto --offline \
  --adapter outputs/checkpoint3/adapter.safetensors \
  --output-dir outputs/checkpoint3_replay
```
