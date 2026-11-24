# halo-sft-demo

A small, real SFT run with [Halo](https://github.com/whitecircle/halo), White Circle's
open-source post-training framework — built as a demo for the
[Halo post-training competition](https://whitecircle.com/halo/competition) (Dec 1–10, 2026).

## What this is

Halo's own smallest single-GPU SFT example is Qwen3-4B LoRA on UltraChat 200K
(`examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`). This repo runs that same
example, scaled down so it finishes in under an hour on one 16 GB GPU:

- **Model:** Qwen/Qwen3-4B-Instruct-2507, LoRA (r=16, alpha=32)
- **Data:** HuggingFaceH4/ultrachat_200k train_sft, `dataset_ratio: 0.005` (~1,000 rows)
- **Training:** 30 steps, max length 1024, packing on, batch 2 x grad-accum 4,
  lr 1e-4 cosine, fp16, SDPA attention
- **Hardware:** Kaggle GPU T4 x2 (one T4 used)

fp16, not bf16: the T4 has no bf16 tensor cores. SDPA, not FlashAttention:
FlashAttention 2 does not support the T4's Turing architecture.

## Results

<!-- Filled in from the real run log once the run finishes. -->

See [results.json](results.json) and the full console output in [run.log](run.log).
The trained LoRA adapter is in `checkpoints/sft-qwen3-4b-ultrachat-lora-demo/`.

## How to reproduce

```bash
git clone https://github.com/whitecircle/halo.git
cd halo && git checkout <commit in HALO_VERSION.md>

# Inside Halo's Docker image (halo:hopper / halo:blackwell), or an
# equivalent Python 3.12 environment with Halo installed:
python scripts/training/sft.py \
  /path/to/this/repo/configs/sft-qwen3-4b-ultrachat-lora-demo.yaml
```

The config is self-contained: model, dataset, LoRA settings, and training
hyperparameters are all in
[`configs/sft-qwen3-4b-ultrachat-lora-demo.yaml`](configs/sft-qwen3-4b-ultrachat-lora-demo.yaml).
Halo version and commit: see [HALO_VERSION.md](HALO_VERSION.md).

## Links

- Halo: https://github.com/whitecircle/halo
- Competition: https://whitecircle.com/halo/competition
- Base model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
