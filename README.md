# halo-sft-demo

A small, real SFT demo with [Halo](https://github.com/whitecircle/halo), White Circle's
open-source post-training framework — built as a demo for the
[Halo post-training competition](https://whitecircle.com/halo/competition) (Dec 1–10, 2026).

## Status (25 Nov 2026)

- **GPU demo — Qwen3-4B LoRA on UltraChat** (the real Halo run): config ready in
  [`configs/`](configs/sft-qwen3-4b-ultrachat-lora-demo.yaml), Kaggle GPU run
  submitted 24 Nov and still **queued** at last check (25 Nov). Its results will go to
  `results-gpu.json` and `checkpoints/` when it lands.
- **CPU demo — real Qwen3-0.6B LoRA: done** (25 Nov 2026, see below). A real
  pretrained model, real tokenizer, 100 chat examples. It ran with plain TRL,
  **not** Halo — Halo's SFT trainer needs a GPU and could not run on this
  CPU-only box. The numbers are real and from the log in this repo.
- **CPU demo — tiny Qwen3 on 24 chat examples: done** (25 Nov 2026, see below).
  Also plain TRL, not Halo (same reason). Random-init smoke test.

## Results — Qwen3-0.6B LoRA run (real, 25 Nov 2026)

Runner: **TRL `SFTTrainer` + LoRA, CPU — not Halo.** Weights downloaded from
Hugging Face [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
(311 tensors, bf16, 0.60B params; 752M including tied embeddings). A 3-step
trial first confirmed it fits (peak RSS 2.11 GB, ~27s/step); the full run
below is 12 steps.

| | |
|---|---|
| Model | Qwen3-0.6B pretrained, bf16, LoRA r=8 alpha=16 (5.05M trainable, 0.85%), real Qwen3 tokenizer |
| Data | 100 chat examples, 97 train / 3 eval: 24 hand-written + 76 from public `tatsu-lab/alpaca` ([`cpu/sft-0.6b.jsonl`](cpu/sft-0.6b.jsonl)) |
| Training | 12 steps, batch 1 x grad-accum 2, lr 2e-4 cosine, warmup 2, max_length 256, grad checkpointing, AdamW on LoRA params only, loss on assistant tokens only |
| Loss, step 1 → step 12 | **2.36 → 0.85** (train avg 1.64; noisy step-to-step because batches vary, clear downward trend — steps 6–12 mostly under 1.4) |
| Eval loss (3 held-out rows) | **1.02** (was 1.83 after 3 steps); eval mean token accuracy 0.79 |
| Time | 332s training, 344s total (~27s/step) |
| Hardware | CPU, 4 cores, 7.9 GB RAM, peak RSS 2.70 GB; torch 2.11.0+cpu, transformers 5.20.2, trl 1.14.2, peft 0.18.1 |

This is a real pretrained model learning: eval loss fell from 1.83 to 1.02
over the run, and token accuracy rose to 0.79. It is still a small demo
(0.25 epoch, 12 steps, no checkpoint saved) — it proves Vinayak can run a
real SFT pipeline end to end, not that the model is useful. No checkpoint
was saved (`save_strategy: no`); the deliverable is the log and the numbers.

Full numbers: [`results-0.6b.json`](results-0.6b.json). Full console log:
[`run-0.6b.log`](run-0.6b.log). Training script:
[`cpu/train_qwen06b_lora.py`](cpu/train_qwen06b_lora.py). Dataset builder:
[`cpu/prep_alpaca_slice.py`](cpu/prep_alpaca_slice.py).

## Results — CPU demo run (real, 25 Nov 2026)

Runner: **TRL `SFTTrainer`, CPU fallback — not Halo.** Halo was tried first:
its SFT script got as far as loading the dataset (24 rows → 21 train / 3 test)
and probing the model config, then failed in its model loader, which
hardcodes `device_map={"": <cuda rank>}` in
`halo/model_loading/parallel/loader.py` (`_load_undistributed_model`) —
transformers raises "Cannot access accelerator device when none is
available". The full failure log is [`cpu/halo-attempt.log`](cpu/halo-attempt.log).
Halo is a GPU framework (its own docs: run everything in the Docker image).

| | |
|---|---|
| Model | Random-init tiny Qwen3, hidden 128, 2 layers, **20.1M params**, real Qwen3-0.6B tokenizer |
| Data | 24 hand-written chat examples, 21 train / 3 eval ([`cpu/tiny-chat.jsonl`](cpu/tiny-chat.jsonl)) |
| Training | 12 steps, batch 2, lr 5e-4 cosine, warmup 2, max_length 256, fp32, loss on assistant tokens only |
| Loss, step 1 → step 12 | **11.93 → 10.96** (train avg 11.33) |
| Eval loss (3 held-out rows) | **10.99**; mean token accuracy 0 → 0.35 over the run |
| Time | 23.6s training, 33.1s total |
| Hardware | CPU, 4 cores, 7.9 GB RAM; torch 2.11.0+cpu, transformers 5.20.2, trl 1.14.2 |

Loss starts at ~11.93, which is exactly random for a 151,936-token vocab
(ln 151936 ≈ 11.93) — the model was randomly initialised, so the absolute
numbers are high. What matters is the trend: loss fell every few steps and
token accuracy rose from 0 to 0.35. This is a smoke-test-scale run that proves
the pipeline works, not a useful model. No checkpoint was saved
(`save_strategy: no`); the deliverable is the log and the numbers.

Full numbers: [`results.json`](results.json). Full console log:
[`run.log`](run.log). Training script:
[`cpu/train_tiny_trl.py`](cpu/train_tiny_trl.py). Halo config that was
attempted: [`cpu/sft-tiny-cpu.yaml`](cpu/sft-tiny-cpu.yaml).

## GPU demo — Qwen3-4B LoRA on UltraChat (queued)

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

## How to reproduce

### CPU demo — Qwen3-0.6B LoRA (real model, ~6 minutes)

```bash
# weights: download Qwen/Qwen3-0.6B into cpu-run/qwen06b/ (config.json,
# generation_config.json, model.safetensors + the Qwen3 tokenizer files);
# dataset: python cpu-run/prep_alpaca_slice.py  (24 hand-written + 76 Alpaca)
# then, in cpu-venv (torch CPU, transformers, trl, peft, datasets):
python cpu/train_qwen06b_lora.py --max-steps 12
# results: results-0.6b.json, log: run-0.6b.log
```

### CPU demo — tiny Qwen3 (smoke test, ~1 minute)

```bash
# in cpu-venv (torch CPU, transformers, trl, datasets) with the tiny model
# and dataset already built (see cpu/train_tiny_trl.py header):
python cpu/train_tiny_trl.py
```

### GPU demo (needs a GPU)

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
- Base model (GPU demo): https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Base model (CPU demo): https://huggingface.co/Qwen/Qwen3-0.6B
- Dataset (GPU demo): https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- Dataset (CPU demo): https://huggingface.co/datasets/tatsu-lab/alpaca
