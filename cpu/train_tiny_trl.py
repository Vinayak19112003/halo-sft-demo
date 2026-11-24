#!/usr/bin/env python3
"""CPU fallback SFT run (plain TRL, NOT Halo).

Why this file exists: Halo's SFT trainer (scripts/training/sft.py) cannot run
on this CPU-only box. It got as far as loading the dataset (24 rows ->
21 train / 3 test) and probing the model config, then failed in its model
loader, which hardcodes device_map={"": <cuda rank>} in
halo/model_loading/parallel/loader.py (_load_undistributed_model), and
transformers raises "Cannot access accelerator device when none is
available". Halo is a GPU framework (its own docs: run everything in the
Docker image); the Kaggle GPU run of the same demo is still queued.
So this script runs the SAME tiny model and SAME 24 examples through
TRL's SFTTrainer on CPU, and the repo README labels it honestly as a
CPU fallback run — it is not claimed to be a Halo run.

Model:  random-init tiny Qwen3 (~20M params, hidden 128, 2 layers),
         cpu-run/tiny-qwen3 (real Qwen3 tokenizer)
Data:    cpu-run/data/tiny-chat.jsonl, 24 chat examples (21 train / 3 eval)
Train:   12 steps, batch 2, lr 5e-4 cosine, warmup 2, max_length 256,
         fp32, loss on assistant tokens only
Hardware: CPU (see results.json for core/RAM counts)
"""
import json
import os
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "tiny-qwen3")
DATA_PATH = os.path.join(HERE, "data", "tiny-chat.jsonl")
OUT_DIR = os.path.join(HERE, "outputs", "tiny-sft-trl")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(HERE, "run-trl.log")
log_f = open(LOG_PATH, "w")


def log(msg):
    print(msg, flush=True)
    log_f.write(msg + "\n")
    log_f.flush()


t0 = time.time()
log(f"torch {torch.__version__} | cpu cores (os.cpu_count) {os.cpu_count()} | "
    f"threads {torch.get_num_threads()}")
log(f"model: {MODEL_DIR}")
log(f"data: {DATA_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
log(f"tokenizer: {type(tokenizer).__name__} vocab={len(tokenizer)} "
    f"chat_template={'yes' if tokenizer.chat_template else 'no'}")

rows = [json.loads(line) for line in open(DATA_PATH)]
train_rows, eval_rows = rows[:21], rows[21:]
log(f"dataset rows: total={len(rows)} train={len(train_rows)} eval={len(eval_rows)}")
train_ds = Dataset.from_list(train_rows)
eval_ds = Dataset.from_list(eval_rows)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
n_params = sum(p.numel() for p in model.parameters())
log(f"model params: {n_params / 1e6:.1f}M | dtype: {next(model.parameters()).dtype}")

config = SFTConfig(
    output_dir=OUT_DIR,
    use_cpu=True,
    bf16=False,
    fp16=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=12,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    max_length=256,
    packing=False,
    assistant_only_loss=True,
    logging_steps=1,
    logging_first_step=True,
    eval_strategy="steps",
    eval_steps=12,
    save_strategy="no",
    report_to=[],
    dataloader_num_workers=0,
    seed=42,
)
log(f"steps=12 batch=2 lr=5e-4 cosine warmup=2 max_length=256 "
    f"assistant_only_loss=True fp32 CPU")

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

train_result = trainer.train()
train_loss = train_result.training_loss
log(f"TRAIN DONE in {time.time() - t0:.1f}s | train_loss(avg)={train_loss:.4f} "
    f"| global_step={train_result.global_step}")
log(f"log history: {json.dumps(trainer.state.log_history)}")

eval_result = trainer.evaluate()
log(f"EVAL: {json.dumps(eval_result)}")

elapsed = time.time() - t0
history = trainer.state.log_history
step_losses = [h["loss"] for h in history if "loss" in h]
results = {
    "runner": "TRL SFTTrainer (CPU fallback — NOT Halo; see README)",
    "halo_status": ("Halo SFT failed on CPU: loader hardcodes "
                    "device_map={'': <cuda rank>} (halo/model_loading/parallel/loader.py); "
                    "Kaggle GPU Halo run still queued"),
    "model": "random-init tiny Qwen3, hidden 128, 2 layers",
    "params_m": round(n_params / 1e6, 1),
    "tokenizer": "Qwen/Qwen3-0.6B tokenizer (real)",
    "dataset": "cpu/data/tiny-chat.jsonl",
    "dataset_rows": {"total": len(rows), "train": len(train_rows), "eval": len(eval_rows)},
    "steps": train_result.global_step,
    "batch_size": 2,
    "learning_rate": 5e-4,
    "lr_scheduler": "cosine",
    "max_length": 256,
    "loss_first_step": step_losses[0] if step_losses else None,
    "loss_last_step": step_losses[-1] if step_losses else None,
    "train_loss_avg": train_loss,
    "eval_loss": eval_result.get("eval_loss"),
    "elapsed_seconds": round(elapsed, 1),
    "hardware": {
        "device": "CPU",
        "cpu_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    },
    "log_history": history,
}
with open(os.path.join(HERE, "results-trl.json"), "w") as f:
    json.dump(results, f, indent=2)
log(f"results written to results-trl.json | total elapsed {elapsed:.1f}s")
log_f.close()
