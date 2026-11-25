#!/usr/bin/env python3
"""Real-model CPU SFT: Qwen3-0.6B (pretrained) + LoRA, plain TRL, NOT Halo.

Why TRL and not Halo: Halo's SFT trainer hardcodes a CUDA device in its
model loader (halo/model_loading/parallel/loader.py) and cannot run on this
CPU-only box (see cpu/halo-attempt.log in the repo). The Kaggle GPU Halo run
of the same demo is still queued. So this is the same demo's CPU track:
a REAL pretrained model, real tokenizer, real LoRA training — labelled
honestly as TRL-on-CPU, never claimed to be Halo.

Model:  Qwen/Qwen3-0.6B (pretrained, bf16 weights, ~0.6B params)
Data:   cpu-run/data/sft-0.6b.jsonl — 24 hand-written chat examples
        (tiny-chat.jsonl) plus, when available, a slice of the public
        tatsu-lab/alpaca dataset converted to chat format (see
        prep_alpaca_slice.py). Dataset file + row counts are logged.
Train:  LoRA r=8 alpha=16 on q/k/v/o/gate/up/down, batch 1 x grad-accum 2,
        lr 2e-4 cosine, warmup 2, max_length 256, assistant-only loss,
        gradient checkpointing, AdamW on LoRA params only (base frozen bf16).
Hardware: CPU only, 4 cores, ~7.7 GB RAM. Trial first: --max-steps 3 to
        measure step time and peak RAM; only then run the full 12-15 steps.
"""
import argparse
import json
import os
import resource
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.path.join(HERE, "qwen06b")  # local dir: curl-downloaded weights + Qwen3 tokenizer
DATA_PATH = os.path.join(HERE, "data", "sft-0.6b.jsonl")
FALLBACK_DATA = os.path.join(HERE, "data", "tiny-chat.jsonl")
OUT_DIR = os.path.join(HERE, "outputs", "qwen06b-lora")
LOG_PATH = os.path.join(HERE, "run-0.6b.log")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--results", default=os.path.join(HERE, "results-0.6b.json"))
    p.add_argument("--log", default=LOG_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_f = open(args.log, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    def peak_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    torch.set_num_threads(4)
    t0 = time.time()
    log(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
        f"cpu cores {os.cpu_count()} | start peak RSS {peak_gb():.2f} GB")
    log(f"model: {MODEL_ID} (pretrained, bf16) | data: {DATA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    log(f"tokenizer: {type(tokenizer).__name__} vocab={len(tokenizer)} "
        f"chat_template={'yes' if tokenizer.chat_template else 'no'}")

    data_path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_DATA
    rows = [json.loads(line) for line in open(data_path)]
    train_rows, eval_rows = rows[:-3], rows[-3:]
    log(f"dataset: {data_path} rows total={len(rows)} "
        f"train={len(train_rows)} eval={len(eval_rows)}")
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = 0
    log(f"model params: {n_params / 1e9:.2f}B | dtype: "
        f"{next(model.parameters()).dtype} | peak RSS after load "
        f"{peak_gb():.2f} GB")

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir=args.out,
        use_cpu=True,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_length=256,
        packing=False,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.max_steps,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=42,
    )
    log(f"LoRA r=8 alpha=16 | steps={args.max_steps} batch=1x2 lr={args.lr} "
        f"cosine warmup=2 max_length=256 grad-ckpt on optim=adamw_torch")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    n_trainable = sum(p.numel() for p in trainer.model.parameters()
                      if p.requires_grad)
    log(f"trainable (LoRA) params: {n_trainable / 1e6:.2f}M "
        f"({100 * n_trainable / n_params:.2f}% of base)")

    train_result = trainer.train()
    train_loss = train_result.training_loss
    log(f"TRAIN DONE in {time.time() - t0:.1f}s | train_loss(avg)={train_loss:.4f} "
        f"| global_step={train_result.global_step} | peak RSS {peak_gb():.2f} GB")
    log(f"log history: {json.dumps(trainer.state.log_history)}")

    eval_result = trainer.evaluate()
    log(f"EVAL: {json.dumps(eval_result)}")

    elapsed = time.time() - t0
    history = trainer.state.log_history
    step_losses = [h["loss"] for h in history if "loss" in h]
    results = {
        "runner": "TRL SFTTrainer + LoRA (CPU — NOT Halo; see README)",
        "halo_status": ("Halo SFT failed on CPU: loader hardcodes "
                        "device_map={'': <cuda rank>} "
                        "(halo/model_loading/parallel/loader.py); "
                        "Kaggle GPU Halo run still queued"),
        "model": "Qwen/Qwen3-0.6B (pretrained)",
        "params_b": round(n_params / 1e9, 2),
        "trainable_params_m": round(n_trainable / 1e6, 2),
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        "tokenizer": "Qwen/Qwen3-0.6B tokenizer (real)",
        "dataset": os.path.basename(data_path),
        "dataset_rows": {"total": len(rows), "train": len(train_rows),
                         "eval": len(eval_rows)},
        "steps": train_result.global_step,
        "batch_size": 1,
        "grad_accum": 2,
        "learning_rate": args.lr,
        "lr_scheduler": "cosine",
        "max_length": 256,
        "loss_first_step": step_losses[0] if step_losses else None,
        "loss_last_step": step_losses[-1] if step_losses else None,
        "train_loss_avg": train_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "elapsed_seconds": round(elapsed, 1),
        "peak_rss_gb": round(peak_gb(), 2),
        "hardware": {
            "device": "CPU",
            "cpu_cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "log_history": history,
    }
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    log(f"results written to {args.results} | total elapsed {elapsed:.1f}s")
    log_f.close()


if __name__ == "__main__":
    main()
