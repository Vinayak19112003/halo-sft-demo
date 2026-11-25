#!/usr/bin/env python3
"""Build sft-0.6b.jsonl: 24 hand-written chat examples (tiny-chat.jsonl)
plus a 76-row slice of the public tatsu-lab/alpaca dataset (streamed,
first 76 train rows, instruction+input -> user, output -> assistant).

If the alpaca download fails, the file is just the 24 hand-written rows
and the script says so — the training script falls back to tiny-chat.jsonl
only if sft-0.6b.jsonl is missing entirely.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TINY = os.path.join(HERE, "data", "tiny-chat.jsonl")
OUT = os.path.join(HERE, "data", "sft-0.6b.jsonl")

rows = [json.loads(line) for line in open(TINY)]
print(f"tiny rows: {len(rows)}")

try:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    added = 0
    for ex in ds:
        if added >= 76:
            break
        user = ex["instruction"].strip()
        if ex.get("input"):
            user += "\n" + ex["input"].strip()
        rows.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex["output"].strip()},
        ]})
        added += 1
    print(f"alpaca rows added: {added}")
except Exception as e:
    print(f"alpaca download failed ({e}); using tiny rows only")

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {OUT} with {len(rows)} rows")
