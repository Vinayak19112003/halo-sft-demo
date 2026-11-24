# Halo version used for this demo

- Repository: https://github.com/whitecircle/halo
- Commit: 3a21b4dbdb56cb840d42a59e98e40f967a1d2807
- Date of commit: 2026-11-24 18:45:16 +0100
- Release: 1.3.0rc1 (pyproject.toml version)
- Stars at time of run: 881

The GPU demo uses Halo's SFT trainer (`scripts/training/sft.py`) with the config in
`configs/`, which is adapted from Halo's own
`examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` at the same commit.
That GPU run is queued on Kaggle; the completed CPU run in this repo used plain
TRL instead (see README), because Halo's SFT trainer needs a GPU.
