#!/usr/bin/env python3
"""
Download HuggingFace datasets and convert to sft-label compatible format.

Usage examples:
    # Download full dataset (raw parquet files)
    python scripts/download_hf_dataset.py open-r1/Mixture-of-Thoughts \
        --output /Volumes/MOVESPEED/datasets/open-source-sft/

    # Download specific subset, convert to sft-label JSON
    python scripts/download_hf_dataset.py open-r1/Mixture-of-Thoughts \
        --subset code --convert --output ./data/

    # Sample N items for quick testing
    python scripts/download_hf_dataset.py open-r1/Mixture-of-Thoughts \
        --subset math --convert --sample 500 --output ./data/

    # Download with HF token for gated datasets
    HF_TOKEN=hf_xxx python scripts/download_hf_dataset.py meta-llama/...

Requires: pip install huggingface_hub datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Format conversion: HF messages → ShareGPT conversations
# ─────────────────────────────────────────────────────────

ROLE_MAP = {
    "user": "human",
    "human": "human",
    "assistant": "gpt",
    "gpt": "gpt",
    "system": "system",
    "tool": "tool",
}


def hf_to_sharegpt(sample: dict) -> dict | None:
    """Convert a HuggingFace messages-format sample to ShareGPT format.

    Handles both:
      - messages format: {"messages": [{"role": ..., "content": ...}]}
      - conversations format: {"conversations": [{"from": ..., "value": ...}]}

    Returns None if the sample has no convertible conversation data.
    """
    conversations = []

    if "messages" in sample:
        for msg in sample["messages"]:
            role = ROLE_MAP.get(msg.get("role", ""), msg.get("role", ""))
            content = msg.get("content", "")
            conversations.append({"from": role, "value": content})
    elif "conversations" in sample:
        # Already ShareGPT-like, normalize role names
        for turn in sample["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            role = ROLE_MAP.get(role, role)
            content = turn.get("value", turn.get("content", ""))
            conversations.append({"from": role, "value": content})
    else:
        return None

    if not conversations:
        return None

    result = {"conversations": conversations}

    # Preserve useful metadata
    for key in ("source", "num_tokens", "id", "dataset", "category"):
        if key in sample:
            result[key] = sample[key]

    return result


# ─────────────────────────────────────────────────────────
# Download: raw snapshot via huggingface_hub
# ─────────────────────────────────────────────────────────

def download_raw(repo_id: str, output_dir: Path, subset: str | None = None,
                  token: str | None = None) -> Path:
    """Download dataset repository as-is (parquet files etc).

    If subset is given, only download files under that subdirectory
    (e.g. subset="code" downloads only code/**).
    """
    from huggingface_hub import snapshot_download

    local_dir = output_dir / repo_id.replace("/", "__")
    label = f"{repo_id}/{subset}" if subset else repo_id
    print(f"Downloading {label} → {local_dir}")

    kwargs = {}
    if subset:
        # Match both subdirectory files (subset/*) and flat files (subset.jsonl, subset.parquet)
        kwargs["allow_patterns"] = [f"{subset}/*", f"{subset}/**", f"{subset}.*", "README*"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
        **kwargs,
    )
    print(f"Done. Files saved to {local_dir}")
    return local_dir


# ─────────────────────────────────────────────────────────
# Convert: load with `datasets` and export as JSON
# ─────────────────────────────────────────────────────────

def download_and_convert(
    repo_id: str,
    output_dir: Path,
    subset: str | None = None,
    split: str = "train",
    sample_n: int | None = None,
    token: str | None = None,
    seed: int = 42,
) -> Path:
    """Load dataset with HF `datasets`, convert to sft-label JSON."""
    from datasets import load_dataset

    print(f"Loading {repo_id}" + (f" (subset={subset})" if subset else ""))
    args = [repo_id]
    if subset:
        args.append(subset)
    ds = load_dataset(*args, split=split, token=token)

    if sample_n and sample_n < len(ds):
        print(f"Sampling {sample_n} / {len(ds)} items (seed={seed})")
        ds = ds.shuffle(seed=seed).select(range(sample_n))

    print(f"Converting {len(ds)} samples to ShareGPT format...")
    converted = []
    skipped = 0
    for item in ds:
        result = hf_to_sharegpt(item)
        if result:
            converted.append(result)
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} samples (no conversation data)")

    # Build output filename
    safe_name = repo_id.replace("/", "__")
    parts = [safe_name]
    if subset:
        parts.append(subset)
    if sample_n:
        parts.append(f"sample{sample_n}")
    out_file = output_dir / (f"{'_'.join(parts)}.json")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=None)
        f.write("\n")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Saved {len(converted)} samples → {out_file} ({size_mb:.1f} MB)")
    return out_file


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets for sft-label testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("repo_id", help="HuggingFace dataset ID (e.g. open-r1/Mixture-of-Thoughts)")
    parser.add_argument("--output", "-o", type=Path, default=Path("./data"),
                        help="Output directory (default: ./data)")
    parser.add_argument("--subset", "-s", help="Dataset subset/config (e.g. code, math, science)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--convert", "-c", action="store_true",
                        help="Convert to sft-label ShareGPT JSON (requires `datasets` library)")
    parser.add_argument("--sample", "-n", type=int, default=None,
                        help="Sample N items (only with --convert)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")

    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    if args.convert:
        download_and_convert(
            repo_id=args.repo_id,
            output_dir=args.output,
            subset=args.subset,
            split=args.split,
            sample_n=args.sample,
            token=token,
            seed=args.seed,
        )
    else:
        download_raw(
            repo_id=args.repo_id,
            output_dir=args.output,
            subset=args.subset,
            token=token,
        )


if __name__ == "__main__":
    main()
