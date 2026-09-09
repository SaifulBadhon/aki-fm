"""
downsample_vitals.py

vitalPeriodic tokens (eICU's automated bedside monitor data, sampled
every ~5 minutes) dominate token counts for eICU patients — one real
batch just showed a sequence length of 26,417 tokens, driven almost
entirely by this. Everything else in a batch gets padded to match,
which is both wasteful and risks exceeding GPU memory on an unlucky batch.

This aggregates vitalPeriodic tokens into coarser time bins (default:
30 minutes) using the mean value per (concept, bin) — clinically
reasonable, since a 30-minute heart-rate average loses little that
matters for trend detection while cutting token volume ~6x relative to
the raw 5-minute stream. All other token types (creatinine, urine
output, chartevents) are left untouched.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 downsample_vitals.py
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

TOKENS_DIR = Path("tokens_output")
INPUT_PATH = TOKENS_DIR / "training_examples.parquet"
OUTPUT_PATH = TOKENS_DIR / "training_examples_downsampled.parquet"

BIN_SIZE_MIN = 30  # aggregate vitalPeriodic into 30-minute bins


def downsample_patient_tokens(context_tokens_json: str) -> tuple[str, int, int]:
    """
    Returns (new_context_tokens_json, original_token_count, new_token_count).
    """
    tokens = json.loads(context_tokens_json)
    original_count = len(tokens)

    vitals_tokens = [t for t in tokens if t["source_table"] == "vitalPeriodic"]
    other_tokens = [t for t in tokens if t["source_table"] != "vitalPeriodic"]

    if not vitals_tokens:
        return context_tokens_json, original_count, original_count

    # Group by (concept, time_bin), average numeric values within each bin.
    # (vitalPeriodic tokens are always numeric — no categorical handling needed here.)
    bins: dict[tuple, list] = defaultdict(list)
    for t in vitals_tokens:
        bin_start = (float(t["time_min"]) // BIN_SIZE_MIN) * BIN_SIZE_MIN
        key = (t["concept"], bin_start)
        try:
            bins[key].append(float(t["value"]))
        except (ValueError, TypeError):
            continue  # skip any unparseable value defensively

    downsampled_vitals = []
    for (concept, bin_start), values in bins.items():
        downsampled_vitals.append({
            "concept": concept,
            "value": sum(values) / len(values),
            "unit": None,
            "time_min": bin_start,
            "source_table": "vitalPeriodic",
        })

    combined = other_tokens + downsampled_vitals
    combined.sort(key=lambda t: t["time_min"])

    return json.dumps(combined), original_count, len(combined)


def main() -> None:
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  {len(df):,} patients")

    print(f"\nDownsampling vitalPeriodic tokens into {BIN_SIZE_MIN}-minute bins...")
    original_counts = []
    new_counts = []
    new_context_tokens = []

    for i, row in enumerate(df.itertuples()):
        if i % 20000 == 0 and i > 0:
            print(f"  ...processed {i:,} patients")
        new_json, orig_n, new_n = downsample_patient_tokens(row.context_tokens)
        new_context_tokens.append(new_json)
        original_counts.append(orig_n)
        new_counts.append(new_n)

    df["context_tokens"] = new_context_tokens
    df["n_tokens"] = new_counts

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"Median tokens BEFORE downsampling: {pd.Series(original_counts).median():.0f}")
    print(f"Median tokens AFTER downsampling:  {pd.Series(new_counts).median():.0f}")
    print(f"Max tokens BEFORE: {max(original_counts):,}")
    print(f"Max tokens AFTER:  {max(new_counts):,}")
    print(f"\nWritten to: {OUTPUT_PATH}")
    print("Review the before/after numbers, then rename this to replace")
    print("training_examples.parquet once you're satisfied with the result.")


if __name__ == "__main__":
    main()
