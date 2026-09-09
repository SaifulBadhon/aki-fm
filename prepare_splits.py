"""
prepare_splits.py

Creates a FIXED train/val/test split, stratified by (source, event_type)
so both hospitals and both AKI/censored outcomes are proportionally
represented in every split — not just randomly hoped for.

Saved once, reused by every future training run. This matters because
aki_model.py's plain random_split() would generate a DIFFERENT split
every time the script runs — meaning validation data could silently
leak into training across different sessions, and no two experiments
could be fairly compared against each other.

Run this ONCE:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 prepare_splits.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

TOKENS_DIR = Path("tokens_output")
SPLIT_PATH = TOKENS_DIR / "splits.parquet"

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # held out, untouched until final evaluation
RANDOM_SEED = 42


def main() -> None:
    print("Loading training_examples.parquet...")
    df = pd.read_parquet(TOKENS_DIR / "training_examples.parquet", columns=["stay_id", "source", "event_type"])
    print(f"  {len(df):,} rows (may include multiple examples per patient — deduplicating for splitting)")

    # CRITICAL: split at the PATIENT level, not the row level. One patient
    # can now contribute multiple examples (different prediction times) —
    # splitting naively on rows could scatter the same patient's examples
    # across train/val/test, which is patient-identity leakage: the model
    # could see part of a patient in training and be "evaluated" on a
    # different snapshot of that SAME patient, inflating val/test performance
    # in a way that won't generalize to genuinely unseen patients.
    patients = df.drop_duplicates(subset=["stay_id"])[["stay_id", "source", "event_type"]]
    print(f"  {len(patients):,} unique patients")

    rng = np.random.RandomState(RANDOM_SEED)
    patients = patients.copy()
    patients["strata"] = patients["source"] + "_" + patients["event_type"]

    splits = []
    for strata_val, group in patients.groupby("strata"):
        n = len(group)
        indices = rng.permutation(n)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        stay_ids = group["stay_id"].values
        train_ids = stay_ids[indices[:n_train]]
        val_ids = stay_ids[indices[n_train:n_train + n_val]]
        test_ids = stay_ids[indices[n_train + n_val:]]

        for sid in train_ids:
            splits.append({"stay_id": sid, "split": "train"})
        for sid in val_ids:
            splits.append({"stay_id": sid, "split": "val"})
        for sid in test_ids:
            splits.append({"stay_id": sid, "split": "test"})

    splits_df = pd.DataFrame(splits)
    splits_df.to_parquet(SPLIT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print("Split sizes (patients, not rows):")
    print(splits_df["split"].value_counts())

    print("\nStratification check — source distribution per split:")
    merged = splits_df.merge(patients[["stay_id", "source", "event_type"]], on="stay_id")
    print(pd.crosstab(merged["split"], merged["source"], normalize="index").round(3))

    print("\nStratification check — AKI rate per split (should be nearly identical across splits):")
    print(pd.crosstab(merged["split"], merged["event_type"], normalize="index").round(3))

    print("\nVerifying no patient appears in more than one split (sanity check)...")
    dup_check = splits_df.groupby("stay_id")["split"].nunique()
    n_bad = (dup_check > 1).sum()
    print(f"  Patients appearing in multiple splits: {n_bad} (must be 0)")

    print(f"\nSplits written to: {SPLIT_PATH}")
    print("NOTE: this file maps stay_id -> split. AKIDataset's split filter already")
    print("works on stay_id, so ALL of a patient's examples correctly inherit one split.")


if __name__ == "__main__":
    main()