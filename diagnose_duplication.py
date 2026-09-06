"""
diagnose_duplication.py

Combined diagnostic for the token-count inflation bug (8,713 tokens
found instead of the expected 236 for stay_id=32128372). Checks every
possible source of duplication in one run: the encounter lookup table,
each raw token file, and the final training example's token breakdown.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 diagnose_duplication.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

TOKENS_DIR = Path("tokens_output")
STAY_ID = "32128372"


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def main() -> None:
    section("1. cohort.parquet — is this patient duplicated at the top level?")
    cohort = pd.read_parquet(TOKENS_DIR / "cohort.parquet")
    match = cohort[cohort["stay_id"] == STAY_ID]
    print(f"Rows matching stay_id: {len(match)} (expect exactly 1)")
    if len(match) > 0:
        print(match[["stay_id", "source", "n_creatinine_values", "aki_onset_time_min"]].to_string())

    section("2. encounter_lookup.parquet — leading hypothesis: duplicate encounter rows")
    lookup_path = TOKENS_DIR / "encounter_lookup.parquet"
    if lookup_path.exists():
        lookup = pd.read_parquet(lookup_path)
        match = lookup[lookup["stay_id"] == STAY_ID]
        print(f"Rows matching stay_id: {len(match)} (expect exactly 1)")
        print(match.to_string())
    else:
        print(f"[MISSING] {lookup_path}")

    section("3. Raw token files — row counts and duplicate check per source")
    for filename in ["creatinine_tokens.parquet", "urine_output_tokens.parquet", "chartevents_tokens.parquet"]:
        path = TOKENS_DIR / filename
        if not path.exists():
            print(f"[MISSING] {path}")
            continue
        df = pd.read_parquet(path)
        match = df[df["stay_id"] == STAY_ID]
        n_total = len(match)
        n_unique = len(match.drop_duplicates())
        print(f"{filename}: {n_total} rows for this stay_id, {n_unique} unique after dropping exact duplicates")

    section("4. grouped_tokens_checkpoint.parquet — duplication introduced at the DuckDB grouping step?")
    checkpoint_path = TOKENS_DIR / "grouped_tokens_checkpoint.parquet"
    if checkpoint_path.exists():
        checkpoint = pd.read_parquet(checkpoint_path)
        match = checkpoint[checkpoint["stay_id"] == STAY_ID]
        print(f"Rows matching stay_id in checkpoint: {len(match)} (expect exactly 1)")
        if len(match) > 0:
            tokens = json.loads(match.iloc[0]["context_tokens_json"])
            print(f"Token count in checkpoint: {len(tokens)}")
    else:
        print(f"[MISSING] {checkpoint_path}")

    section("5. training_examples.parquet — final token breakdown by source_table")
    final_path = TOKENS_DIR / "training_examples.parquet"
    final = pd.read_parquet(final_path)
    match = final[final["stay_id"] == STAY_ID]
    print(f"Rows matching stay_id in final output: {len(match)} (expect exactly 1)")
    if len(match) > 0:
        row = match.iloc[0]
        tokens = json.loads(row["context_tokens"])
        counts = Counter(t["source_table"] for t in tokens)
        print(f"Total tokens: {len(tokens)}")
        print(f"Breakdown by source_table: {dict(counts)}")

        # check for exact duplicate tokens within the list itself
        token_keys = [(t["concept"], t["value"], t["time_min"], t["source_table"]) for t in tokens]
        n_unique_tokens = len(set(token_keys))
        print(f"Unique tokens (by concept+value+time+source): {n_unique_tokens} "
              f"({'DUPLICATES FOUND' if n_unique_tokens < len(tokens) else 'no exact duplicates'})")


    section("6. RAW flattened files (pre-join) — does duplication already exist here?")
    lookup = pd.read_parquet(TOKENS_DIR / "encounter_lookup.parquet")
    match = lookup[lookup["stay_id"] == STAY_ID]
    if len(match) > 0:
        icu_uuid = match.iloc[0]["icu_uuid"]
        parent_uuid = match.iloc[0]["parent_uuid"]
        print(f"icu_uuid={icu_uuid}  parent_uuid={parent_uuid}")

        flattened_dir = Path.home() / "saiful" / "AKI" / "code" / "flattened"
        for prefix, uuid_to_check in [
            ("chartevents", icu_uuid),
            ("outputevents", icu_uuid),
            ("labevents", parent_uuid),
        ]:
            part_files = sorted(flattened_dir.glob(f"{prefix}_part*.parquet"))
            print(f"\n{prefix}: {len(part_files)} part file(s) found in {flattened_dir}")
            matches = []
            for pf in part_files:
                df = pd.read_parquet(pf)
                m = df[df["encounter_uuid"] == uuid_to_check]
                if len(m) > 0:
                    matches.append(m)
                    print(f"  {pf.name}: {len(m)} matching rows")
            if matches:
                combined = pd.concat(matches, ignore_index=True)
                print(f"  TOTAL raw rows across all parts: {len(combined)}, "
                      f"unique: {len(combined.drop_duplicates())}")
            else:
                print("  No matching rows found in any part file.")


if __name__ == "__main__":
    main()