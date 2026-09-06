"""
flatten_eicu.py

eICU equivalent of flatten_to_parquet.py — converts eICU's flat CSV
tables into the same (stay_id, concept, value, unit, time_min,
source_table) token schema MIMIC already uses, so both sources can
eventually feed the same downstream pipeline.

Key differences from MIMIC handled here:
  - Time is ALREADY relative (minutes from ICU admission) via offset
    columns — no admission_start subtraction needed.
  - vitalPeriodic.csv is WIDE format (one row = all vitals at one time)
    and must be melted into long format to match chartevents' shape.
  - celllabel matching for urine output needs normalization
    (.strip().lower()) per concept_map.py's documented traps.
  - stay_id is prefixed with "eicu_" to avoid collision with MIMIC's
    plain-integer stay_ids once the two sources are combined later.

Run this on the server:
    conda activate aki-fm
    python3 flatten_eicu.py
"""

from pathlib import Path

import pandas as pd

from concept_map import CONCEPT_MAP

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
EICU_DIR = DATASETS_ROOT / "eicu"
OUTPUT_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CHUNKSIZE = 200_000


def find_file(filename: str) -> Path | None:
    matches = list(EICU_DIR.rglob(filename))
    return matches[0] if matches else None


def flatten_creatinine() -> None:
    """lab.csv -> creatinine tokens, using concept_map's verified labname match.
    Writes one Parquet part per chunk immediately — if the process crashes,
    everything written so far is preserved instead of lost."""
    print("\nFlattening eICU creatinine (lab.csv)...")
    path = find_file("lab.csv.gz")
    if path is None:
        print("  [MISSING] lab.csv.gz not found")
        return

    include_values = set(CONCEPT_MAP["creatinine_serum"]["eicu"]["include_values"])
    exclude_values = set(CONCEPT_MAP["creatinine_serum"]["eicu"]["exclude_values"])

    part_num = 0
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False):
        norm = chunk["labname"].astype(str).str.strip().str.lower()
        mask = norm.isin(include_values) & ~norm.isin(exclude_values)
        matched = chunk[mask]
        if len(matched) == 0:
            continue

        out = pd.DataFrame({
            "stay_id": "eicu_" + matched["patientunitstayid"].astype(str),
            "concept": "creatinine_serum",
            "value": matched["labresult"],
            "unit": matched["labmeasurenamesystem"],
            "time_min": matched["labresultoffset"],
            "source_table": "lab",
        })
        out_path = OUTPUT_DIR / f"eicu_creatinine_tokens_part{part_num:04d}.parquet"
        out.to_parquet(out_path, index=False)
        part_num += 1
        if part_num % 5 == 0:
            print(f"    ...written {part_num} parts so far")

    print(f"  Done. Wrote {part_num} creatinine part(s).")


def flatten_urine_output() -> None:
    """intakeOutput.csv -> urine output tokens, using concept_map's verified
    include/exclude rules. Writes one Parquet part per chunk immediately."""
    print("\nFlattening eICU urine output (intakeOutput.csv)...")
    path = find_file("intakeOutput.csv.gz")
    if path is None:
        print("  [MISSING] intakeOutput.csv.gz not found")
        return

    cfg = CONCEPT_MAP["urine_output"]["eicu"]
    include_terms = cfg["include_if_contains"]
    exclude_normalized = set(cfg["exclude_values_normalized"])

    part_num = 0
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False):
        norm = chunk["celllabel"].astype(str).str.strip().str.lower()
        include_mask = norm.apply(lambda v: any(term in v for term in include_terms))
        exclude_mask = norm.isin(exclude_normalized)
        matched = chunk[include_mask & ~exclude_mask]
        if len(matched) == 0:
            continue

        out = pd.DataFrame({
            "stay_id": "eicu_" + matched["patientunitstayid"].astype(str),
            "concept": "urine_output",
            "value": matched["cellvaluenumeric"],
            "unit": "ml",
            "time_min": matched["intakeoutputoffset"],
            "source_table": "intakeOutput",
        })
        out_path = OUTPUT_DIR / f"eicu_urine_output_tokens_part{part_num:04d}.parquet"
        out.to_parquet(out_path, index=False)
        part_num += 1
        if part_num % 5 == 0:
            print(f"    ...written {part_num} parts so far")

    print(f"  Done. Wrote {part_num} urine output part(s).")


def flatten_vitals() -> None:
    """
    vitalPeriodic.csv -> melted long-format vitals tokens.
    This is WIDE format (one row = all vitals at one time) — must melt
    into one row per (vital, time) pair to match chartevents' shape.
    """
    print("\nFlattening eICU vitals (vitalPeriodic.csv, melting wide->long)...")
    path = find_file("vitalPeriodic.csv.gz")
    if path is None:
        print("  [MISSING] vitalPeriodic.csv.gz not found")
        return

    id_cols = ["patientunitstayid", "observationoffset"]
    # all other columns are individual vital signs to melt
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    value_cols = [c for c in header_cols if c not in id_cols and c != "vitalperiodicid"]

    part_num = 0
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False):
        melted = chunk.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name="concept",
            value_name="value",
        )
        melted = melted.dropna(subset=["value"])  # wide format is mostly NaN per row
        melted["stay_id"] = "eicu_" + melted["patientunitstayid"].astype(str)
        melted["unit"] = None  # eICU vitalPeriodic has no per-cell unit column
        melted["source_table"] = "vitalPeriodic"
        melted = melted.rename(columns={"observationoffset": "time_min"})
        melted = melted[["stay_id", "concept", "value", "unit", "time_min", "source_table"]]

        out_path = OUTPUT_DIR / f"eicu_vitals_tokens_part{part_num:04d}.parquet"
        melted.to_parquet(out_path, index=False)
        part_num += 1
        if part_num % 5 == 0:
            print(f"    ...written {part_num} parts so far")

    print(f"  Done. Wrote {part_num} vitals part(s).")


def main() -> None:
    existing_creat_parts = list(OUTPUT_DIR.glob("eicu_creatinine_tokens_part*.parquet"))
    existing_urine_parts = list(OUTPUT_DIR.glob("eicu_urine_output_tokens_part*.parquet"))
    existing_vitals_parts = list(OUTPUT_DIR.glob("eicu_vitals_tokens_part*.parquet"))

    if existing_creat_parts:
        print(f"Found {len(existing_creat_parts)} existing creatinine part(s) — skipping (delete them to redo).")
    else:
        flatten_creatinine()

    if existing_urine_parts:
        print(f"Found {len(existing_urine_parts)} existing urine output part(s) — skipping (delete them to redo).")
    else:
        flatten_urine_output()

    if existing_vitals_parts:
        print(f"Found {len(existing_vitals_parts)} existing vitals part(s) — skipping (delete them to redo).")
    else:
        flatten_vitals()

    print("\n\nAll done. eICU tokens are in:", OUTPUT_DIR)
    print("Note: stay_ids are prefixed 'eicu_' to avoid collision with MIMIC's plain-integer stay_ids.")


if __name__ == "__main__":
    main()