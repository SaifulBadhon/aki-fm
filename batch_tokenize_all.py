"""
batch_tokenize_all.py

Scales the per-patient tokenizer to the FULL cohort in one efficient
pass using DuckDB, instead of repeating tens of thousands of independent
filtered queries (which, even at a few seconds each, would still take
many hours run one at a time).

Run this on the server:
    conda activate aki-fm
    pip install duckdb
    python3 batch_tokenize_all.py

NOTE: this is a first pass at the SQL — DuckDB syntax quirks are common
on a first run. If something errors, paste the exact error back and
we'll fix it rather than guessing blind.
"""

import gzip
import json
from pathlib import Path

import duckdb
import pandas as pd

from concept_map import CONCEPT_MAP

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"
FLATTENED_DIR = Path.home() / "saiful" / "AKI" / "code" / "flattened"
OUTPUT_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def build_encounter_lookup() -> Path:
    """
    One pass over EncounterICU (small — tens of thousands of rows, not
    hundreds of millions) to build a lookup table of
    (stay_id, icu_uuid, parent_uuid, admission_start), saved as Parquet
    so DuckDB can join against it directly in the queries below.
    """
    print("Building encounter lookup table...")
    rows = []
    path = FHIR_DIR / "MimicEncounterICU.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            identifiers = record.get("identifier", [])
            stay_id = identifiers[0].get("value") if identifiers else None
            icu_uuid = record["id"]
            parent_ref = record.get("partOf", {}).get("reference")
            parent_uuid = parent_ref.split("/", 1)[1] if parent_ref else None
            admission_start = record.get("period", {}).get("start")
            if stay_id and admission_start:
                rows.append({
                    "stay_id": stay_id,
                    "icu_uuid": icu_uuid,
                    # labs are linked at the HOSPITAL encounter level (see
                    # tokenizer_fast.py's note) — fall back to icu_uuid only
                    # if there's genuinely no parent (shouldn't normally happen)
                    "parent_uuid": parent_uuid or icu_uuid,
                    "admission_start": admission_start,
                })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "encounter_lookup.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"  {len(df)} encounters written to {out_path}")
    return out_path


def main() -> None:
    lookup_path = build_encounter_lookup()
    con = duckdb.connect()

    creatinine_codes = CONCEPT_MAP["creatinine_serum"]["mimic_fhir"]["include_codes"]
    urine_codes = CONCEPT_MAP["urine_output"]["mimic_fhir"]["include_codes"]
    creat_list_sql = ",".join(repr(c) for c in creatinine_codes)
    urine_list_sql = ",".join(repr(c) for c in urine_codes)

    # BUG FIX: previously selected `value_numeric` only, which silently
    # dropped every text-valued observation (e.g. CAM-ICU delirium result
    # "Positive", device checks like "Patent") as NULL — these are real,
    # meaningful categorical values stored in `value_string` in the
    # flattened files, just never selected. COALESCE falls back to the
    # text value whenever the numeric one is absent.

    # Using epoch-seconds subtraction rather than date_diff, since the
    # timestamps carry explicit UTC offsets (e.g. -04:00/-05:00) that can
    # differ across a long stay (daylight saving transitions) — casting
    # to TIMESTAMPTZ and diffing epochs handles this correctly, matching
    # what Python's timezone-aware datetime subtraction already did in
    # tokenizer_fast.py.

    print("\nBuilding creatinine tokens for the full cohort...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                e.stay_id,
                'creatinine_serum' AS concept,
                COALESCE(CAST(l.value_numeric AS VARCHAR), l.value_string) AS value,
                l.value_unit AS unit,
                (epoch(l.time_str::TIMESTAMPTZ) - epoch(e.admission_start::TIMESTAMPTZ)) / 60.0 AS time_min,
                'labevents' AS source_table
            FROM read_parquet('{lookup_path}') e
            JOIN read_parquet('{FLATTENED_DIR}/labevents_part*.parquet') l
                ON l.encounter_uuid = e.parent_uuid
            WHERE l.item_code IN ({creat_list_sql})
        ) TO '{OUTPUT_DIR}/creatinine_tokens.parquet' (FORMAT PARQUET);
    """)
    print("  Done -> creatinine_tokens.parquet")

    print("\nBuilding urine output tokens for the full cohort...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                e.stay_id,
                'urine_output' AS concept,
                COALESCE(CAST(o.value_numeric AS VARCHAR), o.value_string) AS value,
                o.value_unit AS unit,
                (epoch(o.time_str::TIMESTAMPTZ) - epoch(e.admission_start::TIMESTAMPTZ)) / 60.0 AS time_min,
                'outputevents' AS source_table
            FROM read_parquet('{lookup_path}') e
            JOIN read_parquet('{FLATTENED_DIR}/outputevents_part*.parquet') o
                ON o.encounter_uuid = e.icu_uuid
            WHERE o.item_code IN ({urine_list_sql})
        ) TO '{OUTPUT_DIR}/urine_output_tokens.parquet' (FORMAT PARQUET);
    """)
    print("  Done -> urine_output_tokens.parquet")

    print("\nBuilding chartevents tokens for the full cohort (already category-scoped)...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                e.stay_id,
                c.item_display AS concept,
                COALESCE(CAST(c.value_numeric AS VARCHAR), c.value_string) AS value,
                c.value_unit AS unit,
                (epoch(c.time_str::TIMESTAMPTZ) - epoch(e.admission_start::TIMESTAMPTZ)) / 60.0 AS time_min,
                'chartevents' AS source_table
            FROM read_parquet('{lookup_path}') e
            JOIN read_parquet('{FLATTENED_DIR}/chartevents_part*.parquet') c
                ON c.encounter_uuid = e.icu_uuid
        ) TO '{OUTPUT_DIR}/chartevents_tokens.parquet' (FORMAT PARQUET);
    """)
    print("  Done -> chartevents_tokens.parquet")

    print("\nAll done. Full-cohort token tables are in:", OUTPUT_DIR)
    print("Each is keyed by stay_id — filtering to one patient from here is instant.")


if __name__ == "__main__":
    main()