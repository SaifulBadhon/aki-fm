"""
tokenizer_fast.py

Same token schema as tokenizer_prototype.py, but reads from the flattened
Parquet files instead of re-scanning the raw multi-hundred-million-line
gzip files. This is the payoff of flatten_to_parquet.py's one-time cost —
per-patient lookups should now take seconds, not 20+ minutes.

Run this on the server:
    conda activate aki-fm
    python3 tokenizer_fast.py --stay-id 39553978
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

import pyarrow.dataset as ds

from concept_map import CONCEPT_MAP

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"
FLATTENED_DIR = Path.home() / "saiful" / "AKI" / "code" / "flattened"


def parse_fhir_datetime(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str)


def minutes_since_admission(event_time: datetime, admission_start: datetime) -> float:
    return (event_time - admission_start).total_seconds() / 60


def find_encounter(stay_id: str) -> tuple[str, str | None, datetime] | None:
    """
    EncounterICU is small — no need to flatten this one, a direct scan
    is already fast. Returns (icu_uuid, parent_hospital_uuid, admission_start).

    IMPORTANT: MIMIC-IV's underlying schema links labevents at the
    HOSPITAL admission level (hadm_id), not the ICU-stay level
    (icustay_id) — labs are drawn throughout a hospitalization regardless
    of which unit the patient is in. chartevents/outputevents ARE
    ICU-specific (bedside monitoring only happens in the ICU). So labs
    need to be queried against the PARENT hospital encounter, found via
    this ICU encounter's `partOf` reference — not the ICU encounter itself.
    """
    path = FHIR_DIR / "MimicEncounterICU.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            identifiers = record.get("identifier", [])
            if any(ident.get("value") == stay_id for ident in identifiers):
                icu_uuid = record["id"]
                parent_ref = record.get("partOf", {}).get("reference")
                parent_uuid = parent_ref.split("/", 1)[1] if parent_ref else None
                start = parse_fhir_datetime(record["period"]["start"])
                return icu_uuid, parent_uuid, start
    return None


def query_parquet_for_encounter(prefix: str, encounter_uuid: str):
    """
    Read only the rows matching this encounter from the flattened Parquet
    files for one source table (labevents / outputevents / chartevents).
    Uses pyarrow's filter pushdown so this stays fast even across dozens
    of Parquet part files.
    """
    part_files = sorted(FLATTENED_DIR.glob(f"{prefix}_part*.parquet"))
    if not part_files:
        print(f"  [MISSING] No flattened Parquet parts found for prefix={prefix!r}")
        return []

    dataset = ds.dataset([str(p) for p in part_files], format="parquet")
    table = dataset.to_table(filter=ds.field("encounter_uuid") == encounter_uuid)
    return table.to_pylist()


def build_tokens_from_rows(
    rows: list[dict],
    admission_start: datetime,
    source_table: str,
    include_codes: set[str] | None,
    concept_name_override: str | None,
) -> list[dict]:
    """
    Convert flattened Parquet rows into the standard token schema.
    If `include_codes` is given, only keep rows matching those item codes
    (used for labevents/outputevents, which weren't pre-filtered).
    Chartevents rows were already category-filtered at flatten time, so
    no code filtering is needed there.
    """
    tokens = []
    for row in rows:
        if include_codes is not None and row["item_code"] not in include_codes:
            continue
        event_time = parse_fhir_datetime(row["time_str"])
        tokens.append({
            "concept": concept_name_override or row["item_display"],
            "value": row["value_numeric"] if row["value_numeric"] is not None else row["value_string"],
            "unit": row["value_unit"],
            "time_min": minutes_since_admission(event_time, admission_start),
            "source_table": source_table,
        })
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stay-id", required=True, help="ICU stay_id, e.g. 39553978")
    args = parser.parse_args()

    print(f"Looking up encounter for stay_id={args.stay_id!r}...")
    result = find_encounter(args.stay_id)
    if result is None:
        print(f"[NOT FOUND] No encounter with stay_id={args.stay_id!r}.")
        return

    icu_uuid, parent_uuid, admission_start = result
    print(f"Found ICU encounter uuid={icu_uuid}  parent hospital uuid={parent_uuid}  admission_start={admission_start}")

    print("\nQuerying flattened labevents (using PARENT hospital encounter)...")
    lab_query_uuid = parent_uuid or icu_uuid
    lab_rows = query_parquet_for_encounter("labevents", lab_query_uuid)
    creatinine_codes = set(CONCEPT_MAP["creatinine_serum"]["mimic_fhir"]["include_codes"])
    creatinine_tokens = build_tokens_from_rows(
        lab_rows, admission_start, "labevents", creatinine_codes, "creatinine_serum"
    )
    print(f"  {len(lab_rows)} raw rows -> {len(creatinine_tokens)} creatinine tokens")

    print("Querying flattened outputevents (ICU-specific, using ICU encounter)...")
    output_rows = query_parquet_for_encounter("outputevents", icu_uuid)
    urine_codes = set(CONCEPT_MAP["urine_output"]["mimic_fhir"]["include_codes"])
    urine_tokens = build_tokens_from_rows(
        output_rows, admission_start, "outputevents", urine_codes, "urine_output"
    )
    print(f"  {len(output_rows)} raw rows -> {len(urine_tokens)} urine output tokens")

    print("Querying flattened chartevents (ICU-specific, already category-scoped)...")
    chart_rows = query_parquet_for_encounter("chartevents", icu_uuid)
    chartevents_tokens = build_tokens_from_rows(
        chart_rows, admission_start, "chartevents", None, None
    )
    print(f"  {len(chartevents_tokens)} chartevents tokens")

    all_tokens = creatinine_tokens + urine_tokens + chartevents_tokens
    all_tokens.sort(key=lambda t: t["time_min"])

    out_path = Path(f"tokens_{args.stay_id}.txt")
    with open(out_path, "w") as f:
        f.write(f"Full token stream for stay_id={args.stay_id!r} ({len(all_tokens)} tokens)\n")
        f.write("=" * 70 + "\n")
        for token in all_tokens:
            f.write(
                f"  t={token['time_min']:>8.1f}min  "
                f"[{token['source_table']:<12}]  "
                f"{str(token['concept'])[:30]:<30}  "
                f"{token['value']}  {token['unit']}\n"
            )

    print(f"\n{'=' * 70}")
    print("SUMMARY (this is what matters — the detailed list is safely in a file)")
    print(f"{'=' * 70}")
    print(f"  creatinine tokens:  {len(creatinine_tokens)}")
    print(f"  urine output tokens: {len(urine_tokens)}")
    print(f"  chartevents tokens: {len(chartevents_tokens)}")
    print(f"  TOTAL tokens:       {len(all_tokens)}")
    print(f"\nFull detailed token list written to: {out_path.resolve()}")
    print(f"View it with: less {out_path}   or   grep creatinine {out_path}")


if __name__ == "__main__":
    main()