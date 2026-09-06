"""
tokenizer_prototype.py

First working tokenizer, scoped to ONE patient's ICU stay, using
concept_map.py's verified rules. Goal: prove the (concept, value, time,
source) token schema works correctly on a real patient BEFORE tackling
the harder engineering problem of doing this efficiently across
hundreds of millions of rows for the full cohort.

Run this on the server:
    conda activate aki-fm
    python3 tokenizer_prototype.py --stay-id 39553978
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

from concept_map import CONCEPT_MAP, CHARTEVENTS_CATEGORY_SCOPE

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"

MAX_LINES = None  # no cap — for single-patient validation we need the FULL file,
                  # since labevents/chartevents are almost certainly ordered
                  # chronologically across the whole cohort, not grouped by
                  # patient, so one patient's records could be anywhere.


def _should_stop(i: int) -> bool:
    return MAX_LINES is not None and i >= MAX_LINES


def _log_progress(i: int, every: int = 10_000_000) -> None:
    if i % every == 0 and i > 0:
        print(f"    ...scanned {i:,} lines")


def parse_fhir_datetime(dt_str: str) -> datetime:
    """FHIR datetimes look like '2140-01-07T20:01:00-05:00' — fromisoformat handles this directly."""
    return datetime.fromisoformat(dt_str)


def find_encounter(stay_id: str) -> tuple[str, datetime] | None:
    """
    Look up an ICU encounter by its human-readable stay_id (the value
    under `identifier`), and return its internal FHIR uuid + admission
    start time. Observations reference the uuid, not the stay_id, so
    this lookup is required before anything else.
    """
    path = FHIR_DIR / "MimicEncounterICU.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if _should_stop(i):
                break
            record = json.loads(line)
            identifiers = record.get("identifier", [])
            if any(ident.get("value") == stay_id for ident in identifiers):
                uuid = record["id"]
                start = parse_fhir_datetime(record["period"]["start"])
                return uuid, start
    return None


def minutes_since_admission(event_time: datetime, admission_start: datetime) -> float:
    return (event_time - admission_start).total_seconds() / 60


def extract_labevents_tokens(
    encounter_uuid: str, admission_start: datetime
) -> list[dict]:
    """Pull creatinine tokens for this encounter, using concept_map's verified codes."""
    tokens = []
    include_codes = set(CONCEPT_MAP["creatinine_serum"]["mimic_fhir"]["include_codes"])
    target_ref = f"Encounter/{encounter_uuid}"

    path = FHIR_DIR / "MimicObservationLabevents.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if _should_stop(i):
                break
            _log_progress(i)
            record = json.loads(line)
            if record.get("encounter", {}).get("reference") != target_ref:
                continue
            coding = record.get("code", {}).get("coding", [{}])[0]
            if coding.get("code") not in include_codes:
                continue
            # CRITICAL: use `issued` (result-available time), NOT
            # `effectiveDateTime` (specimen-draw time) — see concept_map.py's
            # leakage-risk note. Fall back to effectiveDateTime only if
            # issued is genuinely absent.
            time_str = record.get("issued") or record.get("effectiveDateTime")
            event_time = parse_fhir_datetime(time_str)
            tokens.append({
                "concept": "creatinine_serum",
                "value": record.get("valueQuantity", {}).get("value"),
                "unit": record.get("valueQuantity", {}).get("unit"),
                "time_min": minutes_since_admission(event_time, admission_start),
                "source_table": "labevents",
            })
    return tokens


def extract_outputevents_tokens(
    encounter_uuid: str, admission_start: datetime
) -> list[dict]:
    """Pull urine output tokens for this encounter, using concept_map's verified codes."""
    tokens = []
    include_codes = set(CONCEPT_MAP["urine_output"]["mimic_fhir"]["include_codes"])
    target_ref = f"Encounter/{encounter_uuid}"

    path = FHIR_DIR / "MimicObservationOutputevents.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if _should_stop(i):
                break
            _log_progress(i)
            record = json.loads(line)
            if record.get("encounter", {}).get("reference") != target_ref:
                continue
            coding = record.get("code", {}).get("coding", [{}])[0]
            if coding.get("code") not in include_codes:
                continue
            time_str = record.get("issued") or record.get("effectiveDateTime")
            event_time = parse_fhir_datetime(time_str)
            tokens.append({
                "concept": "urine_output",
                "value": record.get("valueQuantity", {}).get("value"),
                "unit": record.get("valueQuantity", {}).get("unit"),
                "time_min": minutes_since_admission(event_time, admission_start),
                "source_table": "outputevents",
            })
    return tokens


def extract_chartevents_tokens(
    encounter_uuid: str, admission_start: datetime
) -> list[dict]:
    """
    Pull chartevents tokens for this encounter, keeping only categories
    scoped 'include' in concept_map.py (real physiological signal, not
    nursing documentation/workflow noise).
    """
    tokens = []
    included_categories = set(CHARTEVENTS_CATEGORY_SCOPE["include"])
    target_ref = f"Encounter/{encounter_uuid}"

    path = FHIR_DIR / "MimicObservationChartevents.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if _should_stop(i):
                break
            _log_progress(i)
            record = json.loads(line)
            if record.get("encounter", {}).get("reference") != target_ref:
                continue
            category = record.get("category", [{}])[0]
            cat_coding = category.get("coding", [{}])[0]
            cat_display = cat_coding.get("code") or cat_coding.get("display")
            if cat_display not in included_categories:
                continue
            item_coding = record.get("code", {}).get("coding", [{}])[0]
            time_str = record.get("issued") or record.get("effectiveDateTime")
            event_time = parse_fhir_datetime(time_str)
            value = record.get("valueQuantity", {}).get("value", record.get("valueString"))
            unit = record.get("valueQuantity", {}).get("unit", "n/a")
            tokens.append({
                "concept": item_coding.get("display"),
                "value": value,
                "unit": unit,
                "time_min": minutes_since_admission(event_time, admission_start),
                "source_table": "chartevents",
            })
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stay-id", required=True, help="ICU stay_id, e.g. 39553978")
    args = parser.parse_args()

    print(f"Looking up encounter for stay_id={args.stay_id!r}...")
    result = find_encounter(args.stay_id)
    if result is None:
        print(f"[NOT FOUND] No encounter with stay_id={args.stay_id!r} in the full file.")
        return

    encounter_uuid, admission_start = result
    print(f"Found encounter uuid={encounter_uuid}  admission_start={admission_start}")

    print("\nExtracting creatinine tokens...")
    creatinine_tokens = extract_labevents_tokens(encounter_uuid, admission_start)
    print(f"  found {len(creatinine_tokens)} tokens")

    print("Extracting urine output tokens...")
    urine_tokens = extract_outputevents_tokens(encounter_uuid, admission_start)
    print(f"  found {len(urine_tokens)} tokens")

    print("Extracting chartevents tokens (scoped categories only)...")
    chartevents_tokens = extract_chartevents_tokens(encounter_uuid, admission_start)
    print(f"  found {len(chartevents_tokens)} tokens")

    all_tokens = creatinine_tokens + urine_tokens + chartevents_tokens
    all_tokens.sort(key=lambda t: t["time_min"])

    print(f"\n{'=' * 70}")
    print(f"Full token stream for stay_id={args.stay_id!r} ({len(all_tokens)} tokens)")
    print(f"{'=' * 70}")
    for token in all_tokens:
        print(
            f"  t={token['time_min']:>8.1f}min  "
            f"[{token['source_table']:<12}]  "
            f"{token['concept']:<30}  "
            f"{token['value']}  {token['unit']}"
        )


if __name__ == "__main__":
    main()