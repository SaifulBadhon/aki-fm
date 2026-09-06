"""
flatten_to_parquet.py

ONE-TIME preprocessing pass: converts labevents, outputevents, and
chartevents (category-scoped) from nested FHIR JSON into flat, columnar
Parquet files — one row per event, with encounter_uuid as a queryable
column.

Why this exists: tokenizer_prototype.py proved the token schema is
correct, but a full unbounded per-patient scan took 20+ minutes for ONE
patient. Scaling that to 30,000+ patients would take roughly two years.
The fix: scan each huge file ONCE (not once per patient), flatten it,
and store it in a format that supports fast filtered queries afterward.

This pass itself will also take a while (it reads the full file) — but
it's a ONE-TIME cost, not a per-patient cost. That's the entire point.

Run this on the server:
    conda activate aki-fm
    pip install pyarrow
    python3 flatten_to_parquet.py
"""

import gzip
import json
from pathlib import Path

import pandas as pd

from concept_map import CHARTEVENTS_CATEGORY_SCOPE

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"

OUTPUT_DIR = Path.home() / "saiful" / "AKI" / "code" / "flattened"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FLUSH_EVERY = 2_000_000  # write to disk every N records, to keep memory bounded


def _encounter_uuid(record: dict) -> str | None:
    """Extract just the uuid from an 'Encounter/<uuid>' reference string."""
    ref = record.get("encounter", {}).get("reference")
    if ref and ref.startswith("Encounter/"):
        return ref.split("/", 1)[1]
    return None


def flatten_observation_file(
    path: Path,
    out_prefix: str,
    category_filter: set[str] | None = None,
) -> None:
    """
    Stream one FHIR Observation ndjson.gz file ONCE, extract the flat
    fields the tokenizer needs, and write to Parquet in chunks so memory
    stays bounded even for hundreds of millions of records.

    If `category_filter` is given (used for chartevents), only records
    whose category is in it are kept — dropping nursing-documentation
    noise HERE, once, rather than re-filtering it on every future query.
    """
    print(f"\nFlattening {path.name}...")
    buffer: list[dict] = []
    part_num = 0

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i % 10_000_000 == 0 and i > 0:
                print(f"    ...processed {i:,} lines, written {part_num} part(s) so far")

            record = json.loads(line)
            encounter_uuid = _encounter_uuid(record)
            if encounter_uuid is None:
                continue

            if category_filter is not None:
                category = record.get("category", [{}])[0]
                cat_coding = category.get("coding", [{}])[0]
                cat_display = cat_coding.get("code") or cat_coding.get("display")
                if cat_display not in category_filter:
                    continue
            else:
                cat_display = None

            coding = record.get("code", {}).get("coding", [{}])[0]
            time_str = record.get("issued") or record.get("effectiveDateTime")
            if time_str is None:
                continue

            value_qty = record.get("valueQuantity", {})
            buffer.append({
                "encounter_uuid": encounter_uuid,
                "item_code": coding.get("code"),
                "item_display": coding.get("display"),
                "category": cat_display,
                "value_numeric": value_qty.get("value"),
                "value_unit": value_qty.get("unit"),
                "value_string": record.get("valueString"),
                "time_str": time_str,
            })

            if len(buffer) >= FLUSH_EVERY:
                _flush(buffer, out_prefix, part_num)
                part_num += 1
                buffer = []

    if buffer:
        _flush(buffer, out_prefix, part_num)
        part_num += 1

    print(f"  Done. Wrote {part_num} Parquet part(s) for {out_prefix}.")


def _flush(buffer: list[dict], out_prefix: str, part_num: int) -> None:
    df = pd.DataFrame(buffer)
    out_path = OUTPUT_DIR / f"{out_prefix}_part{part_num:04d}.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)


def main() -> None:
    flatten_observation_file(
        FHIR_DIR / "MimicObservationLabevents.ndjson.gz",
        out_prefix="labevents",
    )
    flatten_observation_file(
        FHIR_DIR / "MimicObservationOutputevents.ndjson.gz",
        out_prefix="outputevents",
    )
    flatten_observation_file(
        FHIR_DIR / "MimicObservationChartevents.ndjson.gz",
        out_prefix="chartevents",
        category_filter=set(CHARTEVENTS_CATEGORY_SCOPE["include"]),
    )

    print("\n\nAll done. Flattened Parquet files are in:", OUTPUT_DIR)
    print("Next step: query these filtered by encounter_uuid — this should")
    print("take seconds, not 20+ minutes, per patient.")


if __name__ == "__main__":
    main()
