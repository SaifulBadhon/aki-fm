"""
find_long_stays.py

Quick utility: scan EncounterICU and list a handful of ICU stays longer
than a given threshold, so we have a more representative patient to
validate the tokenizer against (our first test patient's ~10-hour,
clinically stable stay was too short to exercise trend detection).

Run this on the server:
    conda activate aki-fm
    python3 find_long_stays.py --min-hours 72 --n 10
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-hours", type=float, default=72.0)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    path = FHIR_DIR / "MimicEncounterICU.ndjson.gz"
    found = 0

    print(f"Looking for ICU stays longer than {args.min_hours} hours...\n")

    with gzip.open(path, "rt") as f:
        for line in f:
            if found >= args.n:
                break
            record = json.loads(line)
            period = record.get("period", {})
            start_str, end_str = period.get("start"), period.get("end")
            if not start_str or not end_str:
                continue

            start = datetime.fromisoformat(start_str)
            end = datetime.fromisoformat(end_str)
            duration_hours = (end - start).total_seconds() / 3600

            if duration_hours >= args.min_hours:
                found += 1
                identifiers = record.get("identifier", [])
                stay_id = identifiers[0].get("value") if identifiers else "unknown"
                print(f"  stay_id={stay_id}  duration={duration_hours:.1f}h  start={start}")

    if found == 0:
        print(f"  No stays >= {args.min_hours}h found in the scan.")


if __name__ == "__main__":
    main()
