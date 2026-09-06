"""
build_training_example.py

Build ONE complete training example — static tokens, context tokens
(creatinine + urine output + vitals/chartevents, sorted by time), and
the discrete-time hazard label — for a single real patient, using the
now-validated cohort.parquet and token files.

This is the payoff of everything built so far: concept verification ->
tokenization -> labeling -> cohort filtering, all converging into the
actual format the model will train on.

Single-patient prototype, same pattern used throughout this project —
prove correctness on one real example before scaling to the full cohort.

Run this on the server:
    conda activate aki-fm
    python3 build_training_example.py --stay-id 32128372
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TOKENS_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"

HAZARD_BIN_HOURS = 1     # one hazard bin per hour
MAX_HORIZON_HOURS = 72   # how far ahead we bother building bins for


def load_tokens_for_stay(stay_id: str) -> list[dict]:
    """
    Pull every token (creatinine, urine output, vitals/chartevents) for
    one stay_id, from whichever source's files actually contain it.

    NOTE: loads each file fully then filters with plain pandas equality,
    rather than PyArrow's `filters=` predicate pushdown — the latter can
    silently return zero matches if there's any dtype mismatch between
    the filter value and how DuckDB encoded the column, with no error
    raised. This is fine for single-patient prototyping; a full-cohort
    batch version will need a proper join instead of per-patient re-reads.
    """
    is_eicu = stay_id.startswith("eicu_")
    tokens = []

    if is_eicu:
        sources = {
            "creatinine": sorted(TOKENS_DIR.glob("eicu_creatinine_tokens_part*.parquet")),
            "urine_output": sorted(TOKENS_DIR.glob("eicu_urine_output_tokens_part*.parquet")),
            "vitals": sorted(TOKENS_DIR.glob("eicu_vitals_tokens_part*.parquet")),
        }
    else:
        sources = {
            "creatinine": [TOKENS_DIR / "creatinine_tokens.parquet"],
            "urine_output": [TOKENS_DIR / "urine_output_tokens.parquet"],
            "chartevents": [TOKENS_DIR / "chartevents_tokens.parquet"],
        }

    for label, paths in sources.items():
        for path in paths:
            if not path.exists():
                print(f"    [MISSING] {path}")
                continue
            df = pd.read_parquet(path)
            matched = df[df["stay_id"] == stay_id]
            if len(matched) > 0:
                tokens.extend(matched.to_dict("records"))

    return tokens


def build_hazard_bins(aki_onset_time_min, max_time_min: float) -> list:
    """
    Discrete-time hazard bins, one per hour: 0 = no event yet, 1 = AKI
    onset in this bin. We stop generating bins once the event happens.
    NOTE: doesn't yet distinguish death/discharge censoring from AKI —
    that competing-risks piece is a documented future extension.
    """
    n_bins = min(int(max_time_min // (HAZARD_BIN_HOURS * 60)) + 1, MAX_HORIZON_HOURS)
    bins = []

    onset_bin = None
    if aki_onset_time_min is not None:
        onset_bin = int(aki_onset_time_min // (HAZARD_BIN_HOURS * 60))

    for i in range(n_bins):
        if onset_bin is not None and i == onset_bin:
            bins.append(1)
            break
        elif onset_bin is not None and i > onset_bin:
            break
        else:
            bins.append(0)

    return bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stay-id", required=True)
    args = parser.parse_args()
    stay_id = args.stay_id

    cohort = pd.read_parquet(TOKENS_DIR / "cohort.parquet")
    patient_row = cohort[cohort["stay_id"] == stay_id]
    if patient_row.empty:
        print(f"[NOT FOUND] stay_id={stay_id!r} not in cohort.parquet")
        return
    patient_row = patient_row.iloc[0]

    print(f"Building training example for stay_id={stay_id!r} (source={patient_row['source']})")
    print(f"  eligible={patient_row['eligible']}  age={patient_row['age_at_admission']}")

    static_tokens = [
        {"concept": "age", "value": patient_row["age_at_admission"], "type": "numeric"},
        {"concept": "source_hospital", "value": patient_row["source"], "type": "categorical"},
    ]

    context_tokens_raw = load_tokens_for_stay(stay_id)
    context_tokens = sorted(context_tokens_raw, key=lambda t: t["time_min"])
    print(f"  {len(context_tokens)} context tokens found")

    aki_onset = patient_row["aki_onset_time_min"]
    aki_onset = None if pd.isna(aki_onset) else float(aki_onset)
    max_time = max((t["time_min"] for t in context_tokens), default=0)
    hazard_bins = build_hazard_bins(aki_onset, max_time)

    training_example = {
        "stay_id": stay_id,
        "source": patient_row["source"],
        "static_tokens": static_tokens,
        "context_tokens": context_tokens,
        "label": {
            "hazard_bins_hourly": hazard_bins,
            "aki_onset_time_min": aki_onset,
            "event_type": "aki" if aki_onset is not None else "censored",
        },
    }

    out_path = Path(f"training_example_{stay_id}.json")
    with open(out_path, "w") as f:
        json.dump(training_example, f, indent=2, default=str)

    print(f"\nWritten to: {out_path.resolve()}")
    print(f"  static tokens: {len(static_tokens)}")
    print(f"  context tokens: {len(context_tokens)}")
    print(f"  hazard bins: {len(hazard_bins)}  (1 = AKI onset bin, 0 = no event yet)")
    print(f"  event_type: {training_example['label']['event_type']}")


if __name__ == "__main__":
    main()