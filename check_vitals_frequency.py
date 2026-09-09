"""
check_vitals_frequency.py

The vocabulary scan found 129 MILLION heartrate tokens across the
cohort — before trusting that number, check whether it reflects a
genuine, clinically plausible sampling interval (eICU vitalPeriodic is
automated bedside-monitor data, likely denser than manually-charted
vitals) or a remaining duplication bug in flatten_eicu.py's wide-to-long
melt step.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 check_vitals_frequency.py
"""

import json
from pathlib import Path

import pandas as pd

TOKENS_DIR = Path("tokens_output")


def main() -> None:
    df = pd.read_parquet(TOKENS_DIR / "training_examples.parquet")
    eicu_df = df[df["source"] == "eicu"]
    print(f"Checking {min(5, len(eicu_df))} real eICU patients...\n")

    for _, row in eicu_df.head(5).iterrows():
        tokens = json.loads(row["context_tokens"])
        hr_tokens = [t for t in tokens if t["concept"] == "heartrate"]
        if len(hr_tokens) < 2:
            continue

        times = sorted(t["time_min"] for t in hr_tokens)
        span_min = times[-1] - times[0]
        span_hours = span_min / 60
        avg_interval_min = span_min / (len(hr_tokens) - 1) if len(hr_tokens) > 1 else float("nan")

        # check for exact-duplicate (concept, value, time) triples specifically
        hr_keys = [(t["value"], t["time_min"]) for t in hr_tokens]
        n_unique = len(set(hr_keys))

        print(f"stay_id={row['stay_id']}")
        print(f"  heartrate readings: {len(hr_tokens)}  (unique value+time pairs: {n_unique})")
        print(f"  time span: {span_hours:.1f} hours")
        print(f"  average interval between readings: {avg_interval_min:.2f} minutes")
        print(f"  -> plausible if this is roughly 1-5 min (automated monitor); "
              f"suspicious if much less than 1 min or if unique < total\n")


if __name__ == "__main__":
    main()
