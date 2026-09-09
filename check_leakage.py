"""
check_leakage.py

Updated for the prediction-time-based dataset (training_examples_v2,
now swapped in as training_examples.parquet). The relevant invariant
changed: context should never look past `prediction_time_min` (not
admission, and not directly compared to onset anymore) — and for AKI
examples specifically, the real onset must be STRICTLY AFTER the
prediction time, confirming the example genuinely asks the model to
predict something that hasn't happened yet.

Two checks:
  1. context_tokens' max time_min must be <= prediction_time_min, for
     EVERY example (AKI or censored) — this is the core no-future-data rule.
  2. For AKI examples, aki_onset_time_min must be > prediction_time_min.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 check_leakage.py
"""

import json
from pathlib import Path

import pandas as pd

TOKENS_DIR = Path("tokens_output")
N_SAMPLES = 40


def main() -> None:
    df = pd.read_parquet(TOKENS_DIR / "training_examples.parquet")
    if "prediction_time_min" not in df.columns:
        print("[ERROR] 'prediction_time_min' column not found — are you sure you swapped in "
              "the v2 (prediction-time-based) file as training_examples.parquet?")
        return

    sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=42)

    n_context_leaked = 0
    n_onset_leaked = 0
    n_aki_checked = 0

    for _, row in sample.iterrows():
        tokens = json.loads(row["context_tokens"])
        pred_time = row["prediction_time_min"]

        if tokens:
            max_context_time = max(t["time_min"] for t in tokens)
            context_leaked = max_context_time > pred_time + 1e-6  # small tolerance for float rounding
        else:
            max_context_time = None
            context_leaked = False

        if context_leaked:
            n_context_leaked += 1

        onset_leaked = False
        onset = row["aki_onset_time_min"]
        if row["event_type"] == "aki" and pd.notna(onset):
            n_aki_checked += 1
            onset_leaked = onset <= pred_time
            if onset_leaked:
                n_onset_leaked += 1

        flag = ""
        if context_leaked:
            flag += " *** CONTEXT LEAKED ***"
        if onset_leaked:
            flag += " *** ONSET NOT IN FUTURE ***"

        print(f"stay_id={row['stay_id']}  event_type={row['event_type']:<9}  "
              f"pred_time={pred_time:.1f}min  max_context_time={max_context_time}  "
              f"onset={onset}{flag}")

    print(f"\n{'=' * 60}")
    print(f"Context leakage (context extends past prediction time): {n_context_leaked} / {len(sample)}")
    print(f"Onset leakage (AKI onset not strictly after prediction time): {n_onset_leaked} / {n_aki_checked} AKI examples checked")
    print("\nBoth should be 0 if the fix worked correctly.")


if __name__ == "__main__":
    main()