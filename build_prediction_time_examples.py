"""
build_prediction_time_examples.py

Fixes the confirmed label leakage in training_examples.parquet: that
file gave the model each patient's ENTIRE history as context, paired
with one hazard curve — meaning for AKI patients, context routinely
included post-onset/recovery data (20/20 sampled patients confirmed
leaked). This rebuilds examples around explicit PREDICTION TIMES:

  - Sample multiple prediction times per patient through their stay
    (this also multiplies effective training data).
  - Context = only tokens with time_min <= prediction_time.
  - Hazard label = bins measured RELATIVE TO the prediction time, not
    admission — stopping at the real onset if it occurs after t.
  - AKI patients with onset <= 0 (already present at ICU admission,
    not an incident/new-onset case) are EXCLUDED — no valid
    "before it happened" prediction window exists for them.
  - For AKI patients, only prediction times BEFORE the real onset are
    sampled (sampling after onset would just recreate the same leak).

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 build_prediction_time_examples.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TOKENS_DIR = Path("tokens_output")
OUTPUT_PATH = TOKENS_DIR / "training_examples_v2.parquet"

MAX_CONTEXT_TOKENS = 5000
MAX_HORIZON_HOURS = 72
N_PREDICTION_TIMES_PER_PATIENT = 4  # sample this many prediction points per patient
MIN_CONTEXT_HOURS = 2  # don't sample a prediction time with less than this much prior data
RANDOM_SEED = 42


def build_hazard_bins_relative(aki_onset_time_min, pred_time_min: float, max_time_min: float) -> list:
    """
    Same hazard-bin logic as before, but now measured RELATIVE TO
    pred_time_min, not admission. Bin i = [pred_time + i*60, pred_time + (i+1)*60) minutes.
    """
    horizon_min = max_time_min - pred_time_min
    n_bins = min(int(horizon_min // 60) + 1, MAX_HORIZON_HOURS)
    bins = []

    onset_bin = None
    if aki_onset_time_min is not None and not pd.isna(aki_onset_time_min):
        relative_onset = aki_onset_time_min - pred_time_min
        if relative_onset >= 0:
            onset_bin = int(relative_onset // 60)

    for i in range(n_bins):
        if onset_bin is not None and i == onset_bin:
            bins.append(1)
            break
        elif onset_bin is not None and i > onset_bin:
            break
        else:
            bins.append(0)
    return bins


def sample_prediction_times(rng, onset_time_min, max_time_min: float, n_samples: int) -> list:
    """
    Sample prediction times for one patient. For AKI patients, only
    samples BEFORE the real onset. For censored patients, samples
    anywhere with at least MIN_CONTEXT_HOURS of prior data and at
    least a little room left for a meaningful future window.
    """
    min_context_min = MIN_CONTEXT_HOURS * 60

    if onset_time_min is not None and not pd.isna(onset_time_min):
        upper_bound = onset_time_min  # never sample AT or AFTER real onset
    else:
        upper_bound = max_time_min - 60  # leave at least 1 hour of future window for censored patients

    if upper_bound <= min_context_min:
        return []  # no valid window exists for this patient

    return list(rng.uniform(min_context_min, upper_bound, size=n_samples))


def truncate_to_recent(tokens: list, max_tokens: int) -> list:
    if len(tokens) <= max_tokens:
        return tokens
    return sorted(tokens, key=lambda t: t["time_min"])[-max_tokens:]


def main() -> None:
    rng = np.random.RandomState(RANDOM_SEED)

    print("Loading cohort and existing training examples (for their full token histories)...")
    cohort = pd.read_parquet(TOKENS_DIR / "cohort.parquet")
    old_examples = pd.read_parquet(TOKENS_DIR / "training_examples.parquet")
    print(f"  {len(old_examples):,} patients available")

    n_excluded_prevalent = 0
    new_rows = []

    for i, row in enumerate(old_examples.itertuples()):
        if i % 20000 == 0 and i > 0:
            print(f"  ...processed {i:,} patients, {len(new_rows):,} examples built so far")

        onset = row.aki_onset_time_min
        if onset is not None and not pd.isna(onset) and onset <= 0:
            n_excluded_prevalent += 1
            continue  # AKI already present at ICU admission — not an incident case

        all_tokens = json.loads(row.context_tokens)
        if not all_tokens:
            continue
        max_time = max(t["time_min"] for t in all_tokens)

        pred_times = sample_prediction_times(rng, onset, max_time, N_PREDICTION_TIMES_PER_PATIENT)

        for pred_time in pred_times:
            context = [t for t in all_tokens if t["time_min"] <= pred_time]
            if not context:
                continue
            context = truncate_to_recent(context, MAX_CONTEXT_TOKENS)

            hazard_bins = build_hazard_bins_relative(onset, pred_time, max_time)
            if not hazard_bins:
                continue

            event_type = "aki" if (onset is not None and not pd.isna(onset)) else "censored"

            new_rows.append({
                "stay_id": row.stay_id,
                "source": row.source,
                "prediction_time_min": pred_time,
                "static_tokens": row.static_tokens,
                "context_tokens": json.dumps(context),
                "hazard_bins_hourly": json.dumps(hazard_bins),
                "aki_onset_time_min": onset,
                "event_type": event_type,
                "n_tokens": len(context),
            })

    new_df = pd.DataFrame(new_rows)
    new_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"Excluded (AKI already present at admission): {n_excluded_prevalent:,}")
    print(f"Total NEW training examples: {len(new_df):,} (from {len(old_examples):,} patients, "
          f"~{len(new_df) / max(len(old_examples) - n_excluded_prevalent, 1):.1f} examples/patient)")
    print(f"  AKI examples: {(new_df['event_type'] == 'aki').sum():,}")
    print(f"  Censored examples: {(new_df['event_type'] == 'censored').sum():,}")
    print(f"\nWritten to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
