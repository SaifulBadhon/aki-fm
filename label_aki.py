"""
label_aki.py

Apply KDIGO 2020 creatinine-based criteria to the full-cohort creatinine
tokens (from batch_tokenize_all.py) to determine AKI onset time for each
patient stay.

KDIGO criteria (creatinine-only — urine output is a documented future
extension, same limitation flagged in the source AKI paper):
  1. Rise >= 0.3 mg/dL within any trailing 48-hour window
  2. Rise to >= 1.5x baseline (lowest value in trailing 7 days)

AKI onset = the FIRST timestamp at which EITHER criterion is met — this
marks onset, not peak severity (severity/staging is a separate, later
calculation).

Run this on the server:
    conda activate aki-fm
    python3 label_aki.py
"""

from pathlib import Path

import pandas as pd

TOKENS_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"
OUTPUT_PATH = TOKENS_DIR / "aki_labels.parquet"

RISE_ABS_THRESHOLD = 0.3    # mg/dL
RISE_REL_THRESHOLD = 1.5    # x baseline
WINDOW_48H_MIN = 48 * 60    # minutes
WINDOW_7D_MIN = 7 * 24 * 60  # minutes


def load_all_creatinine_tokens() -> pd.DataFrame:
    """
    Load creatinine tokens from BOTH sources at once:
      - MIMIC: single file 'creatinine_tokens.parquet'
      - eICU:  multiple files 'eicu_creatinine_tokens_part*.parquet'
    Since eICU's stay_ids are already prefixed 'eicu_', there's no
    collision risk combining both into one DataFrame here.
    """
    paths = []
    mimic_path = TOKENS_DIR / "creatinine_tokens.parquet"
    if mimic_path.exists():
        paths.append(mimic_path)
    eicu_paths = sorted(TOKENS_DIR.glob("eicu_creatinine_tokens_part*.parquet"))
    paths.extend(eicu_paths)

    if not paths:
        raise FileNotFoundError(f"No creatinine token files found in {TOKENS_DIR}")

    print(
        f"Loading {len(paths)} creatinine file(s): "
        f"{'MIMIC' if mimic_path.exists() else ''} "
        f"{f'+ eICU ({len(eicu_paths)} parts)' if eicu_paths else ''}"
    )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def label_patient(group: pd.DataFrame) -> dict:
    """
    group: creatinine rows for ONE stay_id, with columns [time_min, value],
    NOT necessarily sorted or deduplicated yet.

    Returns the first onset time (if any), which criterion triggered it,
    and the value at that point.
    """
    group = group.sort_values("time_min").reset_index(drop=True)
    times = group["time_min"].to_numpy()
    values = group["value"].to_numpy()

    for i in range(len(values)):
        t_i, v_i = times[i], values[i]

        # Criterion 1: absolute rise >= 0.3 mg/dL within trailing 48h
        window_48h = (times >= t_i - WINDOW_48H_MIN) & (times <= t_i)
        min_48h = values[window_48h].min()
        if v_i - min_48h >= RISE_ABS_THRESHOLD:
            return {
                "aki_onset_time_min": t_i,
                "criterion": "48h_absolute_rise",
                "value_at_onset": v_i,
                "baseline_used": min_48h,
            }

        # Criterion 2: relative rise >= 1.5x baseline within trailing 7 days
        window_7d = (times >= t_i - WINDOW_7D_MIN) & (times <= t_i)
        baseline_7d = values[window_7d].min()
        if baseline_7d > 0 and v_i / baseline_7d >= RISE_REL_THRESHOLD:
            return {
                "aki_onset_time_min": t_i,
                "criterion": "7day_relative_rise",
                "value_at_onset": v_i,
                "baseline_used": baseline_7d,
            }

    return {
        "aki_onset_time_min": None,
        "criterion": None,
        "value_at_onset": None,
        "baseline_used": None,
    }


def main() -> None:
    df = load_all_creatinine_tokens()
    print(f"  {len(df):,} creatinine rows across {df['stay_id'].nunique():,} stays")

    # DEFENSIVE: creatinine 'value' should always be numeric, but now that
    # batch_tokenize_all.py's COALESCE fix correctly recovers text values
    # that used to be silently dropped as NULL, a rare non-numeric result
    # (e.g. a lab comment like "Hemolyzed, unable to process") could
    # surface and crash the arithmetic below. Coerce to numeric and drop
    # anything that fails — these can't be used for KDIGO comparison
    # regardless, and previously would have been NULL/excluded anyway.
    n_before = len(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [INFO] Dropped {n_dropped:,} non-numeric creatinine values "
              f"(e.g. lab comments) before labeling — these can't be used for KDIGO comparison.")

    results = []
    for stay_id, group in df.groupby("stay_id"):
        label = label_patient(group)
        label["stay_id"] = stay_id
        label["n_creatinine_values"] = len(group)
        results.append(label)

    labels_df = pd.DataFrame(results)
    labels_df["source"] = labels_df["stay_id"].apply(
        lambda s: "eicu" if str(s).startswith("eicu_") else "mimic"
    )
    labels_df.to_parquet(OUTPUT_PATH, index=False)

    n_total = len(labels_df)
    n_aki = labels_df["aki_onset_time_min"].notna().sum()
    n_eligible = (labels_df["n_creatinine_values"] >= 2).sum()

    print(f"\n{'=' * 60}")
    print(f"Total stays labeled:        {n_total:,}")
    print(f"Stays with >=2 creatinine:  {n_eligible:,} (cohort-eligible per source paper criteria)")
    print(f"AKI onset detected:         {n_aki:,} ({n_aki / n_total * 100:.1f}%)")
    print(f"No AKI detected:            {n_total - n_aki:,}")
    print(f"\nBreakdown by source:")
    print(labels_df.groupby("source").agg(
        n_stays=("stay_id", "count"),
        n_aki=("aki_onset_time_min", lambda x: x.notna().sum()),
    ))
    print(f"\nBreakdown by triggering criterion:")
    print(labels_df["criterion"].value_counts(dropna=False))
    print(f"\nLabels written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()