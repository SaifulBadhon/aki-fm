"""
check_measurement_density.py

Compares how frequently creatinine gets measured in MIMIC vs eICU,
using the GAP between consecutive draws (not just raw counts per stay,
which are confounded by stay-length differences between sources).

This matters because KDIGO detection can only catch a rise if two
measurements are close enough together in time to see it — sparser
measurement could make a hospital's AKI rate look artificially lower
without any real difference in kidney injury risk.

Run this on the server:
    conda activate aki-fm
    python3 check_measurement_density.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

TOKENS_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"


def load_all_creatinine_tokens() -> pd.DataFrame:
    paths = []
    mimic_path = TOKENS_DIR / "creatinine_tokens.parquet"
    if mimic_path.exists():
        paths.append(mimic_path)
    paths.extend(sorted(TOKENS_DIR.glob("eicu_creatinine_tokens_part*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def median_gap_per_stay(group: pd.DataFrame) -> float:
    times = np.sort(group["time_min"].to_numpy())
    if len(times) < 2:
        return np.nan
    gaps = np.diff(times)
    return np.median(gaps)


def main() -> None:
    df = load_all_creatinine_tokens()

    gaps = df.groupby("stay_id").apply(median_gap_per_stay, include_groups=False)
    gaps_df = gaps.reset_index()
    gaps_df.columns = ["stay_id", "median_gap_min"]
    gaps_df["source"] = gaps_df["stay_id"].apply(lambda s: "eicu" if str(s).startswith("eicu_") else "mimic")

    print("Median time gap between consecutive creatinine draws, by source:")
    print(gaps_df.groupby("source")["median_gap_min"].agg(["median", "mean", "count"]))
    print("\n(in hours, for readability:)")
    print((gaps_df.groupby("source")["median_gap_min"].agg(["median", "mean"]) / 60).round(1))


if __name__ == "__main__":
    main()
