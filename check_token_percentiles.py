"""
check_token_percentiles.py

Median (546) and max (83,612) alone don't tell us where a sensible
max-context cutoff sits — need the actual distribution to make an
informed choice, rather than guessing a cap number.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 check_token_percentiles.py
"""

from pathlib import Path

import pandas as pd

TOKENS_DIR = Path("tokens_output")


def main() -> None:
    df = pd.read_parquet(TOKENS_DIR / "training_examples_downsampled.parquet")
    counts = df["n_tokens"]

    print("Token count distribution (post-downsampling):")
    for p in [50, 75, 90, 95, 99, 99.9, 100]:
        print(f"  p{p}: {counts.quantile(p / 100):.0f}")

    print(f"\nHow many patients would be TRUNCATED at various caps:")
    for cap in [1000, 2000, 3000, 5000, 10000]:
        n_truncated = (counts > cap).sum()
        pct = n_truncated / len(counts) * 100
        print(f"  cap={cap:>6,}: {n_truncated:,} patients truncated ({pct:.2f}% of cohort)")


if __name__ == "__main__":
    main()
