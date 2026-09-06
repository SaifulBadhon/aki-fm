"""
batch_build_training_examples.py

Full-cohort version of build_training_example.py. Uses DuckDB to group
EVERY patient's tokens in one bulk pass (same fix that solved
batch_tokenize_all.py's version of this problem) instead of re-reading
huge Parquet files once per patient, which would take roughly as long
as the original 20-minutes-per-patient problem, just relocated here.

Output: one Parquet file, `training_examples.parquet`, with one row
per ELIGIBLE patient — stay_id, source, and static_tokens/context_tokens/
hazard_bins_hourly all stored as JSON-ENCODED STRINGS (not native nested
Parquet types) — this deliberately avoids a known PyArrow limitation
where nested list-of-struct columns can fail to convert to pandas
("Nested data conversions not implemented for chunked array outputs").
Downstream code should `json.loads()` these columns when reading them.

IMPORTANT: if you previously ran an older version of this script, delete
the stale checkpoint before rerunning — it was written in the old
(broken) nested-struct format, and the skip-if-exists logic will
otherwise silently reuse it:
    rm ~/saiful/AKI/code/tokens_output/grouped_tokens_checkpoint.parquet

Run this on the server:
    conda activate aki-fm
    python3 batch_build_training_examples.py
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

TOKENS_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"
OUTPUT_PATH = TOKENS_DIR / "training_examples.parquet"

HAZARD_BIN_HOURS = 1
MAX_HORIZON_HOURS = 72


def _is_valid_checkpoint(path: Path) -> bool:
    """A checkpoint is only trustworthy if it exists AND has real content —
    an interrupted/crashed run (e.g. the earlier OOM) can leave a 0-byte
    file behind, which `.exists()` alone would wrongly treat as complete."""
    return path.exists() and path.stat().st_size > 0


def _run_grouping_query(con, sources: list, checkpoint_path: Path, eligible_cte: str) -> None:
    """Run one GROUP BY over the given list of source file/glob patterns, writing to its own checkpoint."""
    union_parts = []
    for src in sources:
        union_parts.append(f"""
            SELECT stay_id, concept, CAST(value AS VARCHAR) AS value, unit, time_min, source_table
            FROM read_parquet('{src}')
            WHERE stay_id IN (SELECT stay_id FROM eligible_stays)
        """)
    union_sql = " UNION ALL ".join(union_parts)

    query = f"""
        COPY (
            WITH eligible_stays AS ({eligible_cte}),
            all_tokens AS ({union_sql})
            SELECT
                stay_id,
                to_json(list(struct_pack(
                    concept := concept, value := value, unit := unit,
                    time_min := time_min, source_table := source_table
                ) ORDER BY time_min)) AS context_tokens_json,
                count(*) AS n_tokens,
                max(time_min) AS max_time_min
            FROM all_tokens
            GROUP BY stay_id
        ) TO '{checkpoint_path}' (FORMAT PARQUET);
    """
    con.execute(query)


def build_grouped_tokens_checkpoint() -> Path:
    """
    Two separate DuckDB passes — one per hospital — instead of one giant
    UNION ALL + GROUP BY across both sources at once. The combined query
    exhausted a 300GB memory ceiling on a 503GB-RAM server; splitting by
    hospital (which are already stored in completely separate files)
    roughly halves the peak working set per query, and is combined with
    DuckDB's own suggested tuning (disabling insertion-order preservation,
    capping threads) to further reduce memory pressure.
    """
    final_checkpoint = TOKENS_DIR / "grouped_tokens_checkpoint.parquet"
    if _is_valid_checkpoint(final_checkpoint):
        print(f"Found existing checkpoint at {final_checkpoint} — skipping recomputation.")
        return final_checkpoint

    mimic_checkpoint = TOKENS_DIR / "grouped_tokens_mimic.parquet"
    eicu_checkpoint = TOKENS_DIR / "grouped_tokens_eicu.parquet"
    cohort_path = str(TOKENS_DIR / "cohort.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='300GB'")
    con.execute(f"PRAGMA temp_directory='{TOKENS_DIR}/duckdb_tmp'")
    # DuckDB's own suggested fixes from the OutOfMemoryException:
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("PRAGMA threads=8")

    if not _is_valid_checkpoint(mimic_checkpoint):
        print("Grouping MIMIC tokens for eligible patients (pass 1 of 2)...")
        mimic_sources = [
            str(TOKENS_DIR / "creatinine_tokens.parquet"),
            str(TOKENS_DIR / "urine_output_tokens.parquet"),
            str(TOKENS_DIR / "chartevents_tokens.parquet"),
        ]
        eligible_cte = f"SELECT stay_id FROM read_parquet('{cohort_path}') WHERE eligible = true AND source = 'mimic'"
        _run_grouping_query(con, mimic_sources, mimic_checkpoint, eligible_cte)
        print(f"  MIMIC checkpoint written to {mimic_checkpoint}")
    else:
        print(f"Found existing MIMIC checkpoint — skipping.")

    if not _is_valid_checkpoint(eicu_checkpoint):
        print("Grouping eICU tokens for eligible patients (pass 2 of 2)...")
        eicu_sources = [
            str(TOKENS_DIR / "eicu_creatinine_tokens_part*.parquet"),
            str(TOKENS_DIR / "eicu_urine_output_tokens_part*.parquet"),
            str(TOKENS_DIR / "eicu_vitals_tokens_part*.parquet"),
        ]
        eligible_cte = f"SELECT stay_id FROM read_parquet('{cohort_path}') WHERE eligible = true AND source = 'eicu'"
        _run_grouping_query(con, eicu_sources, eicu_checkpoint, eligible_cte)
        print(f"  eICU checkpoint written to {eicu_checkpoint}")
    else:
        print(f"Found existing eICU checkpoint — skipping.")

    print("Combining both checkpoints into one...")
    combined = pd.concat([
        pd.read_parquet(mimic_checkpoint),
        pd.read_parquet(eicu_checkpoint),
    ], ignore_index=True)
    combined.to_parquet(final_checkpoint, index=False)
    print(f"  Combined checkpoint written to {final_checkpoint}")
    return final_checkpoint


def build_hazard_bins(aki_onset_time_min, max_time_min: float) -> list:
    n_bins = min(int(max_time_min // (HAZARD_BIN_HOURS * 60)) + 1, MAX_HORIZON_HOURS)
    bins = []
    onset_bin = None
    if aki_onset_time_min is not None and not pd.isna(aki_onset_time_min):
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
    checkpoint_path = build_grouped_tokens_checkpoint()

    print("\nLoading grouped tokens checkpoint (already filtered to eligible patients)...")
    # context_tokens_json is a plain string column now (JSON-encoded), so
    # this is a completely ordinary read — no nested-type conversion issues.
    grouped = pd.read_parquet(checkpoint_path)
    print(f"  {len(grouped):,} eligible patients with tokens")

    print("\nLoading cohort for labels...")
    cohort = pd.read_parquet(TOKENS_DIR / "cohort.parquet")
    eligible_cohort = cohort[cohort["eligible"]].copy()

    print("\nJoining tokens with eligible cohort...")
    merged = eligible_cohort.merge(grouped, on="stay_id", how="inner")
    print(f"  {len(merged):,} eligible patients matched with token data")
    dropped = len(eligible_cohort) - len(merged)
    if dropped > 0:
        print(f"  [WARNING] {dropped:,} eligible patients had NO matching tokens — investigate before trusting the final count")

    print("\nComputing hazard bins per patient...")
    # Serialized as JSON strings (not raw Python lists) — the same nested-type
    # issue that hit context_tokens would otherwise resurface here when this
    # final file gets read back later.
    merged["hazard_bins_hourly"] = merged.apply(
        lambda row: json.dumps(build_hazard_bins(row["aki_onset_time_min"], row["max_time_min"])),
        axis=1,
    )
    merged["event_type"] = merged["aki_onset_time_min"].apply(
        lambda x: "aki" if pd.notna(x) else "censored"
    )
    merged["static_tokens"] = merged.apply(
        lambda row: json.dumps([
            {"concept": "age", "value": row["age_at_admission"], "type": "numeric"},
            {"concept": "source_hospital", "value": row["source"], "type": "categorical"},
        ]),
        axis=1,
    )
    merged = merged.rename(columns={"context_tokens_json": "context_tokens"})

    output_cols = [
        "stay_id", "source", "static_tokens", "context_tokens",
        "hazard_bins_hourly", "aki_onset_time_min", "event_type", "n_tokens",
    ]
    final = merged[output_cols]
    final.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"Total training examples written: {len(final):,}")
    print(f"  AKI:      {(final['event_type'] == 'aki').sum():,}")
    print(f"  Censored: {(final['event_type'] == 'censored').sum():,}")
    print(f"  By source:")
    print(final.groupby("source")["event_type"].value_counts())
    print(f"  Median tokens per example: {final['n_tokens'].median():.0f}")
    print(f"\nWritten to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()