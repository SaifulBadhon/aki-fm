"""
build_cohort.py

Apply cohort inclusion/exclusion criteria to the FULL combined dataset
(MIMIC + eICU), producing one unified eligible cohort with a `source`
column so per-hospital breakdowns and leave-one-hospital-out validation
are straightforward later.

Inclusion:
  - age > 18 at admission
  - >= 2 creatinine measurements, spanning >= MIN_CREATININE_GAP_HOURS
    (a real data-sufficiency requirement, replacing the arbitrary >48h
    ICU-stay-length rule from the source paper — see prior discussion)

Exclusion:
  - first creatinine value > 4.0 mg/dL (~353.5 umol/L) — suggests
    pre-existing severe renal impairment at baseline

`is_long_stay` (>48h) is still computed and reported per source, purely
for a stratified sensitivity comparison against the source paper's
protocol — it is NOT a filter.

NOTE on eICU age: "89+" is eICU's own de-identification convention for
patients over 89 (not a MIMIC-style birthDate shift) — parsed as 90 here
for filtering purposes only; don't use this script's age numbers for
anything beyond the >18 threshold check.

Dialysis/transplant exclusion (from the source paper) is still DEFERRED
for both sources — needs a procedure-code lookup not yet built.

Run this on the server:
    conda activate aki-fm
    python3 build_cohort.py
"""

import gzip
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"
EICU_DIR = DATASETS_ROOT / "eicu"
TOKENS_DIR = Path.home() / "saiful" / "AKI" / "code" / "tokens_output"
OUTPUT_PATH = TOKENS_DIR / "cohort.parquet"

MIN_AGE = 18
MIN_CREATININE_GAP_HOURS = 2
LONG_STAY_THRESHOLD_HOURS = 48  # REPORTING only, not a filter
MAX_BASELINE_CREATININE = 4.0   # mg/dL


def find_eicu_file(filename: str) -> Path | None:
    matches = list(EICU_DIR.rglob(filename))
    return matches[0] if matches else None


def build_encounter_details_mimic() -> pd.DataFrame:
    """One pass over MIMIC's EncounterICU + Patient for stay duration + age."""
    print("Scanning MIMIC EncounterICU for stay durations + patient links...")
    rows = []
    path = FHIR_DIR / "MimicEncounterICU.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            identifiers = record.get("identifier", [])
            stay_id = identifiers[0].get("value") if identifiers else None
            period = record.get("period", {})
            start_str, end_str = period.get("start"), period.get("end")
            subject_ref = record.get("subject", {}).get("reference")
            patient_id = subject_ref.split("/", 1)[1] if subject_ref else None
            if stay_id and start_str and end_str:
                start = datetime.fromisoformat(start_str)
                end = datetime.fromisoformat(end_str)
                rows.append({
                    "stay_id": stay_id,
                    "patient_id": patient_id,
                    "admission_year": start.year,
                    "stay_hours": (end - start).total_seconds() / 3600,
                })
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} MIMIC encounters scanned")

    print("Scanning MimicPatient for birth years...")
    birth_years = {}
    path = FHIR_DIR / "MimicPatient.ndjson.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            patient_id = record.get("id")
            birth_date = record.get("birthDate")
            if patient_id and birth_date:
                birth_years[patient_id] = int(birth_date[:4])
    print(f"  {len(birth_years):,} MIMIC patient birth years found")

    df["birth_year"] = df["patient_id"].map(birth_years)
    df["age_at_admission"] = df["admission_year"] - df["birth_year"]
    df["source"] = "mimic"
    return df[["stay_id", "patient_id", "source", "age_at_admission", "stay_hours"]]


def _parse_eicu_age(value) -> float:
    """
    eICU masks ages over 89 as a string like '> 89' for de-identification
    (confirmed from real data — NOT '89+' as originally assumed, which was
    a real bug: it silently misclassified 7,081 elderly patients as
    unparseable/NaN, wrongly excluding them from the cohort entirely).
    Checking for a '>' prefix instead of an exact string match makes this
    robust to minor formatting variants (e.g. '>89' without the space).
    """
    if pd.isna(value):
        return float("nan")
    value = str(value).strip()
    if value.startswith(">"):
        return 90.0
    try:
        return float(value)
    except ValueError:
        return float("nan")


def build_encounter_details_eicu() -> pd.DataFrame:
    """One pass over eICU's patient.csv for stay duration + age."""
    print("\nScanning eICU patient.csv for stay durations + age...")
    path = find_eicu_file("patient.csv.gz")
    if path is None:
        print("  [MISSING] patient.csv.gz not found")
        return pd.DataFrame(columns=["stay_id", "patient_id", "source", "age_at_admission", "stay_hours"])

    df = pd.read_csv(
        path,
        usecols=["patientunitstayid", "uniquepid", "age", "unitdischargeoffset"],
        low_memory=False,
    )
    df["stay_id"] = "eicu_" + df["patientunitstayid"].astype(str)
    df["patient_id"] = df["uniquepid"]
    df["age_at_admission"] = df["age"].apply(_parse_eicu_age)
    df["raw_age"] = df["age"]  # keep the original string for future debugging —
                                # age_at_admission alone can't tell you WHY a
                                # value failed to parse, only THAT it did
    df["stay_hours"] = df["unitdischargeoffset"] / 60
    df["source"] = "eicu"
    print(f"  {len(df):,} eICU encounters scanned")
    return df[["stay_id", "patient_id", "source", "age_at_admission", "raw_age", "stay_hours"]]


def load_all_creatinine_tokens() -> pd.DataFrame:
    """Same combined loader as label_aki.py — MIMIC's single file + eICU's parts."""
    paths = []
    mimic_path = TOKENS_DIR / "creatinine_tokens.parquet"
    if mimic_path.exists():
        paths.append(mimic_path)
    paths.extend(sorted(TOKENS_DIR.glob("eicu_creatinine_tokens_part*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    # DEFENSIVE: same issue as label_aki.py — a rare non-numeric creatinine
    # value (lab comment, unparseable result) can otherwise make this whole
    # column mixed-type, breaking any later numeric comparison (e.g. the
    # baseline-creatinine exclusion check).
    n_before = len(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [INFO] Dropped {n_dropped:,} non-numeric creatinine values before cohort construction.")

    return df


def get_first_creatinine_per_stay(creat_df: pd.DataFrame) -> pd.DataFrame:
    first_vals = (
        creat_df.sort_values("time_min")
        .groupby("stay_id")
        .first()[["value"]]
        .rename(columns={"value": "first_creatinine"})
        .reset_index()
    )
    return first_vals


def get_creatinine_span_per_stay(creat_df: pd.DataFrame) -> pd.DataFrame:
    span = (
        creat_df.groupby("stay_id")["time_min"]
        .agg(["min", "max"])
        .assign(creatinine_span_hours=lambda d: (d["max"] - d["min"]) / 60)
        .reset_index()[["stay_id", "creatinine_span_hours"]]
    )
    return span


def main() -> None:
    mimic_encounters = build_encounter_details_mimic()
    eicu_encounters = build_encounter_details_eicu()
    encounter_df = pd.concat([mimic_encounters, eicu_encounters], ignore_index=True)

    print(f"\nCombined encounters: {len(encounter_df):,} "
          f"({(encounter_df['source'] == 'mimic').sum():,} MIMIC + "
          f"{(encounter_df['source'] == 'eicu').sum():,} eICU)")

    labels_df = pd.read_parquet(TOKENS_DIR / "aki_labels.parquet")
    creat_df = load_all_creatinine_tokens()
    first_creat_df = get_first_creatinine_per_stay(creat_df)
    span_df = get_creatinine_span_per_stay(creat_df)

    cohort = encounter_df.merge(labels_df.drop(columns=["source"], errors="ignore"), on="stay_id", how="left")
    cohort = cohort.merge(first_creat_df, on="stay_id", how="left")
    cohort = cohort.merge(span_df, on="stay_id", how="left")

    cohort["meets_age"] = cohort["age_at_admission"] > MIN_AGE
    cohort["meets_creatinine_count"] = cohort["n_creatinine_values"].fillna(0) >= 2
    cohort["meets_creatinine_span"] = (
        cohort["creatinine_span_hours"].fillna(0) >= MIN_CREATININE_GAP_HOURS
    )
    cohort["excluded_high_baseline"] = cohort["first_creatinine"] > MAX_BASELINE_CREATININE
    cohort["is_long_stay"] = cohort["stay_hours"] > LONG_STAY_THRESHOLD_HOURS

    cohort["eligible"] = (
        cohort["meets_age"]
        & cohort["meets_creatinine_count"]
        & cohort["meets_creatinine_span"]
        & ~cohort["excluded_high_baseline"].fillna(False)
    )

    cohort.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    for source in ["mimic", "eicu"]:
        sub = cohort[cohort["source"] == source]
        n_total = len(sub)
        n_eligible = sub["eligible"].sum()
        n_eligible_aki = sub.loc[sub["eligible"], "aki_onset_time_min"].notna().sum()
        n_long = sub.loc[sub["eligible"], "is_long_stay"].sum()
        print(f"\n--- {source.upper()} ---")
        print(f"  Total stays:              {n_total:,}")
        print(f"  Failed age filter:        {(~sub['meets_age']).sum():,}")
        print(f"  Failed creatinine-count:  {(~sub['meets_creatinine_count']).sum():,}")
        print(f"  Failed creatinine-span:   {(~sub['meets_creatinine_span']).sum():,}")
        print(f"  Excluded (high baseline): {sub['excluded_high_baseline'].sum():,}")
        print(f"  ELIGIBLE:                 {n_eligible:,}")
        print(f"    AKI within eligible:    {n_eligible_aki:,} ({n_eligible_aki / n_eligible * 100:.1f}%)")
        print(f"    >48h subgroup:          {n_long:,} ({n_long / n_eligible * 100:.1f}% of eligible)")

    n_total_eligible = cohort["eligible"].sum()
    n_total_aki = cohort.loc[cohort["eligible"], "aki_onset_time_min"].notna().sum()
    print(f"\n{'=' * 60}")
    print(f"COMBINED ELIGIBLE COHORT: {n_total_eligible:,}")
    print(f"COMBINED AKI RATE:        {n_total_aki:,} ({n_total_aki / n_total_eligible * 100:.1f}%)")
    print(f"\nCohort written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()