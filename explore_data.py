"""
explore_data.py

Quick exploration script: load a handful of records from MIMIC-IV-on-FHIR
and eICU-CRD side by side, so we can see the real field names/structure
before designing the common tokenizer schema.

"""

import gzip
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — update these paths to match your actual server layout.
# Based on your VS Code screenshot, this is a reasonable starting guess.
# ---------------------------------------------------------------------------

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"

FHIR_DIR = DATASETS_ROOT / "fhir"      # contains Mimic*.ndjson.gz files
EICU_DIR = DATASETS_ROOT / "eicu"      # extracted eICU-CRD folder
NWICU_DIR = DATASETS_ROOT / "nwicu"    # extracted NWICU folder (for later)

N_RECORDS = 3  # how many sample records to print per file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def peek_ndjson_gz(path: Path, n: int = N_RECORDS) -> None:
    """Print the first n records of a gzipped ndjson (FHIR) file."""
    print(f"\n{'=' * 70}")
    print(f"FHIR file: {path.name}")
    print(f"{'=' * 70}")

    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            print(f"\n--- record {i} ---")
            # truncate long records so the console stays readable
            print(json.dumps(record, indent=2)[:1500])


def peek_csv(path: Path, n: int = N_RECORDS) -> None:
    """Print the schema + first n rows of an eICU/NWICU CSV table."""
    print(f"\n{'=' * 70}")
    print(f"CSV file: {path.name}")
    print(f"{'=' * 70}")

    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    # only read a small sample — some of these tables are huge
    df = pd.read_csv(path, nrows=1000)
    print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst {n} rows:")
    print(df.head(n).to_string())


def find_file(directory: Path, filename: str) -> Path | None:
    """Find `filename` anywhere under `directory` (handles nested extraction)."""
    if not directory.exists():
        return None
    matches = list(directory.rglob(filename))
    return matches[0] if matches else None


def find_concept_in_fhir(
    path: Path,
    keywords: list[str],
    max_matches: int = 10,
    max_lines: int = 3_000_000,
) -> None:
    """
    Stream a MIMIC-FHIR ndjson.gz Observation file and print every UNIQUE
    (code, display, unit) triple whose display text matches one of `keywords`.
    This gives us verified codes straight from real data instead of guessing.
    """
    print(f"\n--- searching {path.name} for {keywords} ---")
    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    seen = set()
    keywords_lower = [k.lower() for k in keywords]

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines or len(seen) >= max_matches:
                break
            record = json.loads(line)
            coding = record.get("code", {}).get("coding", [{}])[0]
            display = (coding.get("display") or "").lower()
            if any(k in display for k in keywords_lower):
                code = coding.get("code")
                unit = record.get("valueQuantity", {}).get("unit", "n/a")
                key = (code, coding.get("display"), unit)
                if key not in seen:
                    seen.add(key)
                    print(f"  code={code!r}  display={coding.get('display')!r}  unit={unit!r}")

    if not seen:
        print(f"  No matches found in first {max_lines} lines.")


def find_concept_in_eicu(
    path: Path,
    name_column: str,
    keywords: list[str],
    max_matches: int = 10,
    chunksize: int = 200_000,
    value_column: str | None = None,
) -> None:
    """
    Stream an eICU CSV in chunks and print every UNIQUE value in `name_column`
    that matches one of `keywords`, along with its measurement unit column
    if present, and one sample value from `value_column` for a quick
    sanity-check of magnitude/units when no explicit unit column exists.
    """
    print(f"\n--- searching {path.name} column '{name_column}' for {keywords} ---")
    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    seen = set()
    keywords_lower = [k.lower() for k in keywords]
    header_cols = pd.read_csv(path, nrows=0).columns
    unit_col = "labmeasurenamesystem" if "labmeasurenamesystem" in header_cols else None

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if len(seen) >= max_matches:
            break
        mask = chunk[name_column].astype(str).str.lower().apply(
            lambda v: any(k in v for k in keywords_lower)
        )
        matches = chunk[mask]
        for _, row in matches.iterrows():
            unit = row[unit_col] if unit_col else "n/a"
            key = (row[name_column], unit)
            if key not in seen:
                seen.add(key)
                sample_value = row[value_column] if value_column else "n/a"
                print(
                    f"  {name_column}={row[name_column]!r}  unit={unit!r}  "
                    f"sample_value={sample_value!r}"
                )
            if len(seen) >= max_matches:
                break

    if not seen:
        print(f"  No matches found.")


def check_exact_label_values(
    path: Path,
    name_column: str,
    exact_label: str,
    value_column: str,
    n_samples: int = 20,
    chunksize: int = 200_000,
) -> None:
    """
    Pull `n_samples` real values for one EXACT celllabel/labname string,
    to resolve ambiguous cases where the label name alone doesn't tell us
    whether it's a volume or a count (e.g. 'Urine Count' with a suspiciously
    large sample value).
    """
    print(f"\n--- sampling values for exact label {exact_label!r} in {path.name} ---")
    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    values: list[float] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        matches = chunk[chunk[name_column] == exact_label][value_column]
        values.extend(matches.tolist())
        if len(values) >= n_samples:
            break

    if not values:
        print(f"  No rows found with {name_column}={exact_label!r}")
        return

    sample = values[:n_samples]
    print(f"  {len(sample)} sample values: {sample}")
    print(f"  min={min(sample)}  max={max(sample)}  mean={sum(sample) / len(sample):.1f}")
    print(
        "  -> if these look like small integers (0-10), it's likely a COUNT "
        "(exclude). If they look like plausible mL volumes (tens-thousands), "
        "it's likely a genuine urine output entry (include)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Exploring MIMIC-IV-on-FHIR records")
    print("-----------------------------------")

    fhir_files_of_interest = [
        "MimicPatient.ndjson.gz",
        "MimicEncounterICU.ndjson.gz",
        "MimicObservationLabevents.ndjson.gz",
        "MimicObservationChartevents.ndjson.gz",
        "MimicObservationOutputevents.ndjson.gz",
        "MimicMedicationAdministrationICU.ndjson.gz",
    ]
    for filename in fhir_files_of_interest:
        peek_ndjson_gz(FHIR_DIR / filename)

    print("\n\nExploring eICU-CRD tables")
    print("-------------------------")

    eicu_files_of_interest = [
        "patient.csv.gz",
        "lab.csv.gz",
        "vitalPeriodic.csv.gz",
        "medication.csv.gz",
        "intakeOutput.csv.gz",
    ]
    for filename in eicu_files_of_interest:
        path = find_file(EICU_DIR, filename)
        if path is not None:
            peek_csv(path)
        else:
            print(f"\n[MISSING] Could not find '{filename}' anywhere under {EICU_DIR}")

    print("\n\nSearching for verified creatinine + urine output codes")
    print("--------------------------------------------------------")

    find_concept_in_fhir(
        FHIR_DIR / "MimicObservationLabevents.ndjson.gz",
        keywords=["creatinine"],
    )
    creatinine_lab_path = find_file(EICU_DIR, "lab.csv.gz")
    if creatinine_lab_path is not None:
        find_concept_in_eicu(
            creatinine_lab_path,
            name_column="labname",
            keywords=["creatinine"],
        )

    find_concept_in_fhir(
        FHIR_DIR / "MimicObservationOutputevents.ndjson.gz",
        keywords=["urine", "foley", "void"],
    )
    intake_output_path = find_file(EICU_DIR, "intakeOutput.csv.gz")
    if intake_output_path is not None:
        cols = pd.read_csv(intake_output_path, nrows=0).columns.tolist()
        print(f"\nintakeOutput.csv.gz columns: {cols}")
        find_concept_in_eicu(
            intake_output_path,
            name_column="celllabel",
            keywords=["urine", "foley", "void"],
            max_matches=30,
            value_column="cellvaluenumeric",
        )

    print("\n\nResolving the 'Urine Count' ambiguity")
    print("---------------------------------------")
    if intake_output_path is not None:
        check_exact_label_values(
            intake_output_path,
            name_column="celllabel",
            exact_label="Urine Count",
            value_column="cellvaluenumeric",
        )



if __name__ == "__main__":
    main()