"""
inventory_concepts.py

Automated, at-scale inventory of ALL distinct clinical concepts across
MIMIC-FHIR and eICU-CRD — NOT hand-verified like creatinine/urine output
in concept_map.py, just a full census so we know the real scope of
"all features" before deciding how the tokenizer should embed them.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATASETS_ROOT = Path.home() / "saiful" / "AKI" / "datasets"
FHIR_DIR = DATASETS_ROOT / "fhir"
EICU_DIR = DATASETS_ROOT / "eicu"

OUTPUT_DIR = Path.home() / "saiful" / "AKI" / "code" / "inventory_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Fast first pass — bump this up (or remove the cap) later for a full scan.
# Chartevents/labevents can be hundreds of millions of rows, so start small.
MAX_LINES_PER_FHIR_FILE = 5_000_000
TOP_N_TO_PRINT = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inventory_fhir_observations(
    path: Path, max_lines: int = MAX_LINES_PER_FHIR_FILE
) -> Counter:
    """Count occurrences of each (code, display) pair in a MIMIC-FHIR Observation file."""
    counts: Counter = Counter()
    if not path.exists():
        print(f"  [MISSING] {path}")
        return counts

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if i % 1_000_000 == 0 and i > 0:
                print(f"    ...scanned {i:,} lines")
            record = json.loads(line)
            coding = record.get("code", {}).get("coding", [{}])[0]
            key = (coding.get("code"), coding.get("display"))
            counts[key] += 1
    return counts


def inventory_eicu_column(path: Path, column: str, chunksize: int = 200_000) -> Counter:
    """Count occurrences of each distinct (stripped) value in one eICU CSV column."""
    counts: Counter = Counter()
    if not path.exists():
        print(f"  [MISSING] {path}")
        return counts

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=[column]):
        vc = chunk[column].astype(str).str.strip().value_counts()
        counts.update(vc.to_dict())
    return counts


def report(name: str, counts: Counter, out_csv: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"{name}: {len(counts):,} distinct concepts found")
    print(f"{'=' * 70}")
    print(f"Top {TOP_N_TO_PRINT} most frequent:")
    for key, n in counts.most_common(TOP_N_TO_PRINT):
        print(f"  {n:>10,}  {key}")

    # write the FULL list to CSV for offline review — this is the real deliverable
    df = pd.DataFrame(
        [(k, n) for k, n in counts.most_common()],
        columns=["concept", "count"],
    )
    df.to_csv(out_csv, index=False)
    print(f"Full inventory written to: {out_csv}")


def inventory_fhir_category(
    path: Path, max_lines: int = MAX_LINES_PER_FHIR_FILE
) -> Counter:
    """
    Count occurrences of each `category` display value in a MIMIC-FHIR
    Observation file. Used to identify which categories are real
    physiological signal (e.g. "Labs") vs nursing documentation/workflow
    noise (e.g. "Safety", "Skin") that shouldn't be tokenized as features.
    """
    counts: Counter = Counter()
    if not path.exists():
        print(f"  [MISSING] {path}")
        return counts

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            record = json.loads(line)
            category = record.get("category", [{}])[0]
            coding = category.get("coding", [{}])[0]
            display = coding.get("code") or coding.get("display")
            counts[display] += 1
    return counts


def peek_chartevents_by_category(
    path: Path,
    category_name: str,
    n_samples: int = 5,
    max_lines: int = MAX_LINES_PER_FHIR_FILE,
) -> None:
    """
    Pull a few real records from chartevents filtered to one category, to
    figure out what a vaguely-named category (e.g. 'Treatments', 'General')
    actually contains before deciding include/exclude.
    """
    print(f"\n--- sampling category={category_name!r} from {path.name} ---")
    if not path.exists():
        print(f"  [MISSING] {path}")
        return

    found = 0
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines or found >= n_samples:
                break
            record = json.loads(line)
            category = record.get("category", [{}])[0]
            coding = category.get("coding", [{}])[0]
            display = coding.get("code") or coding.get("display")
            if display == category_name:
                found += 1
                item_coding = record.get("code", {}).get("coding", [{}])[0]
                value = record.get("valueQuantity", {}).get("value", record.get("valueString"))
                unit = record.get("valueQuantity", {}).get("unit", "n/a")
                print(
                    f"  item_code={item_coding.get('code')!r}  "
                    f"item_display={item_coding.get('display')!r}  "
                    f"value={value!r}  unit={unit!r}"
                )

    if found == 0:
        print(f"  No records found with category={category_name!r} in first {max_lines:,} lines.")


def check_labs_category_overlap(
    chartevents_path: Path,
    labevents_path: Path,
    max_lines: int = MAX_LINES_PER_FHIR_FILE,
) -> None:
    """
    Check whether item codes under chartevents' 'Labs' category are the
    SAME codes already present in labevents (true duplication — drop the
    chartevents copy) or DIFFERENT codes (genuinely separate point-of-care
    measurements — keep both).
    """
    print("\n--- checking 'Labs' category overlap between chartevents and labevents ---")

    labevents_codes: set[str] = set()
    if not labevents_path.exists():
        print(f"  [MISSING] {labevents_path}")
        return
    with gzip.open(labevents_path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            record = json.loads(line)
            coding = record.get("code", {}).get("coding", [{}])[0]
            code = coding.get("code")
            if code:
                labevents_codes.add(code)

    chart_labs_codes: dict[str, str] = {}
    if not chartevents_path.exists():
        print(f"  [MISSING] {chartevents_path}")
        return
    with gzip.open(chartevents_path, "rt") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            record = json.loads(line)
            category = record.get("category", [{}])[0]
            cat_coding = category.get("coding", [{}])[0]
            cat_display = cat_coding.get("code") or cat_coding.get("display")
            if cat_display == "Labs":
                item_coding = record.get("code", {}).get("coding", [{}])[0]
                code = item_coding.get("code")
                if code:
                    chart_labs_codes[code] = item_coding.get("display")

    overlap = set(chart_labs_codes) & labevents_codes
    only_in_chart = set(chart_labs_codes) - labevents_codes

    print(f"  Distinct item codes under chartevents 'Labs' category: {len(chart_labs_codes)}")
    print(f"  -> also present in labevents (likely TRUE duplicates): {len(overlap)}")
    print(f"  -> ONLY in chartevents (likely genuine point-of-care): {len(only_in_chart)}")

    print("\n  Sample overlapping codes (likely duplicates):")
    for code in list(overlap)[:10]:
        print(f"    {code}  {chart_labs_codes[code]!r}")

    print("\n  Sample chartevents-only codes (likely genuine separate measurements):")
    for code in list(only_in_chart)[:10]:
        print(f"    {code}  {chart_labs_codes[code]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Inventorying MIMIC-FHIR concepts (this may take a few minutes)...")

    labevents_counts = inventory_fhir_observations(
        FHIR_DIR / "MimicObservationLabevents.ndjson.gz"
    )
    report("MIMIC Labevents", labevents_counts, OUTPUT_DIR / "mimic_labevents_inventory.csv")

    chartevents_counts = inventory_fhir_observations(
        FHIR_DIR / "MimicObservationChartevents.ndjson.gz"
    )
    report(
        "MIMIC Chartevents", chartevents_counts, OUTPUT_DIR / "mimic_chartevents_inventory.csv"
    )

    print("\n\nBreaking down chartevents by category (to separate real signal from")
    print("nursing documentation/workflow noise)...")
    category_counts = inventory_fhir_category(
        FHIR_DIR / "MimicObservationChartevents.ndjson.gz"
    )
    report(
        "MIMIC Chartevents categories",
        category_counts,
        OUTPUT_DIR / "mimic_chartevents_categories.csv",
    )

    print("\n\nInventorying eICU-CRD concepts...")

    eicu_lab_path = next(EICU_DIR.rglob("lab.csv.gz"), None)
    if eicu_lab_path:
        lab_counts = inventory_eicu_column(eicu_lab_path, "labname")
        report("eICU lab.csv (labname)", lab_counts, OUTPUT_DIR / "eicu_labname_inventory.csv")
    else:
        print("  [MISSING] lab.csv.gz not found under", EICU_DIR)

    eicu_med_path = next(EICU_DIR.rglob("medication.csv.gz"), None)
    if eicu_med_path:
        med_counts = inventory_eicu_column(eicu_med_path, "drugname")
        report(
            "eICU medication.csv (drugname)",
            med_counts,
            OUTPUT_DIR / "eicu_drugname_inventory.csv",
        )
    else:
        print("  [MISSING] medication.csv.gz not found under", EICU_DIR)

    print("\n\nInspecting ambiguous categories: 'Treatments' and 'General'")
    print("--------------------------------------------------------------")
    peek_chartevents_by_category(FHIR_DIR / "MimicObservationChartevents.ndjson.gz", "Treatments")
    peek_chartevents_by_category(FHIR_DIR / "MimicObservationChartevents.ndjson.gz", "General")

    print("\n\nChecking 'Labs' category duplication vs labevents")
    print("-----------------------------------------------------")
    check_labs_category_overlap(
        FHIR_DIR / "MimicObservationChartevents.ndjson.gz",
        FHIR_DIR / "MimicObservationLabevents.ndjson.gz",
    )

    print("\n\nDone. Check the CSVs in", OUTPUT_DIR, "for the full lists.")
    print("This tells us the real scale of 'all features' across both sources —")
    print("no hand-verification needed here, unlike creatinine/urine output.")


if __name__ == "__main__":
    main()