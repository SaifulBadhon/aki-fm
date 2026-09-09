"""
build_vocab.py

Scans training_examples.parquet to build integer-ID vocabularies for:
  - concept names (e.g. "creatinine_serum", "Heart Rate", "urine_output")
  - source_table values (e.g. "labevents", "chartevents", "vitalPeriodic")

These vocabularies are needed before tokens can be converted into
tensors for the model — an embedding layer needs a fixed integer ID
per distinct concept, not a raw string.

Also flags which concepts are purely numeric vs. which sometimes carry
non-numeric ("categorical"/text) values (e.g. a chartevents entry like
"Patent" for a device check) — the model will need to treat these two
cases differently at the tokenizer/embedding stage.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 build_vocab.py
"""

import json
from pathlib import Path

import pandas as pd

TOKENS_DIR = Path("tokens_output")
VOCAB_PATH = TOKENS_DIR / "vocab.json"


def is_numeric(value) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def main() -> None:
    print("Loading training_examples.parquet...")
    df = pd.read_parquet(TOKENS_DIR / "training_examples.parquet")
    print(f"  {len(df):,} patients")

    concept_counts: dict[str, int] = {}
    concept_numeric_flags: dict[str, set] = {}  # concept -> {True, False} seen
    source_table_counts: dict[str, int] = {}
    categorical_value_counts: dict[str, int] = {}  # global vocab of categorical VALUES

    print("\nScanning all tokens to build vocabulary (this reads every patient's full token list)...")
    for i, context_tokens_json in enumerate(df["context_tokens"]):
        if i % 20000 == 0 and i > 0:
            print(f"  ...scanned {i:,} patients")
        tokens = json.loads(context_tokens_json)
        for t in tokens:
            concept = t["concept"]
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
            value_is_numeric = is_numeric(t["value"])
            concept_numeric_flags.setdefault(concept, set()).add(value_is_numeric)

            src = t["source_table"]
            source_table_counts[src] = source_table_counts.get(src, 0) + 1

            if not value_is_numeric and t["value"] is not None:
                val_str = str(t["value"])
                categorical_value_counts[val_str] = categorical_value_counts.get(val_str, 0) + 1

    # Sort concepts by frequency, most common first — gives smaller/more
    # meaningful IDs to common concepts, though this doesn't matter for
    # correctness, just readability when debugging.
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])
    concept_to_id = {concept: i for i, (concept, _) in enumerate(sorted_concepts)}

    sorted_sources = sorted(source_table_counts.items(), key=lambda x: -x[1])
    source_to_id = {src: i for i, (src, _) in enumerate(sorted_sources)}

    sorted_cat_values = sorted(categorical_value_counts.items(), key=lambda x: -x[1])
    # Reserve ID 0 for an <UNK> token — any categorical value not seen during
    # vocab-building (e.g. a rare value that only appears in held-out data)
    # falls back to this rather than crashing the embedding lookup.
    categorical_value_to_id = {"<UNK>": 0}
    for i, (val, _) in enumerate(sorted_cat_values, start=1):
        categorical_value_to_id[val] = i

    # Flag concepts that are ALWAYS numeric, ALWAYS categorical, or MIXED
    # (some tokens numeric, some not) — mixed concepts need special
    # handling in the tokenizer (e.g. a device-check item that's usually
    # a number but occasionally a text result).
    concept_type = {}
    for concept, flags in concept_numeric_flags.items():
        if flags == {True}:
            concept_type[concept] = "numeric"
        elif flags == {False}:
            concept_type[concept] = "categorical"
        else:
            concept_type[concept] = "mixed"

    vocab = {
        "concept_to_id": concept_to_id,
        "source_to_id": source_to_id,
        "concept_type": concept_type,
        "concept_counts": concept_counts,
        "categorical_value_to_id": categorical_value_to_id,
        "n_concepts": len(concept_to_id),
        "n_sources": len(source_to_id),
        "n_categorical_values": len(categorical_value_to_id),
    }

    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    n_numeric = sum(1 for t in concept_type.values() if t == "numeric")
    n_categorical = sum(1 for t in concept_type.values() if t == "categorical")
    n_mixed = sum(1 for t in concept_type.values() if t == "mixed")

    print(f"\n{'=' * 60}")
    print(f"Total distinct concepts: {len(concept_to_id):,}")
    print(f"  Purely numeric:    {n_numeric:,}")
    print(f"  Purely categorical: {n_categorical:,}")
    print(f"  Mixed (both seen): {n_mixed:,}")
    print(f"Total distinct source tables: {len(source_to_id)}: {list(source_to_id.keys())}")
    print(f"Total distinct categorical values: {len(categorical_value_to_id):,} (includes <UNK>)")
    print(f"\nTop 10 most frequent concepts:")
    for concept, count in sorted_concepts[:10]:
        print(f"  {count:>10,}  {concept}  ({concept_type[concept]})")
    print(f"\nVocab written to: {VOCAB_PATH}")


if __name__ == "__main__":
    main()