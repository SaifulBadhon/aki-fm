"""
aki_dataset.py

PyTorch Dataset for the AKI foundation model. Reads training_examples.parquet
+ vocab.json (both now fully validated) and converts each patient's JSON
token stream into tensors ready for batching.

Each token becomes:
    concept_id            (int)   — from vocab's concept_to_id
    value_numeric          (float) — the numeric value, or NaN if categorical
    value_categorical_id   (int)   — from vocab's categorical_value_to_id, or 0 if numeric
    time_min                (float) — minutes since ICU admission
    source_id                (int)   — from vocab's source_to_id

Variable-length sequences are handled via a custom collate_fn that pads
to the longest sequence in each batch and returns an attention mask.

Run this on the server to sanity-check it works:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 aki_dataset.py
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

TOKENS_DIR = Path("tokens_output")


class AKIDataset(Dataset):
    def __init__(self, parquet_path: Path, vocab_path: Path):
        print(f"Loading {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        print(f"  {len(self.df):,} patients")

        with open(vocab_path) as f:
            self.vocab = json.load(f)

        self.concept_to_id = self.vocab["concept_to_id"]
        self.source_to_id = self.vocab["source_to_id"]
        self.categorical_value_to_id = self.vocab["categorical_value_to_id"]
        self.concept_type = self.vocab["concept_type"]

    def __len__(self) -> int:
        return len(self.df)

    def _encode_token(self, t: dict) -> tuple:
        concept = t["concept"]
        concept_id = self.concept_to_id.get(concept, 0)  # 0 as fallback for unseen concepts
        source_id = self.source_to_id.get(t["source_table"], 0)
        time_min = float(t["time_min"])

        ctype = self.concept_type.get(concept, "categorical")
        value_numeric = float("nan")
        value_categorical_id = 0  # <UNK>

        if ctype in ("numeric", "mixed"):
            try:
                value_numeric = float(t["value"])
            except (ValueError, TypeError):
                pass
        if ctype in ("categorical", "mixed") and (value_numeric != value_numeric):  # NaN check
            val_str = str(t["value"])
            value_categorical_id = self.categorical_value_to_id.get(val_str, 0)

        return concept_id, value_numeric, value_categorical_id, time_min, source_id

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        context_tokens = json.loads(row["context_tokens"])
        encoded = [self._encode_token(t) for t in context_tokens]

        if encoded:
            concept_ids, values_numeric, values_categorical, times, source_ids = zip(*encoded)
        else:
            concept_ids, values_numeric, values_categorical, times, source_ids = [], [], [], [], []

        hazard_bins = json.loads(row["hazard_bins_hourly"])

        return {
            "stay_id": row["stay_id"],
            "source": row["source"],
            "concept_ids": torch.tensor(concept_ids, dtype=torch.long),
            "values_numeric": torch.tensor(values_numeric, dtype=torch.float),
            "values_categorical": torch.tensor(values_categorical, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "hazard_bins": torch.tensor(hazard_bins, dtype=torch.float),
            "event_type": row["event_type"],
        }


def collate_fn(batch: list) -> dict:
    """
    Pads variable-length token sequences to the longest one in this batch,
    and returns an attention_mask so the model knows which positions are
    real tokens vs. padding.
    """
    max_len = max(len(item["concept_ids"]) for item in batch)
    max_hazard_len = max(len(item["hazard_bins"]) for item in batch)
    batch_size = len(batch)

    concept_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    values_numeric = torch.full((batch_size, max_len), float("nan"))
    values_categorical = torch.zeros(batch_size, max_len, dtype=torch.long)
    times = torch.zeros(batch_size, max_len, dtype=torch.float)
    source_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Hazard bins padded with -1 (ignore index) beyond each patient's actual length —
    # patients have different numbers of bins since some are censored earlier than others.
    hazard_bins = torch.full((batch_size, max_hazard_len), -1.0)

    stay_ids, sources, event_types = [], [], []

    for i, item in enumerate(batch):
        n = len(item["concept_ids"])
        concept_ids[i, :n] = item["concept_ids"]
        values_numeric[i, :n] = item["values_numeric"]
        values_categorical[i, :n] = item["values_categorical"]
        times[i, :n] = item["times"]
        source_ids[i, :n] = item["source_ids"]
        attention_mask[i, :n] = True

        h = len(item["hazard_bins"])
        hazard_bins[i, :h] = item["hazard_bins"]

        stay_ids.append(item["stay_id"])
        sources.append(item["source"])
        event_types.append(item["event_type"])

    return {
        "stay_ids": stay_ids,
        "sources": sources,
        "concept_ids": concept_ids,
        "values_numeric": values_numeric,
        "values_categorical": values_categorical,
        "times": times,
        "source_ids": source_ids,
        "attention_mask": attention_mask,
        "hazard_bins": hazard_bins,
        "event_types": event_types,
    }


def main() -> None:
    dataset = AKIDataset(
        TOKENS_DIR / "training_examples.parquet",
        TOKENS_DIR / "vocab.json",
    )

    print(f"\nDataset size: {len(dataset):,} patients")
    print(f"Vocab: {len(dataset.concept_to_id)} concepts, "
          f"{len(dataset.categorical_value_to_id)} categorical values, "
          f"{len(dataset.source_to_id)} source tables")

    print("\nTesting a single example (index 0)...")
    example = dataset[0]
    print(f"  stay_id: {example['stay_id']}")
    print(f"  n_tokens: {len(example['concept_ids'])}")
    print(f"  hazard_bins length: {len(example['hazard_bins'])}")
    print(f"  event_type: {example['event_type']}")

    print("\nTesting a real DataLoader batch (batch_size=4)...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"  concept_ids shape: {batch['concept_ids'].shape}")
    print(f"  values_numeric shape: {batch['values_numeric'].shape}")
    print(f"  hazard_bins shape: {batch['hazard_bins'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  stay_ids in this batch: {batch['stay_ids']}")

    print("\nDataset + DataLoader working correctly.")


if __name__ == "__main__":
    main()
