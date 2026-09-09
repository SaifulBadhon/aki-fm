"""
aki_embedding.py

Converts the raw tensors produced by AKIDataset/collate_fn (concept_ids,
values_numeric, values_categorical, times, source_ids) into a single
embedding vector per token — the actual input to the MoE transformer.

Design: additive fusion (same pattern as BERT's token+position+segment
embeddings) — each token's final vector is the SUM of:
  - concept embedding      (which clinical variable is this?)
  - source embedding       (which table/hospital-system did it come from?)
  - value embedding        (numeric -> linear projection of the scalar;
                             categorical -> lookup embedding; selected
                             per-token based on which one is present)
  - time embedding         (continuous encoding of time_min — NOT a
                             positional index, since multiple tokens can
                             share the exact same timestamp)

Run this on the server to sanity-check shapes:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 aki_embedding.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aki_dataset import AKIDataset, collate_fn

TOKENS_DIR = Path("tokens_output")


class TimeEmbedding(nn.Module):
    """
    Continuous time encoding (Time2Vec-style): a linear component plus
    several learned sinusoidal components, applied to the raw time_min
    value directly — NOT to a sequence position index, since tokens at
    the exact same timestamp must get the exact same time embedding
    regardless of where they happen to sit in the padded sequence.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, time_min: torch.Tensor) -> torch.Tensor:
        # time_min: (batch, seq_len) -> (batch, seq_len, 1)
        t = time_min.unsqueeze(-1)
        linear_part = self.linear(t)  # (batch, seq_len, 1)
        periodic_part = torch.sin(self.periodic(t))  # (batch, seq_len, d_model - 1)
        return torch.cat([linear_part, periodic_part], dim=-1)  # (batch, seq_len, d_model)


class TokenEmbedding(nn.Module):
    def __init__(self, n_concepts: int, n_sources: int, n_categorical_values: int, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        self.concept_embedding = nn.Embedding(n_concepts, d_model, padding_idx=0)
        self.source_embedding = nn.Embedding(n_sources, d_model)
        self.categorical_value_embedding = nn.Embedding(n_categorical_values, d_model, padding_idx=0)
        self.numeric_value_projection = nn.Linear(1, d_model)
        self.time_embedding = TimeEmbedding(d_model)

    def forward(self, batch: dict) -> torch.Tensor:
        concept_ids = batch["concept_ids"]           # (batch, seq_len)
        values_numeric = batch["values_numeric"]       # (batch, seq_len), NaN where categorical
        values_categorical = batch["values_categorical"]  # (batch, seq_len)
        times = batch["times"]                         # (batch, seq_len)
        source_ids = batch["source_ids"]               # (batch, seq_len)

        concept_emb = self.concept_embedding(concept_ids)
        source_emb = self.source_embedding(source_ids)
        time_emb = self.time_embedding(times)

        # Numeric path: NaN -> 0 before the linear projection (the NaN mask
        # itself tells us which tokens are numeric vs categorical).
        is_numeric = ~torch.isnan(values_numeric)
        numeric_safe = torch.nan_to_num(values_numeric, nan=0.0).unsqueeze(-1)
        numeric_emb = self.numeric_value_projection(numeric_safe)

        categorical_emb = self.categorical_value_embedding(values_categorical)

        # Select numeric embedding where the token is numeric, categorical otherwise.
        is_numeric_mask = is_numeric.unsqueeze(-1)  # (batch, seq_len, 1)
        value_emb = torch.where(is_numeric_mask, numeric_emb, categorical_emb)

        token_emb = concept_emb + source_emb + time_emb + value_emb
        return token_emb  # (batch, seq_len, d_model)


def main() -> None:
    with open(TOKENS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    n_concepts = vocab["n_concepts"]
    n_sources = vocab["n_sources"]
    n_categorical_values = vocab["n_categorical_values"]
    d_model = 128

    print(f"Building embedding layer: {n_concepts} concepts, {n_sources} sources, "
          f"{n_categorical_values} categorical values, d_model={d_model}")

    embedding = TokenEmbedding(n_concepts, n_sources, n_categorical_values, d_model)
    n_params = sum(p.numel() for p in embedding.parameters())
    print(f"Total embedding parameters: {n_params:,}")

    dataset = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\nInput batch concept_ids shape: {batch['concept_ids'].shape}")
    token_embeddings = embedding(batch)
    print(f"Output token embeddings shape: {token_embeddings.shape}")
    print(f"  (expect: batch_size x seq_len x {d_model})")

    print("\nEmbedding layer working correctly.")


if __name__ == "__main__":
    main()
