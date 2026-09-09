"""
aki_hazard_head.py

The final two pieces completing a full forward pass:
  1. InterSegmentTransformer — standard multi-head self-attention over
     the SEGMENT sequence (not tokens) — this is what lets a segment
     from hour 40 "see" what happened at hour 2, giving the model full
     patient-history context rather than processing each segment in
     isolation.
  2. HazardForecastHead — takes the most recent segment's representation
     (after it's attended to the whole history) and predicts the full
     discrete-time hazard curve: P(AKI onset in bin i) for every future
     hourly bin, exactly matching training_examples.parquet's
     hazard_bins_hourly label format.

Together with aki_embedding.py + aki_moe.py, this is now a complete,
runnable (though untrained) forward pass of the model.

Run this on the server to see a full forward pass + loss computation
against real labels:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 aki_hazard_head.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aki_dataset import AKIDataset, collate_fn
from aki_embedding import TokenEmbedding
from aki_moe import SegmentPooling, SegmentMoE

TOKENS_DIR = Path("tokens_output")
MAX_HORIZON_HOURS = 72  # matches label_aki.py / batch_build_training_examples.py


class InterSegmentTransformer(nn.Module):
    """Lets segments attend to each other across a patient's full history."""
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, d_ff: int = 256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, segment_embeddings: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        # TransformerEncoder expects True = PAD (opposite convention from our mask)
        padding_mask = ~segment_mask
        return self.encoder(segment_embeddings, src_key_padding_mask=padding_mask)


class HazardForecastHead(nn.Module):
    """
    Predicts P(AKI onset in bin i) for every hourly bin, from the most
    recent segment's representation — the "current patient state" after
    it's attended to everything that came before it.
    """
    def __init__(self, d_model: int, max_horizon: int = MAX_HORIZON_HOURS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, max_horizon)
        )

    def forward(self, last_segment_repr: torch.Tensor) -> torch.Tensor:
        return self.net(last_segment_repr)  # (batch, max_horizon) — raw logits, apply sigmoid for probabilities


def get_last_real_segment(segment_reprs: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
    """Gathers each patient's LAST real (non-padded) segment representation."""
    last_idx = segment_mask.sum(dim=1) - 1  # (batch,) — index of last real segment per patient
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(segment_reprs.size(0))
    return segment_reprs[batch_idx, last_idx]  # (batch, d_model)


def hazard_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy per bin, ignoring padded bins (target == -1,
    from collate_fn's padding — see aki_dataset.py's hazard_bins padding).
    """
    valid_mask = targets != -1
    # align shapes: logits may be longer/shorter than targets if MAX_HORIZON_HOURS
    # differs from this batch's actual max hazard length — truncate to the shorter one
    min_len = min(logits.size(1), targets.size(1))
    logits, targets, valid_mask = logits[:, :min_len], targets[:, :min_len], valid_mask[:, :min_len]

    loss_per_bin = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss_per_bin * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    return loss


def main() -> None:
    with open(TOKENS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    d_model = 128
    embedding = TokenEmbedding(vocab["n_concepts"], vocab["n_sources"], vocab["n_categorical_values"], d_model)
    pooling = SegmentPooling()
    moe = SegmentMoE(d_model, n_experts=8, top_k=2)
    inter_segment = InterSegmentTransformer(d_model)
    hazard_head = HazardForecastHead(d_model)

    dataset = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"stay_ids in this batch: {batch['stay_ids']}")
    print(f"event_types: {batch['event_types']}")

    token_embeddings = embedding(batch)
    segment_embeddings, segment_mask = pooling(token_embeddings, batch["segment_ids"], batch["attention_mask"])
    moe_output, routing_decisions = moe(segment_embeddings)

    print(f"\nAfter MoE, before inter-segment attention: {moe_output.shape}")
    attended_segments = inter_segment(moe_output, segment_mask)
    print(f"After inter-segment attention: {attended_segments.shape}")

    last_segment = get_last_real_segment(attended_segments, segment_mask)
    print(f"Last real segment representation: {last_segment.shape}")

    hazard_logits = hazard_head(last_segment)
    print(f"Hazard logits (raw model output): {hazard_logits.shape}")

    hazard_probs = torch.sigmoid(hazard_logits)
    print(f"\nPredicted risk curve for patient 0 (first 10 hours, UNTRAINED so meaningless magnitude):")
    print(f"  {hazard_probs[0, :10].tolist()}")

    loss = hazard_loss(hazard_logits, batch["hazard_bins"])
    print(f"\nHazard loss vs real labels: {loss.item():.4f}")

    print("\nFull forward pass working correctly, end to end.")


if __name__ == "__main__":
    main()
