"""
aki_moe.py

Two pieces:
  1. SegmentPooling — aggregates token embeddings within each segment
     (using the segment_ids from aki_dataset.py) into one vector per
     segment, correctly excluding padded positions.
  2. SegmentMoE — the actual mixture-of-experts router: each segment
     gets routed to its top-K best-matching expert(s), and the model
     learns over training which physiological patterns each expert
     specializes in — this is the piece the whole project's
     "routing-as-explanation" idea is built on.

Run this on the server to sanity-check shapes and see real routing
decisions on actual patient data:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 aki_moe.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aki_dataset import AKIDataset, collate_fn
from aki_embedding import TokenEmbedding

TOKENS_DIR = Path("tokens_output")


class SegmentPooling(nn.Module):
    """
    Mean-pools token embeddings within each segment. Padded positions
    are redirected to a dedicated 'garbage' slot (one beyond the real
    max segment count) so they never contaminate a real segment's
    average — necessary because padding's segment_id defaults to 0,
    which would otherwise collide with real segment 0.
    """
    def forward(self, token_embeddings: torch.Tensor, segment_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple:
        batch_size, seq_len, d_model = token_embeddings.shape
        max_segments = int(segment_ids.max().item()) + 1

        garbage_slot = max_segments
        routed_ids = segment_ids.clone()
        routed_ids[~attention_mask] = garbage_slot

        sums = torch.zeros(batch_size, max_segments + 1, d_model, device=token_embeddings.device)
        counts = torch.zeros(batch_size, max_segments + 1, device=token_embeddings.device)

        idx_expanded = routed_ids.unsqueeze(-1).expand(-1, -1, d_model)
        sums.scatter_add_(1, idx_expanded, token_embeddings)
        counts.scatter_add_(1, routed_ids, torch.ones_like(routed_ids, dtype=torch.float))

        sums = sums[:, :max_segments, :]
        counts = counts[:, :max_segments]

        segment_embeddings = sums / counts.clamp(min=1).unsqueeze(-1)
        segment_mask = counts > 0  # (batch, max_segments) — which segments are real

        return segment_embeddings, segment_mask


class SegmentMoE(nn.Module):
    """
    Segment-wise Mixture of Experts. Each segment is routed to its
    top-K best-matching expert(s) out of N — sparse activation, same
    mechanism discussed throughout this project: the router scores
    every segment against every expert cheaply, only the chosen
    expert(s) actually process that segment's full representation.
    """
    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2, d_ff: int = 256):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, segment_embeddings: torch.Tensor) -> tuple:
        batch_size, n_segments, d_model = segment_embeddings.shape

        router_logits = self.router(segment_embeddings)  # (batch, n_segments, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize over top-k

        output = torch.zeros_like(segment_embeddings)
        for k in range(self.top_k):
            expert_idx_k = top_k_indices[..., k]  # (batch, n_segments)
            weight_k = top_k_probs[..., k].unsqueeze(-1)  # (batch, n_segments, 1)

            for e in range(self.n_experts):
                mask = (expert_idx_k == e)  # (batch, n_segments)
                if mask.any():
                    expert_out = self.experts[e](segment_embeddings[mask])
                    output[mask] += weight_k[mask] * expert_out

        return output, top_k_indices  # top_k_indices is the routing decision itself — the explanation signal


def main() -> None:
    with open(TOKENS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    d_model = 128
    embedding = TokenEmbedding(vocab["n_concepts"], vocab["n_sources"], vocab["n_categorical_values"], d_model)
    pooling = SegmentPooling()
    moe = SegmentMoE(d_model, n_experts=8, top_k=2)

    dataset = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"Batch concept_ids shape: {batch['concept_ids'].shape}")

    token_embeddings = embedding(batch)
    print(f"Token embeddings shape: {token_embeddings.shape}")

    segment_embeddings, segment_mask = pooling(token_embeddings, batch["segment_ids"], batch["attention_mask"])
    print(f"Segment embeddings shape: {segment_embeddings.shape}")
    print(f"Segment mask shape: {segment_mask.shape}, real segments per patient: {segment_mask.sum(dim=1).tolist()}")

    moe_output, routing_decisions = moe(segment_embeddings)
    print(f"MoE output shape: {moe_output.shape}")
    print(f"Routing decisions shape: {routing_decisions.shape}")

    print("\nReal routing decisions for patient 0 in this batch (stay_id={}):".format(batch["stay_ids"][0]))
    n_real_segments = segment_mask[0].sum().item()
    for seg_idx in range(min(n_real_segments, 10)):
        experts_chosen = routing_decisions[0, seg_idx].tolist()
        print(f"  segment {seg_idx}: routed to expert(s) {experts_chosen}")

    print("\nSegmentPooling + SegmentMoE working correctly.")


if __name__ == "__main__":
    main()
