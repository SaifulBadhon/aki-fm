"""
aki_model.py

Combines everything built so far (embedding, segment pooling, MoE
router, inter-segment attention, hazard head) into one cohesive
nn.Module, plus a real training loop — train/val split, AdamW
optimizer, and a smoke-test run to confirm loss actually decreases.

This is a SMOKE TEST, not a full training run — few steps, meant to
prove the training loop itself works correctly before committing to
real training time/compute.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 aki_model.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from aki_dataset import AKIDataset, collate_fn
from aki_embedding import TokenEmbedding
from aki_moe import SegmentPooling, SegmentMoE
from aki_hazard_head import (
    InterSegmentTransformer, HazardForecastHead,
    get_last_real_segment, hazard_loss, MAX_HORIZON_HOURS,
)

TOKENS_DIR = Path("tokens_output")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


class AKIModel(nn.Module):
    def __init__(self, n_concepts: int, n_sources: int, n_categorical_values: int,
                 d_model: int = 128, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.embedding = TokenEmbedding(n_concepts, n_sources, n_categorical_values, d_model)
        self.pooling = SegmentPooling()
        self.moe = SegmentMoE(d_model, n_experts=n_experts, top_k=top_k)
        self.inter_segment = InterSegmentTransformer(d_model)
        self.hazard_head = HazardForecastHead(d_model)

    def forward(self, batch: dict) -> tuple:
        token_embeddings = self.embedding(batch)
        segment_embeddings, segment_mask = self.pooling(
            token_embeddings, batch["segment_ids"], batch["attention_mask"]
        )
        moe_output, routing_decisions = self.moe(segment_embeddings)
        attended = self.inter_segment(moe_output, segment_mask)
        last_segment = get_last_real_segment(attended, segment_mask)
        hazard_logits = self.hazard_head(last_segment)
        return hazard_logits, routing_decisions


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            for key in ["concept_ids", "values_numeric", "values_categorical",
                        "times", "source_ids", "segment_ids", "attention_mask", "hazard_bins"]:
                batch[key] = batch[key].to(device)
            hazard_logits, _ = model(batch)
            loss = hazard_loss(hazard_logits, batch["hazard_bins"])
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main() -> None:
    with open(TOKENS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AKIModel(vocab["n_concepts"], vocab["n_sources"], vocab["n_categorical_values"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {n_params:,}")

    print("\nLoading FIXED train/val splits (see prepare_splits.py)...")
    train_set = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json", split="train")
    val_set = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json", split="val")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    wandb.init(project="aki-moe", config={
        "d_model": 128, "n_experts": 8, "top_k": 2, "lr": 1e-4, "batch_size": 8,
    })

    N_STEPS = 2000
    EVAL_EVERY = 200
    CHECKPOINT_EVERY = 500

    print(f"\nStarting training run: {N_STEPS} steps, evaluating every {EVAL_EVERY}...")
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")

    for step in range(N_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for key in ["concept_ids", "values_numeric", "values_categorical",
                    "times", "source_ids", "segment_ids", "attention_mask", "hazard_bins"]:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        hazard_logits, _ = model(batch)
        loss = hazard_loss(hazard_logits, batch["hazard_bins"])
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss.item(), "step": step})

        if step % 20 == 0:
            print(f"  step {step:>5}: train_loss = {loss.item():.4f}")

        if step > 0 and step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  step {step:>5}: VAL_LOSS = {val_loss:.4f}")
            wandb.log({"val_loss": val_loss, "step": step})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
                print(f"    -> new best val_loss, checkpoint saved")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"step_{step}.pt")

    print("\nTraining run complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()