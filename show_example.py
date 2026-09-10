"""
show_example.py

Concrete before/after: picks one real AKI patient and one real censored
patient from the validation set, shows exactly what tokens go INTO the
model, and what comes OUT (predicted risk curve), compared against
what actually happened.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 show_example.py
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aki_dataset import AKIDataset, collate_fn
from aki_model import AKIModel

TOKENS_DIR = Path("tokens_output")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
N_RECENT_TOKENS_TO_SHOW = 15
HORIZONS_HOURS = [6, 12, 24, 48]


def show_one_example(model, dataset, idx: int, device):
    row = dataset.df.iloc[idx]
    example = dataset[idx]

    print(f"\n{'=' * 70}")
    print(f"stay_id: {row['stay_id']}  (source: {row['source']})")
    print(f"{'=' * 70}")

    print(f"\n--- INPUT ---")
    print(f"Prediction time: {row['prediction_time_min']:.1f} minutes since admission "
          f"({row['prediction_time_min'] / 60:.1f} hours)")

    static_tokens = json.loads(row["static_tokens"])
    print(f"\nStatic tokens: {static_tokens}")

    context_tokens = json.loads(row["context_tokens"])
    print(f"\nTotal context tokens: {len(context_tokens)}")
    print(f"Most recent {min(N_RECENT_TOKENS_TO_SHOW, len(context_tokens))} tokens "
          f"(i.e. the data right up to the prediction moment):")
    for t in context_tokens[-N_RECENT_TOKENS_TO_SHOW:]:
        print(f"  t={t['time_min']:>8.1f}min  [{t['source_table']:<12}]  "
              f"{str(t['concept'])[:30]:<30}  value={t['value']}")

    print(f"\n--- GROUND TRUTH (what actually happened) ---")
    print(f"event_type: {row['event_type']}")
    if row["event_type"] == "aki":
        onset = row["aki_onset_time_min"]
        hours_until_onset = (onset - row["prediction_time_min"]) / 60
        print(f"Real AKI onset time: {onset:.1f}min  "
              f"({hours_until_onset:.1f} hours AFTER this prediction point)")
    else:
        print("No AKI occurred in this patient's observed data (censored).")

    print(f"\n--- MODEL OUTPUT ---")
    batch = collate_fn([example])
    for key in ["concept_ids", "values_numeric", "values_categorical",
                "times", "source_ids", "segment_ids", "attention_mask", "hazard_bins"]:
        batch[key] = batch[key].to(device)

    with torch.no_grad():
        hazard_logits, routing_decisions = model(batch)
    hazard_probs = torch.sigmoid(hazard_logits)[0]

    print("Predicted per-hour risk (first 10 hours from this prediction point):")
    for h in range(min(10, hazard_probs.size(0))):
        print(f"  hour {h}: {hazard_probs[h].item():.4f}")

    print("\nCumulative predicted risk at standard horizons:")
    for h in HORIZONS_HOURS:
        p_bins = hazard_probs[:h]
        cum_prob = 1 - torch.prod(1 - p_bins).item()
        print(f"  within {h}h: {cum_prob:.4f}")

    n_segments = int(example["segment_ids"].max().item()) + 1 if len(example["segment_ids"]) > 0 else 0
    if n_segments > 0:
        last_seg_routing = routing_decisions[0, min(n_segments - 1, routing_decisions.size(1) - 1)].tolist()
        print(f"\nMost recent segment routed to expert(s): {last_seg_routing}")
        print("(NOTE: model is only partially trained so far — this routing")
        print(" is not yet expected to correspond to a clinically meaningful pattern.)")


def main() -> None:
    with open(TOKENS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AKIModel(vocab["n_concepts"], vocab["n_sources"], vocab["n_categorical_values"]).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    val_set = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json", split="val")

    aki_idx = val_set.df[val_set.df["event_type"] == "aki"].index[0]
    censored_idx = val_set.df[val_set.df["event_type"] == "censored"].index[0]

    print("\n\n########## EXAMPLE 1: A REAL AKI CASE ##########")
    show_one_example(model, val_set, aki_idx, device)

    print("\n\n########## EXAMPLE 2: A REAL CENSORED (NO AKI) CASE ##########")
    show_one_example(model, val_set, censored_idx, device)


if __name__ == "__main__":
    main()
