"""
check_discrimination.py

Loss can look deceptively low under severe class imbalance (hazard
sequences are mostly 0s with one 1) without the model having learned
any real discriminative signal. This checks the thing that actually
matters: does the model assign HIGHER predicted risk to examples where
AKI genuinely occurs soon, vs. examples where it doesn't (or occurs
much later / not at all)?

Computes AUROC for "will AKI occur within N hours of the prediction
time" at a few horizons, using the saved best checkpoint.

Run this on the server:
    conda activate aki-fm
    cd ~/saiful/AKI/code
    python3 check_discrimination.py
"""

import json
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from aki_dataset import AKIDataset, collate_fn
from aki_model import AKIModel

TOKENS_DIR = Path("tokens_output")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
HORIZONS_HOURS = [6, 12, 24, 48]


def main() -> None:
    with open(TOKENS_DIR / "vocab.json") as f:
        vocab = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AKIModel(vocab["n_concepts"], vocab["n_sources"], vocab["n_categorical_values"]).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    val_set = AKIDataset(TOKENS_DIR / "training_examples.parquet", TOKENS_DIR / "vocab.json", split="val")
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=collate_fn)

    all_probs = []   # cumulative P(AKI within horizon), per horizon
    all_labels = {}  # per horizon: did AKI actually occur within that horizon
    for h in HORIZONS_HOURS:
        all_labels[h] = []
    all_probs = {h: [] for h in HORIZONS_HOURS}

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i % 200 == 0:
                print(f"  ...batch {i}")

            for key in ["concept_ids", "values_numeric", "values_categorical",
                        "times", "source_ids", "segment_ids", "attention_mask", "hazard_bins"]:
                batch[key] = batch[key].to(device)

            hazard_logits, _ = model(batch)
            hazard_probs = torch.sigmoid(hazard_logits)  # (batch, 72) per-bin hazard

            for b in range(hazard_probs.size(0)):
                targets = batch["hazard_bins"][b]
                valid_targets = targets[targets != -1]
                true_onset_bin = None
                if len(valid_targets) > 0 and valid_targets[-1].item() == 1:
                    true_onset_bin = len(valid_targets) - 1  # last valid bin is the onset

                for h in HORIZONS_HOURS:
                    # Cumulative P(event within first h bins), assuming independence
                    # across bins: 1 - product(1 - p_i) for i in [0, h)
                    p_bins = hazard_probs[b, :h]
                    cum_prob = 1 - torch.prod(1 - p_bins).item()

                    label = 1 if (true_onset_bin is not None and true_onset_bin < h) else 0
                    all_probs[h].append(cum_prob)
                    all_labels[h].append(label)

    print(f"\n{'=' * 60}")
    for h in HORIZONS_HOURS:
        labels = all_labels[h]
        probs = all_probs[h]
        n_positive = sum(labels)
        if n_positive == 0 or n_positive == len(labels):
            print(f"  {h}h horizon: cannot compute AUROC (only one class present, n_positive={n_positive}/{len(labels)})")
            continue
        auroc = roc_auc_score(labels, probs)
        print(f"  {h}h horizon: AUROC = {auroc:.4f}  (n={len(labels)}, {n_positive} positive)")

    print("\nAUROC ~0.5 = no better than random (real problem).")
    print("AUROC meaningfully > 0.5 (e.g. 0.65+) = genuine discriminative signal.")


if __name__ == "__main__":
    main()
