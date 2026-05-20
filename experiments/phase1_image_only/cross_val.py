"""
Phase 1 — leave-one-set-out cross-validation for ResNet18 image-only model.

Reads best hyperparams from results/grid_search.json (run grid_search.py first).
Val rotation: fold i → test=sets[i], val=sets[(i+1)%13], train=remaining 11.

Run from the thesis root:
    python experiments/phase1_image_only/cross_val.py

Saves results/cross_val_results.json.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MATWIDataset
from src.models.image_model import build_resnet18_regressor
from src.utils.cv_utils import make_loso_folds
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

BATCH_SIZE  = 16
EPOCHS      = 60
NUM_WORKERS = 0
# ─────────────────────────────────────────────────────────────────────────────


def train_fold(fold: dict, lr: float, weight_decay: float, device: str) -> dict:
    train_ds = MATWIDataset(DATA_ROOT, sets=fold["train"], train_mode=True)
    val_ds   = MATWIDataset(DATA_ROOT, sets=fold["val"],   train_mode=False)
    test_ds  = MATWIDataset(DATA_ROOT, sets=fold["test"],  train_mode=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = build_resnet18_regressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_mae  = float("inf")
    tmp_ckpt      = CKPT_DIR / "cv_tmp_phase1.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                preds_list.append(model(images.to(device)).squeeze(1))
                targets_list.append(targets.to(device))
        val_mae_v = mae(torch.cat(preds_list), torch.cat(targets_list))
        scheduler.step(val_mae_v)

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            torch.save(model.state_dict(), tmp_ckpt)

    # Evaluate best checkpoint on test fold
    model.load_state_dict(torch.load(tmp_ckpt, map_location=device, weights_only=True))
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            preds_list.append(model(images.to(device)).squeeze(1))
            targets_list.append(targets.to(device))

    all_preds   = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)
    errors      = (all_preds - all_targets).abs()

    return {
        "fold":       fold["fold"],
        "test_sets":  fold["test"],
        "val_sets":   fold["val"],
        "n_test":     len(test_ds),
        "test_mae":   round(errors.mean().item(), 4),
        "test_std":   round(errors.std().item(),  4),
        "best_val_mae": round(best_val_mae, 4),
    }


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    # Load best hyperparams from grid search
    gs_path = RESULTS_DIR / "grid_search.json"
    if gs_path.exists():
        with open(gs_path) as f:
            best = json.load(f)["best"]
        lr, wd = best["lr"], best["weight_decay"]
        print(f"Using grid-search best: lr={lr:.0e}  weight_decay={wd:.0e}\n")
    else:
        lr, wd = 1e-4, 0.0
        print(f"grid_search.json not found — using defaults: lr={lr:.0e}  weight_decay={wd:.0e}\n")

    folds        = make_loso_folds()
    fold_results = []

    for fold in folds:
        print(f"Fold {fold['fold']:2d}  test={fold['test']}  val={fold['val']}  "
              f"train_sets={fold['train']}")
        result = train_fold(fold, lr, wd, device)
        fold_results.append(result)
        print(f"         test MAE = {result['test_mae']:.2f} ± {result['test_std']:.2f} µm  "
              f"(n={result['n_test']})\n")

    all_maes = [r["test_mae"] for r in fold_results]
    mean_mae = sum(all_maes) / len(all_maes)
    std_mae  = (sum((m - mean_mae) ** 2 for m in all_maes) / len(all_maes)) ** 0.5

    print("=" * 60)
    print(f"LOSO-CV  mean MAE = {mean_mae:.2f} µm  std across folds = {std_mae:.2f} µm")
    print("=" * 60)

    output = {
        "model":        "phase1_resnet18_image_only",
        "lr":           lr,
        "weight_decay": wd,
        "n_folds":      len(fold_results),
        "mean_mae":     round(mean_mae, 4),
        "std_mae":      round(std_mae,  4),
        "folds":        fold_results,
    }
    out_path = RESULTS_DIR / "cross_val_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    run()
