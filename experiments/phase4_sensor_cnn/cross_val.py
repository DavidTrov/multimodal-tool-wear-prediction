"""
Phase 4 — leave-one-set-out cross-validation for MultiScaleSensorCNN.

Reads best hyperparams from results/grid_search.json (run grid_search.py first).
Val rotation: fold i → test=sets[i], val=sets[(i+1)%13], train=remaining 11.

Run from the thesis root:
    python experiments/phase4_sensor_cnn/cross_val.py

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

from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset
from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN
from src.utils.cv_utils import make_loso_folds
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

BATCH_SIZE  = 16
EPOCHS      = 100
NUM_WORKERS = 0
# ─────────────────────────────────────────────────────────────────────────────


def train_fold(fold: dict, lr: float, weight_decay: float, device: str) -> dict:
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, sets=fold["train"])
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, sets=fold["val"])
    test_ds  = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, sets=fold["test"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = MultiScaleSensorCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=20.0)

    best_val_mae = float("inf")
    tmp_ckpt     = CKPT_DIR / "cv_tmp_phase4.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for scalograms, targets in train_loader:
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(scalograms), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for scalograms, targets in val_loader:
                preds_list.append(model(scalograms.to(device)).squeeze(1))
                targets_list.append(targets.to(device))
        val_mae_v = mae(torch.cat(preds_list), torch.cat(targets_list))

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            torch.save(model.state_dict(), tmp_ckpt)

    # Evaluate best checkpoint on test fold
    model.load_state_dict(torch.load(tmp_ckpt, map_location=device, weights_only=True))
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for scalograms, targets in test_loader:
            preds_list.append(model(scalograms.to(device)).squeeze(1))
            targets_list.append(targets.to(device))

    all_preds   = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)
    errors      = (all_preds - all_targets).abs()

    return {
        "fold":         fold["fold"],
        "test_sets":    fold["test"],
        "val_sets":     fold["val"],
        "n_test":       len(test_ds),
        "test_mae":     round(errors.mean().item(), 4),
        "test_std":     round(errors.std().item(),  4),
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

    gs_path = RESULTS_DIR / "grid_search.json"
    if gs_path.exists():
        with open(gs_path) as f:
            best = json.load(f)["best"]
        lr, wd = best["lr"], best["weight_decay"]
        print(f"Using grid-search best: lr={lr:.0e}  weight_decay={wd:.0e}\n")
    else:
        lr, wd = 1e-3, 5e-3
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
        "model":        "phase4_multiscale_sensor_cnn",
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
