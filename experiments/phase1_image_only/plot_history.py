"""
Plot training history from results/history.json.
Usage: python experiments/phase1_image_only/plot_history.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

with open(RESULTS_DIR / "history.json") as f:
    history = json.load(f)

epochs    = [h["epoch"]      for h in history]
train_loss = [h["train_loss"] for h in history]
val_mae   = [h["val_mae"]    for h in history]
val_std   = [h["val_mae_std"] for h in history]
lrs       = [h.get("lr", None) for h in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ── Left: Train loss ──────────────────────────────────────────────────────────
axes[0].plot(epochs, train_loss, color="steelblue")
axes[0].set_title("Training Loss (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# ── Right: Val MAE ────────────────────────────────────────────────────────────
val_mae_arr = [v for v in val_mae]
val_std_arr = [v for v in val_std]

axes[1].plot(epochs, val_mae_arr, color="tomato", label="Val MAE")
axes[1].fill_between(
    epochs,
    [m - s for m, s in zip(val_mae_arr, val_std_arr)],
    [m + s for m, s in zip(val_mae_arr, val_std_arr)],
    color="tomato", alpha=0.15, label="± std"
)
best_epoch = epochs[val_mae_arr.index(min(val_mae_arr))]
best_mae   = min(val_mae_arr)
axes[1].axhline(best_mae, color="tomato", linestyle="--", linewidth=0.8, label=f"Best: {best_mae:.1f} µm (epoch {best_epoch})")
axes[1].axhline(19, color="green", linestyle="--", linewidth=0.8, label="Paper: 19 µm")
axes[1].set_title("Validation MAE")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE (µm)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = RESULTS_DIR / "training_curve.png"
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
