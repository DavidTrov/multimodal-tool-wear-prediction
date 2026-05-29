#!/usr/bin/env bash
# Compression pipeline — remaining steps after 90%/95% runs completed.
#
# Usage (from thesis root):
#   bash run_compression.sh 2>&1 | tee compression.log

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "========================================================"
echo " Compression pipeline — starting $(date)"
echo "========================================================"

# ── QAT on 90% distilled model (best test MAE: 23.54 µm) ─────────────────────
echo ""
echo ">>> [QAT] Phase 3b — Quantization-Aware Training (90% model)"
python experiments/compression/resnet/phase3_quantization/qat.py \
    --input-ckpt checkpoints/resnet_distilled_90.pt

# ── Budget run: push to ≤2M params (≤2048 KB INT8, on-device threshold) ──────
# Uses --target-params to binary-search sparsity scaling so the pruned model
# actually fits in flash. The 95% sensitivity map is used as the starting
# point; scale_sparsity_map_to_target() then scales it up (including
# previously-protected layers) until param count ≤ 2,000,000.
echo ""
echo ">>> [budget] Phase 1 — Pruning (target ≤2M params)"
python experiments/compression/resnet/phase1_pruning/train.py \
    --sparsity 0.95 \
    --target-params 2000000 \
    --output-suffix _budget

echo ""
echo ">>> [budget] Phase 2 — Distillation"
python experiments/compression/resnet/phase2_distillation/train.py \
    --student-ckpt checkpoints/resnet_pruned_budget.pt \
    --output-suffix _budget

echo ""
echo ">>> [budget] Phase 3a — PTQ Quantization"
python experiments/compression/resnet/phase3_quantization/quantize.py \
    --input-ckpt checkpoints/resnet_distilled_budget.pt \
    --output-suffix _budget

echo ""
echo ">>> [budget] Phase 3b — QAT"
python experiments/compression/resnet/phase3_quantization/qat.py \
    --input-ckpt checkpoints/resnet_distilled_budget.pt

# ── Aggregate all results + plots ────────────────────────────────────────────
echo ""
echo ">>> Aggregating results"
python experiments/compression/resnet/aggregate_results.py

echo ""
echo "========================================================"
echo " All done! $(date)"
echo "========================================================"
