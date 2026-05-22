#!/usr/bin/env bash
# Compression pipeline — 1M params target
# Run from the thesis root:
#   bash run_compression_1m.sh 2>&1 | tee logs/compression_1m.log
#
# Phases:
#   1. Prune ResNet-18 to ≤1,000,000 parameters (binary-search target)
#   2. Knowledge-distillation fine-tune
#   3. PTQ (dynamic INT8)
#   4. QAT (INT8-aware fine-tuning)
#   5. Aggregate all results and regenerate plots

set -euo pipefail

PYENV_ROOT="${HOME}/.pyenv"
export PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
pyenv local thesis 2>/dev/null || true

THESIS_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${THESIS_ROOT}"

mkdir -p logs

echo "================================================================"
echo "  1/5  Pruning  →  ≤1 M params  (suffix _1m)"
echo "================================================================"
python experiments/compression/resnet/phase1_pruning/train.py \
    --sparsity 0.95 \
    --target-params 1000000 \
    --output-suffix _1m

echo ""
echo "================================================================"
echo "  2/5  Distillation  →  resnet_distilled_1m.pt"
echo "================================================================"
python experiments/compression/resnet/phase2_distillation/train.py \
    --student-ckpt checkpoints/resnet_pruned_1m.pt \
    --output-suffix _1m

echo ""
echo "================================================================"
echo "  3/5  PTQ (dynamic INT8)  →  quantization_results_1m.json"
echo "================================================================"
python experiments/compression/resnet/phase3_quantization/quantize.py \
    --input-ckpt checkpoints/resnet_distilled_1m.pt \
    --output-suffix _1m

echo ""
echo "================================================================"
echo "  4/5  QAT  →  resnet_qat_int8_1m.pt"
echo "================================================================"
python experiments/compression/resnet/phase3_quantization/qat.py \
    --input-ckpt checkpoints/resnet_distilled_1m.pt

echo ""
echo "================================================================"
echo "  5/5  Aggregate all results"
echo "================================================================"
python experiments/compression/resnet/aggregate_results.py

echo ""
echo "All done.  Check experiments/compression/resnet/results/ for outputs."
