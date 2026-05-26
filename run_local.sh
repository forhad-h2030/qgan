#!/bin/bash
# ============================================================
#  Local Mac test run — CPU, 2 layers, 50 epochs
#  Just confirms the pipeline works before submitting to Rivanna.
#
#  Usage: bash ~/qgan/run_local.sh
# ============================================================

set -euo pipefail

QGAN_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$QGAN_DIR/results_local"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Run : QGAN LOCAL TEST  |  CPU  |  2 layers  |  50 epochs"
echo "  Dir : $QGAN_DIR"
echo "  Start : $(date)"
echo "=============================================="

JAX_PLATFORMS="cpu" \
JAX_ENABLE_X64="true" \
PYTHONPATH="$QGAN_DIR" \
python3 -u "$QGAN_DIR/run_repro.py" \
  --seed           42  \
  --bond_dim       32  \
  --layer          2   \
  --stride         5   \
  --n_qubits       10  \
  --epochs         50  \
  --path           "$RESULTS_DIR" \
  --alpha_acf      0.0 \
  --alpha_leverage 0.0 \
  --alpha_emd      1.0 \
  --eval_every     10

echo "=============================================="
echo "  Done : $(date)"
echo "  Results in: $RESULTS_DIR"
echo "=============================================="
