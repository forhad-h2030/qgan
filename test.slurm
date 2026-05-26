#!/bin/bash
# ============================================================
#  Quick test run — 100 epochs, 2 layers
#  Verifies the environment, data, and circuit work correctly.
#  Not meant to produce meaningful metrics — just confirms the pipeline runs.
#  Results go to ~/qgan/results_test/
#
#  Submit: sbatch ~/qgan/test.slurm
# ============================================================
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o logs/qgan_test_%j.out
#SBATCH -e logs/qgan_test_%j.err
#SBATCH -J qgan_test

set -euo pipefail
module purge
module load apptainer pytorch/2.7.0

REPRO_DIR="$HOME/qgan"
RESULTS_DIR="$REPRO_DIR/results_test"
PKG_DIR="$HOME/jax_packages"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"

mkdir -p "$REPRO_DIR/logs" "$RESULTS_DIR"

echo "=============================================="
echo "  Job    : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "  Run    : QGAN TEST  |  100 epochs  |  2 layers  |  Seed 42"
echo "  Start  : $(date)"
echo "=============================================="

apptainer exec --nv --cleanenv \
  --bind "$REPRO_DIR:$REPRO_DIR" \
  --bind "$PKG_DIR:$PKG_DIR" \
  --env PYTHONPATH="$REPRO_DIR:$PKG_DIR" \
  --env JAX_ENABLE_X64="true" \
  --env JAX_PLATFORMS="cuda" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE="false" \
  "$SIF" python3 -u "$REPRO_DIR/run_repro.py" \
    --seed           42   \
    --bond_dim       32   \
    --layer          2    \
    --stride         5    \
    --n_qubits       10   \
    --epochs         100  \
    --path           "$RESULTS_DIR" \
    --alpha_acf      0.0  \
    --alpha_leverage 0.0  \
    --alpha_emd      1.0  \
    --eval_every     10

echo "=============================================="
echo "  Done : $(date)"
echo "  Results in: $RESULTS_DIR"
echo "=============================================="
