#!/bin/bash
# ============================================================
#  Reproduce Dechant et al. Table 1 — JAX+Quimb QGAN
#
#  Hyperparameters (from paper, confirmed by L18_B32 filenames):
#    n_layers=18, bond_dim=32, stride=5, n_qubits=10
#    pure WGAN: alpha_acf=0, alpha_leverage=0
#    epochs=8001, batch_size=N//10 (auto)
#
#  Prerequisites: bash ~/qgan/jax_setup.sh  (once)
#  Submit:        sbatch ~/qgan/repro.slurm
# ============================================================
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o logs/qgan_repro_%j.out
#SBATCH -e logs/qgan_repro_%j.err
#SBATCH -J qgan_repro

set -euo pipefail
module purge
module load apptainer pytorch/2.7.0

REPRO_DIR="$HOME/qgan"
RESULTS_DIR="$REPRO_DIR/results"
PKG_DIR="$HOME/jax_packages"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"

mkdir -p "$REPRO_DIR/logs" "$RESULTS_DIR"

echo "=============================================="
echo "  Job    : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "  Run    : Dechant Repro  |  Seed 42  |  L18 B32"
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
    --layer          18   \
    --stride         5    \
    --n_qubits       10   \
    --epochs         8001 \
    --path           "$RESULTS_DIR" \
    --alpha_acf      0.0  \
    --alpha_leverage 0.0  \
    --alpha_emd      1.0  \
    --eval_every     50

echo "=============================================="
echo "  Done : $(date)"
echo "  Results in: $RESULTS_DIR"
echo "=============================================="
