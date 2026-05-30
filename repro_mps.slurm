#!/bin/bash
# ============================================================
#  Dechant QGAN — MPS variant (quimb CircuitMPS, bond_dim ACTIVE)
#
#  CPU job: the quimb MPS path was swapped out of the statevec runs
#  "for GPU compatibility", so MPS runs on CPU (JAX_PLATFORMS=cpu).
#  Parameterized over BOND (and optionally SEED/LAYER/EPOCHS) via env:
#     sbatch --job-name=qgan_mps_b4 --export=ALL,BOND=4 ~/qgan/repro_mps.slurm
#  Or sweep all bond dims at once:
#     bash ~/qgan/submit_mps_sweep.sh
#
#  NOTE: at 10 qubits, bond_dim=32 == exact full-state. Low bond dims
#  (2,4,8,16) are the interesting, genuinely-approximate regime.
#  Prerequisites: bash ~/qgan/jax_setup.sh  (installs quimb into jax_packages)
# ============================================================
#SBATCH -A spinquest_standard
#SBATCH --partition=standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -J qgan_mps

set -euo pipefail

: "${BOND:?set BOND, e.g. --export=ALL,BOND=4}"
SEED="${SEED:-42}"
LAYER="${LAYER:-18}"
EPOCHS="${EPOCHS:-2001}"   # CPU MPS is slow; size from the smoke-test per-epoch time

module purge
module load apptainer pytorch/2.7.0

REPRO_DIR="$HOME/qgan"
RESULTS_DIR="$REPRO_DIR/results_mps"
PKG_DIR="$HOME/jax_packages"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"

mkdir -p "$REPRO_DIR/logs" "$RESULTS_DIR"

echo "=============================================="
echo "  Job    : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "  Run    : Dechant QGAN MPS  |  seed=$SEED  L=$LAYER  bond_dim=$BOND"
echo "  Backend: CPU (quimb CircuitMPS)  |  epochs=$EPOCHS"
echo "  Start  : $(date)"
echo "=============================================="

# CPU MPS: no --nv, no --gres; force the JAX CPU backend inside the container.
apptainer exec --cleanenv \
  --bind "$REPRO_DIR:$REPRO_DIR" \
  --bind "$PKG_DIR:$PKG_DIR" \
  --env PYTHONPATH="$REPRO_DIR:$PKG_DIR" \
  --env JAX_ENABLE_X64="true" \
  --env JAX_PLATFORMS="cpu" \
  "$SIF" python3 -u "$REPRO_DIR/run_repro.py" \
    --use_mps             \
    --seed           "$SEED"   \
    --bond_dim       "$BOND"   \
    --layer          "$LAYER"  \
    --stride         5         \
    --n_qubits       10        \
    --epochs         "$EPOCHS" \
    --path           "$RESULTS_DIR" \
    --alpha_acf      0.0       \
    --alpha_leverage 0.0       \
    --alpha_emd      1.0       \
    --eval_every     50

echo "=============================================="
echo "  Done : $(date)"
echo "  Results in: $RESULTS_DIR  (bond_dim=$BOND)"
echo "=============================================="
