#!/bin/bash
# ============================================================
#  GPU COMPATIBILITY TEST — Dechant QGAN with the MPS generator on A100.
#
#  Goal: answer two things quickly --
#    (1) does quimb's CircuitMPS run on the JAX CUDA backend at all?
#        (the statevec swap was reportedly done "for GPU compatibility",
#         so this is the open question)
#    (2) if it runs, what is the per-epoch wall time on an A100?
#
#  Short run: L18, bond_dim=32, only 10 epochs. If it errors on cuda we
#  find out in minutes; if it runs, the per-epoch time tells us whether a
#  full training sweep is feasible on GPU.
#
#  Prerequisites: bash ~/qgan/jax_setup.sh  (installs jax+quimb)
#  Submit:        sbatch ~/qgan/repro_mps_gputest.slurm
# ============================================================
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/qgan_mps_gputest_%j.out
#SBATCH -e logs/qgan_mps_gputest_%j.err
#SBATCH -J qgan_mps_gputest

set -euo pipefail
module purge
module load apptainer pytorch/2.7.0

REPRO_DIR="${SLURM_SUBMIT_DIR}"   # submit from the qgan dir; matches QFinance convention
RESULTS_DIR="$REPRO_DIR/results_mps_gputest"
PKG_DIR="$HOME/jax_packages"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"

mkdir -p "$REPRO_DIR/logs" "$RESULTS_DIR"

echo "=============================================="
echo "  Job    : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo "  Test   : MPS generator on GPU (cuda)  |  L18  bond_dim=32  |  10 epochs"
echo "  Start  : $(date)"
echo "=============================================="

# --nv exposes the GPU; JAX_PLATFORMS=cuda forces the GPU backend.
# If quimb cannot run on cuda, this is where it will error.
apptainer exec --nv --cleanenv \
  --bind "$REPRO_DIR:$REPRO_DIR" \
  --bind "$PKG_DIR:$PKG_DIR" \
  --env PYTHONPATH="$REPRO_DIR:$PKG_DIR" \
  --env JAX_ENABLE_X64="true" \
  --env JAX_PLATFORMS="cuda" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE="false" \
  "$SIF" python3 -u "$REPRO_DIR/run_repro.py" \
    --use_mps             \
    --seed           42   \
    --bond_dim       32   \
    --layer          18   \
    --stride         5    \
    --n_qubits       10   \
    --epochs         10   \
    --path           "$RESULTS_DIR" \
    --alpha_acf      0.0  \
    --alpha_leverage 0.0  \
    --alpha_emd      1.0  \
    --eval_every     5

echo "=============================================="
echo "  Done : $(date)"
echo "  Verdict checklist:"
echo "    - Look above for 'JAX backend : gpu' and no quimb/CUDA errors."
echo "    - Note the per-epoch seconds to judge full-run feasibility."
echo "  Results in: $RESULTS_DIR"
echo "=============================================="
