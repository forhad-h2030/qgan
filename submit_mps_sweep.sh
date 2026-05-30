#!/bin/bash
# ============================================================
#  Submit the Dechant-QGAN MPS bond-dimension sweep (CPU) on Rivanna.
#  Default grid: bond_dim in {2, 4, 8, 16, 32}, single seed.
#  At 10 qubits, bond_dim=32 reproduces the exact (statevec) result;
#  the low-bond-dim runs show the MPS approximation error.
#
#  Usage:
#     bash ~/qgan/submit_mps_sweep.sh
#  Override:
#     BONDS="2 4 8" SEED=1 EPOCHS=2001 bash ~/qgan/submit_mps_sweep.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/repro_mps.slurm"

BONDS="${BONDS:-2 4 8 16 32}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-2001}"

cd "$SCRIPT_DIR"
echo "Submitting MPS bond sweep: bonds=[$BONDS]  seed=$SEED  epochs=$EPOCHS"
for BOND in $BONDS; do
  sbatch --job-name="qgan_mps_b${BOND}" \
         --export=ALL,BOND="${BOND}",SEED="${SEED}",EPOCHS="${EPOCHS}",LAYER=18 \
         "$JOB"
done
echo "Done. Status: squeue -u \"$USER\""
