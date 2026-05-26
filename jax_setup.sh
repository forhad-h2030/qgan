#!/bin/bash
# ============================================================
#  ONE-TIME SETUP — Install JAX (CUDA) + Quimb + Flax + Optax
#  into $HOME/jax_packages using the PyTorch apptainer container.
#
#  Run interactively:   bash scripts/jax_setup.sh
#  Or queue it:         sbatch scripts/jax_setup.sh
# ============================================================
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/jax_setup_%j.out
#SBATCH -e logs/jax_setup_%j.err
#SBATCH -J jax_setup

set -euo pipefail
module purge
module load apptainer pytorch/2.7.0

PKG_DIR="$HOME/jax_packages"
SIF="$CONTAINERDIR/pytorch-2.7.0.sif"
mkdir -p "$PKG_DIR" "${SLURM_SUBMIT_DIR:-$PWD}/logs"

echo "=============================================="
echo "  Installing JAX (CUDA 12) + Quimb into: $PKG_DIR"
echo "  Container : $SIF"
echo "  Start     : $(date)"
echo "=============================================="

apptainer exec --nv --cleanenv \
  --bind "$PKG_DIR:$PKG_DIR" \
  "$SIF" pip install --target="$PKG_DIR" \
    "jax[cuda12]==0.4.30" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

apptainer exec --nv --cleanenv \
  --bind "$PKG_DIR:$PKG_DIR" \
  "$SIF" pip install --target="$PKG_DIR" \
    "quimb==1.11.0" \
    "flax==0.8.5" \
    "optax==0.2.5" \
    "pandas>=1.5" \
    "scipy>=1.10" \
    "matplotlib>=3.7" \
    "seaborn>=0.12"

echo "=============================================="
echo "  Install complete : $(date)"
echo "  Verifying GPU access..."
echo "=============================================="

apptainer exec --nv --cleanenv \
  --bind "$PKG_DIR:$PKG_DIR" \
  --env PYTHONPATH="$PKG_DIR" \
  --env JAX_PLATFORMS="cuda" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE="false" \
  "$SIF" python3 -c "
import jax
print('jax version :', jax.__version__)
print('devices     :', jax.devices())
print('backend     :', jax.default_backend())
import jaxlib; print('jaxlib      :', jaxlib.__version__)
"

echo "=============================================="
echo "  Done. If devices shows GPU above, setup is good."
echo "  Verify full environment with:  bash scripts/jax_check.sh"
echo "=============================================="
