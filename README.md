# QGAN — Reproduction of Dechant et al.

Reproduces the QGAN results from:
> Dechant et al., "Quantum generative modeling for financial time series with temporal correlations", arXiv:2507.22035, JAX+Quimb implementation.

Target metrics to match (Table 1 of the paper):
- `D_EMD` (Wasserstein distance to real S&P 500 marginals)
- `D_ACF` (volatility clustering ACF mismatch)
- `D_Lev` (leverage effect mismatch)

---

## Files

| File | Origin | Purpose |
|---|---|---|
| `JAX_QUIMB_Composite.py` | Dechant (modified) | Main QGAN model — generator, discriminator, training loop |
| `data_handling.py` | Dechant (modified) | Loads and preprocesses S&P 500 data |
| `stylized.py` | Dechant (unchanged) | Computes stylized fact metrics (EMD, ACF, leverage) |
| `jax_statevec.py` | Ours | Pure-JAX statevector circuit simulator — replaces Quimb on GPU |
| `run_repro.py` | Ours | CLI wrapper — passes seed, path, hyperparameters as arguments |
| `repro.slurm` | Ours | Rivanna job submission script |

### Changes made to Dechant's code

1. **`JAX_QUIMB_Composite.py`** — swapped `batched_quimb_circuit` → `batched_statevec_circuit` (from `jax_statevec.py`) for GPU compatibility; added `eval_every` to avoid computing metrics every epoch.
2. **`data_handling.py`** — replaced Dechant's hardcoded machine path with a portable relative path using `__file__`.

---

## How to run locally (Mac)

For a quick sanity check before submitting to Rivanna:

```bash
bash ~/qgan/run_local.sh
```

Runs 50 epochs, 2 layers on CPU. Should complete in a few minutes. Requires JAX installed (`pip install jax`).

---

## How to run on Rivanna

### Step 1 — One-time setup (install packages)

Run this once on Rivanna to install JAX, Quimb, Flax, Optax into `~/jax_packages`:

```bash
bash ~/qgan/jax_setup.sh
```

### Step 2 — Submit the job

Run a quick 100-epoch test first to confirm the environment and data are working:

```bash
sbatch ~/qgan/test.slurm
```

Logs: `~/qgan/logs/qgan_test_<jobid>.out` — should complete in ~30 minutes.

Once that passes, submit the full run:

```bash
sbatch ~/qgan/repro.slurm
```

Logs go to `~/qgan/logs/qgan_repro_<jobid>.out`
Results go to `~/qgan/results/`

### Step 3 — Monitor

```bash
squeue -u $USER
tail -f ~/qgan/logs/qgan_repro_<jobid>.out
```

---

## Hyperparameters (exact paper values)

| Parameter | Value |
|---|---|
| `n_layers` | 18 |
| `bond_dim` | 32 |
| `n_qubits` | 10 |
| `stride` | 5 |
| `epochs` | 8001 |
| `batch_size` | N // 10 (auto) |
| `alpha_acf` | 0.0 (pure WGAN) |
| `alpha_leverage` | 0.0 (pure WGAN) |
| `alpha_emd` | 1.0 |
| `seed` | 42 |

---

## Expected runtime

~48 hours on A100 GPU.
