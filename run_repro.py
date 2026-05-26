"""
Thin wrapper around JAX_QUIMB_Composite.py that adds:
  - configurable --seed (overrides the hardcoded PRNGKey values)
  - --results_path override
  - prints key hyperparameters at startup for log traceability

Usage (from this directory, with SP500 symlinked here):
  python3 run_repro.py --seed 42 --bond_dim 32 --layer 18 \
      --epochs 8001 --stride 5 --path /path/to/results

See JAX_QUIMB_Composite_managaer.py for the full arg list.
"""
import os
os.environ["JAX_ENABLE_X64"] = "true"

import argparse
import sys

import jax
import jax.numpy as jnp

print(f"  JAX version : {jax.__version__}")
print(f"  JAX devices : {jax.devices()}")
print(f"  JAX backend : {jax.default_backend()}")

import JAX_QUIMB_Composite as jmm

parser = argparse.ArgumentParser()
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--bond_dim",     type=int,   required=True)
parser.add_argument("--layer",        type=int,   required=True)
parser.add_argument("--stride",       type=int,   default=5)
parser.add_argument("--n_qubits",     type=int,   default=10)
parser.add_argument("--epochs",       type=int,   default=8001)
parser.add_argument("--path",         type=str,   default="./results")
parser.add_argument("--alpha_acf",    type=float, default=0.0)
parser.add_argument("--alpha_leverage", type=float, default=0.0)
parser.add_argument("--alpha_emd",    type=float, default=1.0)
parser.add_argument("--eval_every",   type=int,   default=50)
args = parser.parse_args()

print("=" * 60)
print(f"  Dechant JAX+Quimb Reproduction Run")
print(f"  seed={args.seed}  bond_dim={args.bond_dim}  n_layers={args.layer}")
print(f"  n_qubits={args.n_qubits}  chopsize={2*args.n_qubits}  stride={args.stride}")
print(f"  epochs={args.epochs}  eval_every={args.eval_every}  alpha_acf={args.alpha_acf}  alpha_lev={args.alpha_leverage}")
print(f"  path={args.path}")
print("=" * 60)

chopsize = 2 * args.n_qubits
gan_params = {
    "chopsize":         chopsize,
    "stride":           args.stride,
    "n_qubits":         args.n_qubits,
    "n_layers":         args.layer,
    "epochs":           args.epochs,
    "save":             True,
    "path":             args.path,
    "time_inc":         "SP500",
    "bond_dim":         args.bond_dim,
    "ckpt_dir":         "",
    "resume_from":      False,
    "alpha_acf":        args.alpha_acf,
    "alpha_leverage":   args.alpha_leverage,
    "alpha_emd":        args.alpha_emd,
    "eval_every":       args.eval_every,
}

gan = jmm.quantum_GAN(**gan_params)
gan.critic_model = jmm.Critic(dropout_rate=0.5)

# Override hardcoded PRNGKey(0) / PRNGKey(42) with seed-derived keys
key = jax.random.PRNGKey(args.seed)
key_gen, key_disc = jax.random.split(key, 2)

gan.generator_params = jmm.init_generator_params(key_gen, args.layer, args.n_qubits)
dummy = jnp.ones((1, chopsize, 1), dtype=jnp.float32)
vars_ = gan.critic_model.init(key_disc, dummy, train=True)
gan.critic_params = vars_["params"]

print(f"  Generator params initialized with seed={args.seed}  (PRNGKey derived)")
print(f"  Discriminator params initialized with seed={args.seed}+1")
print("  Starting training ...\n")

gan.train()

print("\n  Training complete.")
