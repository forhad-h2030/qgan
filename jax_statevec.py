"""
Pure-JAX statevector quantum circuit simulator.
Drop-in replacement for single_quimb_circuit / batched_quimb_circuit.

Why: Quimb CircuitMPS uses Python objects + for-loops — not JIT/vmap-able.
     For n_qubits=10, statevector dim = 2^10 = 1024 (tiny).
     bond_dim=32 is already EXACT for 10 qubits (max Schmidt rank = 2^5 = 32),
     so statevec and MPS produce identical results.

Speed: pure JAX ops on GPU with vmap → ~1000x faster than sequential Quimb.
"""
import jax
import jax.numpy as jnp
from functools import partial

DTYPE = jnp.complex128


# ── Gate matrices ────────────────────────────────────────────────────────────

def _rx(theta):
    c = jnp.cos(theta / 2).astype(DTYPE)
    s = jnp.sin(theta / 2).astype(DTYPE)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=DTYPE)

def _ry(theta):
    c = jnp.cos(theta / 2).astype(DTYPE)
    s = jnp.sin(theta / 2).astype(DTYPE)
    return jnp.array([[c, -s], [s, c]], dtype=DTYPE)

def _rz(theta):
    e = jnp.exp(-1j * theta / 2).astype(DTYPE)
    return jnp.array([[e, 0.+0j], [0.+0j, jnp.conj(e)]], dtype=DTYPE)

_CNOT_T = jnp.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=DTYPE
).reshape(2, 2, 2, 2)

_Z = jnp.array([[1.+0j, 0.+0j], [0.+0j, -1.+0j]], dtype=DTYPE)
_X = jnp.array([[0.+0j, 1.+0j], [1.+0j,  0.+0j]], dtype=DTYPE)


# ── State operations (all JIT+vmap compatible) ────────────────────────────────

def _apply_1q(state, gate, q, n):
    """Apply 2×2 gate to qubit q of n-qubit statevector (length 2^n)."""
    s = state.reshape([2] * n)
    s = jnp.tensordot(gate, s, axes=([1], [q]))   # gate contracts on axis q
    s = jnp.moveaxis(s, 0, q)                      # move result axis back to q
    return s.reshape(2 ** n)


def _apply_cnot(state, c, t, n):
    """Apply CNOT(control=c, target=t) to n-qubit statevector."""
    non_ct = sorted(set(range(n)) - {c, t})
    s = state.reshape([2] * n)
    # Contract CNOT tensor (output axes 0,1; input axes 2,3) with state at (c, t)
    s = jnp.tensordot(_CNOT_T, s, axes=([2, 3], [c, t]))
    # After tensordot: axes are [new_c(0), new_t(1), non_ct dims in original order]
    # Build transpose permutation to restore original axis order
    perm = [0] * n
    perm[c] = 0
    perm[t] = 1
    for k, nc in enumerate(non_ct):
        perm[nc] = k + 2
    s = jnp.transpose(s, perm)
    return s.reshape(2 ** n)


def _expect_pauli(state, pauli, q, n):
    """<ψ|P_q|ψ> — expectation of single-qubit Pauli acting on qubit q."""
    s = state.reshape([2] * n)
    Ps = jnp.tensordot(pauli, s, axes=([1], [q]))
    Ps = jnp.moveaxis(Ps, 0, q)
    return jnp.real(jnp.sum(jnp.conj(s) * Ps))


# ── Circuit (matches single_quimb_circuit exactly) ───────────────────────────

def single_statevec_circuit(params, x, lambdas, n_layers, n_qubits):
    """
    Statevector simulation of Dechant's PQC.
    Matches single_quimb_circuit gate sequence exactly.

    Args:
      params   : (n_layers+1, n_qubits, 3) — RX/RY/RZ angles per layer/qubit
      x        : (n_qubits,) — data encoding input
      lambdas  : (n_qubits * n_layers,) — data scaling factors
      n_layers : int (static)
      n_qubits : int (static)

    Returns: (2*n_qubits,) — [Z0, X0, Z1, X1, ..., Z_{n-1}, X_{n-1}]
    """
    dim = 2 ** n_qubits
    state = jnp.zeros(dim, dtype=DTYPE).at[0].set(1.0 + 0j)   # |0...0>

    for layer in range(n_layers):
        # Parametric rotation layer
        for q in range(n_qubits):
            state = _apply_1q(state, _rx(params[layer, q, 0]), q, n_qubits)
            state = _apply_1q(state, _ry(params[layer, q, 1]), q, n_qubits)
            state = _apply_1q(state, _rz(params[layer, q, 2]), q, n_qubits)
        # Entanglement layer
        for q in range(n_qubits - 1):
            state = _apply_cnot(state, q, q + 1, n_qubits)
        # Data encoding layer
        for q in range(n_qubits):
            lam_idx = layer * n_qubits + q
            state = _apply_1q(state, _rx(lambdas[lam_idx] * x[q]), q, n_qubits)

    # Final rotation layer
    for q in range(n_qubits):
        state = _apply_1q(state, _rx(params[n_layers, q, 0]), q, n_qubits)
        state = _apply_1q(state, _ry(params[n_layers, q, 1]), q, n_qubits)
        state = _apply_1q(state, _rz(params[n_layers, q, 2]), q, n_qubits)

    # Measure: [Z0, X0, Z1, X1, ..., Z_{n-1}, X_{n-1}]
    expecs = []
    for q in range(n_qubits):
        expecs.append(_expect_pauli(state, _Z, q, n_qubits))
        expecs.append(_expect_pauli(state, _X, q, n_qubits))

    return jnp.stack(expecs)


def batched_statevec_circuit(params, x_batch, lambdas, n_layers, bond_dim=None):
    """
    Vmapped statevec circuit. Drop-in replacement for batched_quimb_circuit.
    bond_dim is accepted but ignored (statevec is always exact for 10 qubits).
    """
    n_qubits = x_batch.shape[1]   # Python int — static at trace time
    fn = partial(single_statevec_circuit, n_layers=n_layers, n_qubits=n_qubits)
    return jax.vmap(fn, in_axes=(None, 0, None))(params, x_batch, lambdas)


# ── Shared helpers for circuit variants ──────────────────────────────────────

def _zero_state(n_qubits):
    return jnp.zeros(2 ** n_qubits, dtype=DTYPE).at[0].set(1.0 + 0j)


def _rot_layer(state, params, layer, n_qubits):
    for q in range(n_qubits):
        state = _apply_1q(state, _rx(params[layer, q, 0]), q, n_qubits)
        state = _apply_1q(state, _ry(params[layer, q, 1]), q, n_qubits)
        state = _apply_1q(state, _rz(params[layer, q, 2]), q, n_qubits)
    return state


def _fwd_chain(state, n_qubits):
    for q in range(n_qubits - 1):
        state = _apply_cnot(state, q, q + 1, n_qubits)
    return state


def _bwd_chain(state, n_qubits):
    for q in range(n_qubits - 2, -1, -1):
        state = _apply_cnot(state, q + 1, q, n_qubits)
    return state


def _rx_encode(state, x, lambdas, layer, n_qubits):
    for q in range(n_qubits):
        idx = layer * n_qubits + q
        state = _apply_1q(state, _rx(lambdas[idx] * x[q]), q, n_qubits)
    return state


def _measure_zx(state, n_qubits):
    expecs = []
    for q in range(n_qubits):
        expecs.append(_expect_pauli(state, _Z, q, n_qubits))
        expecs.append(_expect_pauli(state, _X, q, n_qubits))
    return jnp.stack(expecs)


# ── Variant 1: Circular entanglement ─────────────────────────────────────────

def single_statevec_circuit_circular(params, x, lambdas, n_layers, n_qubits):
    """Linear CNOT chain + CNOT(n-1 → 0) closing the ring each layer.
    One extra gate per layer; breaks the open-boundary asymmetry of the baseline."""
    state = _zero_state(n_qubits)
    for layer in range(n_layers):
        state = _rot_layer(state, params, layer, n_qubits)
        state = _fwd_chain(state, n_qubits)
        state = _apply_cnot(state, n_qubits - 1, 0, n_qubits)  # ring closure
        state = _rx_encode(state, x, lambdas, layer, n_qubits)
    state = _rot_layer(state, params, n_layers, n_qubits)
    return _measure_zx(state, n_qubits)


# ── Variant 2: Strongly Entangling Layers (brick-wall) ───────────────────────

def single_statevec_circuit_sel(params, x, lambdas, n_layers, n_qubits):
    """Forward chain + backward chain per layer (brick-wall pattern).
    2× CNOT depth vs baseline; bidirectional information flow means q0 and q9
    share entanglement after a single layer instead of needing n_layers layers."""
    state = _zero_state(n_qubits)
    for layer in range(n_layers):
        state = _rot_layer(state, params, layer, n_qubits)
        state = _fwd_chain(state, n_qubits)   # 0→1→2→...→9
        state = _bwd_chain(state, n_qubits)   # 8→9 then 7→8 ... then 0→1 (reversed controls)
        state = _rx_encode(state, x, lambdas, layer, n_qubits)
    state = _rot_layer(state, params, n_layers, n_qubits)
    return _measure_zx(state, n_qubits)


# ── Variant 3: Double-angle encoding ─────────────────────────────────────────

def single_statevec_circuit_doubleenc(params, x, lambdas, n_layers, n_qubits):
    """RX(λ₁·x) · RZ(λ₂·x) encoding per qubit per layer instead of RX alone.
    Requires lambdas of shape (2 * n_layers * n_qubits,):
      first  n_layers*n_qubits entries → RX scaling
      second n_layers*n_qubits entries → RZ scaling"""
    n_lam = n_layers * n_qubits
    lam_rx = lambdas[:n_lam]
    lam_rz = lambdas[n_lam:]
    state = _zero_state(n_qubits)
    for layer in range(n_layers):
        state = _rot_layer(state, params, layer, n_qubits)
        state = _fwd_chain(state, n_qubits)
        for q in range(n_qubits):
            idx = layer * n_qubits + q
            state = _apply_1q(state, _rx(lam_rx[idx] * x[q]), q, n_qubits)
            state = _apply_1q(state, _rz(lam_rz[idx] * x[q]), q, n_qubits)
    state = _rot_layer(state, params, n_layers, n_qubits)
    return _measure_zx(state, n_qubits)


# ── Variant 4: Alternating CNOT direction ────────────────────────────────────

def single_statevec_circuit_altcnot(params, x, lambdas, n_layers, n_qubits):
    """Even PQC layers use forward chain (0→1→...); odd layers use backward chain.
    Same gate count as baseline; alternating direction breaks the one-way bias
    and lets distant qubits influence each other within fewer layers."""
    state = _zero_state(n_qubits)
    for layer in range(n_layers):
        state = _rot_layer(state, params, layer, n_qubits)
        if layer % 2 == 0:
            state = _fwd_chain(state, n_qubits)
        else:
            state = _bwd_chain(state, n_qubits)
        state = _rx_encode(state, x, lambdas, layer, n_qubits)
    state = _rot_layer(state, params, n_layers, n_qubits)
    return _measure_zx(state, n_qubits)


# ── Circuit dispatcher ────────────────────────────────────────────────────────

_CIRCUIT_REGISTRY = {
    "linear":      single_statevec_circuit,
    "circular":    single_statevec_circuit_circular,
    "sel":         single_statevec_circuit_sel,
    "double_enc":  single_statevec_circuit_doubleenc,
    "altcnot":     single_statevec_circuit_altcnot,
}

CIRCUIT_CHOICES = list(_CIRCUIT_REGISTRY.keys())


def get_circuit_fn(name: str):
    if name not in _CIRCUIT_REGISTRY:
        raise ValueError(f"Unknown circuit '{name}'. Choose from {CIRCUIT_CHOICES}")
    return _CIRCUIT_REGISTRY[name]
