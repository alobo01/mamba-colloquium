"""Numerical contract for every SSD / attention identity shown on screen.

Run before rendering any Act III or slide-24 scene:

    python -I math_contracts/test_ssd_equivalence.py      # plain
    python -m pytest math_contracts                         # or with pytest

Conventions (one SSD head, matching PRESENTATION_VISUAL_SCRIPT.md):
    a : (T,)    scalar decay per step, S_t = a_t S_{t-1} + k_t v_t^T
    K : (T, N)  write vectors  (b_t in Mamba-2 notation)
    Q : (T, N)  read vectors   (c_t in Mamba-2 notation)
    V : (T, P)  inputs         (x_t in Mamba-2 notation)
    y_t = S_t^T q_t
"""

import numpy as np

RTOL = ATOL = 1e-10


# ---------------------------------------------------------------- references
def ssd_recurrent(a, K, Q, V, S0=None):
    """Route 1, the notebook: constant-size state, one step per token."""
    T, N = K.shape
    P = V.shape[1]
    S = np.zeros((N, P)) if S0 is None else S0.copy()
    Y = np.zeros((T, P))
    for t in range(T):
        S = a[t] * S + np.outer(K[t], V[t])
        Y[t] = Q[t] @ S
    return Y


def decay_mask(a):
    """L[t, s] = prod_{u=s+1}^{t} a_u for s <= t, else 0 (1-semiseparable)."""
    T = len(a)
    L = np.zeros((T, T))
    for t in range(T):
        survival = 1.0
        for s in range(t, -1, -1):
            L[t, s] = survival
            survival *= a[s]
    return L


def ssd_matrix(a, K, Q, V, S0=None):
    """Route 2, the attention-shaped matrix: Y = (L ⊙ QK^T) V (+ initial-state term)."""
    Y = (decay_mask(a) * (Q @ K.T)) @ V
    if S0 is not None:
        Y += np.cumprod(a)[:, None] * (Q @ S0)
    return Y


def ssd_chunked(a, K, Q, V, C):
    """The four SSD steps shown in slide 16, chunk size C."""
    T, N = K.shape
    P = V.shape[1]
    Y = np.zeros((T, P))
    S_in = np.zeros((N, P))  # state entering the current chunk
    for start in range(0, T, C):
        sl = slice(start, min(start + C, T))
        a_c, K_c, Q_c, V_c = a[sl], K[sl], Q[sl], V[sl]
        # 1. intra-chunk outputs assuming zero incoming state (C×C masked matmul)
        Y[sl] = (decay_mask(a_c) * (Q_c @ K_c.T)) @ V_c
        # 2. chunk's own outgoing state, assuming zero incoming state
        tail = np.array([np.prod(a_c[s + 1:]) for s in range(len(a_c))])
        S_local = (K_c * tail[:, None]).T @ V_c
        # 4. correction: incoming state decayed to each position, read by q_t
        head = np.cumprod(a_c)
        Y[sl] += head[:, None] * (Q_c @ S_in)
        # 3. state passing across chunk boundaries (scan over T/C chunks)
        S_in = np.prod(a_c) * S_in + S_local
    return Y


def causal_linear_attention(K, Q, V):
    """Katharopoulos et al. 2020 without feature map: tril(QK^T) V."""
    return np.tril(Q @ K.T) @ V


# ---------------------------------------------------------------- tests
def _random(T, N=5, P=7, seed=2026):
    rng = np.random.default_rng(seed + T)
    a = rng.uniform(0.1, 0.95, size=T)
    K, Q = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    V = rng.normal(size=(T, P))
    return a, K, Q, V


def test_recurrent_equals_matrix():
    for T in (1, 3, 8, 17):
        a, K, Q, V = _random(T)
        np.testing.assert_allclose(ssd_recurrent(a, K, Q, V),
                                   ssd_matrix(a, K, Q, V), rtol=RTOL, atol=ATOL)


def test_nonzero_initial_state():
    a, K, Q, V = _random(9)
    S0 = np.random.default_rng(7).normal(size=(K.shape[1], V.shape[1]))
    np.testing.assert_allclose(ssd_recurrent(a, K, Q, V, S0),
                               ssd_matrix(a, K, Q, V, S0), rtol=RTOL, atol=ATOL)


def test_chunked_equals_recurrent():
    for T in (1, 7, 16, 23):
        a, K, Q, V = _random(T)
        ref = ssd_recurrent(a, K, Q, V)
        for C in (1, 2, 4, 5, 16):
            np.testing.assert_allclose(ssd_chunked(a, K, Q, V, C), ref,
                                       rtol=RTOL, atol=ATOL)


def test_linear_attention_is_ssd_with_no_decay():
    a, K, Q, V = _random(11)
    ones = np.ones_like(a)
    lin = causal_linear_attention(K, Q, V)
    np.testing.assert_allclose(ssd_recurrent(ones, K, Q, V), lin, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ssd_matrix(ones, K, Q, V), lin, rtol=RTOL, atol=ATOL)
    # associativity (non-causal): (QK^T)V == Q(K^T V)
    np.testing.assert_allclose((Q @ K.T) @ V, Q @ (K.T @ V), rtol=RTOL, atol=ATOL)


def test_on_screen_4x4_mask():
    # a_1 is irrelevant when S0 = 0; slides show a_2=.8, a_3=.5, a_4=.75
    L = decay_mask(np.array([0.9, 0.8, 0.5, 0.75]))
    expected = np.array([[1, 0, 0, 0],
                         [.8, 1, 0, 0],
                         [.4, .5, 1, 0],
                         [.3, .375, .75, 1]])
    np.testing.assert_allclose(L, expected, rtol=RTOL, atol=ATOL)
    assert np.isclose(L[3, 1], 0.5 * 0.75)        # L_{4,2} = a_3 a_4
    assert np.isclose(L[3, 0], 0.8 * 0.5 * 0.75)  # L_{4,1} = a_2 a_3 a_4


def test_off_diagonal_block_rank_at_most_N():
    T, N, C = 16, 2, 4
    a, K, Q, V = _random(T, N=N)
    M = decay_mask(a) * (Q @ K.T)
    for I in range(1, T // C):
        for J in range(I):
            block = M[I * C:(I + 1) * C, J * C:(J + 1) * C]
            assert np.linalg.matrix_rank(block, tol=1e-9) <= N


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"{len(tests)} SSD contract tests passed")
