"""
Python port of include/AMGOperators.hpp for Option A (algebraic AMG on the
PolyDG matrix), plus a proper two-grid convergence-factor computation.

Faithful to the C++:
  - compute_strength_matrix : -a_ij >= theta * max_k(-a_ik), off-diagonal only
  - classical_cf_splitting  : greedy Ruge-Stueben (lambda = # strong conns from i)
  - compute_baseline_prolongation : direct interpolation, row-normalized

Deviation from the C++ (documented): the C++ compute_gauss_seidel_smoother
returns the identity, which makes the two-grid error operator a projection with
spectral radius == 1 (useless as a rho label). Here we use a real weighted-Jacobi
smoother so rho = spectral_radius(M) lands in (0,1). The same smoother is stored
as S in the p_value NPZ for self-consistency.

NOTE on scaling: eigenvalues use dense linear algebra (fine for the small
inspection matrices, n ~ few thousand). For large matrices switch rho to a
power-iteration / sparse eigensolver.
"""

import numpy as np
import scipy.sparse as sp


def compute_strength_matrix(A: sp.csr_matrix, theta: float) -> sp.csr_matrix:
    """Binary strength-of-connection matrix (row i -> its strong neighbors)."""
    A = A.tocsr()
    n = A.shape[0]
    rows, cols = [], []
    for i in range(n):
        start, end = A.indptr[i], A.indptr[i + 1]
        cols_i = A.indices[start:end]
        vals_i = A.data[start:end]
        # max_k(-a_ik) over off-diagonal entries
        max_offdiag = 0.0
        for j, v in zip(cols_i, vals_i):
            if j != i:
                max_offdiag = max(max_offdiag, -v)
        if max_offdiag <= 0.0:
            continue
        thresh = theta * max_offdiag
        for j, v in zip(cols_i, vals_i):
            if j != i and (-v) >= thresh:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def classical_cf_splitting(A: sp.csr_matrix, theta: float) -> np.ndarray:
    """Greedy Ruge-Stueben C/F splitting. Returns markers: 1=coarse, 0=fine."""
    S = compute_strength_matrix(A, theta)
    n = A.shape[0]
    markers = np.zeros(n, dtype=np.int64)  # 0=undecided during the loop

    # lambda[i] = number of strong connections FROM i (matches the C++ row count)
    lam = np.diff(S.indptr).astype(np.int64)

    undecided = set(range(n))
    decided = np.zeros(n, dtype=bool)  # coarse or fine

    while undecided:
        # node with maximum lambda among the undecided
        max_node, max_lam = -1, -1
        for i in undecided:
            if lam[i] > max_lam:
                max_lam, max_node = lam[i], i
        if max_node == -1:
            break

        markers[max_node] = 1  # coarse
        decided[max_node] = True
        undecided.discard(max_node)

        # strong neighbors of max_node become fine
        s0, s1 = S.indptr[max_node], S.indptr[max_node + 1]
        for j in S.indices[s0:s1]:
            if j in undecided:
                markers[j] = -1  # fine
                decided[j] = True
                undecided.discard(j)
                # bump lambda of j's still-undecided strong neighbors
                t0, t1 = S.indptr[j], S.indptr[j + 1]
                for k in S.indices[t0:t1]:
                    if not decided[k]:
                        lam[k] += 1

    # binary: 1=coarse, 0=fine
    return (markers == 1).astype(np.int64)


def extract_coarse_nodes(cf_markers: np.ndarray) -> np.ndarray:
    return np.where(cf_markers == 1)[0].astype(np.int64)


def compute_baseline_prolongation(A: sp.csr_matrix,
                                  cf_markers: np.ndarray) -> sp.csr_matrix:
    """Direct-interpolation prolongation P (n x n_coarse), row-normalized."""
    A = A.tocsr()
    n = A.shape[0]
    coarse_nodes = extract_coarse_nodes(cf_markers)
    n_coarse = len(coarse_nodes)
    coarse_to_col = {c: k for k, c in enumerate(coarse_nodes)}

    rows, cols, data = [], [], []
    for i in range(n):
        if cf_markers[i] == 1:  # coarse -> identity row
            rows.append(i); cols.append(coarse_to_col[i]); data.append(1.0)
            continue
        # fine node: gather strongly/algebraically connected coarse neighbors
        s0, s1 = A.indptr[i], A.indptr[i + 1]
        nbr_cols, weights = [], []
        for j, v in zip(A.indices[s0:s1], A.data[s0:s1]):
            if j != i and cf_markers[j] == 1:
                nbr_cols.append(j)
                weights.append(-v)  # off-diagonal assumed negative
        if not nbr_cols:
            if n_coarse > 0:  # fallback: attach to first coarse node
                rows.append(i); cols.append(0); data.append(1.0)
            continue
        total = sum(weights)
        if abs(total) > 1e-12:
            for j, w in zip(nbr_cols, weights):
                rows.append(i); cols.append(coarse_to_col[j]); data.append(w / total)

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n_coarse))


def estimate_jacobi_omega(A: sp.csr_matrix) -> tuple:
    """Smoothing-optimal weighted-Jacobi omega = 4 / (3 * lambda_max(D^-1 A)).

    Weighted Jacobi S = I - omega D^-1 A converges only if omega*lambda_max < 2.
    For a standard 2D FEM Laplacian lambda_max(D^-1 A) = 2, giving the textbook
    omega = 2/3. SIP-DG penalty terms push lambda_max well above 2, so a FIXED
    omega = 2/3 yields omega*lambda_max > 2 and a DIVERGENT smoother (spectral
    radius > 1) -> two-grid rho > 1 regardless of coarsening. Scaling by the
    actual lambda_max is therefore essential on DG matrices.

    Returns (omega, lambda_max). lambda_max is computed on the symmetric
    D^-1/2 A D^-1/2 (similar to D^-1 A, so same real positive spectrum).
    """
    A_d = A.toarray()
    d = np.diag(A_d).copy()
    d[np.abs(d) < 1e-14] = 1.0
    dinv_sqrt = 1.0 / np.sqrt(np.abs(d))
    B = (A_d * dinv_sqrt[:, None]) * dinv_sqrt[None, :]  # symmetric ~ D^-1 A
    lam_max = float(np.linalg.eigvalsh(B)[-1])
    return 4.0 / (3.0 * lam_max), lam_max


def jacobi_smoother_matrix(A: sp.csr_matrix, omega: float = None) -> np.ndarray:
    """Weighted-Jacobi iteration matrix S = I - omega D^{-1} A (dense).

    If omega is None it is auto-set to the smoothing-optimal value via
    estimate_jacobi_omega (required for DG matrices; see its docstring)."""
    A_d = A.toarray()
    d = np.diag(A_d).copy()
    d[np.abs(d) < 1e-14] = 1.0
    if omega is None:
        omega, _ = estimate_jacobi_omega(A)
    return np.eye(A_d.shape[0]) - omega * (A_d / d[:, None])


def spectral_radius(M: np.ndarray) -> float:
    """max |eigenvalue| of a dense matrix (diagnostic helper)."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# ===========================================================================
# rho label matching the ORIGINAL definition (DiffusionModel.hpp): the average
# residual-reduction-per-iteration of an AMG-preconditioned CG solve, with
# theta entering as the classical strong-threshold. On an SPD matrix this
# converges by construction, so rho in (0,1). We use pyamg's classical
# Ruge-Stueben AMG (paper Def 2.1) in place of HYPRE BoomerAMG -- same AMG
# family, not bit-identical, so polygonal rho is on a slightly different scale
# than the original FEM datasets.
# ===========================================================================

def amg_convergence_factor(A: sp.csr_matrix, theta: float,
                           max_iter: int = 1000, tol: float = 1e-12,
                           seed: int = 42) -> tuple:
    """rho = (||r_k|| / ||r_0||)^(1/k) for AMG-preconditioned CG.

    theta is the classical strong-threshold. Returns (rho, n_iter). rho=1.0 if
    the solve makes no progress (degenerate). Uses a fixed random RHS so the
    measurement is reproducible and RHS-independent (asymptotic factor)."""
    import pyamg

    A = A.tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)

    ml = pyamg.ruge_stuben_solver(
        A, strength=("classical", {"theta": theta}), max_coarse=10
    )
    residuals = []
    ml.solve(b, x0=np.zeros(n), tol=tol, maxiter=max_iter,
             accel="cg", residuals=residuals)

    k = len(residuals) - 1
    if k < 1 or residuals[0] == 0.0:
        return 1.0, 0
    rho = float((residuals[-1] / residuals[0]) ** (1.0 / k))
    return rho, k


def amg_report_metrics(A: sp.csr_matrix, theta: float,
                       max_iter: int = 1000, tol: float = 1e-12,
                       seed: int = 42) -> dict:
    """Full metric set for a D.csv report row, from ONE pyamg solve:
    rho, iterations, elapsed_sec, n_levels. Mirrors the AMG-DL DiffusionModel
    getters (get_convergence_factor / _linear_solve_elapsed_sec /
    _amg_hierarchy_levels) but with pyamg's classical Ruge-Stueben instead of
    HYPRE BoomerAMG -- values are on pyamg's scale, not directly comparable to
    the paper's HYPRE numbers, though theta-trends and the rho-time correlation
    hold."""
    import time
    import pyamg

    A = A.tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)

    ml = pyamg.ruge_stuben_solver(
        A, strength=("classical", {"theta": theta}), max_coarse=10
    )
    n_levels = len(ml.levels)

    residuals = []
    t0 = time.perf_counter()
    ml.solve(b, x0=np.zeros(n), tol=tol, maxiter=max_iter,
             accel="cg", residuals=residuals)
    elapsed_sec = time.perf_counter() - t0

    k = len(residuals) - 1
    rho = (float((residuals[-1] / residuals[0]) ** (1.0 / k))
           if k >= 1 and residuals[0] != 0.0 else 1.0)
    return {"rho": rho, "iterations": max(k, 0),
            "elapsed_sec": elapsed_sec, "n_levels": n_levels}


def two_grid_convergence_factor(A: sp.csr_matrix, P: sp.csr_matrix,
                                S_iter: np.ndarray,
                                nu_pre: int = 1, nu_post: int = 1) -> float:
    """rho = spectral radius of M = S^nu_post (I - P Ac^{-1} P^T A) S^nu_pre."""
    A_d = A.toarray()
    P_d = P.toarray()
    R_d = P_d.T
    Ac = R_d @ A_d @ P_d
    Ac_inv = np.linalg.inv(Ac + 1e-12 * np.eye(Ac.shape[0]))
    CGC = np.eye(A_d.shape[0]) - P_d @ Ac_inv @ R_d @ A_d
    M = np.linalg.matrix_power(S_iter, nu_post) @ CGC @ np.linalg.matrix_power(S_iter, nu_pre)
    eig = np.linalg.eigvals(M)
    return float(np.max(np.abs(eig)))


# ===========================================================================
# Block / nodal coarsening for DG systems.
#
# Classical pointwise RS diverges on DG matrices: they are block systems whose
# intra-element off-diagonals are POSITIVE, so the signed strength measure
# (-a_ij >= theta*max(-a_ik)) ignores the dominant within-polytope coupling.
# Here we coarsen at the POLYTOPE level. DG dofs are element-local, so dof
# block [I*dpp:(I+1)*dpp] belongs to polytope I (dpp = dofs per polytope).
# Connection strength between polytopes I,J is the Frobenius norm of the
# off-diagonal block A_IJ (unsigned -> positive couplings now count).
# ===========================================================================

def infer_block_size(n_dofs: int, n_poly: int) -> int:
    assert n_dofs % n_poly == 0, \
        f"n_dofs {n_dofs} not divisible by n_poly {n_poly}"
    return n_dofs // n_poly


def verify_block_contiguous(A: sp.csr_matrix, n_poly: int, dpp: int) -> bool:
    """Sanity-check the assumed per-polytope diagonal blocks: DG blocks should
    be (nearly) dense and symmetric. Returns False if the contiguous-blocking
    assumption looks wrong (then we need an explicit dof->polytope map)."""
    A = A.tocsr()
    for I in range(n_poly):
        s = slice(I * dpp, (I + 1) * dpp)
        blk = A[s, s].toarray()
        if np.count_nonzero(blk) < dpp:          # implausibly sparse block
            return False
        if not np.allclose(blk, blk.T, atol=1e-6):
            return False
    return True


def block_offdiag_norms(A: sp.csr_matrix, n_poly: int, dpp: int) -> np.ndarray:
    """n_poly x n_poly Frobenius norms of off-diagonal blocks (O(nnz))."""
    A = A.tocoo()
    W = np.zeros((n_poly, n_poly))
    bi = A.row // dpp
    bj = A.col // dpp
    np.add.at(W, (bi, bj), A.data ** 2)   # accumulate squared entries per block
    np.fill_diagonal(W, 0.0)
    return np.sqrt(W)


def block_cf_splitting(W: np.ndarray, theta: float) -> np.ndarray:
    """Greedy Ruge-Stueben on the polytope strength graph W (block norms).
    Polytope I strongly connected to J if W[I,J] >= theta * max_k W[I,k].
    Returns polytope markers: 1=coarse, 0=fine."""
    n = W.shape[0]
    # binary strength graph
    row_max = W.max(axis=1)
    strong = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if row_max[i] > 0:
            strong[i] = W[i] >= theta * row_max[i]
    strong[np.arange(n), np.arange(n)] = False

    markers = np.zeros(n, dtype=np.int64)     # 0 undecided
    lam = strong.sum(axis=1).astype(np.int64)
    undecided = set(range(n))
    decided = np.zeros(n, dtype=bool)

    while undecided:
        max_node, max_lam = -1, -1
        for i in undecided:
            if lam[i] > max_lam:
                max_lam, max_node = lam[i], i
        if max_node == -1:
            break
        markers[max_node] = 1
        decided[max_node] = True
        undecided.discard(max_node)
        for j in np.where(strong[max_node])[0]:
            if j in undecided:
                markers[j] = -1
                decided[j] = True
                undecided.discard(j)
                for k in np.where(strong[j])[0]:
                    if not decided[k]:
                        lam[k] += 1
    return (markers == 1).astype(np.int64)


def block_prolongation(A: sp.csr_matrix, poly_markers: np.ndarray,
                       dpp: int) -> tuple:
    """Block interpolation. Coarse polytopes inject their whole dof block
    (identity). Fine polytopes interpolate from strongly connected coarse
    polytopes with weights from off-diagonal block norms (row-normalized),
    applied block-diagonally so constants are preserved.

    Returns (P as csr n x (n_coarse*dpp), coarse_dof_indices)."""
    n_poly = len(poly_markers)
    n = A.shape[0]
    W = block_offdiag_norms(A, n_poly, dpp)
    coarse_poly = np.where(poly_markers == 1)[0]
    poly_to_ccol = {p: k for k, p in enumerate(coarse_poly)}
    n_coarse = len(coarse_poly)

    I_blk = np.eye(dpp)
    rows, cols, data = [], [], []

    def add_block(bi, bj, block):
        r0, c0 = bi * dpp, bj * dpp
        nz = np.nonzero(block)
        for a, b in zip(*nz):
            rows.append(r0 + a); cols.append(c0 + b); data.append(block[a, b])

    for I in range(n_poly):
        if poly_markers[I] == 1:                       # coarse: identity block
            add_block(I, poly_to_ccol[I], I_blk)
            continue
        strong_c = [(J, W[I, J]) for J in coarse_poly if W[I, J] > 0]
        if not strong_c:                               # fallback: nearest coarse
            J = coarse_poly[np.argmax(W[I, coarse_poly])] if n_coarse else None
            if J is not None:
                add_block(I, poly_to_ccol[J], I_blk)
            continue
        total = sum(w for _, w in strong_c)
        for J, w in strong_c:
            add_block(I, poly_to_ccol[J], (w / total) * I_blk)

    P = sp.csr_matrix((data, (rows, cols)), shape=(n, n_coarse * dpp))
    # coarse dof indices (for the NPZ coarse_nodes field): all dofs of coarse polys
    coarse_dofs = np.concatenate(
        [np.arange(p * dpp, (p + 1) * dpp) for p in coarse_poly]
    ).astype(np.int64) if n_coarse else np.array([], dtype=np.int64)
    return P, coarse_dofs
