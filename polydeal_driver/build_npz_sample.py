"""
Complete one PolyDG training sample (Option A) from the exported matrix.

Reads:  <prefix>_matrix.mtx  (+ optional <prefix>_geometry.csv)
Runs :  strength -> C/F splitting -> baseline P -> S -> rho, per theta
Writes: theta_gnn NPZ (Stage 1)  and  p_value NPZ (Stage 2), in the EXACT
        schema data_loader_npy.py expects.

theta_gnn keys : edge_index(2,E) int, edge_attr(E,) f, theta(1,) f,
                 y(1,)=rho f, metadata=[n, rho, h, epsilon]
p_value  keys  : A_{values,col_idx,row_ptr}, coarse_nodes,
                 P_{values,col_idx,row_ptr}, S_{values,col_idx,row_ptr},
                 metadata=[n, theta, rho, h]

Usage:
    python build_npz_sample.py polydg_D_matrix.mtx \
        --geometry polydg_D_geometry.csv \
        --thetas 0.1 0.25 0.5 \
        --out-theta datasets_poly/train/raw/theta_gnn_npy/train_D \
        --out-pvalue datasets_poly/train/raw/p_value_npy/train_D
"""

import argparse
import os
import numpy as np
import scipy.io
import scipy.sparse as sp

import amg_operators as amg


def read_geometry(geometry_csv):
    """Return (h = max diameter, n_polytopes) or (None, None) if no file."""
    if geometry_csv and os.path.exists(geometry_csv):
        g = np.genfromtxt(geometry_csv, delimiter=",", names=True)
        h = float(np.max(g["diameter"]))
        n_poly = int(g["polytope_index"].shape[0]) if g.ndim else 1
        return h, n_poly
    return None, None


def pool_matrix(A_csr, m=50):
    """50x50 pooled image of A, matching include/Pooling.hpp exactly:
    uneven block partition (q=n//m, p=n%m, t=(q+1)*p), SUM op, then
    std_normalize over all m*m entries. Used as the CNN input."""
    n = A_csr.shape[0]
    if n <= 0:
        return np.zeros((m, m), dtype=np.float64)
    q = n // m
    p = n % m
    t = (q + 1) * p

    def bucket(idx):
        idx = np.asarray(idx)
        lo = idx < t
        out = np.empty_like(idx)
        # low region: blocks of size q+1; high region: blocks of size q offset by p
        out[lo] = idx[lo] // (q + 1)
        out[~lo] = (idx[~lo] - t) // q + p
        return np.clip(out, 0, m - 1)

    coo = A_csr.tocoo()
    I = bucket(coo.row)
    J = bucket(coo.col)
    V = np.zeros((m, m), dtype=np.float64)
    np.add.at(V, (I, J), coo.data)  # SUM pooling

    # std_normalize over all m*m entries
    mean = V.mean()
    std = V.std()
    if std > 0:
        V = (V - mean) / std
    else:
        V = V - mean
    return V


def save_theta_cnn(out_dir, idx, pooled, rho, n, h, theta,
                   pattern_id, epsilon, refinement, iterations):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"sample_{idx:05d}.npz"),
        pooled_matrix=pooled.astype(np.float64),
        y=np.array([rho], dtype=np.float64),
        # schema: [n, rho, h, theta, pattern_id, epsilon, refinement, iterations]
        metadata=np.array([n, rho, h, theta, pattern_id, epsilon,
                           refinement, iterations], dtype=np.float64),
    )


def cpp_amg_solve(solver_bin, mtx_path, rhs_path, thetas):
    """Call polydg_amg_solve for the whole theta grid; return
    {theta: (rho, iterations, elapsed_sec, n_levels)}. Requires the paired
    <prefix>_rhs.mtx to exist next to the matrix."""
    import subprocess
    theta_arg = ",".join(f"{t:.10g}" for t in thetas)
    out = subprocess.run([solver_bin, mtx_path, rhs_path, theta_arg],
                         check=True, capture_output=True, text=True)
    result = {}
    for line in out.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 5:
            continue
        th, rho, it, el, lvl = parts
        result[float(th)] = (float(rho), int(float(it)),
                             float(el), int(float(lvl)))
    return result


def save_theta_gnn(out_dir, idx, A_coo, theta, rho, n, h, epsilon,
                   pattern_id, refinement, iterations):
    os.makedirs(out_dir, exist_ok=True)
    edge_index = np.vstack([A_coo.row, A_coo.col]).astype(np.int64)
    edge_attr = A_coo.data.astype(np.float64)
    np.savez(
        os.path.join(out_dir, f"sample_{idx:05d}.npz"),
        edge_index=edge_index,
        edge_attr=edge_attr,
        theta=np.array([theta], dtype=np.float64),
        y=np.array([rho], dtype=np.float64),
        # schema: [n, rho, h, epsilon, pattern_id, refinement, iterations]
        metadata=np.array([n, rho, h, epsilon, pattern_id, refinement,
                           iterations], dtype=np.float64),
    )


def save_pvalue(out_dir, idx, A, P, S, coarse_nodes, theta, rho, n, h,
                pattern_id, epsilon, refinement, iterations):
    os.makedirs(out_dir, exist_ok=True)
    A = A.tocsr(); P = P.tocsr(); S = sp.csr_matrix(S)
    np.savez(
        os.path.join(out_dir, f"sample_{idx:05d}.npz"),
        A_values=A.data.astype(np.float64),
        A_col_idx=A.indices.astype(np.int64),
        A_row_ptr=A.indptr.astype(np.int64),
        coarse_nodes=coarse_nodes.astype(np.int64),
        P_values=P.data.astype(np.float64),
        P_col_idx=P.indices.astype(np.int64),
        P_row_ptr=P.indptr.astype(np.int64),
        S_values=S.data.astype(np.float64),
        S_col_idx=S.indices.astype(np.int64),
        S_row_ptr=S.indptr.astype(np.int64),
        # schema: [n, theta, rho, h, pattern_id, epsilon, refinement, iterations]
        metadata=np.array([n, theta, rho, h, pattern_id, epsilon,
                           refinement, iterations], dtype=np.float64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mtx")
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5])
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="PDE coefficient tag stored in metadata")
    ap.add_argument("--pattern-id", type=int, default=-1,
                    help="DiffusionPattern enum value (0-3); stored in metadata")
    ap.add_argument("--refinement", type=int, default=-1,
                    help="n_subdomains used as size-axis label; stored in metadata")
    ap.add_argument("--out-theta", default=None,
                    help="output dir for theta_gnn_npy; omit to skip GNN samples")
    ap.add_argument("--out-pvalue", default=None,
                    help="output dir for p_value_npy; omit to skip Stage-2 samples")
    ap.add_argument("--out-cnn", default=None,
                    help="output dir for theta_cnn_npy samples; requires --amg-backend cpp")
    ap.add_argument("--amg-backend", choices=["pyamg", "cpp"], default="pyamg",
                    help="pyamg: Python Ruge-Stuben (no HYPRE, eps<=5); "
                         "cpp: polydg_amg_solve binary with HYPRE BoomerAMG")
    ap.add_argument("--amg-solver", default=None,
                    help="path to polydg_amg_solve binary (required for --amg-backend cpp)")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--coarsening", choices=["block", "pointwise"],
                    default="block",
                    help="block = polytope-level (correct for DG); "
                         "pointwise = classical RS (diverges on DG, for comparison)")
    ap.add_argument("--n-poly", type=int, default=None,
                    help="polytope count (else read from geometry file)")
    args = ap.parse_args()

    if args.amg_backend == "cpp" and not args.amg_solver:
        raise SystemExit("--amg-backend cpp requires --amg-solver <path>")
    if args.out_cnn and args.amg_backend != "cpp":
        raise SystemExit("--out-cnn requires --amg-backend cpp (needs the rhs.mtx file)")
    if not any([args.out_theta, args.out_pvalue, args.out_cnn]):
        raise SystemExit("at least one of --out-theta / --out-pvalue / --out-cnn must be specified")

    need_cf = args.out_theta is not None or args.out_pvalue is not None

    A = scipy.io.mmread(args.mtx).tocsr()
    n = A.shape[0]
    A_coo = A.tocoo()
    h, n_poly = read_geometry(args.geometry)
    if h is None:
        h = 1.0 / np.sqrt(n)  # crude fallback
        print(f"[warn] no geometry; using fallback h={h:.4f}")
    if args.n_poly is not None:
        n_poly = args.n_poly

    dpp = None
    if args.coarsening == "block":
        if n_poly is None:
            raise SystemExit("block coarsening needs --n-poly or a geometry file")
        dpp = amg.infer_block_size(n, n_poly)
        if not amg.verify_block_contiguous(A, n_poly, dpp):
            print("[warn] per-polytope diagonal blocks don't look contiguous/"
                  "symmetric; block coarsening assumes dof block I*dpp:(I+1)*dpp "
                  "belongs to polytope I. Proceeding anyway.")
        print(f"block coarsening: n_poly={n_poly}, dofs/polytope={dpp}")

    print(f"matrix {n}x{n}, nnz={A.nnz}, h={h:.4f}")

    # Pre-pool the matrix once (same for every theta). Only needed for CNN.
    pooled = None
    if args.out_cnn:
        pooled = pool_matrix(A, m=50)
        print(f"pooled matrix: 50x50, mean={pooled.mean():.3f}, std={pooled.std():.3f}")

    # If using the C++ backend, run all thetas in one subprocess call.
    cpp_results = {}
    if args.amg_backend == "cpp":
        rhs_path = args.mtx.replace("_matrix.mtx", "_rhs.mtx")
        if not os.path.exists(rhs_path):
            raise SystemExit(f"RHS file not found: {rhs_path}\n"
                             "Rebuild the driver (polydg_diffusion_hetero.cc) to export _rhs.mtx.")
        print(f"cpp AMG backend: {args.amg_solver}")
        cpp_results = cpp_amg_solve(args.amg_solver, args.mtx, rhs_path, args.thetas)

    idx = args.start_index
    for theta in args.thetas:
        # ---- AMG metrics (rho, iterations) --------------------------------
        if args.amg_backend == "cpp":
            if theta not in cpp_results:
                match = min(cpp_results.keys(), key=lambda k: abs(k - theta),
                            default=None)
                if match is None or abs(match - theta) > 1e-8:
                    print(f"  theta={theta:.3f}: no cpp result, skipping")
                    continue
                rho, n_iter, elapsed, n_levels = cpp_results[match]
            else:
                rho, n_iter, elapsed, n_levels = cpp_results[theta]
        else:
            rho, n_iter = amg.amg_convergence_factor(A, theta)
            n_levels = -1

        # ---- C/F splitting + baseline P + strength S (skip if CNN-only) ---
        coarse = None
        if need_cf:
            if args.coarsening == "block":
                W = amg.block_offdiag_norms(A, n_poly, dpp)
                poly_markers = amg.block_cf_splitting(W, theta)
                P, coarse = amg.block_prolongation(A, poly_markers, dpp)
            else:
                cf = amg.classical_cf_splitting(A, theta)
                coarse = amg.extract_coarse_nodes(cf)
                P = amg.compute_baseline_prolongation(A, cf)
            if P.shape[1] == 0:
                print(f"  theta={theta:.3f}: no coarse nodes, skipping")
                continue
            S = amg.compute_strength_matrix(A, theta)

        # ---- write samples -------------------------------------------------
        if args.out_theta is not None:
            save_theta_gnn(args.out_theta, idx, A_coo, theta, rho, n, h,
                           args.epsilon, args.pattern_id, args.refinement, n_iter)
        if args.out_pvalue is not None:
            save_pvalue(args.out_pvalue, idx, A, P, S, coarse, theta, rho, n, h,
                        args.pattern_id, args.epsilon, args.refinement, n_iter)
        if args.out_cnn is not None and pooled is not None:
            save_theta_cnn(args.out_cnn, idx, pooled, rho, n, h, theta,
                           args.pattern_id, args.epsilon, args.refinement, n_iter)

        n_coarse_str = f"n_coarse={len(coarse)} " if coarse is not None else ""
        print(f"  theta={theta:.3f}: {n_coarse_str}"
              f"cg_iters={n_iter} rho={rho:.4f} n_levels={n_levels} "
              f"-> sample_{idx:05d}.npz")
        idx += 1

    print(f"wrote {idx - args.start_index} samples")


if __name__ == "__main__":
    main()
