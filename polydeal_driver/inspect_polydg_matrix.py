"""
Phase 0d sanity check: read the PolyDG diffusion matrix exported by
polydg_diffusion_export.cc and confirm it round-trips into the CSR
representation used by the existing NPZ pipeline.

Usage:
    python inspect_polydg_matrix.py polydg_D_matrix.mtx polydg_D_geometry.csv
"""

import sys
import numpy as np
import scipy.io
import scipy.sparse as sp


def inspect(mtx_path, geom_path=None):
    # One-line read of the MatrixMarket file -> CSR (matches A_values /
    # A_row_ptr / A_col_idx in the existing p-value NPZ schema).
    A = scipy.io.mmread(mtx_path).tocsr()
    n, m = A.shape

    print(f"matrix file        : {mtx_path}")
    print(f"shape              : {n} x {m}")
    print(f"nnz                : {A.nnz}")
    print(f"symmetric (A==A^T) : {np.allclose((A - A.T).data, 0.0)}")

    # SIP-DG diffusion should be SPD-ish: positive diagonal, near-zero row sums
    # away from the boundary (consistency on constants up to the penalty).
    diag = A.diagonal()
    print(f"min diagonal       : {diag.min():.6e}")
    print(f"max |row sum|      : {np.abs(np.asarray(A.sum(axis=1)).ravel()).max():.6e}")

    # Rough symmetry -> eigenvalues via symmetric solver on the dense form
    # (fine for the small inspection meshes).
    if n <= 4000:
        w = np.linalg.eigvalsh(A.toarray())
        print(f"eigenvalue range   : [{w.min():.6e}, {w.max():.6e}]")
        print(f"SPD                : {w.min() > -1e-8}")

    # CSR triplet ready for the existing NPZ writer:
    print("\nCSR arrays for NPZ schema:")
    print(f"  A_values  : shape {A.data.shape}")
    print(f"  A_col_idx : shape {A.indices.shape}")
    print(f"  A_row_ptr : shape {A.indptr.shape}")

    if geom_path:
        geom = np.genfromtxt(geom_path, delimiter=",", names=True)
        print(f"\ngeometry file      : {geom_path}")
        print(f"  n_polytopes      : {geom.shape[0]}")
        print(f"  diameter range   : [{geom['diameter'].min():.4f}, {geom['diameter'].max():.4f}]")
        print(f"  volume range     : [{geom['volume'].min():.4e}, {geom['volume'].max():.4e}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inspect(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
