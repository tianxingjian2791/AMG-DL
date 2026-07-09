#!/usr/bin/env python3
"""
Batch-generate a polygonal PolyDG diffusion dataset (Option A).

Variety follows the paper (2111.01629v2): four coefficient PATTERNS x a range
of contrast EPSILONs (high = 10^epsilon), plus mesh variety (n_subdomains,
refinement, METIS seed). For each config:
  1. run the C++ driver  -> <tmp>/mesh_XXXX_matrix.mtx + _geometry.csv
  2. run build_npz_sample -> theta_gnn + p_value NPZ samples (theta grid)
into  <out_root>/{train,test}/raw/{theta_gnn_npy,p_value_npy}/<problem>/
with globally continuous sample indices. Configs are split train/test by ratio.

Patterns: 0 vertical_stripes, 1 checkerboard_2x2, 2 vertical_stripes_4,
          3 checkerboard_4x4.

Example:
  python generate_dataset.py \
      --driver ./examples/polydg_diffusion_hetero.g \
      --builder build_npz_sample.py \
      --out-root datasets_poly --problem D \
      --patterns 0 1 2 3 --epsilons 1 2 3 4 \
      --n-subdomains 120 --refinements 6 --seeds 0 1 2 \
      --degree 1 --n-thetas 15 --test-frac 0.3
"""

import argparse
import os
import subprocess
import tempfile
import numpy as np


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, help="path to polydg_diffusion_hetero.g")
    ap.add_argument("--builder", required=True, help="path to build_npz_sample.py")
    ap.add_argument("--out-root", default="datasets_poly")
    ap.add_argument("--problem", default="D")
    ap.add_argument("--n-subdomains", type=int, nargs="+", default=[120])
    ap.add_argument("--refinements", type=int, nargs="+", default=[6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patterns", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="0 vstripes, 1 check2x2, 2 vstripes4, 3 check4x4")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1, 2, 3, 4],
                    help="contrast exponents; high = 10^epsilon")
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--n-thetas", type=int, default=15)
    ap.add_argument("--theta-lo", type=float, default=0.02)
    ap.add_argument("--theta-hi", type=float, default=0.9)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--tmp", default=None, help="scratch dir for .mtx files")
    ap.add_argument("--amg-backend", choices=["pyamg", "cpp"], default="pyamg",
                    help="AMG backend for rho labels. 'cpp' uses HYPRE BoomerAMG "
                         "(same scale as FEM data, handles eps=9.5); requires "
                         "--amg-solver and emits theta_cnn_npy in addition to GNN/p-value.")
    ap.add_argument("--amg-solver", default=None,
                    help="path to polydg_amg_solve binary (required for --amg-backend cpp)")
    ap.add_argument("--formats", nargs="+", default=["gnn", "pvalue"],
                    choices=["gnn", "pvalue", "cnn"],
                    help="which sample schemas to write. 'cnn' requires --amg-backend cpp. "
                         "CNN-only (--formats cnn) skips block C/F + strength + p_value "
                         "writes -> much faster and far less disk.")
    args = ap.parse_args()

    thetas = np.linspace(args.theta_lo, args.theta_hi, args.n_thetas)
    theta_str = [f"{t:.4f}" for t in thetas]

    # Enumerate configs (pattern x epsilon x mesh), shuffle, split train/test.
    configs = [(pat, eps, ns, r, s)
               for pat in args.patterns
               for eps in args.epsilons
               for ns in args.n_subdomains
               for r in args.refinements
               for s in args.seeds]
    rng = np.random.default_rng(12345)
    rng.shuffle(configs)
    n_test = int(round(len(configs) * args.test_frac))
    split = {"test": configs[:n_test], "train": configs[n_test:]}
    print(f"{len(configs)} configs -> "
          f"{len(split['train'])} train / {len(split['test'])} test, "
          f"{args.n_thetas} thetas each")

    tmp = args.tmp or tempfile.mkdtemp(prefix="polydg_")
    os.makedirs(tmp, exist_ok=True)

    if args.amg_backend == "cpp" and not args.amg_solver:
        raise SystemExit("--amg-backend cpp requires --amg-solver <path>")

    for phase in ("train", "test"):
        idx = 0  # continuous sample index within this split
        out_theta = os.path.join(args.out_root, phase, "raw",
                                 "theta_gnn_npy", f"{phase}_{args.problem}")
        out_pval = os.path.join(args.out_root, phase, "raw",
                                "p_value_npy", f"{phase}_{args.problem}")
        out_cnn = os.path.join(args.out_root, phase, "raw",
                               "theta_cnn_npy", f"{phase}_{args.problem}")
        for mi, (pat, eps, ns, r, s) in enumerate(split[phase]):
            prefix = os.path.join(tmp, f"{phase}_cfg_{mi:04d}")
            # driver CLI: n_sub degree refine pattern epsilon prefix
            # (mesh variety comes from n_subdomains; METIS is deterministic so
            #  --seeds does NOT vary the matrix -- avoid duplicate configs.)
            run([args.driver, str(ns), str(args.degree), str(r),
                 str(pat), str(eps), prefix])
            builder_cmd = ["python3", args.builder, prefix + "_matrix.mtx",
                           "--geometry", prefix + "_geometry.csv",
                           "--coarsening", "block",
                           "--thetas", *theta_str,
                           "--epsilon", str(eps),
                           "--pattern-id", str(pat),
                           "--refinement", str(ns),
                           "--start-index", str(idx)]
            if "gnn" in args.formats:
                builder_cmd += ["--out-theta", out_theta]
            if "pvalue" in args.formats:
                builder_cmd += ["--out-pvalue", out_pval]
            if args.amg_backend == "cpp":
                builder_cmd += ["--amg-backend", "cpp",
                                "--amg-solver", args.amg_solver]
                if "cnn" in args.formats:
                    builder_cmd += ["--out-cnn", out_cnn]
            run(builder_cmd)
            idx += args.n_thetas
        print(f"[{phase}] wrote ~{idx} samples to {out_theta}")


if __name__ == "__main__":
    main()
