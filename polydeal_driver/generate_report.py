#!/usr/bin/env python3
"""
Generate the polygonal-mesh analogue of AMG-DL's diffusion_reports/D.csv.

For every (pattern, epsilon, refinement, theta) it runs the PolyDG driver to
assemble the SIP-DG matrix, then a pyamg AMG-preconditioned CG solve to record
the same 14 columns as the paper's report:

  scale,scale_id,pattern,pattern_id,epsilon,refinement,h,theta,rho,
  iterations,elapsed_sec,n_levels,n,nnz

Output goes to <out-root>/raw/diffusion_reports/D.csv, i.e. exactly the layout
the AMG-DL scripts/ expect (they glob diffusion_reports/*.csv).

Two AMG backends (--amg-backend):
  pyamg : classical Ruge-Stueben, capped at eps<=5 (breaks down higher).
  cpp   : HYPRE BoomerAMG via polydg_amg_solve -- SAME backend/scale as the
          FEM training data and the polygonal CNN dataset, reaches eps=9.5.
          Requires --amg-solver and the driver to export <prefix>_rhs.mtx.

Example:
  python generate_report.py \
      --driver ./examples/polydg_diffusion_hetero.g \
      --out-root results_poly/datasets/diffusion/large \
      --scale large --n-subdomains 150 250 400 600 \
      --patterns 0 1 2 3 --epsilons 0 0.8 1.6 2.4 5.0 9.5 \
      --refinement 7 --n-thetas 10

Note: n_subdomains is the SIZE AXIS (matrix size and h track it); refinement is
a single fixed value that only resolves each polytope. This differs from the FEM
report, where refinement drove size -- see the --n-subdomains help.
"""

import argparse
import os
import subprocess
import tempfile
import numpy as np
import scipy.io

import amg_operators as amg

PATTERN_NAMES = {0: "vertical_stripes", 1: "checkerboard_2x2",
                 2: "vertical_stripes_4", 3: "checkerboard_4x4"}
HEADER = ("scale,scale_id,pattern,pattern_id,epsilon,refinement,h,theta,rho,"
          "iterations,elapsed_sec,n_levels,n,nnz\n")


def read_h(geometry_csv):
    g = np.genfromtxt(geometry_csv, delimiter=",", names=True)
    return float(np.max(g["diameter"]))


def cpp_solve_all(solver_bin, mtx_path, rhs_path, thetas):
    """Run polydg_amg_solve over the whole theta grid; return
    {theta: {rho, iterations, elapsed_sec, n_levels}}."""
    theta_arg = ",".join(f"{t:.10g}" for t in thetas)
    out = subprocess.run([solver_bin, mtx_path, rhs_path, theta_arg],
                         check=True, capture_output=True, text=True)
    res = {}
    for line in out.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 5:
            continue
        th, rho, it, el, lvl = parts
        res[float(th)] = {"rho": float(rho), "iterations": int(float(it)),
                          "elapsed_sec": float(el), "n_levels": int(float(lvl))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out-root", required=True,
                    help="dataset root; report written to <root>/raw/diffusion_reports/D.csv")
    ap.add_argument("--scale", default="large")
    ap.add_argument("--scale-id", type=int, default=1)
    ap.add_argument("--problem", default="D")
    ap.add_argument("--n-subdomains", type=int, nargs="+",
                    default=[150, 250, 400, 600],
                    help="SIZE AXIS: matrix size = n_subdomains * dofs_per_poly, "
                         "and h (max polytope diameter) shrinks as it grows. This "
                         "is the polytopic analogue of the paper's refinement sweep "
                         "(refinement here only resolves each polytope, not size).")
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--patterns", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0, 0.8, 1.6, 2.4, 5.0, 9.5])
    ap.add_argument("--refinement", type=int, default=7,
                    help="fixed global refinement; must give >= 4*max(n_subdomains) "
                         "fine cells so every polytope is a real agglomerate "
                         "(refine=7 -> 16384 cells, ok up to n_subdomains~4000)")
    ap.add_argument("--n-thetas", type=int, default=10)
    ap.add_argument("--theta-lo", type=float, default=0.02)
    ap.add_argument("--theta-hi", type=float, default=0.9)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--amg-backend", choices=["pyamg", "cpp"], default="pyamg",
                    help="'cpp' uses HYPRE BoomerAMG via polydg_amg_solve -- same "
                         "scale as FEM training data, reaches eps=9.5. Requires "
                         "--amg-solver. 'pyamg' is classical RS, capped at eps<=5.")
    ap.add_argument("--amg-solver", default=None,
                    help="path to polydg_amg_solve binary (required for --amg-backend cpp)")
    args = ap.parse_args()

    if args.amg_backend == "cpp" and not args.amg_solver:
        raise SystemExit("--amg-backend cpp requires --amg-solver <path>")

    thetas = np.linspace(args.theta_lo, args.theta_hi, args.n_thetas)
    report_dir = os.path.join(args.out_root, "raw", "diffusion_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{args.problem}.csv")

    tmp = args.tmp or tempfile.mkdtemp(prefix="polydg_report_")
    os.makedirs(tmp, exist_ok=True)

    n_rows = 0
    with open(report_path, "w") as report:
        report.write(HEADER)
        for pat in args.patterns:
            for eps in args.epsilons:
                for ns in args.n_subdomains:
                    prefix = os.path.join(tmp, f"p{pat}_e{eps}_n{ns}")
                    # driver CLI: n_sub degree refine pattern epsilon prefix
                    # (refinement fixed; n_subdomains is the size axis)
                    subprocess.run(
                        [args.driver, str(ns), str(args.degree),
                         str(args.refinement), str(pat), str(eps), prefix],
                        check=True)
                    A = scipy.io.mmread(prefix + "_matrix.mtx").tocsr()
                    h = read_h(prefix + "_geometry.csv")
                    n, nnz = A.shape[0], A.nnz

                    if args.amg_backend == "cpp":
                        results = cpp_solve_all(
                            args.amg_solver,
                            prefix + "_matrix.mtx",
                            prefix + "_rhs.mtx",
                            thetas)
                    else:
                        results = None  # use amg.amg_report_metrics per theta

                    for theta in thetas:
                        if results is not None:
                            m = results.get(theta)
                            if m is None:
                                # float key mismatch — find nearest
                                key = min(results.keys(),
                                          key=lambda k: abs(k - theta))
                                m = results[key]
                        else:
                            m = amg.amg_report_metrics(A, float(theta))
                        # 'refinement' column carries the size label (n_subdomains)
                        # so the AMG-DL scripts' (refinement,h) grouping is 1:1.
                        report.write(
                            f"{args.scale},{args.scale_id},"
                            f"{PATTERN_NAMES[pat]},{pat},{eps},{ns},{h:.8g},"
                            f"{theta:.8g},{m['rho']:.8g},{m['iterations']},"
                            f"{m['elapsed_sec']:.8g},{m['n_levels']},{n},{nnz}\n")
                        report.flush()
                        n_rows += 1
                    backend_tag = "BoomerAMG" if args.amg_backend == "cpp" else "pyamg"
                    print(f"pat={PATTERN_NAMES[pat]} eps={eps} n_sub={ns} "
                          f"n={n} h={h:.4g} [{backend_tag}] -> {args.n_thetas} rows")
    print(f"wrote {n_rows} rows to {report_path}")


if __name__ == "__main__":
    main()
