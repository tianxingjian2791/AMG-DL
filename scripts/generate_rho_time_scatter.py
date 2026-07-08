#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data import SampleRecordRepository


DEFAULT_INPUT_GLOB = "datasets/diffusion/large/**/raw/diffusion_reports/*.csv"
DEFAULT_H_VALUES = (
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
)
H_COLORS = {
    0.125: "#440154",
    0.0625: "#253494",
    0.03125: "#1f9eb7",
    0.015625: "#008837",
    0.0078125: "#31a354",
    0.00390625: "#b8a900",
    0.001953125: "#e6550d",
    0.0009765625: "#b30000",
}


@dataclass(frozen=True)
class Point:
    h: float
    pattern: str
    epsilon: float
    theta: float
    rho: float
    elapsed: float
    count: int


def parse_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(v.strip()) for v in raw.split(",") if v.strip())


def load_points(input_glob: str) -> tuple[list[Point], list[str]]:
    paths = sorted(glob.glob(input_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matched {input_glob}")
    records = SampleRecordRepository.from_glob(input_glob).all()
    grouped: dict[tuple[float, str, float, float], list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        try:
            meta = record.sample_meta
            metrics = record.metrics
            key = (
                round(float(meta["h"]), 8),
                str(meta.get("pattern", "")),
                round(float(meta["epsilon"]), 8),
                round(float(meta["theta"]), 8),
            )
            grouped[key].append((float(metrics["rho"]), float(metrics["elapsed_sec"])))
        except (KeyError, TypeError, ValueError):
            continue
    points = [
        Point(
            h=key[0],
            pattern=key[1],
            epsilon=key[2],
            theta=key[3],
            rho=statistics.mean(v[0] for v in values),
            elapsed=statistics.mean(v[1] for v in values),
            count=len(values),
        )
        for key, values in grouped.items()
    ]
    return points, paths


def normalize_by_test_case(points: list[Point]) -> tuple[list[Point], list[float], list[float], int, int]:
    groups: dict[tuple[float, str, float], list[Point]] = defaultdict(list)
    for point in points:
        groups[(point.h, point.pattern, point.epsilon)].append(point)

    kept: list[Point] = []
    norm_rho: list[float] = []
    norm_time: list[float] = []
    skipped = 0
    used_groups = 0
    for group in groups.values():
        if len(group) < 2:
            skipped += len(group)
            continue
        rhos = [p.rho for p in group]
        times = [p.elapsed for p in group]
        rho_mean = statistics.mean(rhos)
        time_mean = statistics.mean(times)
        rho_std = statistics.pstdev(rhos)
        time_std = statistics.pstdev(times)
        if rho_std <= 0 or time_std <= 0:
            skipped += len(group)
            continue
        used_groups += 1
        for point in group:
            kept.append(point)
            norm_rho.append((point.rho - rho_mean) / rho_std)
            norm_time.append((point.elapsed - time_mean) / time_std)
    return kept, norm_rho, norm_time, skipped, used_groups


def axis_limits(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-9)
    return lo - 0.08 * span, hi + 0.08 * span


def ticks(lo: float, hi: float) -> list[int]:
    return [v for v in range(math.ceil(lo), math.floor(hi) + 1) if lo <= v <= hi]


def fmt_h(h: float) -> str:
    return f"{h:.3e}"


def write_csv(path: Path, points: list[Point], x: list[float], y: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "h",
                "pattern",
                "epsilon",
                "theta",
                "rho",
                "elapsed_sec",
                "normalized_rho",
                "normalized_elapsed_sec",
                "count",
            ]
        )
        for point, nx, ny in zip(points, x, y):
            writer.writerow([point.h, point.pattern, point.epsilon, point.theta, point.rho, point.elapsed, nx, ny, point.count])


def write_png(path: Path, points: list[Point], x_values: list[float], y_values: list[float], h_values: tuple[float, ...], scale: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG output uses matplotlib. Install project requirements, e.g. `pip install -r requirements.txt`, "
            "or run with the repo .venv if matplotlib is installed there."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    x_lo, x_hi = axis_limits(x_values)
    y_lo, y_hi = axis_limits(y_values)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )
    dpi = 100 * max(1, scale)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=dpi)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.13)

    for h in h_values:
        h_key = round(h, 8)
        xs = [x for point, x in zip(points, x_values) if round(point.h, 8) == h_key]
        ys = [y for point, y in zip(points, y_values) if round(point.h, 8) == h_key]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=7.0,
            c=H_COLORS.get(h_key, "#222222"),
            edgecolors="black",
            linewidths=0.25,
            label=f"h={fmt_h(h)}",
            zorder=3,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"normalized $\rho$", fontsize=9)
    ax.set_ylabel(r"normalized $t$", fontsize=9)
    ax.set_xticks(ticks(x_lo, x_hi))
    ax.set_yticks(ticks(y_lo, y_hi))
    ax.tick_params(axis="both", labelsize=8, length=3)
    ax.grid(True, color="#d4d4d4", linewidth=0.55, zorder=0)
    legend = ax.legend(
        loc="upper right",
        fontsize=6.2,
        frameon=True,
        framealpha=0.9,
        fancybox=False,
        edgecolor="#777777",
        markerscale=0.8,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.35,
    )
    legend.get_frame().set_linewidth(0.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rho/time scatter normalized inside each same test case: h + pattern + epsilon."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--out-dir", default="results/figures/rho_time_scatter")
    parser.add_argument("--h-values", default=",".join(str(v) for v in DEFAULT_H_VALUES))
    parser.add_argument("--png-name", default="rho_time_scatter_large.png")
    parser.add_argument("--csv-name", default="rho_time_scatter_large.csv")
    parser.add_argument("--png-scale", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    h_values = parse_floats(args.h_values, DEFAULT_H_VALUES)
    h_set = {round(v, 8) for v in h_values}

    points, paths = load_points(args.input_glob)
    points = [point for point in points if round(point.h, 8) in h_set]
    if not points:
        raise ValueError("No compatible points found for the requested h values.")

    points, norm_rho, norm_time, skipped, used_groups = normalize_by_test_case(points)
    if not points:
        raise ValueError("No same-test-case subset had enough variation to normalize.")

    print(f"Loaded {len(points)} averaged points from {len(paths)} file(s).")
    print(f"Normalization: each same test case independently: h + pattern + epsilon ({used_groups} groups).")
    if skipped:
        print(f"WARNING: skipped {skipped} point(s) from same-test-case subsets without variation.")
    if max((p.count for p in points), default=0) < 2:
        print("WARNING: no repeated samples were found; each averaged point is based on one record.")

    csv_path = out_dir / args.csv_name
    png_path = out_dir / args.png_name
    write_csv(csv_path, points, norm_rho, norm_time)
    write_png(png_path, points, norm_rho, norm_time, h_values, max(1, args.png_scale))
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
