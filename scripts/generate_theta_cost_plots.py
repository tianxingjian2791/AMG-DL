#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data import SampleRecordRepository


DEFAULT_INPUT_GLOB = "datasets/diffusion/large/**/diffusion_reports/*.csv"
DEFAULT_PATTERNS = ("vertical_stripes_4", "checkerboard_4x4")
DEFAULT_EPSILONS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 7.0, 9.5)
DEFAULT_H_VALUES = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625)
PATTERN_TITLES = {
    "vertical_stripes": "vertical split",
    "vertical_stripes_4": "stripes",
    "checkerboard_2x2": "checkerboard 2x2",
    "checkerboard_4x4": "checkerboard 4x4",
}
SVG_FONT_FAMILY = "Times New Roman"
INTENSE_COLORS = (
    "#001f9e",
    "#0057ff",
    "#007a99",
    "#008c3a",
    "#00b020",
    "#b8a900",
    "#ff7a00",
    "#d93000",
    "#a00000",
    "#4b0082",
    "#111111",
    "#7a3b00",
)


@dataclass(frozen=True)
class CostStats:
    mean: float
    std: float
    count: int


def _parse_csv_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw or raw.strip().lower() == "auto":
        return default
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _parse_csv_text(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw or raw.strip().lower() == "auto":
        return default
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _key(pattern: str, epsilon: float, h: float, theta: float) -> tuple[str, float, float, float]:
    return (pattern, round(epsilon, 8), round(h, 8), round(theta, 8))


def _format_float(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.10g}"


def _format_h_title(h: float) -> str:
    return f"{h:.2e}"


def _load_cost_stats(input_glob: str) -> tuple[dict[tuple[str, float, float, float], CostStats], list[str]]:
    matches = sorted(glob.glob(input_glob, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No files matched --input-glob: {input_glob}")

    records = SampleRecordRepository.from_glob(input_glob).all()
    if not records:
        raise ValueError(
            "No compatible diffusion records were loaded. Expected report CSVs with "
            "scale, pattern, epsilon, h, theta, rho, elapsed_sec columns."
        )

    grouped: dict[tuple[str, float, float, float], list[float]] = defaultdict(list)
    for record in records:
        meta = record.sample_meta
        metrics = record.metrics
        try:
            pattern = str(meta["pattern"])
            epsilon = float(meta["epsilon"])
            h = float(meta["h"])
            theta = float(meta["theta"])
            elapsed = float(metrics["elapsed_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[_key(pattern, epsilon, h, theta)].append(elapsed)

    stats = {}
    for key, values in grouped.items():
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[key] = CostStats(mean=statistics.mean(values), std=std, count=len(values))
    return stats, matches


def _available_thetas(
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
) -> tuple[float, ...]:
    eps_set = {round(value, 8) for value in epsilons}
    h_set = {round(value, 8) for value in h_values}
    values = {
        theta
        for pat, epsilon, h, theta in stats
        if pat == pattern and epsilon in eps_set and h in h_set
    }
    return tuple(sorted(values))


def _available_patterns(stats: dict[tuple[str, float, float, float], CostStats]) -> tuple[str, ...]:
    return tuple(sorted({pattern for pattern, _, _, _ in stats}))


def _available_epsilons(
    stats: dict[tuple[str, float, float, float], CostStats],
    patterns: tuple[str, ...],
) -> tuple[float, ...]:
    pattern_set = set(patterns)
    return tuple(sorted({epsilon for pattern, epsilon, _, _ in stats if pattern in pattern_set}))


def _available_h_values(
    stats: dict[tuple[str, float, float, float], CostStats],
    patterns: tuple[str, ...],
    epsilons: tuple[float, ...],
) -> tuple[float, ...]:
    pattern_set = set(patterns)
    epsilon_set = {round(value, 8) for value in epsilons}
    return tuple(sorted({h for pattern, epsilon, h, _ in stats if pattern in pattern_set and epsilon in epsilon_set}, reverse=True))


def _coverage(
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    thetas: tuple[float, ...],
) -> tuple[int, int, int]:
    total = len(epsilons) * len(h_values) * len(thetas)
    present = sum(
        1
        for epsilon in epsilons
        for h in h_values
        for theta in thetas
        if _key(pattern, epsilon, h, theta) in stats
    )
    max_repeats = max(
        (cell.count for key, cell in stats.items() if key[0] == pattern),
        default=0,
    )
    return present, total, max_repeats


def _write_summary_csv(
    destination: Path,
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    thetas: tuple[float, ...],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pattern", "epsilon", "h", "theta", "mean_elapsed_sec", "std_elapsed_sec", "count"])
        for h in h_values:
            for epsilon in epsilons:
                for theta in thetas:
                    cell = stats.get(_key(pattern, epsilon, h, theta))
                    if cell is None:
                        continue
                    writer.writerow([pattern, epsilon, h, theta, cell.mean, cell.std, cell.count])


def _legend_entries(epsilons: tuple[float, ...]) -> str:
    return ",".join(f"$\\epsilon={epsilon:.1f}$" for epsilon in epsilons)


def _axis_block(
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    h: float,
    epsilons: tuple[float, ...],
    thetas: tuple[float, ...],
    *,
    legend: bool,
) -> str:
    lines = [
        "\\nextgroupplot[",
        f"title={{h={_format_h_title(h)}}},",
        "xlabel={$\\theta$},",
        "ylabel={solve time [s]},",
        "xmin=0, xmax=0.92,",
        "grid=major,",
        "tick label style={font=\\scriptsize},",
        "label style={font=\\scriptsize},",
        "title style={font=\\scriptsize},",
        "]",
    ]
    marks = ("*", "triangle*", "square*", "diamond*", "pentagon*", "otimes*", "oplus*", "star", "x", "+")
    colors = (
        "blue",
        "cyan",
        "teal",
        "green!70!black",
        "lime!70!black",
        "yellow!70!black",
        "orange",
        "red!70!black",
        "red",
        "black",
    )
    for index, epsilon in enumerate(epsilons):
        coords = []
        for theta in thetas:
            cell = stats.get(_key(pattern, epsilon, h, theta))
            if cell is None:
                continue
            coords.append(
                f"({_format_float(theta)},{_format_float(cell.mean)}) +- (0,{_format_float(cell.std)})"
            )
        if not coords:
            continue
        color = colors[index % len(colors)]
        mark = marks[index % len(marks)]
        lines.append(
            "\\addplot+["
            f"{color}, mark={mark}, mark size=1.1pt, line width=0.45pt, "
            "error bars/.cd, y dir=both, y explicit"
            "] coordinates {"
        )
        lines.append(" ".join(coords))
        lines.append("};")
    if legend:
        lines.append(f"\\legend{{{_legend_entries(epsilons)}}}")
    return "\n".join(lines)


def _figure_tex(
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    thetas: tuple[float, ...],
) -> str:
    title = PATTERN_TITLES.get(pattern, pattern.replace("_", " "))
    columns = 2 if len(h_values) > 1 else 1
    rows = max(1, math.ceil(len(h_values) / columns))
    axes = [
        _axis_block(stats, pattern, h, epsilons, thetas, legend=(index == len(h_values) - 1))
        for index, h in enumerate(h_values)
    ]
    return "\n".join(
        [
            f"\\section*{{CPU time vs. \\theta$: {title}}}",
            "\\begin{center}",
            "\\begin{tikzpicture}",
            "\\begin{groupplot}[",
            f"group style={{group size={columns} by {rows}, horizontal sep=1.4cm, vertical sep=1.2cm}},",
            "width=0.44\\textwidth,",
            f"height={0.98 / rows:.3f}\\textheight,",
            "]",
            *axes,
            "\\end{groupplot}",
            "\\end{tikzpicture}",
            "\\end{center}",
        ]
    )


def _document_tex(figures: list[str]) -> str:
    return "\n\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[margin=0.55in]{geometry}",
            "\\usepackage{pgfplots}",
            "\\usepgfplotslibrary{groupplots}",
            "\\pgfplotsset{compat=1.18}",
            "\\begin{document}",
            *figures,
            "\\end{document}",
        ]
    ) + "\n"


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "middle") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="{SVG_FONT_FAMILY}">{escaped}</text>'


def _write_svg(
    destination: Path,
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    thetas: tuple[float, ...],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = 2 if len(h_values) > 1 else 1
    rows = max(1, math.ceil(len(h_values) / columns))
    width = 900
    height = 40 + rows * 275
    panel_w = 360
    panel_h = 230
    lefts = (70, 500) if columns == 2 else (270,)
    tops = tuple(40 + row * 275 for row in range(rows))
    plot_pad_l = 48
    plot_pad_r = 16
    plot_pad_t = 28
    plot_pad_b = 42
    colors = INTENSE_COLORS
    marks = ("circle", "triangle", "square", "diamond")
    all_y = [
        cell.mean + cell.std
        for key, cell in stats.items()
        if key[0] == pattern and key[1] in {round(e, 8) for e in epsilons}
    ]
    global_max = max(all_y, default=1.0)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    x_min, x_max = 0.0, 0.92
    for index, h in enumerate(h_values):
        col = index % 2
        row = index // 2
        x0 = lefts[col]
        y0 = tops[row]
        px0 = x0 + plot_pad_l
        py0 = y0 + plot_pad_t
        pw = panel_w - plot_pad_l - plot_pad_r
        ph = panel_h - plot_pad_t - plot_pad_b
        values = [
            cell.mean + cell.std
            for epsilon in epsilons
            for theta in thetas
            for cell in [stats.get(_key(pattern, epsilon, h, theta))]
            if cell is not None
        ]
        y_max = max(values) if values else global_max
        y_max = max(y_max * 1.12, 1e-12)

        def sx(theta: float) -> float:
            return px0 + (theta - x_min) / (x_max - x_min) * pw

        def sy(value: float) -> float:
            return py0 + ph - value / y_max * ph

        svg.append(f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="black" stroke-width="1"/>')
        svg.append(_svg_text(px0 + pw / 2, y0 + 14, f"h={_format_h_title(h)}", size=11))
        svg.append(_svg_text(px0 + pw / 2, y0 + panel_h - 8, "theta", size=10))
        svg.append(f'<text x="{x0 + 12:.1f}" y="{py0 + ph / 2:.1f}" font-size="10" text-anchor="middle" font-family="{SVG_FONT_FAMILY}" transform="rotate(-90 {x0 + 12:.1f} {py0 + ph / 2:.1f})">solve time [s]</text>')
        for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
            x = sx(tick)
            svg.append(f'<line x1="{x:.1f}" y1="{py0 + ph:.1f}" x2="{x:.1f}" y2="{py0 + ph + 4:.1f}" stroke="black" stroke-width="0.7"/>')
            svg.append(_svg_text(x, py0 + ph + 16, f"{tick:.1f}", size=8))
        for tick_index in range(5):
            value = y_max * tick_index / 4
            y = sy(value)
            svg.append(f'<line x1="{px0 - 4:.1f}" y1="{y:.1f}" x2="{px0:.1f}" y2="{y:.1f}" stroke="black" stroke-width="0.7"/>')
            svg.append(_svg_text(px0 - 7, y + 3, f"{value:.2g}", size=8, anchor="end"))

        for eps_index, epsilon in enumerate(epsilons):
            points = []
            for theta in thetas:
                cell = stats.get(_key(pattern, epsilon, h, theta))
                if cell is None:
                    continue
                x = sx(theta)
                y = sy(cell.mean)
                points.append((x, y, cell))
            if not points:
                continue
            color = colors[eps_index % len(colors)]
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.2"/>')
            for x, y, cell in points:
                if cell.std > 0:
                    y_low = sy(max(cell.mean - cell.std, 0.0))
                    y_high = sy(cell.mean + cell.std)
                    svg.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="0.8"/>')
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.0" fill="{color}"/>')

        if index == len(h_values) - 1:
            legend_cols = 4
            legend_col_w = 62
            legend_row_h = 13
            legend_rows = math.ceil(len(epsilons) / legend_cols)
            legend_w = legend_cols * legend_col_w + 8
            legend_h = legend_rows * legend_row_h + 8
            legend_x = px0 + pw - legend_w - 8
            legend_y = py0 + ph - legend_h - 8
            svg.append(
                f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" width="{legend_w:.1f}" height="{legend_h:.1f}" '
                'fill="white" fill-opacity="0.92" stroke="#b0b0b0" stroke-width="0.7"/>'
            )
            for legend_index, epsilon in enumerate(epsilons):
                x = legend_x + 8 + (legend_index % legend_cols) * legend_col_w
                y = legend_y + 10 + (legend_index // legend_cols) * legend_row_h
                color = colors[legend_index % len(colors)]
                svg.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 14:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="1.1"/>')
                svg.append(f'<circle cx="{x + 7:.1f}" cy="{y:.1f}" r="1.6" fill="{color}"/>')
                svg.append(_svg_text(x + 17, y + 3, f"e={epsilon:.1f}", size=6, anchor="start"))
    svg.append("</svg>")
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_png(
    destination: Path,
    stats: dict[tuple[str, float, float, float], CostStats],
    pattern: str,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    thetas: tuple[float, ...],
    *,
    scale: int = 3,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG output now uses matplotlib. Install project requirements, e.g. `pip install -r requirements.txt`, "
            "or run without --png."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    dpi = 100 * max(1, scale)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )

    columns = 2 if len(h_values) > 1 else 1
    rows = max(1, math.ceil(len(h_values) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2, 2.175 * rows), dpi=dpi, squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.075, hspace=0.58, wspace=0.38)

    x_min, x_max = 0.0, 0.92
    axes_flat = axes.ravel()
    for index, (ax, h) in enumerate(zip(axes_flat, h_values)):
        values = [
            cell.mean + cell.std
            for epsilon in epsilons
            for theta in thetas
            for cell in [stats.get(_key(pattern, epsilon, h, theta))]
            if cell is not None
        ]
        y_pad = 1.28 if index == len(h_values) - 1 else 1.12
        y_max = max(max(values, default=1.0) * y_pad, 1e-12)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_max)
        ax.set_title(f"h={_format_h_title(h)}", fontsize=7, pad=2)
        ax.set_xlabel(r"$\theta$", fontsize=7, labelpad=1)
        ax.set_ylabel("solve time [s]", fontsize=7, labelpad=1)
        ax.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8))
        ax.set_yticks([y_max * tick / 4 for tick in range(5)])
        ax.tick_params(axis="both", labelsize=6, length=2.5, pad=1)
        ax.grid(True, color="#d0d0d0", linewidth=0.45)

        handles = []
        labels = []
        for eps_index, epsilon in enumerate(epsilons):
            xs = []
            means = []
            errors = []
            for theta in thetas:
                cell = stats.get(_key(pattern, epsilon, h, theta))
                if cell is None:
                    continue
                xs.append(theta)
                means.append(cell.mean)
                errors.append(cell.std)
            if not xs:
                continue
            line = ax.errorbar(
                xs,
                means,
                yerr=errors if any(error > 0 for error in errors) else None,
                color=INTENSE_COLORS[eps_index % len(INTENSE_COLORS)],
                marker="o",
                markersize=2.0,
                linewidth=1.0,
                elinewidth=0.55,
                capsize=1.2,
                capthick=0.55,
            )
            handles.append(line.lines[0])
            labels.append(f"e={epsilon:.1f}")

        if index == len(h_values) - 1:
            legend = ax.legend(
                handles,
                labels,
                loc="upper left",
                ncol=4,
                fontsize=3.4,
                frameon=True,
                framealpha=0.82,
                fancybox=False,
                edgecolor="#777777",
                handlelength=0.65,
                handletextpad=0.16,
                columnspacing=0.28,
                borderpad=0.14,
                labelspacing=0.10,
                markerscale=0.55,
            )
            legend.get_frame().set_linewidth(0.45)

    for ax in axes_flat[len(h_values) :]:
        ax.set_visible(False)

    fig.savefig(destination, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _compile_pdf(tex_path: Path) -> None:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={tex_path.parent}",
        str(tex_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        log_path = tex_path.with_suffix(".log")
        hint = f" See {log_path}" if log_path.exists() else ""
        raise RuntimeError(
            "pdflatex failed. The TeX backend requires pgfplots; install the LaTeX pgfplots package "
            f"or use the generated SVG files.{hint}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scalable PGFPlots figures for theta vs diffusion solve time."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--out-dir", default="results/figures/theta_cost_relation")
    parser.add_argument("--patterns", default="auto")
    parser.add_argument("--epsilons", default="auto")
    parser.add_argument("--h-values", default="auto")
    parser.add_argument("--thetas", default="", help="Optional comma-separated theta grid. Defaults to values found in input.")
    parser.add_argument("--tex-name", default="theta_cost_plots.tex")
    parser.add_argument("--svg", action="store_true", help="Also write one scalable SVG per pattern.")
    parser.add_argument("--png", action="store_true", help="Also write one high-resolution PNG per pattern.")
    parser.add_argument("--png-scale", type=int, default=3, help="Raster scale factor for PNG output.")
    parser.add_argument("--compile", action="store_true", help="Compile the generated TeX to PDF with pdflatex.")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected table cell is missing.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, matches = _load_cost_stats(args.input_glob)

    print(f"Loaded {len(stats)} grouped cost records from {len(matches)} file(s).")

    auto_patterns = _available_patterns(stats)
    patterns = _parse_csv_text(args.patterns, auto_patterns or DEFAULT_PATTERNS)
    epsilons = _parse_csv_floats(args.epsilons, _available_epsilons(stats, patterns) or DEFAULT_EPSILONS)
    h_values = _parse_csv_floats(args.h_values, _available_h_values(stats, patterns, epsilons) or DEFAULT_H_VALUES)
    print(
        "Using "
        f"{len(patterns)} pattern(s), {len(epsilons)} epsilon value(s), and {len(h_values)} h value(s)."
    )

    figures = []
    for pattern in patterns:
        thetas = _parse_csv_floats(args.thetas, ()) if args.thetas else _available_thetas(stats, pattern, epsilons, h_values)
        if not thetas:
            message = f"No theta values found for pattern '{pattern}'."
            if args.strict:
                raise ValueError(message)
            print(f"WARNING: {message}")
            continue
        present, total, max_repeats = _coverage(stats, pattern, epsilons, h_values, thetas)
        print(f"{pattern}: coverage {present}/{total}; max repeats per cell: {max_repeats}.")
        if present < total:
            message = f"{pattern} is missing {total - present} expected cells."
            if args.strict:
                raise ValueError(message)
            print(f"WARNING: {message}")
        if max_repeats < 2:
            print(f"WARNING: {pattern} has no repeated samples, so error bars will be zero.")
        _write_summary_csv(out_dir / f"{pattern}_theta_cost_summary.csv", stats, pattern, epsilons, h_values, thetas)
        if args.svg:
            svg_path = out_dir / f"{pattern}_theta_cost.svg"
            _write_svg(svg_path, stats, pattern, epsilons, h_values, thetas)
            print(f"Wrote {svg_path}")
        if args.png:
            png_path = out_dir / f"{pattern}_theta_cost.png"
            try:
                _write_png(png_path, stats, pattern, epsilons, h_values, thetas, scale=max(1, args.png_scale))
            except RuntimeError as exc:
                print(f"WARNING: could not write {png_path}: {exc}")
                svg_path = out_dir / f"{pattern}_theta_cost.svg"
                _write_svg(svg_path, stats, pattern, epsilons, h_values, thetas)
                print(f"Wrote {svg_path}")
            else:
                print(f"Wrote {png_path}")
        figures.append(_figure_tex(stats, pattern, epsilons, h_values, thetas))

    if not figures:
        fallback_patterns = _available_patterns(stats)
        if args.strict or tuple(patterns) == fallback_patterns:
            raise ValueError("No figures were generated. Check --input-glob and --patterns.")
        print("WARNING: requested patterns produced no figures; retrying with all available patterns.")
        patterns = fallback_patterns
        epsilons = _available_epsilons(stats, patterns) or epsilons
        h_values = _available_h_values(stats, patterns, epsilons) or h_values
        for pattern in patterns:
            thetas = _parse_csv_floats(args.thetas, ()) if args.thetas else _available_thetas(stats, pattern, epsilons, h_values)
            if not thetas:
                continue
            _write_summary_csv(out_dir / f"{pattern}_theta_cost_summary.csv", stats, pattern, epsilons, h_values, thetas)
            if args.svg:
                svg_path = out_dir / f"{pattern}_theta_cost.svg"
                _write_svg(svg_path, stats, pattern, epsilons, h_values, thetas)
                print(f"Wrote {svg_path}")
            if args.png:
                png_path = out_dir / f"{pattern}_theta_cost.png"
                try:
                    _write_png(png_path, stats, pattern, epsilons, h_values, thetas, scale=max(1, args.png_scale))
                except RuntimeError as exc:
                    print(f"WARNING: could not write {png_path}: {exc}")
                    svg_path = out_dir / f"{pattern}_theta_cost.svg"
                    _write_svg(svg_path, stats, pattern, epsilons, h_values, thetas)
                    print(f"Wrote {svg_path}")
                else:
                    print(f"Wrote {png_path}")
            figures.append(_figure_tex(stats, pattern, epsilons, h_values, thetas))
        if not figures:
            raise ValueError("No figures were generated from the available records.")

    tex_path = out_dir / args.tex_name
    tex_path.write_text(_document_tex(figures), encoding="utf-8")
    print(f"Wrote {tex_path}")
    if args.compile:
        try:
            _compile_pdf(tex_path)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        else:
            print(f"Wrote {tex_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
