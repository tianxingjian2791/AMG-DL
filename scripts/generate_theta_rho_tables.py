#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import SampleRecordRepository


DEFAULT_THETAS = (0.24, 0.48, 0.72)
DEFAULT_EPSILONS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 7.0, 9.5)
DEFAULT_H_VALUES = (0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625)


@dataclass(frozen=True)
class TableCell:
    rho: float
    iterations: int


def _parse_csv_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw or raw.strip().lower() == "auto":
        return default
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _key(theta: float, epsilon: float, h: float) -> tuple[float, float, float]:
    return (round(theta, 8), round(epsilon, 8), round(h, 8))


def _records_by_grid(
    input_glob: str,
    pattern_filters: tuple[str, ...],
) -> dict[tuple[float, float, float], TableCell]:
    records = SampleRecordRepository.from_glob(input_glob).all()
    cells: dict[tuple[float, float, float], TableCell] = {}
    for record in records:
        meta = record.sample_meta
        metrics = record.metrics
        if pattern_filters and str(meta.get("pattern")) not in pattern_filters:
            continue
        try:
            theta = float(meta["theta"])
            epsilon = float(meta["epsilon"])
            h = float(meta["h"])
            rho = float(metrics["rho"])
            iterations = int(metrics["iterations"])
        except (KeyError, TypeError, ValueError):
            continue
        cells[_key(theta, epsilon, h)] = TableCell(rho=rho, iterations=iterations)
    return cells


def _load_tables(
    input_glob: str | None,
    vertical_split_glob: str | None,
    stripes_glob: str | None,
    checker2x2_glob: str | None,
    checker_glob: str | None,
) -> dict[str, dict[tuple[float, float, float], TableCell]]:
    tables: dict[str, dict[tuple[float, float, float], TableCell]] = {}
    if input_glob:
        vertical_split_glob = vertical_split_glob or input_glob
        stripes_glob = stripes_glob or input_glob
        checker2x2_glob = checker2x2_glob or input_glob
        checker_glob = checker_glob or input_glob
    if vertical_split_glob:
        tables["vertical_split"] = _records_by_grid(
            vertical_split_glob,
            ("vertical_split", "vertical_stripes"),
        )
    if stripes_glob:
        tables["vertical_stripes4"] = _records_by_grid(
            stripes_glob,
            ("vertical_stripes4", "vertical_stripes_4"),
        )
    if checker2x2_glob:
        tables["checker2x2"] = _records_by_grid(
            checker2x2_glob,
            ("checker2x2", "checkerboard_2x2"),
        )
    if checker_glob:
        tables["checker4x4"] = _records_by_grid(
            checker_glob,
            ("checker4x4", "checkerboard_4x4"),
        )
    if not tables:
        raise ValueError("Provide at least one input glob")
    return tables


def _rho_range(
    tables: dict[str, dict[tuple[float, float, float], TableCell]],
    thetas: tuple[float, ...],
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
) -> tuple[float, float]:
    values: list[float] = []
    for cells in tables.values():
        for theta in thetas:
            for epsilon in epsilons:
                for h in h_values:
                    cell = cells.get(_key(theta, epsilon, h))
                    if cell is not None:
                        values.append(cell.rho)
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def _coverage(
    cells: dict[tuple[float, float, float], TableCell],
    thetas: tuple[float, ...],
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
) -> tuple[int, int]:
    total = len(thetas) * len(epsilons) * len(h_values)
    present = sum(
        1
        for theta in thetas
        for epsilon in epsilons
        for h in h_values
        if cells.get(_key(theta, epsilon, h)) is not None
    )
    return present, total


def _grid_from_tables(
    tables: dict[str, dict[tuple[float, float, float], TableCell]],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    theta_values: set[float] = set()
    epsilon_values: set[float] = set()
    h_values: set[float] = set()
    for cells in tables.values():
        for theta, epsilon, h in cells:
            theta_values.add(theta)
            epsilon_values.add(epsilon)
            h_values.add(h)
    return (
        tuple(sorted(theta_values)),
        tuple(sorted(epsilon_values)),
        tuple(sorted(h_values, reverse=True)),
    )


def _cell_color(rho: float, rho_min: float, rho_max: float) -> str:
    if rho_max <= rho_min:
        t = 0.0
    else:
        t = (rho - rho_min) / (rho_max - rho_min)
    low = (0.78, 0.82, 1.00)
    high = (1.00, 0.78, 0.78)
    r = low[0] + (high[0] - low[0]) * t
    g = low[1] + (high[1] - low[1]) * t
    b = low[2] + (high[2] - low[2]) * t
    return f"{r:.3f},{g:.3f},{b:.3f}"


def _format_h(h: float) -> str:
    return f"{h:.2e}"


def _format_theta(theta: float) -> str:
    return f"{theta:.6g}"


def _theta_stem(theta: float) -> str:
    return _format_theta(theta).replace(".", "p").replace("-", "m")


def _format_cell(cell: TableCell | None, rho_min: float, rho_max: float) -> str:
    if cell is None:
        return "--"
    color = _cell_color(cell.rho, rho_min, rho_max)
    return f"\\cellcolor[rgb]{{{color}}}{cell.rho:.3f}({cell.iterations})"


def _write_csv(
    destination: Path,
    cells: dict[tuple[float, float, float], TableCell],
    theta: float,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epsilon/h", *(_format_h(h) for h in h_values)])
        for epsilon in epsilons:
            row = [epsilon]
            for h in h_values:
                cell = cells.get(_key(theta, epsilon, h))
                row.append("" if cell is None else f"{cell.rho:.6f}({cell.iterations})")
            writer.writerow(row)


def _latex_table(
    title: str,
    label: str,
    pattern_name: str,
    cells: dict[tuple[float, float, float], TableCell],
    theta: float,
    epsilons: tuple[float, ...],
    h_values: tuple[float, ...],
    rho_min: float,
    rho_max: float,
) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}. Pattern: {pattern_name}. Fixed strong threshold $\\theta={_format_theta(theta)}$.}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{c" + "c" * len(h_values) + "}",
        "\\toprule",
        "$\\varepsilon / h$ & " + " & ".join(_format_h(h) for h in h_values) + " \\\\",
        "\\midrule",
    ]
    for epsilon in epsilons:
        values = [_format_cell(cells.get(_key(theta, epsilon, h)), rho_min, rho_max) for h in h_values]
        lines.append(f"{epsilon:.1f} & " + " & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        default=None,
        help="Shared diffusion report glob. The script separates coefficient patterns automatically.",
    )
    parser.add_argument("--vertical_split_glob", default=None)
    parser.add_argument("--stripes_glob", default=None)
    parser.add_argument("--checker2x2_glob", default=None)
    parser.add_argument("--checker_glob", default=None)
    parser.add_argument("--out_dir", default="results/figures/theta_rho_relation")
    parser.add_argument("--thetas", default="auto", help="Comma-separated theta grid or 'auto' to read it from the input reports.")
    parser.add_argument("--epsilons", default="auto", help="Comma-separated epsilon grid or 'auto' to read it from the input reports.")
    parser.add_argument("--h_values", default="auto", help="Comma-separated h grid or 'auto' to read it from the input reports.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(
        args.input_glob,
        args.vertical_split_glob,
        args.stripes_glob,
        args.checker2x2_glob,
        args.checker_glob,
    )
    auto_thetas, auto_epsilons, auto_h_values = _grid_from_tables(tables)
    thetas = _parse_csv_floats(args.thetas, auto_thetas or DEFAULT_THETAS)
    epsilons = _parse_csv_floats(args.epsilons, auto_epsilons or DEFAULT_EPSILONS)
    h_values = _parse_csv_floats(args.h_values, auto_h_values or DEFAULT_H_VALUES)
    rho_min, rho_max = _rho_range(tables, thetas, epsilons, h_values)

    preamble = [
        "\\documentclass{article}",
        "\\usepackage[table]{xcolor}",
        "\\usepackage{booktabs}",
        "\\usepackage{geometry}",
        "\\geometry{margin=0.65in}",
        "\\begin{document}",
    ]
    latex_sections: list[str] = list(preamble)

    for pattern_name, cells in tables.items():
        present, total = _coverage(cells, thetas, epsilons, h_values)
        if present < total:
            print(f"WARNING: {pattern_name} coverage is {present}/{total} table cells.")
        for theta in thetas:
            stem = f"{pattern_name}_theta_{_theta_stem(theta)}"
            latex_sections.append(
                _latex_table(
                    title="Approximate convergence factor $\\rho$ with preconditioned CG iterations in parentheses",
                    label=f"tab:{stem}",
                    pattern_name=pattern_name.replace("_", " "),
                    cells=cells,
                    theta=theta,
                    epsilons=epsilons,
                    h_values=h_values,
                    rho_min=rho_min,
                    rho_max=rho_max,
                )
            )
            _write_csv(out_dir / f"{stem}.csv", cells, theta, epsilons, h_values)

    latex_sections.append("\\end{document}")
    tex_path = out_dir / "theta_rho_tables.tex"
    tex_path.write_text("\n\n".join(latex_sections) + "\n", encoding="utf-8")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
