

"""Build a diagnostic plot for theta versus the number of multigrid levels.

This script reads diffusion report CSV files from a dataset root, normalizes
the ``n_levels`` column, runs a per-test-case linear regression, and writes a
summary figure plus CSV/JSON statistics into the output directory.

Typical usage:

    python scripts/theta_vs_nlevels_plot.py \
        --dataset-root datasets/diffusion/small \
        --output-dir results/figures

The generated files are useful for checking whether ``theta`` correlates with
hierarchy depth across the selected train/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_COLUMNS = ["split", "pattern", "epsilon", "refinement", "h"]
REQUIRED_COLUMNS = {
    "pattern",
    "epsilon",
    "refinement",
    "h",
    "theta",
    "n_levels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create theta vs number of levels plot."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/diffusion/small"),
        help="Dataset root containing raw/diffusion_reports directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/theta_vs_nlevels"),
        help="Directory for the generated figure and statistics files.",
    )
    parser.add_argument(
        "--figure-name",
        default="theta_vs_nlevels.png",
        help="Output image filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=["all"],
        choices=["all", "train", "test"],
        help="Dataset splits to include.",
    )
    return parser.parse_args()


def report_candidates(dataset_root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    requested_splits = list(splits)
    if requested_splits == ["all"]:
        return [
            ("all", dataset_root / "raw" / "diffusion_reports"),
            ("all", dataset_root / "diffusion_reports"),
        ]
    return [
        (split, dataset_root / split / "raw" / "diffusion_reports")
        for split in requested_splits
    ]


def fallback_split_candidates(dataset_root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    if list(splits) != ["all"]:
        return []
    return [
        (split, dataset_root / split / "raw" / "diffusion_reports")
        for split in ("train", "test")
    ]


def load_reports(dataset_root: Path, splits: list[str]) -> Any:
    import pandas as pd

    frames = []
    for split, report_dir in report_candidates(dataset_root, splits):
        for path in sorted(report_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            frame["split"] = split
            frame["source_file"] = str(path)
            frames.append(frame)

    if not frames:
        for split, report_dir in fallback_split_candidates(dataset_root, splits):
            for path in sorted(report_dir.glob("*.csv")):
                frame = pd.read_csv(path)
                frame["split"] = split
                frame["source_file"] = str(path)
                frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No diffusion report CSVs found below {dataset_root}. "
            "Checked unsplit raw/diffusion_reports, direct diffusion_reports, "
            f"and requested split folders {splits}."
        )

    data = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required report columns: {sorted(missing)}")

    for column in ["epsilon", "refinement", "h", "theta", "n_levels"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data


def load_report_rows(dataset_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, report_dir in report_candidates(dataset_root, splits):
        for path in sorted(report_dir.glob("*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Missing required report columns in {path}: {sorted(missing)}")
                for row in reader:
                    row["split"] = split
                    row["source_file"] = str(path)
                    rows.append(row)

    if not rows:
        for split, report_dir in fallback_split_candidates(dataset_root, splits):
            for path in sorted(report_dir.glob("*.csv")):
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
                    if missing:
                        raise ValueError(f"Missing required report columns in {path}: {sorted(missing)}")
                    for row in reader:
                        row["split"] = split
                        row["source_file"] = str(path)
                        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No diffusion report CSVs found below {dataset_root}. "
            "Checked unsplit raw/diffusion_reports, direct diffusion_reports, "
            f"and requested split folders {splits}."
        )

    for row in rows:
        for column in ["epsilon", "refinement", "h", "theta", "n_levels"]:
            row[column] = float(row[column])
    return rows


def add_normalized_levels(data: Any) -> Any:
    data = data.copy()
    lo = data["n_levels"].min()
    hi = data["n_levels"].max()
    if hi == lo:
        raise ValueError("Cannot min-max normalize n_levels because all values are equal.")
    data["normalized_levels"] = (data["n_levels"] - lo) / (hi - lo)
    return data


def add_normalized_levels_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    levels = [float(row["n_levels"]) for row in rows]
    lo = min(levels)
    hi = max(levels)
    if hi == lo:
        raise ValueError("Cannot min-max normalize n_levels because all values are equal.")
    normalized = []
    for row in rows:
        item = dict(row)
        item["normalized_levels"] = (float(item["n_levels"]) - lo) / (hi - lo)
        normalized.append(item)
    return normalized


def regression_by_test_case(data: Any) -> Any:
    import pandas as pd
    from scipy.stats import linregress

    records = []
    for key, group in data.groupby(GROUP_COLUMNS, dropna=False):
        group = group.sort_values("theta")
        if group["theta"].nunique() < 3 or group["normalized_levels"].nunique() < 2:
            continue
        result = linregress(group["theta"], group["normalized_levels"])
        records.append(
            {
                **dict(zip(GROUP_COLUMNS, key)),
                "n_points": int(len(group)),
                "p_value": float(result.pvalue),
                "r_squared": float(result.rvalue**2),
                "slope": float(result.slope),
                "intercept": float(result.intercept),
            }
        )
    return pd.DataFrame.from_records(records)


def regression_by_test_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in GROUP_COLUMNS)].append(row)

    records: list[dict[str, Any]] = []
    for key, group in groups.items():
        group = sorted(group, key=lambda row: float(row["theta"]))
        xs = [float(row["theta"]) for row in group]
        ys = [float(row["normalized_levels"]) for row in group]
        if len(set(xs)) < 3 or len(set(ys)) < 2:
            continue
        x_mean = statistics.mean(xs)
        y_mean = statistics.mean(ys)
        sxx = sum((x - x_mean) ** 2 for x in xs)
        syy = sum((y - y_mean) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            continue
        sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        slope = sxy / sxx
        intercept = y_mean - slope * x_mean
        r_squared = (sxy * sxy) / (sxx * syy)
        records.append(
            {
                **dict(zip(GROUP_COLUMNS, key)),
                "n_points": len(group),
                "p_value": "",
                "r_squared": r_squared,
                "slope": slope,
                "intercept": intercept,
            }
        )
    return records


def settings_report(data: Any, regressions: Any) -> dict:
    report = {
        "rows": int(len(data)),
        "splits": sorted(data["split"].unique().tolist()),
        "patterns": sorted(data["pattern"].unique().tolist()),
        "epsilon_values": sorted(data["epsilon"].unique().tolist()),
        "refinements": sorted(data["refinement"].unique().tolist()),
        "theta_values": sorted(data["theta"].unique().tolist()),
        "n_levels_min": int(data["n_levels"].min()),
        "n_levels_max": int(data["n_levels"].max()),
        "regression_test_cases": int(len(regressions)),
        "warnings": [],
    }

    if len(report["theta_values"]) < 10:
        report["warnings"].append(
            "Only a small theta grid is available; the paper figure used many theta samples."
        )
    if len(data) < 1000:
        report["warnings"].append(
            "This is a smoke-sized dataset, so KDE and histogram shapes are not paper-faithful."
        )
    if report["n_levels_max"] > 20:
        report["warnings"].append(
            "n_levels values are unusually large for hierarchy levels; this may be an older report generated with the coarse-node proxy."
        )

    return report


def settings_report_rows(rows: list[dict[str, Any]], regressions: list[dict[str, Any]]) -> dict:
    report = {
        "rows": len(rows),
        "splits": sorted({str(row["split"]) for row in rows}),
        "patterns": sorted({str(row["pattern"]) for row in rows}),
        "epsilon_values": sorted({float(row["epsilon"]) for row in rows}),
        "refinements": sorted({float(row["refinement"]) for row in rows}),
        "theta_values": sorted({float(row["theta"]) for row in rows}),
        "n_levels_min": int(min(float(row["n_levels"]) for row in rows)),
        "n_levels_max": int(max(float(row["n_levels"]) for row in rows)),
        "regression_test_cases": len(regressions),
        "warnings": [
            "Used standard-library fallback; p_value is omitted because scipy is unavailable.",
        ],
    }

    if len(report["theta_values"]) < 10:
        report["warnings"].append(
            "Only a small theta grid is available; the paper figure used many theta samples."
        )
    if len(rows) < 1000:
        report["warnings"].append(
            "This is a smoke-sized dataset, so KDE and histogram shapes are not paper-faithful."
        )
    if report["n_levels_max"] > 20:
        report["warnings"].append(
            "n_levels values are unusually large for hierarchy levels; this may be an older report generated with the coarse-node proxy."
        )
    return report


def plot_figure(data: pd.DataFrame, regressions: pd.DataFrame, output_path: Path, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG output uses matplotlib, seaborn, and statsmodels. Install project requirements, "
            "or use the CSV/JSON diagnostics written by this script."
        ) from exc

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), constrained_layout=True)

    scatter_ax, p_ax, r2_ax = axes
    sns.scatterplot(
        data=data,
        x="theta",
        y="normalized_levels",
        hue="split",
        s=22,
        alpha=0.65,
        linewidth=0,
        ax=scatter_ax,
    )
    sns.kdeplot(
        data=data,
        x="theta",
        y="normalized_levels",
        levels=7,
        color="#2f6fb2",
        linewidths=1.1,
        fill=False,
        ax=scatter_ax,
    )
    smoothed = lowess(data["normalized_levels"], data["theta"], frac=0.35, return_sorted=True)
    scatter_ax.plot(smoothed[:, 0], smoothed[:, 1], color="#d62728", linewidth=1.8, label="LOWESS")
    scatter_ax.set_xlabel(r"$\theta$")
    scatter_ax.set_ylabel("normalized # levels")
    scatter_ax.legend(frameon=True, fontsize=7)

    if regressions.empty:
        p_ax.text(0.5, 0.5, "No valid regressions", ha="center", va="center")
        r2_ax.text(0.5, 0.5, "No valid regressions", ha="center", va="center")
    else:
        sns.histplot(regressions["p_value"], bins="auto", stat="density", ax=p_ax)
        p_ax.set_xlabel(r"$p$-value")
        p_ax.set_ylabel("density")

        sns.histplot(regressions["r_squared"], bins="auto", stat="density", ax=r2_ax)
        r2_ax.set_xlabel(r"$R^2$")
        r2_ax.set_ylabel("density")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / args.figure_name

    try:
        data = add_normalized_levels(load_reports(args.dataset_root, args.include_splits))
        regressions = regression_by_test_case(data)
        report = settings_report(data, regressions)
    except ModuleNotFoundError as exc:
        print(f"WARNING: using standard-library fallback because optional dependency is missing: {exc.name}")
        rows = add_normalized_levels_rows(load_report_rows(args.dataset_root, args.include_splits))
        regression_rows = regression_by_test_case_rows(rows)
        report = settings_report_rows(rows, regression_rows)
        csv_path = args.output_dir / "regression_stats.csv"
        fieldnames = [*GROUP_COLUMNS, "n_points", "p_value", "r_squared", "slope", "intercept"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(regression_rows)
        with (args.output_dir / "settings_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote {csv_path}")
        print(f"Wrote {args.output_dir / 'settings_report.json'}")
        print(f"WARNING: could not write {figure_path}: plotting dependencies are unavailable.")
    else:
        regressions.to_csv(args.output_dir / "regression_stats.csv", index=False)
        with (args.output_dir / "settings_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote {args.output_dir  / 'regression_stats.csv'}")
        print(f"Wrote {args.output_dir / 'settings_report.json'}")
        try:
            plot_figure(data, regressions, figure_path, args.dpi)
        except RuntimeError as exc:
            print(f"WARNING: could not write {figure_path}: {exc}")
        else:
            print(f"Wrote {figure_path}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
