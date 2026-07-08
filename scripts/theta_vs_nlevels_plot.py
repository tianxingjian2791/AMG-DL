

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
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess


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


def load_reports(dataset_root: Path, splits: list[str]) -> pd.DataFrame:
    frames = []
    for split in splits:
        report_dir = (
            dataset_root / "raw" / "diffusion_reports"
            if split == "all"
            else dataset_root / split / "raw" / "diffusion_reports"
        )
        for path in sorted(report_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            frame["split"] = split
            frame["source_file"] = str(path)
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No diffusion report CSVs found below {dataset_root} for splits {splits}."
        )

    data = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required report columns: {sorted(missing)}")

    for column in ["epsilon", "refinement", "h", "theta", "n_levels"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    return data


def add_normalized_levels(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    lo = data["n_levels"].min()
    hi = data["n_levels"].max()
    if hi == lo:
        raise ValueError("Cannot min-max normalize n_levels because all values are equal.")
    data["normalized_levels"] = (data["n_levels"] - lo) / (hi - lo)
    return data


def regression_by_test_case(data: pd.DataFrame) -> pd.DataFrame:
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


def settings_report(data: pd.DataFrame, regressions: pd.DataFrame) -> dict:
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


def plot_figure(data: pd.DataFrame, regressions: pd.DataFrame, output_path: Path, dpi: int) -> None:
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
    data = add_normalized_levels(load_reports(args.dataset_root, args.include_splits))
    regressions = regression_by_test_case(data)
    report = settings_report(data, regressions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / args.figure_name

    plot_figure(data, regressions, figure_path, args.dpi)

    regressions.to_csv(args.output_dir / "regression_stats.csv", index=False)
    with (args.output_dir / "settings_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote {figure_path}")
    print(f"Wrote {args.output_dir  / 'regression_stats.csv'}")
    print(f"Wrote {args.output_dir / 'settings_report.json'}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
