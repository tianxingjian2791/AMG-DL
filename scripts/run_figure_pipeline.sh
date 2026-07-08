#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_large_figure_pipeline.sh [options]

Options:
  --dataset-root DIR   Dataset root containing raw/diffusion_reports.
                       Default: datasets/diffusion/large
  --output-root DIR    Root directory for generated figures.
                       Default: results/figures
  --no-pdf             Skip pdflatex compilation.
  --help               Show this message.

The script runs the four scripts in scripts/ against the selected dataset and
compiles generated .tex files to PDF when possible.
USAGE
}

dataset_root="datasets/diffusion/large"
output_root="results/figures"
compile_pdf=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      dataset_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --no-pdf)
      compile_pdf=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

report_glob="${dataset_root%/}/**/raw/diffusion_reports/*.csv"
mkdir -p "$output_root"
failures=0

echo "Dataset root: $dataset_root"
echo "Output root:  $output_root"
echo "Report glob:  $report_glob"
echo

run_step() {
  local name="$1"
  shift
  echo "==> $name"
  if "$@"; then
    :
  else
    echo "WARNING: step failed: $name" >&2
    failures=$((failures + 1))
  fi
  echo
}

compile_tex_tree() {
  local root="$1"
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "WARNING: pdflatex not found; skipping TeX to PDF compilation." >&2
    return 0
  fi

  local tex_file
  while IFS= read -r tex_file; do
    local tex_dir
    local tex_name
    tex_dir="$(dirname "$tex_file")"
    tex_name="$(basename "$tex_file")"
    echo "==> Compiling $tex_file"
    if (
      cd "$tex_dir"
      pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
      pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
    ); then
      echo "Wrote ${tex_file%.tex}.pdf"
    else
      echo "WARNING: Failed to compile $tex_file. Check ${tex_file%.tex}.log." >&2
      failures=$((failures + 1))
    fi
    echo
  done < <(find "$root" -name '*.tex' -type f | sort)
}

run_step "theta vs nlevels diagnostic" \
  python3 scripts/theta_vs_nlevels_plot.py \
    --dataset-root "$dataset_root" \
    --output-dir "$output_root/theta_nlevels" \
    --include-splits all

run_step "theta-rho tables" \
  python3 scripts/generate_theta_rho_tables.py \
    --input_glob "$report_glob" \
    --out_dir "$output_root/theta_rho_relation"

run_step "theta-cost plots" \
  python3 scripts/generate_theta_cost_plots.py \
    --input-glob "$report_glob" \
    --out-dir "$output_root/theta_cost_relation" \
    --png

run_step "rho-time scatter" \
  python3 scripts/generate_rho_time_scatter.py \
    --input-glob "$report_glob" \
    --out-dir "$output_root/rho_time_scatter"

if [[ "$compile_pdf" -eq 1 ]]; then
  compile_tex_tree "$output_root"
fi

echo "Done. Outputs are under $output_root"
if [[ "$failures" -gt 0 ]]; then
  echo "Completed with $failures failure(s)." >&2
  exit 1
fi
