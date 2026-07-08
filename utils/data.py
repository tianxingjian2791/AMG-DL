from __future__ import annotations

import ast
import csv
import glob
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DIFFUSION_REPORT_GLOBS = (
    "datasets/diffusion/*/raw/diffusion_reports/*.csv",
    "datasets/diffusion/*/raw/diffusion_reports/**/*.csv",
    "datasets/diffusion/*/*/raw/diffusion_reports/*.csv",
    "datasets/diffusion/*/*/raw/diffusion_reports/**/*.csv",
    "datasets/diffusion/*/raw/theta_gnn_npy/*/*.npz",
    "datasets/diffusion/*/raw/theta_cnn_npy/*/*.npz",
    "datasets/diffusion/*/raw/p_value_npy/*/*.npz",
    "datasets/unified/diffusion/*/*/raw/diffusion_reports/*.csv",
    "datasets/unified/diffusion/*/*/raw/diffusion_reports/**/*.csv",
    "datasets/unified/*/raw/theta_gnn_npy/*_D/*.npz",
    "datasets/unified/*/raw/theta_cnn_npy/*_D/*.npz",
    "datasets/unified/*/raw/p_value_npy/*_D/*.npz",
)
DIFFUSION_REPORT_COLUMNS = {
    "scale",
    "pattern",
    "epsilon",
    "h",
    "theta",
    "rho",
}


def _coerce_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _coerce_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class SampleRecord:
    sample_meta: dict[str, object]
    metrics: dict[str, object]
    source_path: Path | None = None


def record_join_key(record: SampleRecord) -> tuple[object, ...]:
    meta = record.sample_meta
    return (
        meta.get("scale"),
        meta.get("scale_id"),
        meta.get("pattern"),
        meta.get("pattern_id"),
        meta.get("epsilon"),
        meta.get("h"),
        meta.get("refinement"),
    )


class SampleRecordRepository:
    def __init__(self, records: Iterable[SampleRecord]):
        self._records = list(records)

    @classmethod
    def from_glob(cls, pattern: str) -> "SampleRecordRepository":
        matches = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
        return cls(_load_records(matches))

    @classmethod
    def from_directory(cls, directory: str | Path) -> "SampleRecordRepository":
        root = Path(directory)
        if not root.exists():
            return cls([])
        matches = sorted(path for path in root.rglob("*.csv") if path.is_file())
        return cls(_load_records(matches))

    def all(self) -> list[SampleRecord]:
        return list(self._records)


def matched_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(Path(path) for path in glob.glob(pattern, recursive=True)):
            if match in seen or not match.is_file():
                continue
            seen.add(match)
            paths.append(match)
    return paths


def diffusion_report_input_error(patterns: Iterable[str], *, required_metric: str | None = None) -> str:
    pattern_list = tuple(patterns)
    paths = matched_paths(pattern_list)
    metric_text = f" with metric '{required_metric}'" if required_metric else ""
    if paths:
        preview = ", ".join(str(path) for path in paths[:4])
        if len(paths) > 4:
            preview += f", ... ({len(paths)} files total)"
        if any(path.suffix.lower() == ".npz" for path in paths):
            return (
                f"No compatible diffusion records{metric_text} were found in: {preview}. "
                "NPZ fallback loading supports rho/theta/h and reconstructs legacy epsilon/refinement indices; "
                "elapsed_sec is only available in diffusion report CSVs."
            )
        return (
            f"No compatible diffusion report records{metric_text} were found in: {preview}. "
            "Expected CSV columns include "
            f"{', '.join(sorted(DIFFUSION_REPORT_COLUMNS))}. "
            "If these are legacy theta/p-value training CSVs, regenerate diffusion reports first."
        )
    return (
        "No diffusion report CSV or NPZ files matched: "
        f"{', '.join(pattern_list)}. "
        "Generate them first, for example: "
        "build/generate_amg_data -p D -f all -c small --use-npy"
    )


def _load_records(paths: Iterable[Path]) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if path.suffix.lower() == ".npz":
            records.extend(_load_npz_sample(path))
        else:
            records.extend(_load_csv_report(path))
    _fill_legacy_npz_meta(records)
    return records


def _load_csv_report(path: Path) -> list[SampleRecord]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            fieldnames = {name.strip() for name in reader.fieldnames if name}
            required = {"scale", "pattern", "epsilon", "h", "theta", "rho"}
            if not required.issubset(fieldnames):
                return []

            records: list[SampleRecord] = []
            for row in reader:
                records.append(_row_to_record(row, path))
            return records
    except OSError:
        return []


def _load_npz_sample(path: Path) -> list[SampleRecord]:
    try:
        with zipfile.ZipFile(path) as data:
            names = set(data.namelist())
            if "metadata.npy" not in names:
                return []
            metadata = _read_npy_values(data.read("metadata.npy"))

            npy_kind = path.parts[-3]
            theta = _npz_scalar(data, "theta.npy")
            rho = _npz_scalar(data, "y.npy")

            if npy_kind == "theta_gnn_npy":
                if len(metadata) >= 7:
                    n, rho_meta, h, epsilon, pattern_id, refinement, iterations = metadata[:7]
                    rho = float(rho if rho is not None else rho_meta)
                elif len(metadata) >= 4:
                    n, rho_meta, h, _legacy_value = metadata[:4]
                    rho = float(rho if rho is not None else rho_meta)
                    epsilon = 0.0
                    pattern_id = 0
                    refinement = 0
                    iterations = -1
                else:
                    return []
            elif npy_kind == "p_value_npy":
                if len(metadata) >= 8:
                    n, theta_meta, rho_meta, h, pattern_id, epsilon, refinement, iterations = metadata[:8]
                    rho = float(rho if rho is not None else rho_meta)
                    theta = float(theta if theta is not None else theta_meta)
                elif len(metadata) >= 4:
                    n, theta_meta, rho_meta, h = metadata[:4]
                    rho = float(rho if rho is not None else rho_meta)
                    theta = float(theta if theta is not None else theta_meta)
                    epsilon = 0.0
                    pattern_id = 0
                    refinement = 0
                    iterations = -1
                else:
                    return []
            else:
                if len(metadata) >= 8:
                    n, rho_meta, h, theta_meta, pattern_id, epsilon, refinement, iterations = metadata[:8]
                    rho = float(rho if rho is not None else rho_meta)
                    theta = float(theta if theta is not None else theta_meta)
                elif len(metadata) >= 4:
                    n, rho_meta, h, theta_meta = metadata[:4]
                    rho = float(rho if rho is not None else rho_meta)
                    theta = float(theta if theta is not None else theta_meta)
                    epsilon = 0.0
                    pattern_id = 0
                    refinement = 0
                    iterations = -1
                else:
                    return []
            if theta is None:
                return []

            sample_meta = {
                "scale": _scale_from_path(path),
                "scale_id": 0,
                "pattern": _pattern_name(int(pattern_id)),
                "pattern_id": int(pattern_id),
                "epsilon": float(epsilon),
                "refinement": int(refinement),
                "h": float(h),
                "theta": float(theta),
            }
            metrics = {
                "rho": float(rho),
                "iterations": int(iterations),
                "n": int(n),
            }
            if "coarse_nodes.npy" in names:
                metrics["n_levels"] = _npy_size(data.read("coarse_nodes.npy"))
            if "edge_attr.npy" in names:
                metrics["nnz"] = _npy_size(data.read("edge_attr.npy"))
            if int(iterations) < 0:
                metrics.pop("iterations", None)
                sample_meta["_legacy_compact_npz"] = True
            return [SampleRecord(sample_meta=sample_meta, metrics=metrics, source_path=path)]
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return []


def _npz_scalar(data: zipfile.ZipFile, name: str) -> float | None:
    if name not in data.namelist():
        return None
    values = _read_npy_values(data.read(name), max_values=1)
    if not values:
        return None
    return float(values[0])


def _read_npy_values(blob: bytes, *, max_values: int | None = None) -> list[float]:
    header, offset = _read_npy_header(blob)
    count = _shape_size(header.get("shape", ()))
    if max_values is not None:
        count = min(count, max_values)
    descr = str(header.get("descr", ""))
    fmt = _struct_format(descr)
    if fmt is None or count <= 0:
        return []
    size = struct.calcsize(fmt)
    raw = blob[offset : offset + count * size]
    return [float(value) for value in struct.unpack("<" + fmt * count, raw)]


def _npy_size(blob: bytes) -> int:
    header, _offset = _read_npy_header(blob)
    return _shape_size(header.get("shape", ()))


def _read_npy_header(blob: bytes) -> tuple[dict, int]:
    if not blob.startswith(b"\x93NUMPY"):
        raise ValueError("not an npy file")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", blob[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"unsupported npy version {major}")
    header_text = blob[offset : offset + header_len].decode("latin1").strip()
    return ast.literal_eval(header_text), offset + header_len


def _shape_size(shape: object) -> int:
    if isinstance(shape, int):
        return shape
    if not isinstance(shape, tuple):
        return 0
    return math.prod(int(dim) for dim in shape)


def _struct_format(descr: str) -> str | None:
    formats = {
        "<f8": "d",
        "|f8": "d",
        "<f4": "f",
        "|f4": "f",
        "<i8": "q",
        "|i8": "q",
        "<i4": "i",
        "|i4": "i",
        "<u8": "Q",
        "|u8": "Q",
        "<u4": "I",
        "|u4": "I",
    }
    return formats.get(descr)


def _scale_from_path(path: Path) -> str:
    parts = path.parts
    if "diffusion" in parts:
        idx = parts.index("diffusion")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "legacy"


def _pattern_name(pattern_id: int) -> str:
    names = {
        0: "vertical_split",
        1: "checker2x2",
        2: "vertical_stripes",
        3: "checker4x4",
    }
    return names.get(pattern_id, f"pattern_{pattern_id}")


def _fill_legacy_npz_meta(records: list[SampleRecord]) -> None:
    by_dir: dict[Path, list[SampleRecord]] = {}
    for record in records:
        if not record.sample_meta.get("_legacy_compact_npz") or record.source_path is None:
            continue
        by_dir.setdefault(record.source_path.parent, []).append(record)

    for group in by_dir.values():
        group.sort(key=lambda record: record.source_path.name if record.source_path else "")
        theta_count = len({round(float(record.sample_meta["theta"]), 8) for record in group})
        h_count = len({round(float(record.sample_meta["h"]), 8) for record in group})
        if theta_count <= 0:
            continue
        h_count = max(h_count, 1)
        block = theta_count * h_count
        for index, record in enumerate(group):
            record.sample_meta["epsilon"] = float(index // block)
            record.sample_meta["refinement"] = int((index // theta_count) % h_count)


def _row_to_record(row: dict[str, str], path: Path) -> SampleRecord:
    sample_meta: dict[str, object] = {}
    metrics: dict[str, object] = {}

    meta_fields = {
        "scale": _coerce_text(row.get("scale")),
        "scale_id": _coerce_int(row.get("scale_id")),
        "pattern": _coerce_text(row.get("pattern")),
        "pattern_id": _coerce_int(row.get("pattern_id")),
        "epsilon": _coerce_float(row.get("epsilon")),
        "refinement": _coerce_int(row.get("refinement")),
        "h": _coerce_float(row.get("h")),
        "theta": _coerce_float(row.get("theta")),
    }
    metric_fields = {
        "rho": _coerce_float(row.get("rho")),
        "iterations": _coerce_int(row.get("iterations")),
        "elapsed_sec": _coerce_float(row.get("elapsed_sec")),
        "n_levels": _coerce_int(row.get("n_levels")),
        "n": _coerce_int(row.get("n")),
        "nnz": _coerce_int(row.get("nnz")),
    }

    for key, value in meta_fields.items():
        if value is not None:
            sample_meta[key] = value
    for key, value in metric_fields.items():
        if value is not None:
            metrics[key] = value

    return SampleRecord(sample_meta=sample_meta, metrics=metrics, source_path=path)
