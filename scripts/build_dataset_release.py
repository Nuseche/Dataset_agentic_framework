#!/usr/bin/env python3.11
"""Build the paper data release from local agentic mesher artifacts.

The release is intentionally archive-based because several meshes and raw CAD
assets are too large for convenient direct GitHub storage. Each archive is
split into <=95 MiB parts and gets a JSON manifest with SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MESHER_ROOT = Path("/scratch/negishi/nusechec/mesher_work")
FRAMEWORK_ROOT = MESHER_ROOT / "agentic_mesher_framework"
DATASET_REPO = MESHER_ROOT / "Dataset_agentic_framework"
BUILD_ROOT = MESHER_ROOT / "_dataset_agentic_build"
STAGING_ROOT = MESHER_ROOT / "_dataset_agentic_staging"

VPMR_RUN = FRAMEWORK_ROOT / "benchmarks/faulty_cad/runs/vpmr_strict_100_20260608"
VPMR_SPLIT_ZIP_DIR = MESHER_ROOT / "_external_sources/vgmr_dataset_parts"
THINGI_RUN = (
    FRAMEWORK_ROOT
    / "benchmarks/faulty_cad/runs/thingi10k_100_repair_mesh_family_20260615"
)
THINGI_FOLLOWUP_RUN = (
    FRAMEWORK_ROOT / "benchmarks/faulty_cad/runs/thingi10k_100_hxt8_rerun_20260618_live"
)
SNES_V2_RAW = MESHER_ROOT / "cads/snes_v2"
SNES_V2_RUNS = [
    MESHER_ROOT / "runs/snes_a_robust_current_20260504_patch1",
    MESHER_ROOT / "runs/snes_y_robust_current_20260504_patch1",
    MESHER_ROOT / "runs/snes_y_a_robust_current_20260504_patch1",
    MESHER_ROOT / "runs/v2_letters_robust_current_20260504_patch1",
]
PHYSIO_BUNDLE = (
    MESHER_ROOT / "final_bundles/physio_final_results_20260429_fsi_spine_closure_v1"
)
HEART_V004_RUN = MESHER_ROOT / "runs/heart_heart_shell_v004_smooth_20260504"
HEART_FULL_MESH = HEART_V004_RUN / "attempt_01_tol_1e-08/mesh_shell_reconstructed.msh"

SPLIT_SIZE = "95m"
VPMR_SOURCE_URL = "https://github.com/VisualGuidedMeshRepair/dataset.git"
VPMR_SOURCE_COMMIT = "801dc65041bcadac6ade6932e491a1d6ef301b44"
DATASET_REMOTE = "https://github.com/Nuseche/Dataset_agentic_framework.git"


@dataclass(frozen=True)
class ArchiveSpec:
    name: str
    staged_dir: Path
    description: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve_framework_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return FRAMEWORK_ROOT / path


def link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_if_exists(src: Path, dst: Path, manifest: list[dict[str, Any]]) -> bool:
    if not src.exists():
        return False
    link_or_copy(src, dst)
    manifest.append(file_record(dst, source=src))
    return True


def file_record(path: Path, *, source: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "source_path": str(source) if source else None,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def clean_generated() -> None:
    for rel in ["archives", "manifests", "docs"]:
        path = DATASET_REPO / rel
        if path.exists():
            shutil.rmtree(path)
    for path in [BUILD_ROOT, STAGING_ROOT]:
        if path.exists():
            shutil.rmtree(path)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    (DATASET_REPO / "archives").mkdir(parents=True, exist_ok=True)
    (DATASET_REPO / "manifests").mkdir(parents=True, exist_ok=True)
    (DATASET_REPO / "docs").mkdir(parents=True, exist_ok=True)


def rel_from_anchor(path: Path, anchor_name: str) -> Path:
    parts = path.parts
    if anchor_name not in parts:
        return Path(path.name)
    idx = parts.index(anchor_name)
    return Path(*parts[idx:])


def write_dataset_manifest(dataset_root: Path, extra: dict[str, Any]) -> None:
    records = []
    for path in iter_files(dataset_root):
        records.append(
            {
                "path": str(path.relative_to(dataset_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        dataset_root / "DATASET_MANIFEST.json",
        {
            **extra,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_count": len(records),
            "total_size_bytes": sum(r["size_bytes"] for r in records),
            "files": records,
        },
    )


def materialize_split_zip(split_dir: Path) -> Path:
    parts = sorted(split_dir.glob("data.zip.[0-9][0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(f"No VPMR split ZIP parts found in {split_dir}")
    combined = BUILD_ROOT / "vpmr_source_combined.zip"
    expected_size = sum(part.stat().st_size for part in parts)
    if combined.exists() and combined.stat().st_size == expected_size:
        return combined
    with combined.open("wb") as out:
        for part in parts:
            with part.open("rb") as inp:
                shutil.copyfileobj(inp, out, length=1024 * 1024 * 16)
    return combined


def is_vpmr_raw_member(name: str) -> bool:
    lower = name.lower()
    if lower.endswith(("_input_origin.obj", "_input_origin.obj.mtl")):
        return True
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        return True
    return False


def vpmr_row_from_member(name: str) -> str | None:
    for part in Path(name).parts:
        if re.fullmatch(r"\d{3}", part):
            return part
    match = re.search(r"(?:^|/)(\d{3})_", name)
    if match:
        return match.group(1)
    return None


def extract_vpmr_raw_500() -> ArchiveSpec:
    dataset = STAGING_ROOT / "vpmr_faulty_synthetic_raw_available_500"
    dataset.mkdir(parents=True, exist_ok=True)
    combined_zip = materialize_split_zip(VPMR_SPLIT_ZIP_DIR)
    extracted_rows: set[str] = set()
    extracted_files: list[dict[str, Any]] = []
    row_re = re.compile(r"(?:^|/)(\d{3})_input_origin\.obj(?:\.mtl)?$", re.IGNORECASE)

    with zipfile.ZipFile(combined_zip) as zf:
        members = sorted(
            (info for info in zf.infolist() if not info.is_dir()),
            key=lambda info: info.filename,
        )
        source_rows: set[str] = set()
        for info in members:
            match = re.search(r"(?:^|/)(\d{3})_input_origin\.obj$", info.filename)
            if match:
                source_rows.add(match.group(1))

        for info in members:
            if not is_vpmr_raw_member(info.filename):
                continue
            match = row_re.search(info.filename)
            row = match.group(1) if match else vpmr_row_from_member(info.filename)
            if row:
                dst = dataset / f"row_{row}" / "raw" / Path(info.filename).name
                extracted_rows.add(row)
            else:
                dst = dataset / "shared_assets" / Path(info.filename).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024 * 8)
            extracted_files.append(
                {
                    "path": str(dst.relative_to(dataset)),
                    "zip_member": info.filename,
                    "size_bytes": dst.stat().st_size,
                    "sha256": sha256_file(dst),
                }
            )

    write_json(
        dataset / "SOURCE.json",
        {
            "dataset": "VPMR faulty synthetic raw geometries",
            "source_repository": VPMR_SOURCE_URL,
            "source_commit": VPMR_SOURCE_COMMIT,
            "split_zip_dir": str(VPMR_SPLIT_ZIP_DIR),
            "raw_rows_available_in_source": len(extracted_rows),
            "raw_rows_detected_in_zip": sorted(extracted_rows),
            "requested_faulty_synthetic_count_from_user": 874,
            "local_findings": (
                "The split VPMR source ZIP available in this workspace contains 500 "
                "raw rows (000-499). No local 874-row raw source was found during "
                "the release inventory, so this archive packages all raw VPMR rows "
                "that are actually available from the referenced source."
            ),
            "extracted_file_count": len(extracted_files),
            "extracted_files": extracted_files,
        },
    )
    write_dataset_manifest(
        dataset,
        {
            "dataset": "vpmr_faulty_synthetic_raw_available_500",
            "description": "All raw VPMR faulty synthetic geometries available locally/source-side.",
        },
    )
    return ArchiveSpec(
        "vpmr_faulty_synthetic_raw_available_500",
        dataset,
        "Raw VPMR faulty synthetic OBJ/MTL assets available from the source split ZIP.",
    )


def accepted_vpmr_rows(summary: dict[str, Any]) -> list[str]:
    rows = []
    by_category = summary.get("rows_by_category", {})
    rows.extend(by_category.get("accepted_strict", []))
    rows.extend(by_category.get("accepted_with_caveats_full_body_mesh", []))
    return sorted(set(str(row).zfill(3) for row in rows))


def copy_vpmr_attempt_companions(src_path: Path, dst_dir: Path, manifest: list[dict[str, Any]]) -> None:
    attempt_dir = None
    for parent in src_path.parents:
        if parent.name.startswith("attempt_"):
            attempt_dir = parent
            break
    if attempt_dir is None:
        return
    names = {
        "validation_report.json",
        "meshability_acceptance_metrics.json",
        "mesh_quality_report.json",
        "surface_topology_metrics.json",
        "diagnostic_geometry_preservation_metrics.json",
        "point_sampling_similarity_metrics.json",
        "candidate_tradeoff_report.json",
        "repair_operation_audit.json",
        "tetgen_worker_status.json",
        "row_meshability.json",
        "low_quality_tetrahedra_audit.json",
    }
    for path in sorted(attempt_dir.rglob("*")):
        if not path.is_file() or path.name not in names:
            continue
        copy_if_exists(path, dst_dir / "audit" / path.name, manifest)


def copy_vpmr_evidence_bundle(
    evidence: dict[str, Any],
    dst_dir: Path,
    copied: list[dict[str, Any]],
    *,
    include_mesh: bool,
) -> dict[str, Any]:
    copied_keys: list[str] = []
    for key, subdir in [
        ("surface_path", "surface"),
        ("quality_report_path", "validation"),
        ("source_path", "validation"),
    ]:
        src = resolve_framework_path(evidence.get(key))
        if src and src.exists():
            copy_if_exists(src, dst_dir / subdir / src.name, copied)
            copied_keys.append(key)
            if key == "surface_path":
                copy_vpmr_attempt_companions(src, dst_dir, copied)
    if include_mesh:
        src = resolve_framework_path(evidence.get("mesh_path"))
        if src and src.exists():
            copy_if_exists(src, dst_dir / "mesh" / src.name, copied)
            copied_keys.append("mesh_path")
    return {
        "selected_strategy": evidence.get("selected_strategy"),
        "selected_variant": evidence.get("selected_variant"),
        "tetra_count": evidence.get("tetra_count"),
        "mesh_accepted_old_gate": evidence.get("mesh_accepted_old_gate"),
        "mesh_generated_clean": evidence.get("mesh_generated_clean"),
        "quality_caveats": evidence.get("quality_caveats", []),
        "copied_keys": copied_keys,
    }


def build_vpmr_final_results() -> ArchiveSpec:
    dataset = STAGING_ROOT / "vpmr_strict_100_final_results"
    dataset.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    summary = read_json(VPMR_RUN / "meshability_relaxed_summary.json")
    rows = accepted_vpmr_rows(summary)

    summary_files = [
        "meshability_relaxed_summary.json",
        "meshability_relaxed_summary.md",
        "strict_campaign_summary.json",
        "strict_campaign_summary.md",
        "strict_campaign_manifest.json",
        "strict_accepted_runs_normalized/run_manifest.json",
        "strict_accepted_runs_normalized/migration_report.md",
        "reports/vpmr_meshability_relaxed_full_report.md",
        "reports/full_body_pipeline_bottleneck_audit_20260611.md",
        "reports/vpmr_four_phase_completion_report_20260611.md",
        "reports/tables/row_status.csv",
    ]
    for rel in summary_files:
        copy_if_exists(VPMR_RUN / rel, dataset / "campaign_reports" / rel, copied)

    row_records = []
    row_by_id = {str(item.get("row", "")).zfill(3): item for item in summary.get("rows", [])}
    for row in rows:
        record = row_by_id.get(row, {"row": row})
        row_root = dataset / "rows" / f"row_{row}"
        write_json(row_root / "row_result_summary.json", record)
        repaired = record.get("repaired_full_body_surface_path") or record.get("repaired_surface_path")
        mesh = record.get("full_body_mesh_path")
        quality = record.get("quality_report_path")
        if repaired:
            repaired_path = resolve_framework_path(repaired)
            if repaired_path.exists():
                copy_if_exists(repaired_path, row_root / "final" / repaired_path.name, copied)
                copy_vpmr_attempt_companions(repaired_path, row_root / "final", copied)
        if mesh:
            mesh_path = resolve_framework_path(mesh)
            if mesh_path.exists():
                copy_if_exists(mesh_path, row_root / "final" / mesh_path.name, copied)
        if quality:
            quality_path = resolve_framework_path(quality)
            if quality_path.exists():
                copy_if_exists(quality_path, row_root / "final" / quality_path.name, copied)

        full_body_evidence_records = []
        for idx, evidence in enumerate(record.get("evidence", []), start=1):
            full_body_evidence_records.append(
                copy_vpmr_evidence_bundle(
                    evidence,
                    row_root / "full_body_mesh_evidence" / f"evidence_{idx:03d}",
                    copied,
                    include_mesh=True,
                )
            )

        diagnostic_evidence_records = []
        for idx, evidence in enumerate(record.get("excluded_mesh_evidence", []), start=1):
            diagnostic_evidence_records.append(
                copy_vpmr_evidence_bundle(
                    evidence,
                    row_root / "diagnostic_mesh_evidence" / f"evidence_{idx:03d}",
                    copied,
                    include_mesh=True,
                )
            )
        if full_body_evidence_records or diagnostic_evidence_records:
            write_json(
                row_root / "evidence_copy_manifest.json",
                {
                    "full_body_mesh_evidence": full_body_evidence_records,
                    "diagnostic_mesh_evidence": diagnostic_evidence_records,
                    "diagnostic_note": (
                        "diagnostic_mesh_evidence may contain componentwise, partial, "
                        "isolated, split, or legacy-preservation meshes. These are "
                        "packaged as final evidence but are not global full-body "
                        "tetrahedral success unless also listed under full_body_mesh_evidence."
                    ),
                },
            )

        raw_source = VPMR_RUN / "extraction" / f"row_{row}" / "raw"
        if raw_source.exists():
            for path in iter_files(raw_source):
                copy_if_exists(path, row_root / "raw" / path.relative_to(raw_source), copied)
        row_records.append(
            {
                "row": row,
                "status": record.get("status"),
                "has_full_body_mesh": bool(mesh),
                "tetra_count": record.get("tetra_count"),
                "repaired_surface_path_present": bool(
                    repaired and resolve_framework_path(repaired) and resolve_framework_path(repaired).exists()
                ),
                "full_body_mesh_path_present": bool(
                    mesh and resolve_framework_path(mesh) and resolve_framework_path(mesh).exists()
                ),
                "full_body_evidence_count": len(full_body_evidence_records),
                "diagnostic_mesh_evidence_count": len(diagnostic_evidence_records),
                "caveats": record.get("caveats", []),
            }
        )

    write_json(
        dataset / "ACCEPTANCE_MANIFEST.json",
        {
            "dataset": "vpmr_strict_100_final_results",
            "source_run": str(VPMR_RUN),
            "acceptance_gate": (
                "Only a tetrahedral mesh for the full repaired body is functional "
                "full-body mesh evidence. Componentwise, isolated, split, or "
                "matching-only meshes are not counted as global success."
            ),
            "summary_counts": summary.get("counts", {}),
            "accepted_rows_packaged": rows,
            "functional_full_body_mesh_evidence_rows": summary.get("rows_by_category", {}).get(
                "functional_full_body_mesh_evidence_rows", []
            ),
            "row_records": row_records,
            "copied_file_count": len(copied),
            "copied_files": copied,
        },
    )
    write_dataset_manifest(
        dataset,
        {
            "dataset": "vpmr_strict_100_final_results",
            "description": "VPMR strict-100 raw selected rows plus final audited results for accepted rows.",
        },
    )
    return ArchiveSpec(
        "vpmr_strict_100_final_results",
        dataset,
        "Accepted VPMR strict-100 rows and their final audited artifacts.",
    )


def selected_thingi_artifact(path: Path) -> bool:
    name = path.name
    if name.endswith("_selected_repaired.stl"):
        return True
    if "tetgen_meshability" in path.parts and (
        name.endswith(".msh")
        or name in {"mesh_quality_report.json", "variant_attempt_result.json"}
    ):
        return True
    if name in {
        "validation_report.json",
        "meshability_acceptance_metrics.json",
        "surface_topology_metrics.json",
        "diagnostic_geometry_preservation_metrics.json",
        "candidate_tradeoff_report.json",
        "repair_operation_audit.json",
        "candidate_audit.json",
    }:
        return True
    return False


def case_id_from_path(path: Path) -> str:
    match = re.search(r"(thingi10k_\d+)", str(path))
    if match:
        return match.group(1)
    return "unknown_case"


def copy_thingi_outputs_from_validation(
    validation_path: Path,
    dst_cases: Path,
    copied: list[dict[str, Any]],
) -> dict[str, Any]:
    if not validation_path.exists():
        return {"validation_path": str(validation_path), "present": False}
    validation = read_json(validation_path)
    per_case: dict[str, list[str]] = {}
    for execution in validation.get("executions", []):
        artifacts = execution.get("artifacts", [])
        for artifact in artifacts:
            src_raw = artifact.get("path") if isinstance(artifact, dict) else artifact
            if not src_raw:
                continue
            src = Path(src_raw)
            if not src.exists() or not selected_thingi_artifact(src):
                continue
            case_id = case_id_from_path(src)
            if "outputs" in src.parts:
                rel = rel_from_anchor(src, "outputs")
            elif "artifacts" in src.parts:
                rel = rel_from_anchor(src, "artifacts")
            else:
                rel = Path(src.name)
            dst = dst_cases / case_id / rel
            copy_if_exists(src, dst, copied)
            per_case.setdefault(case_id, []).append(str(dst.relative_to(dst_cases / case_id)))
    return {
        "validation_path": str(validation_path),
        "present": True,
        "execution_count": len(validation.get("executions", [])),
        "case_count_with_selected_outputs": len(per_case),
        "outputs_by_case": {key: sorted(value) for key, value in sorted(per_case.items())},
    }


def build_thingi10k_results() -> ArchiveSpec:
    dataset = STAGING_ROOT / "thingi10k_100_raw_and_audited_outputs"
    dataset.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for rel in [
        "case_manifest.json",
        "download_manifest.json",
        "download_manifest.md",
        "commands_manifest.md",
        "repair_mesh_summary.json",
        "repair_mesh_summary.md",
        "repair_mesh_summary.csv",
        "reports/thingi10k_100_blocker_report.md",
        "reports/thingi10k_100_repair_mesh_report.md",
        "repair_mesh_runs_sequential_screening_v4_20260615/validation_summary.json",
        "repair_mesh_runs_sequential_screening_v4_20260615/next_stage_run_report.md",
        "repair_mesh_runs_sequential_screening_v4_20260615/run_manifest.json",
    ]:
        copy_if_exists(THINGI_RUN / rel, dataset / "campaign_reports" / rel, copied)

    case_manifest = read_json(THINGI_RUN / "case_manifest.json")
    raw_files = []
    input_files = []
    for case in case_manifest.get("cases", []):
        raw = Path(case.get("raw_geometry_path") or case.get("geometry_path", ""))
        if raw.exists():
            raw_files.append(raw)
        input_file = Path(case.get("input_file", ""))
        if input_file.exists():
            input_files.append(input_file)
    for raw in sorted(set(raw_files)):
        copy_if_exists(raw, dataset / "raw" / raw.name, copied)
    for input_file in sorted(set(input_files)):
        copy_if_exists(input_file, dataset / "inputs" / input_file.name, copied)

    validation_reports = [
        copy_thingi_outputs_from_validation(
            THINGI_RUN / "repair_mesh_runs_sequential_screening_v4_20260615/validation_summary.json",
            dataset / "cases",
            copied,
        )
    ]

    if THINGI_FOLLOWUP_RUN.exists():
        for rel in [
            "validation_summary.json",
            "repair_mesh_summary.json",
            "repair_mesh_summary.md",
            "case_manifest.json",
        ]:
            copy_if_exists(THINGI_FOLLOWUP_RUN / rel, dataset / "followup_hxt8_20260618" / rel, copied)
        validation_reports.append(
            copy_thingi_outputs_from_validation(
                THINGI_FOLLOWUP_RUN / "validation_summary.json",
                dataset / "followup_hxt8_20260618/cases",
                copied,
            )
        )

    summary = read_json(THINGI_RUN / "repair_mesh_summary.json")
    write_json(
        dataset / "ACCEPTANCE_MANIFEST.json",
        {
            "dataset": "thingi10k_100_raw_and_audited_outputs",
            "source_run": str(THINGI_RUN),
            "raw_count": len(raw_files),
            "summary_counts": {
                "case_count": summary.get("case_count"),
                "accepted_count": summary.get("accepted_count"),
                "failed_count": summary.get("failed_count"),
            },
            "acceptance_note": (
                "This audited Thingi10K 100-case run has accepted_count=0 under "
                "the campaign's full tetrahedral mesh gate. The archive still "
                "packages raw inputs plus final selected repaired surfaces and "
                "validation/meshability reports as failure/diagnostic outputs."
            ),
            "validation_reports": validation_reports,
            "copied_file_count": len(copied),
            "copied_files": copied,
        },
    )
    write_dataset_manifest(
        dataset,
        {
            "dataset": "thingi10k_100_raw_and_audited_outputs",
            "description": "Thingi10K 100 raw STL inputs and audited final repair/meshability outputs.",
        },
    )
    return ArchiveSpec(
        "thingi10k_100_raw_and_audited_outputs",
        dataset,
        "Thingi10K 100 raw inputs and final audited repair/meshability artifacts.",
    )


def build_snes_v2_results() -> ArchiveSpec:
    dataset = STAGING_ROOT / "snes_v2_raw_and_final_results"
    dataset.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for raw_name in ["snes-controller-v2.step", "V2.STEP"]:
        copy_if_exists(SNES_V2_RAW / raw_name, dataset / "raw" / raw_name, copied)
    wanted_names = {
        "mesh_final.msh",
        "mesh_final.xdmf",
        "mesh_final_boundaries.xdmf",
        "mesh_final_physical_groups.json",
        "pipeline_report.md",
        "run_usage_report.json",
        "agent_trace_report.json",
    }
    for run_dir in SNES_V2_RUNS:
        case_dir = dataset / "cases" / run_dir.name
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(run_dir)
            if path.name in wanted_names or path.name.startswith("geometry_sanitized_"):
                copy_if_exists(path, case_dir / rel, copied)
            elif rel.parts and rel.parts[0] == "review" and path.suffix.lower() in {
                ".json",
                ".md",
                ".png",
            }:
                copy_if_exists(path, case_dir / rel, copied)
    write_json(
        dataset / "RESULTS_MANIFEST.json",
        {
            "dataset": "snes_v2_raw_and_final_results",
            "raw_source": str(SNES_V2_RAW),
            "case_runs": [str(path) for path in SNES_V2_RUNS],
            "copied_file_count": len(copied),
            "copied_files": copied,
        },
    )
    write_dataset_manifest(
        dataset,
        {
            "dataset": "snes_v2_raw_and_final_results",
            "description": "SNES and V2 raw STEP CADs plus final healed geometry and generated meshes.",
        },
    )
    return ArchiveSpec(
        "snes_v2_raw_and_final_results",
        dataset,
        "SNES/V2 raw CADs plus final healed geometry, meshes, and validation.",
    )


def copy_tree_selected(
    src_root: Path,
    dst_root: Path,
    copied: list[dict[str, Any]],
    *,
    include_suffixes: set[str] | None = None,
    exclude_suffixes: set[str] | None = None,
) -> None:
    include_suffixes = include_suffixes or set()
    exclude_suffixes = exclude_suffixes or set()
    for path in iter_files(src_root):
        suffix = path.suffix.lower()
        if suffix in exclude_suffixes:
            continue
        if include_suffixes and suffix not in include_suffixes:
            continue
        copy_if_exists(path, dst_root / path.relative_to(src_root), copied)


def build_physio_core_results() -> ArchiveSpec:
    dataset = STAGING_ROOT / "physiological_cads_final_results_core"
    dataset.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []

    for rel in [
        "MANIFEST.files.tsv",
        "repo_docs/README.md",
        "repo_docs/context.md",
        "repo_docs/physio_results_manifest_20260429.md",
    ]:
        copy_if_exists(PHYSIO_BUNDLE / rel, dataset / rel, copied)

    physio_relative_roots = [
        Path("fsi_spine/assembly"),
        Path("fsi_spine/volume"),
        Path("heart_v003_aggregate"),
        Path("lung_v007_combined"),
    ]
    excluded_suffixes = {".vtk"}
    for rel_root in physio_relative_roots:
        src_root = PHYSIO_BUNDLE / rel_root
        if src_root.exists():
            copy_tree_selected(
                src_root,
                dataset / rel_root,
                copied,
                exclude_suffixes=excluded_suffixes,
            )

    fsi_surfaces = PHYSIO_BUNDLE / "fsi_spine/assembly/surfaces"
    if fsi_surfaces.exists():
        copy_tree_selected(
            fsi_surfaces,
            dataset / "fsi_spine/assembly/surfaces",
            copied,
            exclude_suffixes={".vtk"},
        )

    heart_surface_candidates = [
        HEART_V004_RUN / "geometry_sanitized_smoothed_for_reconstruction.stl",
        HEART_V004_RUN / "geometry_sanitized_repaired_for_reconstruction.stl",
        HEART_V004_RUN / "surface_artifacts/surface_external.stl",
        HEART_V004_RUN / "surface_artifacts/surface_external.ply",
        HEART_V004_RUN / "surface_artifacts/surface_render.png",
        HEART_V004_RUN / "physio_shell_reconstruction_report.json",
        HEART_V004_RUN / "physio_shell_reconstruction_report.md",
        HEART_V004_RUN / "heart_v004_stream_validation.json",
        HEART_V004_RUN / "surface_smoothing_report.json",
        HEART_V004_RUN / "run_manifest.txt",
    ]
    for src in heart_surface_candidates:
        if src.exists():
            rel = src.relative_to(HEART_V004_RUN)
            copy_if_exists(src, dataset / "heart_v004_surface_and_validation" / rel, copied)

    write_json(
        dataset / "RESULTS_MANIFEST.json",
        {
            "dataset": "physiological_cads_final_results_core",
            "source_bundle": str(PHYSIO_BUNDLE),
            "heart_v004_run": str(HEART_V004_RUN),
            "note": (
                "Core physiological release includes final meshes/surfaces/reports "
                "for FSI-spine and lung, aggregate heart reports, and heart v004 "
                "surface/validation files. The 12.5 GB full heart MSH is packaged "
                "separately in physiological_heart_full_mesh_msh."
            ),
            "excluded_duplicate_format": "VTK files are omitted when an MSH/STL/PLY equivalent is included.",
            "copied_file_count": len(copied),
            "copied_files": copied,
        },
    )
    write_dataset_manifest(
        dataset,
        {
            "dataset": "physiological_cads_final_results_core",
            "description": "Physiological CAD final surfaces, meshes, validation reports, and audit evidence.",
        },
    )
    return ArchiveSpec(
        "physiological_cads_final_results_core",
        dataset,
        "Physiological CAD final results without duplicate VTK payloads.",
    )


def build_heart_full_mesh() -> ArchiveSpec:
    dataset = STAGING_ROOT / "physiological_heart_full_mesh_msh"
    dataset.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    copy_if_exists(HEART_FULL_MESH, dataset / "heart_v004" / HEART_FULL_MESH.name, copied)
    for src in [
        HEART_V004_RUN / "heart_v004_stream_validation.json",
        HEART_V004_RUN / "physio_shell_reconstruction_report.json",
        HEART_V004_RUN / "physio_shell_reconstruction_report.md",
        HEART_V004_RUN / "run_manifest.txt",
    ]:
        if src.exists():
            copy_if_exists(src, dataset / "heart_v004" / src.name, copied)
    (dataset / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # Physiological Heart Full Mesh

            This archive contains the full generated Heart v004 tetrahedral mesh
            as MSH plus its validation/report files.

            The duplicate VTK export is intentionally omitted because the MSH is
            the generated mesh needed for the paper data release and the VTK copy
            would add roughly 9.7 GB of duplicate payload.

            Source MSH:
            `{HEART_FULL_MESH}`
            """
        )
    )
    write_json(
        dataset / "RESULTS_MANIFEST.json",
        {
            "dataset": "physiological_heart_full_mesh_msh",
            "source_mesh": str(HEART_FULL_MESH),
            "duplicate_vtk_omitted": str(
                HEART_V004_RUN / "attempt_01_tol_1e-08/mesh_shell_reconstructed.vtk"
            ),
            "copied_file_count": len(copied),
            "copied_files": copied,
        },
    )
    write_dataset_manifest(
        dataset,
        {
            "dataset": "physiological_heart_full_mesh_msh",
            "description": "Full Heart v004 generated tetrahedral MSH and validation reports.",
        },
    )
    return ArchiveSpec(
        "physiological_heart_full_mesh_msh",
        dataset,
        "Full Heart v004 generated MSH split for GitHub storage.",
    )


def split_archive(spec: ArchiveSpec) -> dict[str, Any]:
    archive_dir = DATASET_REPO / "archives" / spec.name
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = BUILD_ROOT / f"{spec.name}.tar.zst"
    if archive_path.exists():
        archive_path.unlink()

    run(
        [
            "tar",
            "--use-compress-program",
            "zstd -T0 -6",
            "-cf",
            str(archive_path),
            "-C",
            str(spec.staged_dir.parent),
            spec.staged_dir.name,
        ]
    )

    combined_sha = sha256_file(archive_path)
    run(
        [
            "split",
            "-b",
            SPLIT_SIZE,
            "-d",
            "-a",
            "4",
            str(archive_path),
            str(archive_dir / f"{spec.name}.tar.zst.part-"),
        ]
    )
    parts = []
    for part in sorted(archive_dir.glob(f"{spec.name}.tar.zst.part-*")):
        parts.append(
            {
                "path": str(part.relative_to(DATASET_REPO)),
                "size_bytes": part.stat().st_size,
                "sha256": sha256_file(part),
            }
        )
    manifest = {
        "archive": spec.name,
        "description": spec.description,
        "dataset_dir_name": spec.staged_dir.name,
        "compression": "tar.zst",
        "zstd_level": 6,
        "split_size": SPLIT_SIZE,
        "combined_archive_size_bytes": archive_path.stat().st_size,
        "combined_archive_sha256": combined_sha,
        "part_count": len(parts),
        "parts": parts,
        "reconstruct_command": (
            f"cat archives/{spec.name}/{spec.name}.tar.zst.part-* > {spec.name}.tar.zst && "
            f"sha256sum {spec.name}.tar.zst && "
            f"tar --use-compress-program zstd -xf {spec.name}.tar.zst"
        ),
    }
    write_json(DATASET_REPO / "manifests" / f"{spec.name}.archive_manifest.json", manifest)
    archive_path.unlink()
    return manifest


def write_release_docs(archive_manifests: list[dict[str, Any]]) -> None:
    archive_table = "\n".join(
        f"| `{item['archive']}` | {item['part_count']} | {item['combined_archive_size_bytes']} | `{item['combined_archive_sha256']}` |"
        for item in archive_manifests
    )
    readme = f"""# Dataset Agentic Framework

This repository stores the paper data release for the agentic meshing framework.
The payload is archive-based: each dataset is packaged as `tar.zst` and split
into GitHub-safe 95 MiB parts under `archives/`.

## Contents

| Archive | Parts | Combined bytes | Combined SHA-256 |
| --- | ---: | ---: | --- |
{archive_table}

## Reconstructing An Archive

Use the command from the matching `manifests/*.archive_manifest.json`, for example:

```bash
cat archives/vpmr_strict_100_final_results/vpmr_strict_100_final_results.tar.zst.part-* > vpmr_strict_100_final_results.tar.zst
sha256sum vpmr_strict_100_final_results.tar.zst
tar --use-compress-program zstd -xf vpmr_strict_100_final_results.tar.zst
```

## Dataset Notes

- VPMR raw faulty synthetic source available in this workspace contains 500 rows
  (`000`-`499`) from `{VPMR_SOURCE_URL}` at commit `{VPMR_SOURCE_COMMIT}`. The
  user-requested 874-row source was not present locally during this inventory,
  so all available raw VPMR rows were packaged and the limitation is recorded in
  the VPMR raw archive `SOURCE.json`.
- VPMR strict-100 final results follow the audited full-body gate from
  `meshability_relaxed_summary.json`: 23 accepted/caveat rows are packaged, but
  only rows with a full-body tetrahedral mesh path are labeled as functional
  full-body mesh evidence.
- Thingi10K 100 is packaged as raw inputs plus final audited repair and
  validation outputs. Its audited `accepted_count` is 0 under the full
  tetrahedral mesh gate, so these are diagnostic/failure outputs rather than
  claimed successful meshes.
- Physiological CADs are split into a core archive and a separate full Heart MSH
  archive. Duplicate VTK exports are omitted when MSH/STL/PLY equivalents are
  present.

Generated at: {datetime.now(timezone.utc).isoformat()}
"""
    (DATASET_REPO / "README.md").write_text(readme)

    index = {
        "repository": DATASET_REMOTE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_count": len(archive_manifests),
        "archives": archive_manifests,
    }
    write_json(DATASET_REPO / "manifests/release_index.json", index)

    (DATASET_REPO / "docs/release_scope.md").write_text(
        textwrap.dedent(
            f"""\
            # Release Scope

            This release keeps raw inputs, final generated/healed geometry,
            tetrahedral meshes, mesh validation, and compact audit reports.
            Intermediate prompt traces, large execution JSON logs, and duplicate
            visualization formats are excluded unless they are the only compact
            evidence for a result.

            The acceptance language is intentionally conservative:

            - A full repaired-body tetrahedral mesh is required before a VPMR or
              Thingi10K case is called a full-body mesh success.
            - Componentwise, isolated, split, or matching-only mesh artifacts are
              diagnostic evidence and are not labeled as global success.
            - Quality caveats are kept in the per-dataset manifests.

            Build script: `scripts/build_dataset_release.py`
            """
        )
    )


def verify_split_parts(archive_manifests: list[dict[str, Any]]) -> None:
    failures = []
    for manifest in archive_manifests:
        for part in manifest["parts"]:
            path = DATASET_REPO / part["path"]
            if not path.exists():
                failures.append(f"missing part {path}")
                continue
            if path.stat().st_size != part["size_bytes"]:
                failures.append(f"size mismatch {path}")
            if path.stat().st_size > 100 * 1024 * 1024:
                failures.append(f"part exceeds 100 MiB {path}")
            actual_sha = sha256_file(path)
            if actual_sha != part["sha256"]:
                failures.append(f"sha mismatch {path}")
    if failures:
        raise RuntimeError("\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-heart-full",
        action="store_true",
        help="Build every archive except the separate 12.5 GB heart MSH archive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_generated()
    builders = [
        extract_vpmr_raw_500,
        build_vpmr_final_results,
        build_thingi10k_results,
        build_snes_v2_results,
        build_physio_core_results,
    ]
    if not args.skip_heart_full:
        builders.append(build_heart_full_mesh)

    specs = []
    for builder in builders:
        print(f"== staging {builder.__name__}", flush=True)
        specs.append(builder())

    archive_manifests = []
    for spec in specs:
        print(f"== archiving {spec.name}", flush=True)
        archive_manifests.append(split_archive(spec))

    write_release_docs(archive_manifests)
    verify_split_parts(archive_manifests)
    print("== release build complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
