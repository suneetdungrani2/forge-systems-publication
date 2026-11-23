"""Reanalyze selected numerical-refinement trials from sanitized records only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analysis.verify import SHA256_RE, ArtifactError, _file_integrity, _read_json, load_artifact

TRIAL_FIELDS = (
    "profile", "design_id", "disturbance_id", "original_status", "refined_status",
    "original_dt_s", "refined_dt_s", "original_solver_iterations",
    "refined_solver_iterations", "cavity_witness_status",
)
STATUSES = {"PASS", "FAIL", "UNKNOWN", "CRASH"}
PROFILES = {"reference", "historical_fast"}
DESIGNS = {"original", "historical_v4_lead_in"}
REFINEMENT_MANIFEST_SCHEMA = "forge.v6.refinement-inputs/1"
SELECTION_RECEIPT_SCHEMA = "forge.v6-e2-refinement-selection/1"
PINNED_SELECTION_SHA256 = "86af8b7eedd2a7a769676d69421546db1461fc3b962494a8f5767dc1b1dce5ab"
PINNED_INTERPRETATION_SHA256 = "105d5a1094af2047ec6fe3394f31725e1318408fecccbc640a9d55f8515bd9dc"
PINNED_SOURCE_MANIFEST_SHA256 = "f76dbf46c77426ab3da9060c9b9fa48ee27e7e5099346c8a58840ba786fbf5ca"
REFINEMENT_FILES = {
    "records/refinement_trials.csv": "refinement_trials.csv",
    "records/refinement_facets48.csv": "refinement_facets48.csv",
    "records/refinement_selection.json": "refinement_selection.json",
}


class RefinementError(ValueError):
    """Raised when refinement records violate their declared data contract."""


def _read_csv(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise RefinementError(f"required input is not a regular file: {path.name}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None or len(header) != len(set(header)) or set(header) != set(expected):
                raise RefinementError(f"{path.name}: columns must be exactly {list(expected)} (order may vary)")
            rows: list[dict[str, str]] = []
            for number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise RefinementError(f"{path.name}:{number}: expected {len(header)} fields, found {len(values)}")
                row = dict(zip(header, values, strict=True))
                row["_row_id"] = f"{path.name}:{number}"
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise RefinementError(f"cannot parse {path.name}: {exc}") from exc


def _text(row: dict[str, str], field: str) -> str:
    value = row[field].strip()
    if not value:
        raise RefinementError(f"{row['_row_id']}: {field} must not be blank")
    return value


def _positive_number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise RefinementError(f"{row['_row_id']}: {field} must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise RefinementError(f"{row['_row_id']}: {field} must be finite and > 0")
    return value


def _positive_integer(row: dict[str, str], field: str) -> int:
    value = _positive_number(row, field)
    if not value.is_integer():
        raise RefinementError(f"{row['_row_id']}: {field} must be a positive integer")
    return int(value)


def _load_canonical_holdout(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    outcomes: dict[tuple[str, str, str], str] = {}
    for row in rows:
        if row["phase"].strip() != "holdout":
            continue
        profile = _text(row, "control_profile")
        design = _text(row, "design_id")
        disturbance = _text(row, "disturbance_id")
        status = _text(row, "outcome")
        if status not in STATUSES:
            raise RefinementError(f"{row['_row_id']}: unsupported canonical outcome {status!r}")
        key = profile, design, disturbance
        if key in outcomes:
            raise RefinementError(f"{row['_row_id']}: duplicate canonical holdout key {key!r}")
        outcomes[key] = status
    return outcomes


def _read_trials(path: Path, canonical: dict[tuple[str, str, str], str]) -> list[dict[str, Any]]:
    rows = _read_csv(path, TRIAL_FIELDS)
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        row_id = row["_row_id"]
        profile = _text(row, "profile")
        design = _text(row, "design_id")
        disturbance = _text(row, "disturbance_id")
        if profile not in PROFILES:
            raise RefinementError(f"{row_id}: unsupported profile {profile!r}")
        if design not in DESIGNS:
            raise RefinementError(f"{row_id}: unsupported design_id {design!r}")
        key = profile, design, disturbance
        if key in seen:
            raise RefinementError(f"{row_id}: duplicate refinement key {key!r}")
        seen.add(key)
        original = _text(row, "original_status")
        refined = _text(row, "refined_status")
        witness = _text(row, "cavity_witness_status")
        for field, status in (("original_status", original), ("refined_status", refined),
                              ("cavity_witness_status", witness)):
            if status not in STATUSES:
                raise RefinementError(f"{row_id}: unsupported {field} {status!r}")
        if key not in canonical:
            raise RefinementError(f"{row_id}: no matching canonical holdout trial for {key!r}")
        if original != canonical[key]:
            raise RefinementError(
                f"{row_id}: original_status {original!r} disagrees with canonical status {canonical[key]!r}"
            )

        original_dt = _positive_number(row, "original_dt_s")
        refined_dt = _positive_number(row, "refined_dt_s")
        original_iterations = _positive_integer(row, "original_solver_iterations")
        refined_iterations = _positive_integer(row, "refined_solver_iterations")
        if not math.isclose(refined_dt, original_dt / 2, rel_tol=1e-9, abs_tol=1e-12):
            raise RefinementError(f"{row_id}: refined_dt_s must be exactly half of original_dt_s")
        if refined_iterations != original_iterations * 2:
            raise RefinementError(f"{row_id}: refined_solver_iterations must be double the original")
        result.append({
            "profile": profile,
            "design_id": design,
            "disturbance_id": disturbance,
            "original_status": original,
            "refined_status": refined,
            "cavity_witness_status": witness,
        })
    return result


def _count_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = Counter(row["original_status"] for row in rows)
    refined = Counter(row["refined_status"] for row in rows)
    flips = [row for row in rows if row["original_status"] != row["refined_status"]]
    profiles: dict[str, Any] = {}
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row)
    for profile in sorted(by_profile):
        group = by_profile[profile]
        profiles[profile] = {
            "trial_count": len(group),
            "flip_count": sum(row["original_status"] != row["refined_status"] for row in group),
            "original_status_counts": {status: sum(row["original_status"] == status for row in group)
                                        for status in sorted(STATUSES)},
            "refined_status_counts": {status: sum(row["refined_status"] == status for row in group)
                                      for status in sorted(STATUSES)},
        }
    return {
        "trial_count": len(rows),
        "flip_count": len(flips),
        "status_counts": {
            "original": {status: original[status] for status in sorted(STATUSES)},
            "refined": {status: refined[status] for status in sorted(STATUSES)},
        },
        "profiles": profiles,
        "flips": [{
            "profile": row["profile"], "design_id": row["design_id"],
            "disturbance_id": row["disturbance_id"],
            "original_status": row["original_status"],
            "refined_status": row["refined_status"],
        } for row in flips],
    }


def _compare_refinements(
    primary: list[dict[str, Any]], facets: list[dict[str, Any]]
) -> dict[str, Any]:
    primary_by_key = {
        (row["profile"], row["design_id"], row["disturbance_id"]): row["refined_status"]
        for row in primary
    }
    facets_by_key = {
        (row["profile"], row["design_id"], row["disturbance_id"]): row["refined_status"]
        for row in facets
    }
    if primary_by_key.keys() != facets_by_key.keys():
        raise RefinementError("48-facet comparison requires identical 24-facet case keys")
    changed_rows = []
    for profile, design, disturbance in sorted(primary_by_key):
        status_24 = primary_by_key[(profile, design, disturbance)]
        status_48 = facets_by_key[(profile, design, disturbance)]
        if status_24 != status_48:
            changed_rows.append({
                "profile": profile,
                "design_id": design,
                "disturbance_id": disturbance,
                "refinement24_status": status_24,
                "facets48_status": status_48,
            })
    return {
        "matched_case_count": len(primary_by_key),
        "changed_case_count": len(changed_rows),
        "changed_rows": changed_rows,
    }


def _validate_selection_receipt(value: Any) -> dict[str, Any]:
    required = {
        "schema", "created_after_primary_results", "rule_frozen_before_primary",
        "interpretation_receipt_sha256", "source_manifest_sha256", "selected_seeds",
        "change", "primary_results_replaced", "prespecified_receipt",
    }
    allowed = required | {"exploratory_geometry_assessment"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - allowed:
        raise RefinementError("selection receipt does not match its strict schema")
    if value["schema"] != SELECTION_RECEIPT_SCHEMA:
        raise RefinementError(f"unsupported selection receipt schema {value['schema']!r}")
    if (value["created_after_primary_results"] is not True
            or value["rule_frozen_before_primary"] is not True
            or value["primary_results_replaced"] is not False):
        raise RefinementError("selection receipt lacks the frozen-before-primary, nonreplacement guarantees")
    if value["interpretation_receipt_sha256"] != PINNED_INTERPRETATION_SHA256:
        raise RefinementError("selection receipt does not bind the pinned pre-holdout interpretation")
    if value["source_manifest_sha256"] != PINNED_SOURCE_MANIFEST_SHA256:
        raise RefinementError("selection receipt does not bind the pinned E2 source manifest")
    change = value["change"]
    if (not isinstance(change, dict) or set(change) != {
            "timestep_multiplier", "solver_iteration_multiplier"}
            or isinstance(change["timestep_multiplier"], bool)
            or not isinstance(change["timestep_multiplier"], (int, float))
            or change["timestep_multiplier"] != 0.5
            or isinstance(change["solver_iteration_multiplier"], bool)
            or not isinstance(change["solver_iteration_multiplier"], int)
            or change["solver_iteration_multiplier"] != 2):
        raise RefinementError("selection receipt does not specify the frozen timestep/iteration refinement")
    if not isinstance(value["prespecified_receipt"], dict):
        raise RefinementError("selection receipt must embed the pre-holdout interpretation receipt")
    seeds = value["selected_seeds"]
    if not isinstance(seeds, dict) or set(seeds) != PROFILES:
        raise RefinementError("selection receipt selected_seeds must map both control profiles")
    for profile, values in seeds.items():
        if not isinstance(values, list) or not values:
            raise RefinementError(f"selection receipt selected_seeds[{profile}] must be nonempty")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in values):
            raise RefinementError(f"selection receipt selected_seeds[{profile}] must contain positive integers")
        if len(values) != len(set(values)):
            raise RefinementError(f"selection receipt selected_seeds[{profile}] contains duplicates")
    return value


def _expected_keys_from_selection(receipt: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (profile, design, str(seed))
        for profile, seeds in receipt["selected_seeds"].items()
        for design in sorted(DESIGNS)
        for seed in seeds
    }


def _refinement_inputs(
    data_root: Path, artifact: dict[str, Any]
) -> tuple[set[tuple[str, str, str]], Path, dict[str, str]]:
    listed = {entry["path"] for entry in artifact["manifest"]["files"]}
    if "operation_trials.csv" not in listed or "operation_trials.csv" not in artifact["datasets"]:
        raise RefinementError("canonical operation_trials.csv must be listed in manifest.json")
    if "refinement_manifest.json" not in listed:
        raise RefinementError("refinement_manifest.json must be listed in manifest.json")

    records = data_root / "records"
    if records.is_symlink() or not records.is_dir():
        raise RefinementError("records must be a real, non-symlink directory")
    manifest_path = data_root / "refinement_manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except ArtifactError as exc:
        raise RefinementError(str(exc)) from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "files", "primary_selection"}:
        raise RefinementError("refinement_manifest.json must contain exactly schema, files, and primary_selection")
    if manifest["schema"] != REFINEMENT_MANIFEST_SCHEMA:
        raise RefinementError(f"unsupported refinement manifest schema {manifest['schema']!r}")

    entries = manifest["files"]
    if not isinstance(entries, list):
        raise RefinementError("refinement manifest files must be an array")
    file_hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise RefinementError("each refinement file entry must contain exactly path, sha256, and bytes")
        relative = entry["path"]
        if not isinstance(relative, str) or relative not in REFINEMENT_FILES:
            raise RefinementError(f"refinement manifest path is not allowlisted: {relative!r}")
        if relative in file_hashes:
            raise RefinementError(f"duplicate refinement manifest path: {relative}")
        digest, size = entry["sha256"], entry["bytes"]
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise RefinementError(f"{relative}: sha256 must be a lowercase SHA-256 digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RefinementError(f"{relative}: bytes must be a nonnegative integer")
        path = data_root / relative
        if path.is_symlink() or not path.is_file() or path.parent.resolve(strict=True) != records.resolve(strict=True):
            raise RefinementError(f"{relative}: listed input must be a regular file below a real records directory")
        try:
            _file_integrity(path, relative, size, digest)
        except ArtifactError as exc:
            raise RefinementError(str(exc)) from exc
        file_hashes[relative] = digest
    required_paths = {"records/refinement_trials.csv", "records/refinement_selection.json"}
    if not required_paths.issubset(file_hashes):
        raise RefinementError("refinement manifest must list both refinement_trials.csv and refinement_selection.json")
    facets_path = records / "refinement_facets48.csv"
    if (facets_path.exists() or facets_path.is_symlink()) != ("records/refinement_facets48.csv" in file_hashes):
        raise RefinementError("optional refinement_facets48.csv must be listed if present and present if listed")

    selection_path = records / "refinement_selection.json"
    if file_hashes["records/refinement_selection.json"] != PINNED_SELECTION_SHA256:
        raise RefinementError("refinement selection receipt hash does not match the independently pinned receipt")
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != PINNED_SELECTION_SHA256:
        raise RefinementError("refinement selection receipt bytes differ from the independently pinned receipt")
    try:
        selection_receipt = _read_json(selection_path)
    except ArtifactError as exc:
        raise RefinementError(str(exc)) from exc
    selection_receipt = _validate_selection_receipt(selection_receipt)

    selection = manifest["primary_selection"]
    expected_selection_fields = {
        "expected_cases", "interpretation_receipt_sha256", "selection_receipt_sha256",
    }
    if not isinstance(selection, dict) or set(selection) != expected_selection_fields:
        raise RefinementError("primary_selection must contain expected_cases and both receipt hashes")
    for field in ("interpretation_receipt_sha256", "selection_receipt_sha256"):
        if not isinstance(selection[field], str) or not SHA256_RE.fullmatch(selection[field]):
            raise RefinementError(f"primary_selection.{field} must be a lowercase SHA-256 digest")
    if (selection["interpretation_receipt_sha256"] != PINNED_INTERPRETATION_SHA256
            or selection["selection_receipt_sha256"] != PINNED_SELECTION_SHA256):
        raise RefinementError("primary_selection receipt hashes do not match the pinned receipts")
    receipt_expected_keys = _expected_keys_from_selection(selection_receipt)
    cases = selection["expected_cases"]
    if not isinstance(cases, list) or not cases:
        raise RefinementError("primary_selection.expected_cases must be a nonempty array")
    expected_keys: set[tuple[str, str, str]] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or set(case) != {"profile", "design_id", "disturbance_id"}:
            raise RefinementError(f"expected case {index} must contain profile, design_id, and disturbance_id")
        key = tuple(case[field].strip() if isinstance(case[field], str) else ""
                    for field in ("profile", "design_id", "disturbance_id"))
        profile, design, disturbance = key
        if not profile or not disturbance or profile not in PROFILES or design not in DESIGNS:
            raise RefinementError(f"expected case {index} has an invalid identity")
        if key in expected_keys:
            raise RefinementError(f"primary_selection.expected_cases contains duplicate key {key!r}")
        expected_keys.add(key)
    if expected_keys != receipt_expected_keys:
        raise RefinementError("primary_selection.expected_cases differs from the pinned selection receipt")
    top_hashes = {entry["path"]: entry["sha256"] for entry in artifact["manifest"]["files"]}
    source_hashes = {
        "operation_trials.csv": top_hashes["operation_trials.csv"],
        "refinement_manifest.json": top_hashes["refinement_manifest.json"],
        **{REFINEMENT_FILES[path]: digest for path, digest in file_hashes.items()},
    }
    return receipt_expected_keys, data_root / "records" / "refinement_trials.csv", source_hashes


def analyze_refinement(artifact_dir: str | Path) -> dict[str, Any]:
    """Recompute selected refinement outcomes and return a path-free summary."""
    root = Path(artifact_dir)
    if root.is_symlink() or not root.is_dir():
        raise RefinementError("artifact directory must be a real directory, not a symlink")
    root = root.resolve(strict=True)
    layouts = [candidate for candidate in (root, root / "data")
               if (candidate / "operation_trials.csv").exists()]
    if len(layouts) != 1:
        raise RefinementError("artifact must contain exactly one operation_trials.csv layout")
    data_root = layouts[0]
    try:
        artifact = load_artifact(data_root)
    except ArtifactError as exc:
        raise RefinementError(str(exc)) from exc
    expected_keys, refinement_path, source_hashes = _refinement_inputs(data_root, artifact)
    canonical = _load_canonical_holdout(artifact["datasets"]["operation_trials.csv"])
    primary = _read_trials(refinement_path, canonical)
    observed_keys = {(row["profile"], row["design_id"], row["disturbance_id"]) for row in primary}
    if observed_keys != expected_keys:
        missing, unexpected = expected_keys - observed_keys, observed_keys - expected_keys
        raise RefinementError(
            f"refinement_trials.csv keyset differs from frozen selection (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    result = {
        "schema": "forge.v6.refinement-summary/1",
        "status": "ANALYZED" if primary else "PRESENT_EMPTY",
        "selection_complete": bool(primary) and observed_keys == expected_keys,
        "expected_case_count": len(expected_keys),
        "expected_cases": [
            {"profile": profile, "design_id": design, "disturbance_id": disturbance}
            for profile, design, disturbance in sorted(expected_keys)
        ],
        "verified_selection_receipt": True,
        "selection_receipt_sha256": PINNED_SELECTION_SHA256,
        "interpretation_receipt_sha256": PINNED_INTERPRETATION_SHA256,
        "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
        **_count_trials(primary),
        "selection_note": "Selected sensitivity subset; descriptive only and not a replacement for primary holdout results.",
        "interval": None,
        "interval_note": "No IID confidence interval: refinement cases were selected by protocol and may be related.",
        "source_file_hashes": source_hashes,
    }
    facets_relative = "records/refinement_facets48.csv"
    if "refinement_facets48.csv" in source_hashes:
        facets_path = data_root / facets_relative
        facets = _read_trials(facets_path, canonical)
        facet_keys = {(row["profile"], row["design_id"], row["disturbance_id"]) for row in facets}
        if facet_keys != expected_keys:
            raise RefinementError("refinement_facets48.csv keyset differs from the frozen primary selection")
        result["exploratory_facets48"] = {
            "status": "ANALYZED" if facets else "PRESENT_EMPTY",
            **_count_trials(facets),
            "cavity_witness_status_counts": {
                status: sum(row["cavity_witness_status"] == status for row in facets)
                for status in sorted(STATUSES)
            },
            "comparison_to_refinement24": _compare_refinements(primary, facets),
            "expected_case_count": len(expected_keys),
            "expected_cases": result["expected_cases"],
            "selection_complete": bool(facets) and facet_keys == expected_keys,
            "role": "exploratory reanalysis of the same frozen cases; excluded from primary refinement counts",
        }
    return result


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    try:
        _write_json(args.output_json, analyze_refinement(args.artifact_dir))
    except (OSError, RefinementError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
