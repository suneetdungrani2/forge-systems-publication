"""Validate a sanitized study-artifact directory without loading Forge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """Raised when an artifact violates its declared input contract."""


SCHEMA_VERSION = 1
FILE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "evaluator_cases.csv": (
        "case_id", "group_id", "fault_family", "truth", "checker",
        "status", "evidence_scope", "runtime_ms",
    ),
    "operation_trials.csv": (
        "phase", "control_profile", "design_id", "disturbance_id", "outcome",
        "legacy_success", "attempt_duration_s", "model_sha256", "protocol_sha256",
    ),
    "production_runs.csv": (
        "scenario_id", "paired_seed", "condition", "good_units", "horizon_s",
        "flow_time_s", "check_status", "protocol_sha256",
    ),
}
EPISODE_SCHEMA = (
    "task_id", "family", "method", "repeat_id", "episode_id",
    "selected_candidate_id", "outcome", "wall_s", "provider_model", "protocol_sha256",
)
OPTIONAL_DATA_FILES = {"episodes.csv"}
OPTIONAL_JSON_FILES = {"protocol.json", "refinement_manifest.json"}
AUXILIARY_FILES = {
    "records/geometry_sensitivity.json",
    "records/operation_model_contracts.json",
    "records/operation_witnesses.csv",
    "records/operation_witnesses.manifest.json",
    "records/e1_protocol.json",
    "records/candidate_attempts.json",
    "records/resource_accounting.json",
    "records/resource_summary.json",
    "records/result_identities.json",
    "records/first_candidate_outcomes.json",
    "records/primary_candidate_cohort.json",
    "records/first_candidate_cohort.json",
    "records/primary_results_seal.json",
    "records/study_provenance.json",
    "records/fixture_case.json",
}
ALLOWLIST = set(FILE_SCHEMAS) | OPTIONAL_DATA_FILES | OPTIONAL_JSON_FILES | AUXILIARY_FILES
PHASES = {"development", "qualification", "holdout"}
DESIGNS = {"original", "historical_v4_lead_in"}
CONTROL_PROFILES = {"reference", "historical_fast"}
OUTCOMES = {"PASS", "FAIL", "UNKNOWN", "CRASH"}
CHECK_STATUSES = {"PASS", "FAIL", "UNKNOWN", "ERROR"}
CONDITIONS = {"M00", "M01", "M10", "M11"}
METHODS = {"A0", "A1", "A2"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIRST_CANDIDATE_SCHEMA = "forge.v6.e1-first-results/1"
FIRST_CANDIDATE_FIELDS = (
    "episode_id", "task_id", "method", "repeat_id", "first_candidate_id",
    "final_candidate_id", "first_outcome", "final_outcome", "reused_primary",
    "first_result_sha256", "final_result_sha256",
)
FIRST_CANDIDATE_IDENTITY_FIELDS = (
    "schema", "amendment_sha256", "source_commit", "source_manifest_sha256",
    "primary_protocol_sha256", "primary_cohort_sha256", "first_cohort_sha256",
    "assessment_script_sha256", "primary_results_seal_sha256",
)
PINNED_E1_SOURCE_COMMIT = "1e8e388afda1c98eb2c6ccf2de52d59d7b23b259"
PINNED_E1_SOURCE_MANIFEST_SHA256 = "72968ebb370eb941c7e5cf971acdadcc6161effb6d3f1a0ccf6cfd8a351f2deb"
PINNED_E1_PROTOCOL_SHA256 = "54cded11ebdb1b67ff3db7f454166161413af62d7ae59d2c6e0b04a5635d370c"
PINNED_E1_AMENDMENT_SHA256 = "39a6bc9450b4185478fa87db98708346617c69e500ca0796ee46a1fa3468ae3f"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ArtifactError(f"non-finite JSON value is not allowed: {value}")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {path.name}: {exc}") from exc


def _number(value: str, field: str, row_id: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{row_id}: {field} must be numeric") from exc
    if not math.isfinite(number) or number < minimum:
        raise ArtifactError(f"{row_id}: {field} must be finite and >= {minimum}")
    return number


def _required_text(row: dict[str, str], fields: tuple[str, ...], row_id: str) -> None:
    for field in fields:
        if not row[field].strip():
            raise ArtifactError(f"{row_id}: {field} must not be blank")
        row[field] = row[field].strip()


def _validate_digest(value: str, field: str, row_id: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise ArtifactError(f"{row_id}: {field} must be a lowercase SHA-256 digest")


def _validate_rows(filename: str, rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, ...]] = set()
    if filename == "evaluator_cases.csv":
        for row in rows:
            row_id = row["_row_id"]
            _required_text(row, FILE_SCHEMAS[filename][:-1], row_id)
            if row["truth"] not in {"valid", "invalid"}:
                raise ArtifactError(f"{row_id}: truth must be valid or invalid")
            if row["status"] not in CHECK_STATUSES:
                raise ArtifactError(f"{row_id}: status must be one of {sorted(CHECK_STATUSES)}")
            _number(row["runtime_ms"], "runtime_ms", row_id)
            key = (row["case_id"],)
            if key in seen:
                raise ArtifactError(f"{row_id}: duplicate evaluator case_id {key[0]!r}")
            seen.add(key)
    elif filename == "operation_trials.csv":
        protocol_by_pair: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            row_id = row["_row_id"]
            _required_text(row, FILE_SCHEMAS[filename][:-3], row_id)
            if row["phase"] not in PHASES:
                raise ArtifactError(f"{row_id}: unsupported phase {row['phase']!r}")
            if row["control_profile"] not in CONTROL_PROFILES:
                raise ArtifactError(f"{row_id}: unsupported control_profile {row['control_profile']!r}")
            if row["design_id"] not in DESIGNS:
                raise ArtifactError(f"{row_id}: unsupported design_id {row['design_id']!r}")
            if row["outcome"] not in OUTCOMES:
                raise ArtifactError(f"{row_id}: unsupported outcome {row['outcome']!r}")
            if row["legacy_success"] not in {"true", "false", "unknown"}:
                raise ArtifactError(f"{row_id}: legacy_success must be true, false, or unknown")
            if row["attempt_duration_s"].strip():
                _number(row["attempt_duration_s"], "attempt_duration_s", row_id)
            elif row["outcome"] not in {"UNKNOWN", "CRASH"}:
                raise ArtifactError(f"{row_id}: attempt_duration_s is required for {row['outcome']}")
            _validate_digest(row["model_sha256"], "model_sha256", row_id)
            _validate_digest(row["protocol_sha256"], "protocol_sha256", row_id)
            key = (row["phase"], row["control_profile"], row["design_id"], row["disturbance_id"])
            if key in seen:
                raise ArtifactError(f"{row_id}: duplicate operation trial key {key!r}")
            seen.add(key)
            pair_key = (row["phase"], row["control_profile"], row["disturbance_id"])
            protocol_by_pair.setdefault(pair_key, set()).add(row["protocol_sha256"])
        for key, digests in protocol_by_pair.items():
            if len(digests) != 1:
                raise ArtifactError(
                    f"operation pair {key!r} mixes protocol_sha256 values across designs"
                )
    elif filename == "production_runs.csv":
        for row in rows:
            row_id = row["_row_id"]
            _required_text(row, ("scenario_id", "paired_seed", "condition", "horizon_s", "check_status", "protocol_sha256"), row_id)
            if row["condition"] not in CONDITIONS:
                raise ArtifactError(f"{row_id}: condition must be one of {sorted(CONDITIONS)}")
            if row["check_status"] not in CHECK_STATUSES:
                raise ArtifactError(f"{row_id}: unsupported check_status {row['check_status']!r}")
            if row["good_units"].strip():
                units = _number(row["good_units"], "good_units", row_id)
                if not units.is_integer():
                    raise ArtifactError(f"{row_id}: good_units must be an integer")
            elif row["check_status"] == "PASS":
                raise ArtifactError(f"{row_id}: a PASS requires measured good_units")
            _number(row["horizon_s"], "horizon_s", row_id, minimum=0.0)
            if float(row["horizon_s"]) == 0:
                raise ArtifactError(f"{row_id}: horizon_s must be > 0")
            if row["flow_time_s"].strip():
                _number(row["flow_time_s"], "flow_time_s", row_id)
            elif row["check_status"] == "PASS":
                raise ArtifactError(f"{row_id}: a PASS requires measured flow_time_s")
            _validate_digest(row["protocol_sha256"], "protocol_sha256", row_id)
            key = (row["scenario_id"], row["paired_seed"], row["condition"])
            if key in seen:
                raise ArtifactError(f"{row_id}: duplicate production run key {key!r}")
            seen.add(key)

    if filename == "production_runs.csv":
        by_run: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            by_run.setdefault((row["scenario_id"], row["paired_seed"]), set()).add(
                row["protocol_sha256"]
            )
        for key, digests in by_run.items():
            if len(digests) != 1:
                raise ArtifactError(f"production pair {key!r} mixes protocol_sha256 values")
    elif filename == "episodes.csv":
        for row in rows:
            row_id = row["_row_id"]
            _required_text(row, tuple(field for field in EPISODE_SCHEMA if field not in {
                "selected_candidate_id", "wall_s"
            }), row_id)
            if row["method"] not in METHODS:
                raise ArtifactError(f"{row_id}: method must be one of {sorted(METHODS)}")
            if row["outcome"] not in OUTCOMES:
                raise ArtifactError(f"{row_id}: outcome must be one of {sorted(OUTCOMES)}")
            candidate = row["selected_candidate_id"].strip()
            row["selected_candidate_id"] = candidate
            if not candidate and row["outcome"] not in {"FAIL", "CRASH"}:
                raise ArtifactError(f"{row_id}: selected_candidate_id may be blank only for FAIL or CRASH")
            if row["wall_s"].strip():
                _number(row["wall_s"], "wall_s", row_id)
            elif row["outcome"] not in {"UNKNOWN", "CRASH"}:
                raise ArtifactError(f"{row_id}: wall_s is required for {row['outcome']}")
            _validate_digest(row["protocol_sha256"], "protocol_sha256", row_id)
            key = (row["family"], row["task_id"], row["method"], row["repeat_id"], row["episode_id"])
            if key in seen:
                raise ArtifactError(f"{row_id}: duplicate episode key {key!r}")
            seen.add(key)


def _read_csv(path: Path, filename: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None:
                raise ArtifactError(f"{filename}: CSV is empty")
            if len(set(header)) != len(header):
                raise ArtifactError(f"{filename}: duplicate CSV column names")
            expected = FILE_SCHEMAS.get(filename)
            if filename == "episodes.csv":
                expected = EPISODE_SCHEMA
            if expected is not None and (len(header) != len(expected) or set(header) != set(expected)):
                raise ArtifactError(
                    f"{filename}: columns must be exactly {list(expected)} (order may vary)"
                )
            if filename in OPTIONAL_DATA_FILES and any(not name.strip() for name in header):
                raise ArtifactError(f"{filename}: column names must not be blank")
            rows: list[dict[str, str]] = []
            for data_number, values in enumerate(reader, start=1):
                row_id = f"{filename}:{data_number + 1}"
                if len(values) != len(header):
                    raise ArtifactError(f"{row_id}: expected {len(header)} fields, found {len(values)}")
                row = dict(zip(header, values, strict=True))
                row["_row_id"] = row_id
                rows.append(row)
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise ArtifactError(f"cannot parse {filename}: {exc}") from exc
    if filename in FILE_SCHEMAS or filename in OPTIONAL_DATA_FILES:
        _validate_rows(filename, rows)
    return rows


def _file_integrity(path: Path, filename: str, expected_bytes: int, expected_digest: str) -> None:
    try:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ArtifactError(
                f"{filename}: byte count mismatch (manifest {expected_bytes}, actual {actual_bytes})"
            )
        with path.open("rb") as stream:
            actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise ArtifactError(f"cannot verify {filename}: {exc}") from exc
    if actual_digest != expected_digest:
        raise ArtifactError(f"{filename}: SHA-256 mismatch")


def load_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Read, hash-check, and validate all allowlisted inputs in an artifact."""
    root = Path(artifact_dir)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("artifact directory must be a real directory, not a symlink")
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactError("manifest.json must be a regular, non-symlink file")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise ArtifactError("manifest.json must contain exactly schema_version and files")
    manifest_schema = manifest["schema_version"]
    if (
        isinstance(manifest_schema, bool)
        or not isinstance(manifest_schema, int)
        or manifest_schema != SCHEMA_VERSION
    ):
        raise ArtifactError(f"unsupported manifest schema_version {manifest['schema_version']!r}")
    entries = manifest["files"]
    if not isinstance(entries, list):
        raise ArtifactError("manifest files must be an array")

    datasets: dict[str, list[dict[str, str]]] = {}
    auxiliary_json: dict[str, Any] = {}
    protocol: dict[str, Any] | None = None
    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise ArtifactError("each manifest file entry must contain exactly path, sha256, bytes")
        name, digest, byte_count = entry["path"], entry["sha256"], entry["bytes"]
        if not isinstance(name, str) or name not in ALLOWLIST:
            raise ArtifactError(f"manifest path is not allowlisted: {name!r}")
        if name in listed:
            raise ArtifactError(f"duplicate manifest path: {name}")
        listed.add(name)
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ArtifactError(f"{name}: sha256 must be a lowercase SHA-256 digest")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ArtifactError(f"{name}: bytes must be a nonnegative integer")
        path = root / name
        if name in AUXILIARY_FILES:
            records_dir = root / "records"
            if records_dir.is_symlink() or not records_dir.is_dir():
                raise ArtifactError("records must be a real directory, not a symlink")
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"{name}: listed file must exist as a regular non-symlink file")
        expected_parent = root / "records" if name in AUXILIARY_FILES else root
        if path.resolve(strict=True).parent != expected_parent.resolve(strict=True):
            raise ArtifactError(f"{name}: listed file escapes artifact directory")
        _file_integrity(path, name, byte_count, digest)
        if name in FILE_SCHEMAS or name in OPTIONAL_DATA_FILES:
            datasets[name] = _read_csv(path, name)
        elif name in AUXILIARY_FILES and name.endswith(".json"):
            auxiliary_json[name] = _read_json(path)
        elif name == "protocol.json":
            protocol_value = _read_json(path)
            if not isinstance(protocol_value, dict) or set(protocol_value) - {
                "schema_version", "planned_holdout_trials", "expected_production_seeds"
            }:
                raise ArtifactError("protocol.json has unsupported fields")
            protocol_schema = protocol_value.get("schema_version")
            if (
                isinstance(protocol_schema, bool)
                or not isinstance(protocol_schema, int)
                or protocol_schema != SCHEMA_VERSION
            ):
                raise ArtifactError("protocol.json schema_version must be 1")
            planned = protocol_value.get("planned_holdout_trials")
            if planned is not None:
                if not isinstance(planned, dict) or set(planned) != CONTROL_PROFILES:
                    raise ArtifactError("protocol.json planned_holdout_trials must map both control profiles")
                for profile, designs in planned.items():
                    if not isinstance(designs, dict) or set(designs) != DESIGNS:
                        raise ArtifactError(f"protocol.json planned_holdout_trials[{profile}] must map both designs")
                    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in designs.values()):
                        raise ArtifactError("planned holdout counts must be nonnegative integers")
            expected_seeds = protocol_value.get("expected_production_seeds")
            if expected_seeds is not None and (
                not isinstance(expected_seeds, dict) or not expected_seeds
                or any(not isinstance(k, str) or not k.strip() or isinstance(v, bool)
                       or not isinstance(v, int) or v < 0 for k, v in expected_seeds.items())
            ):
                raise ArtifactError("protocol.json expected_production_seeds must map scenarios to nonnegative integers")
            protocol = protocol_value
    return {
        "root": root, "manifest": manifest, "datasets": datasets,
        "protocol": protocol, "auxiliary_json": auxiliary_json,
    }


def first_candidate_records(artifact: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the manifest-bound first-candidate amendment and its E1 join."""
    path = "records/first_candidate_outcomes.json"
    entries = {entry["path"]: entry for entry in artifact["manifest"]["files"]}
    if path not in entries:
        return None
    value = artifact.get("auxiliary_json", {}).get(path)
    if not isinstance(value, dict) or set(value) != {
        *FIRST_CANDIDATE_IDENTITY_FIELDS, "rows"
    }:
        raise ArtifactError("first-candidate result has unsupported top-level fields")
    if value["schema"] != FIRST_CANDIDATE_SCHEMA:
        raise ArtifactError("unsupported first-candidate result schema")
    expected_identity = {
        "amendment_sha256": PINNED_E1_AMENDMENT_SHA256,
        "source_commit": PINNED_E1_SOURCE_COMMIT,
        "source_manifest_sha256": PINNED_E1_SOURCE_MANIFEST_SHA256,
        "primary_protocol_sha256": PINNED_E1_PROTOCOL_SHA256,
    }
    for field, expected in expected_identity.items():
        if value.get(field) != expected:
            raise ArtifactError(f"first-candidate result has an unexpected {field}")
    for field in (
        "amendment_sha256", "source_manifest_sha256", "primary_protocol_sha256",
        "primary_cohort_sha256", "first_cohort_sha256", "assessment_script_sha256",
        "primary_results_seal_sha256",
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ArtifactError(f"first-candidate result {field} must be a lowercase SHA-256 digest")

    manifest_digests = {entry["path"]: entry["sha256"] for entry in artifact["manifest"]["files"]}
    result_identities = artifact.get("auxiliary_json", {}).get("records/result_identities.json")
    primary_cohort = artifact.get("auxiliary_json", {}).get("records/primary_candidate_cohort.json")
    first_cohort = artifact.get("auxiliary_json", {}).get("records/first_candidate_cohort.json")
    primary_results_seal = artifact.get("auxiliary_json", {}).get("records/primary_results_seal.json")
    if (not isinstance(result_identities, dict) or set(result_identities) != {
        "schema", "protocol_sha256", "cohort_freeze_sha256", "episodes",
    } or result_identities.get("schema") != "forge.v6-e1-result-identities/1"
            or result_identities.get("protocol_sha256") != PINNED_E1_PROTOCOL_SHA256):
        raise ArtifactError("first-candidate result requires the pinned primary result-identities record")
    result_identity_rows = result_identities.get("episodes")
    if not isinstance(result_identity_rows, list) or len(result_identity_rows) != 54:
        raise ArtifactError("primary result-identities record must contain all 54 E1 episode links")
    result_identity_by_episode: dict[str, dict[str, str]] = {}
    for index, item in enumerate(result_identity_rows):
        if not isinstance(item, dict) or set(item) != {
            "episode_id", "episode_receipt_sha256", "heldout_result_sha256",
        }:
            raise ArtifactError(f"primary result-identities row {index} has an unsupported schema")
        episode_id = item["episode_id"]
        if not isinstance(episode_id, str) or not episode_id.strip() or episode_id in result_identity_by_episode:
            raise ArtifactError(f"primary result-identities row {index} has a duplicate or invalid episode_id")
        for field in ("episode_receipt_sha256", "heldout_result_sha256"):
            if not isinstance(item[field], str) or not SHA256_RE.fullmatch(item[field]):
                raise ArtifactError(f"primary result-identities row {index} has an invalid {field}")
        result_identity_by_episode[episode_id] = item
    primary_digest = manifest_digests.get("records/primary_candidate_cohort.json")
    first_digest = manifest_digests.get("records/first_candidate_cohort.json")
    if (primary_digest is None or primary_digest != result_identities["cohort_freeze_sha256"]
            or primary_digest != value["primary_cohort_sha256"]):
        raise ArtifactError("first-candidate primary cohort hash does not match the frozen result identity")
    if first_digest is None or first_digest != value["first_cohort_sha256"]:
        raise ArtifactError("first-candidate cohort hash does not match its manifest-listed bytes")
    primary_results_seal_digest = manifest_digests.get("records/primary_results_seal.json")
    if primary_results_seal_digest is None or primary_results_seal_digest != value["primary_results_seal_sha256"]:
        raise ArtifactError("first-candidate primary-results seal hash does not match its manifest-listed bytes")
    if not isinstance(primary_cohort, dict) or set(primary_cohort) != {
        "schema", "protocol_sha256", "state", "episodes"
    } or primary_cohort.get("schema") != "forge.v6-e1-cohort-freeze/1":
        raise ArtifactError("primary candidate cohort has an unsupported schema")
    if (primary_cohort.get("protocol_sha256") != PINNED_E1_PROTOCOL_SHA256
            or primary_cohort.get("state") != "ALL_CANDIDATES_FROZEN"):
        raise ArtifactError("primary candidate cohort does not match the frozen E1 protocol")
    if not isinstance(primary_results_seal, dict) or set(primary_results_seal) != {
        "schema", "state", "created_at", "source_commit", "source_manifest_sha256",
        "primary_protocol_sha256", "primary_cohort_sha256", "episodes",
    } or primary_results_seal.get("schema") != "forge.v6.e1-primary-results-seal/1":
        raise ArtifactError("primary-results seal has an unsupported schema")
    if (not isinstance(primary_results_seal.get("created_at"), str)
            or not primary_results_seal["created_at"].strip()
            or primary_results_seal.get("state") != "ALL_PRIMARY_RESULTS_FROZEN"
            or primary_results_seal.get("source_commit") != value["source_commit"]
            or primary_results_seal.get("source_manifest_sha256") != value["source_manifest_sha256"]
            or primary_results_seal.get("primary_protocol_sha256") != PINNED_E1_PROTOCOL_SHA256
            or primary_results_seal.get("primary_cohort_sha256") != primary_digest):
        raise ArtifactError("primary-results seal does not match the frozen E1 source and cohort")
    if not isinstance(first_cohort, dict) or set(first_cohort) != {
        "schema", "amendment_sha256", "source_commit", "source_manifest_sha256",
        "primary_protocol_sha256", "primary_cohort_sha256", "assessment_script_sha256", "rows",
    } or first_cohort.get("schema") != "forge.v6.e1-first-cohort/1":
        raise ArtifactError("first-candidate cohort has an unsupported schema")
    for field in ("amendment_sha256", "source_commit", "source_manifest_sha256", "primary_protocol_sha256"):
        if first_cohort.get(field) != value[field]:
            raise ArtifactError(f"first-candidate cohort has an unexpected {field}")
    if (first_cohort.get("primary_cohort_sha256") != primary_digest
            or first_cohort.get("assessment_script_sha256") != value["assessment_script_sha256"]):
        raise ArtifactError("first-candidate cohort identity does not match its results")

    rows = value.get("rows")
    primary_rows = primary_cohort.get("episodes")
    primary_result_rows = primary_results_seal.get("episodes")
    first_rows = first_cohort.get("rows")
    episodes = artifact["datasets"].get("episodes.csv")
    if not isinstance(rows, list) or len(rows) != 54:
        raise ArtifactError("first-candidate result must contain all 54 E1 episode rows")
    if not isinstance(primary_rows, list) or len(primary_rows) != 54:
        raise ArtifactError("primary candidate cohort must contain all 54 E1 episode identities")
    if not isinstance(primary_result_rows, list) or len(primary_result_rows) != 54:
        raise ArtifactError("primary-results seal must contain all 54 E1 episode identities")
    if not isinstance(first_rows, list) or len(first_rows) != 54:
        raise ArtifactError("first-candidate cohort must contain all 54 E1 episode identities")
    if episodes is None or len(episodes) != 54:
        raise ArtifactError("first-candidate result requires all 54 canonical E1 episodes")

    by_episode: dict[tuple[str, str, str, str], dict[str, str]] = {}
    primary_identity_by_episode: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    canonical_key_by_episode_id: dict[str, tuple[str, str, str, str]] = {}
    for row in episodes:
        if row["protocol_sha256"] != PINNED_E1_PROTOCOL_SHA256:
            raise ArtifactError(f"{row['_row_id']}: protocol_sha256 does not match the pinned E1 protocol")
        key = tuple(row[field] for field in ("episode_id", "task_id", "method", "repeat_id"))
        if key in by_episode:
            raise ArtifactError(f"duplicate canonical E1 identity {key!r}")
        primary_identity = (key[0], key[1])
        if primary_identity in primary_identity_by_episode:
            raise ArtifactError(f"duplicate canonical E1 primary identity {primary_identity!r}")
        if key[0] in canonical_key_by_episode_id:
            raise ArtifactError(f"duplicate canonical E1 episode_id {key[0]!r}")
        by_episode[key] = row
        primary_identity_by_episode[primary_identity] = key
        canonical_key_by_episode_id[key[0]] = key

    def index_cohort(items: list[Any], label: str, candidate_field: str) -> dict[tuple[str, ...], dict[str, Any]]:
        indexed = {}
        identity_fields = (("episode_id", "task_id") if label == "primary candidate cohort"
                           else ("episode_id", "task_id", "method", "repeat_id"))
        expected_fields = set(identity_fields)
        if label == "primary candidate cohort":
            expected_fields.update((
                "episode_sha256", "selected_sha256", "candidate_raw_sha256", "model_sha256", "candidate_id",
            ))
        else:
            expected_fields.update((
                "first_ordinal", "first_candidate_id", "first_model_sha256",
                "first_candidate_raw_sha256", "episode_receipt_sha256", "eligible_for_heldout", "completion",
            ))
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != expected_fields or not all(
                isinstance(item.get(field), str) and item[field].strip() for field in identity_fields
            ):
                raise ArtifactError(f"{label} row {index} has an invalid schema or episode identity")
            key = tuple(item[field] for field in identity_fields)
            if key in indexed:
                raise ArtifactError(f"duplicate {label} identity {key!r}")
            candidate = item.get(candidate_field)
            if candidate is not None and (not isinstance(candidate, str) or not candidate.strip()):
                raise ArtifactError(f"{label} row {index} has an invalid {candidate_field}")
            digest_fields = (
                ("episode_sha256", "selected_sha256", "candidate_raw_sha256", "model_sha256")
                if label == "primary candidate cohort"
                else ("first_model_sha256", "first_candidate_raw_sha256", "episode_receipt_sha256")
            )
            for field in digest_fields:
                digest = item.get(field)
                if digest is not None and (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
                    raise ArtifactError(f"{label} row {index} has an invalid {field}")
            if label == "primary candidate cohort" and item.get("episode_sha256") is None:
                raise ArtifactError(f"primary candidate cohort row {index} lacks its episode receipt digest")
            if label == "first-candidate cohort" and item.get("episode_receipt_sha256") is None:
                raise ArtifactError(f"first-candidate cohort row {index} lacks its episode receipt digest")
            if label == "first-candidate cohort":
                if (type(item.get("first_ordinal")) is not int or item["first_ordinal"] != 1
                        or type(item.get("eligible_for_heldout")) is not bool
                        or item.get("completion") not in {"COMPLETED", "FAIL", "CRASH"}):
                    raise ArtifactError(f"first-candidate cohort row {index} is not an eligible ordinal-one record")
                if item.get("first_candidate_id") is None and any(
                    item.get(field) is not None for field in ("first_model_sha256", "first_candidate_raw_sha256")
                ):
                    raise ArtifactError(f"first-candidate cohort row {index} has fields for a missing candidate")
                if item["eligible_for_heldout"] and any(
                    item.get(field) is None for field in (
                        "first_candidate_id", "first_model_sha256", "first_candidate_raw_sha256"
                    )
                ):
                    raise ArtifactError(f"first-candidate cohort row {index} lacks an eligible candidate identity")
                if item["eligible_for_heldout"] and item["completion"] != "COMPLETED":
                    raise ArtifactError(f"first-candidate cohort row {index} is eligible without completion")
            if label == "primary candidate cohort":
                candidate_fields = ("selected_sha256", "candidate_raw_sha256", "model_sha256", "candidate_id")
                if candidate is None and any(item.get(field) is not None for field in candidate_fields):
                    raise ArtifactError(f"{label} row {index} has fields for a missing selected candidate")
                if candidate is not None and any(item.get(field) is None for field in candidate_fields):
                    raise ArtifactError(f"{label} row {index} has an incomplete selected candidate identity")
            indexed[key] = item
        return indexed

    primary_by_identity = index_cohort(primary_rows, "primary candidate cohort", "candidate_id")
    first_by_episode = index_cohort(first_rows, "first-candidate cohort", "first_candidate_id")
    primary_result_by_episode: dict[str, dict[str, Any]] = {}
    primary_result_fields = {
        "episode_id", "task_id", "episode_receipt_sha256", "selected_candidate_id",
        "heldout_result_sha256", "heldout_sidecar_sha256",
    }
    for index, item in enumerate(primary_result_rows):
        if not isinstance(item, dict) or set(item) != primary_result_fields:
            raise ArtifactError(f"primary-results seal row {index} has unsupported fields")
        episode_id = item["episode_id"]
        if not isinstance(episode_id, str) or not episode_id.strip() or episode_id in primary_result_by_episode:
            raise ArtifactError(f"primary-results seal row {index} has an invalid episode identity")
        if not isinstance(item["task_id"], str) or not item["task_id"].strip():
            raise ArtifactError(f"primary-results seal row {index} has an invalid task_id")
        for field in ("episode_receipt_sha256", "heldout_result_sha256", "heldout_sidecar_sha256"):
            if not isinstance(item[field], str) or not SHA256_RE.fullmatch(item[field]):
                raise ArtifactError(f"primary-results seal row {index} has an invalid {field}")
        candidate = item["selected_candidate_id"]
        if candidate is not None and (not isinstance(candidate, str) or not candidate.strip()):
            raise ArtifactError(f"primary-results seal row {index} has an invalid selected_candidate_id")
        primary_result_by_episode[episode_id] = item
    expected_episode_ids = {key[0] for key in by_episode}
    if (set(result_identity_by_episode) != expected_episode_ids
            or set(primary_result_by_episode) != expected_episode_ids):
        raise ArtifactError("primary result-identities links do not cover the canonical E1 episode set")
    for episode_id, result_identity in result_identity_by_episode.items():
        canonical_key = canonical_key_by_episode_id[episode_id]
        primary = primary_by_identity.get((canonical_key[0], canonical_key[1]))
        primary_result = primary_result_by_episode[episode_id]
        if (primary is None
                or result_identity["episode_receipt_sha256"] != primary.get("episode_sha256")
                or result_identity["episode_receipt_sha256"] != primary_result["episode_receipt_sha256"]
                or result_identity["heldout_result_sha256"] != primary_result["heldout_result_sha256"]):
            raise ArtifactError(f"primary result-identities row for {episode_id!r} disagrees with frozen result seals")
    seen: set[tuple[str, str, str, str]] = set()
    joined = []
    for json_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(FIRST_CANDIDATE_FIELDS):
            raise ArtifactError(f"first-candidate row {json_index} has unsupported fields")
        key_values = ("episode_id", "task_id", "method", "repeat_id")
        if any(not isinstance(row[field], str) or not row[field].strip() for field in key_values):
            raise ArtifactError(f"first-candidate row {json_index} has a blank identity")
        key = tuple(row[field] for field in key_values)
        if key in seen:
            raise ArtifactError(f"duplicate first-candidate E1 identity {key!r}")
        seen.add(key)
        canonical = by_episode.get(key)
        if canonical is None:
            raise ArtifactError(f"first-candidate row {json_index} does not match canonical E1 identity")
        primary = primary_by_identity.get((key[0], key[1]))
        first = first_by_episode.get(key)
        primary_result = primary_result_by_episode.get(key[0])
        if primary is None or first is None or primary_result is None:
            raise ArtifactError(f"first-candidate row {json_index} is missing from a frozen cohort")
        for field in ("first_candidate_id", "final_candidate_id"):
            if row[field] is not None and (not isinstance(row[field], str) or not row[field].strip()):
                raise ArtifactError(f"first-candidate row {json_index} {field} must be null or nonblank text")
        for field in ("first_outcome", "final_outcome"):
            if row[field] not in OUTCOMES:
                raise ArtifactError(f"first-candidate row {json_index} has invalid {field}")
        if (row["first_outcome"] == "PASS"
                and (row["first_candidate_id"] is None or not first["eligible_for_heldout"])):
            raise ArtifactError(f"first-candidate row {json_index} passes without an eligible ordinal-one candidate")
        if type(row["reused_primary"]) is not bool:
            raise ArtifactError(f"first-candidate row {json_index} reused_primary must be boolean")
        for field in ("first_result_sha256", "final_result_sha256"):
            digest = row[field]
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ArtifactError(f"first-candidate row {json_index} has invalid {field}")

        final_candidate = canonical["selected_candidate_id"] or None
        if (row["final_candidate_id"] != final_candidate or row["final_outcome"] != canonical["outcome"]
                or primary.get("candidate_id") != final_candidate
                or primary_result.get("task_id") != canonical["task_id"]
                or primary_result.get("selected_candidate_id") != final_candidate
                or primary_result.get("heldout_result_sha256") != row["final_result_sha256"]
                or first.get("first_candidate_id") != row["first_candidate_id"]
                or first.get("episode_receipt_sha256") != primary.get("episode_sha256")
                or primary_result.get("episode_receipt_sha256") != primary.get("episode_sha256")):
            raise ArtifactError(f"first-candidate row {json_index} disagrees with canonical final E1 result")
        same_nonnull_candidate = (
            row["first_candidate_id"] is not None
            and row["first_candidate_id"] == row["final_candidate_id"]
        )
        if row["reused_primary"]:
            if not same_nonnull_candidate or any(row[field] is None for field in (
                "first_result_sha256", "final_result_sha256"
            )) or row["first_result_sha256"] != row["final_result_sha256"] or row["first_outcome"] != row["final_outcome"]:
                raise ArtifactError(f"first-candidate row {json_index} has invalid primary-result reuse")
        elif same_nonnull_candidate:
            raise ArtifactError(f"first-candidate row {json_index} failed to mark reusable primary result")
        joined.append({
            **row,
            "family": canonical["family"],
            "_json_index": json_index,
            "_row_id": canonical["_row_id"],
        })

    if (seen != set(by_episode) or set(primary_by_identity) != set(primary_identity_by_episode)
            or set(first_by_episode) != set(by_episode)
            or set(primary_result_by_episode) != {key[0] for key in by_episode}):
        raise ArtifactError("first-candidate rows do not cover the canonical E1 episode set")
    method_counts: dict[str, int] = {}
    task_method_counts: dict[tuple[str, str, str], int] = {}
    task_clusters = set()
    for episode_id, task_id, method, repeat_id in seen:
        method_counts[method] = method_counts.get(method, 0) + 1
        family = by_episode[(episode_id, task_id, method, repeat_id)]["family"]
        task_clusters.add((family, task_id))
        task_key = (family, task_id, method)
        task_method_counts[task_key] = task_method_counts.get(task_key, 0) + 1
    if method_counts != {"A0": 18, "A1": 18, "A2": 18}:
        raise ArtifactError("first-candidate result must retain 18 episodes in each E1 arm")
    if len(task_clusters) != 9 or len(task_method_counts) != 27 or any(
        count != 2 for count in task_method_counts.values()
    ):
        raise ArtifactError("first-candidate result must match nine task clusters and two sessions per arm")

    return {"identity": {key: value[key] for key in FIRST_CANDIDATE_IDENTITY_FIELDS}, "rows": joined}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        artifact = load_artifact(args.artifact_dir)
    except ArtifactError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "VALID" if artifact["datasets"] else "VALID_NO_STUDY_DATA",
        "datasets": {name: len(rows) for name, rows in sorted(artifact["datasets"].items())},
        "protocol_present": artifact["protocol"] is not None,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
