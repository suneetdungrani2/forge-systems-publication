"""Offline arithmetic check for a bounded frozen E2 observation export.

This module uses only the Python standard library and sanitized CSV/JSON inputs.
It reclassifies frozen aggregate measurements; it does not run Forge or physics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "86308de844dbca9d0a9ee3103a23e977c097d24c"
SOURCE_MANIFEST_SHA256 = "f76dbf46c77426ab3da9060c9b9fa48ee27e7e5099346c8a58840ba786fbf5ca"
MANIFEST_SCHEMA = "forge.v6.e2-operation-witness-export/1"
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
PROFILES = ("reference", "historical_fast")
DESIGNS = ("original", "historical_v4_lead_in")
SEEDS = tuple(range(630100, 630200))
LIMITS = {
    "abs_z_error_mm": 0.4,
    "xy_error_mm": 1.0,
    "abs_yaw_error_deg": 3.3,
    "tilt_deg": 2.0,
}
SAFE_REASONS = {
    "invalid_model_timestep", "incomplete_task_and_post_task_pose_grid",
    "invalid_pose_sample_order_or_shape", "missing_duplicate_or_out_of_order_native_pose_sample",
    "no_samples_in_final_hold_window", "nonfinite_trial_result", "missing_observation",
    "nonfinite_observation", "missing_or_nonfinite_numerics", "missing_or_nonzero_warning",
    "missing_pose_samples_or_task_duration", "missing_or_incomplete_sample_trace",
    "missing_or_nonfinite_fixture_truth", "missing_or_nonfinite_hold_pose",
    "legacy_terminal_failure", "finite_hold_out_of_limits", "missing_qpos_replay_trace",
    "nonfinite_qpos_replay_trace", "crash", "other_unknown",
}
WITNESS_FIELDS = (
    "phase", "control_profile", "design_id", "disturbance_id", "outcome", "legacy_success",
    "n_hold_samples", "reason_code", "worst_abs_z_error_mm", "limit_abs_z_error_mm",
    "worst_xy_error_mm", "limit_xy_error_mm", "worst_abs_yaw_error_deg", "limit_abs_yaw_error_deg",
    "worst_tilt_deg", "limit_tilt_deg", "observation_sha256", "replay_sha256",
    "model_sha256", "protocol_sha256",
)
TRIAL_FIELDS = (
    "phase", "control_profile", "design_id", "disturbance_id", "outcome",
    "legacy_success", "attempt_duration_s", "model_sha256", "protocol_sha256",
)
METRIC_COLUMNS = (
    ("worst_abs_z_error_mm", "limit_abs_z_error_mm", "abs_z_error_mm"),
    ("worst_xy_error_mm", "limit_xy_error_mm", "xy_error_mm"),
    ("worst_abs_yaw_error_deg", "limit_abs_yaw_error_deg", "abs_yaw_error_deg"),
    ("worst_tilt_deg", "limit_tilt_deg", "tilt_deg"),
)


class ObservationError(ValueError):
    """Input data violates the frozen witness or source-trial contract."""


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ObservationError("input file is missing or unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
        raise ObservationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ObservationError("optional manifest contains a duplicate JSON field")
        value[key] = item
    return value


def _reject_constant(_value: str) -> None:
    raise ObservationError("optional manifest contains a non-finite JSON value")


def _finite_number(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ObservationError(f"{label} must be finite and nonnegative")
    return number


def classify(outcome: str, legacy_success: str, observed: dict[str, float] | None,
             limits: dict[str, float] | None) -> str:
    """Recompute only resolved composite outcomes; leave UNKNOWN/CRASH unresolved."""
    if outcome in {"UNKNOWN", "CRASH"}:
        return outcome
    if outcome not in {"PASS", "FAIL"} or observed is None or limits is None:
        raise ObservationError("resolved outcome lacks a complete aggregate observation")
    passed = legacy_success == "true" and all(observed[key] <= limits[key] for key in LIMITS)
    return "PASS" if passed else "FAIL"


def _read_csv(path: Path, expected: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ObservationError(f"{label} is missing or unsafe")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if tuple(reader.fieldnames or ()) != expected:
                raise ObservationError(f"{label} has an unexpected column contract")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ObservationError(f"{label} contains a malformed row")
                rows.append(row)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ObservationError(f"cannot read {label}") from exc


def _identity(row: dict[str, str], label: str) -> tuple[str, str, str]:
    if (row.get("phase") != "holdout" or row.get("control_profile") not in PROFILES
            or row.get("design_id") not in DESIGNS):
        raise ObservationError(f"{label} has an unexpected holdout identity")
    seed = row.get("disturbance_id", "")
    try:
        parsed = int(seed)
    except ValueError as exc:
        raise ObservationError(f"{label} disturbance_id must be a canonical holdout seed") from exc
    if str(parsed) != seed or parsed not in SEEDS:
        raise ObservationError(f"{label} disturbance_id is outside the frozen holdout denominator")
    return row["control_profile"], row["design_id"], seed


def _validate_witness(row: dict[str, str], row_number: int) -> tuple[tuple[str, str, str], str]:
    label = f"witness row {row_number}"
    key = _identity(row, label)
    outcome, legacy = row["outcome"], row["legacy_success"]
    if outcome not in {"PASS", "FAIL", "UNKNOWN", "CRASH"}:
        raise ObservationError(f"{label} has an unsupported outcome")
    if legacy not in {"true", "false", "unknown"}:
        raise ObservationError(f"{label} has an unsupported legacy terminal status")
    _sha(row["model_sha256"], f"{label} model_sha256")
    _sha(row["protocol_sha256"], f"{label} protocol_sha256")
    for field in ("observation_sha256", "replay_sha256"):
        if row[field]:
            _sha(row[field], f"{label} {field}")
    if outcome in {"PASS", "FAIL"}:
        try:
            count = int(row["n_hold_samples"])
        except ValueError as exc:
            raise ObservationError(f"{label} n_hold_samples must be a positive integer") from exc
        if str(count) != row["n_hold_samples"] or count < 1:
            raise ObservationError(f"{label} n_hold_samples must be a positive integer")
        if not row["observation_sha256"] or not row["replay_sha256"]:
            raise ObservationError(f"{label} resolved observation lacks artifact digests")
        observed = {}
        limits = {}
        for metric, limit, key_name in METRIC_COLUMNS:
            observed[key_name] = _finite_number(row[metric], f"{label} {metric}")
            limits[key_name] = _finite_number(row[limit], f"{label} {limit}")
            if limits[key_name] != LIMITS[key_name]:
                raise ObservationError(f"{label} {limit} differs from the frozen protocol")
        recomputed = classify(outcome, legacy, observed, limits)
        if recomputed != outcome:
            raise ObservationError(f"{label} source outcome differs from aggregate recomputation")
    else:
        if row["n_hold_samples"] or any(row[metric] for metric, _, _ in METRIC_COLUMNS):
            raise ObservationError(f"{label} unresolved outcome must retain blank aggregates")
        expected_reason = "crash" if outcome == "CRASH" else row["reason_code"]
        if row["reason_code"] not in SAFE_REASONS or row["reason_code"] != expected_reason:
            raise ObservationError(f"{label} unresolved outcome has an unsafe reason code")
        for _, limit, key_name in METRIC_COLUMNS:
            if _finite_number(row[limit], f"{label} {limit}") != LIMITS[key_name]:
                raise ObservationError(f"{label} {limit} differs from the frozen protocol")
    resolved_reasons = {"", "legacy_terminal_failure", "finite_hold_out_of_limits"}
    if outcome in {"PASS", "FAIL"} and row["reason_code"] not in resolved_reasons:
        raise ObservationError(f"{label} has an unsupported resolved reason code")
    return key, outcome


def _validate_manifest(path: Path, witness_path: Path) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise ObservationError("optional manifest is missing or unsafe")
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_pairs,
                              parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObservationError("optional manifest is unreadable") from exc
    expected_keys = {"schema", "row_count", "scope", "units", "frozen_source", "inputs",
                     "operation_witnesses_sha256", "limits"}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ObservationError("optional manifest does not match the export contract")
    source_identity = {"commit": SOURCE_COMMIT, "manifest_sha256": SOURCE_MANIFEST_SHA256}
    if (manifest["schema"] != MANIFEST_SCHEMA or manifest["row_count"] != 400
            or manifest["scope"] != "frozen aggregate observations; not raw pose-grid coverage"
            or manifest["frozen_source"] != source_identity
            or manifest["units"] != {"abs_z_error_mm": "mm", "xy_error_mm": "mm",
                                      "abs_yaw_error_deg": "deg", "tilt_deg": "deg"}
            or manifest["limits"] != LIMITS):
        raise ObservationError("optional manifest changes frozen scope, units, source, or limits")
    if not isinstance(manifest["inputs"], dict) or set(manifest["inputs"]) != set(PROFILES):
        raise ObservationError("optional manifest source profiles are incomplete")
    input_fields = {"protocol_sha256", "study_status_sha256", "summary_sha256", "trials_sha256"}
    for profile in PROFILES:
        profile_input = manifest["inputs"][profile]
        if not isinstance(profile_input, dict) or set(profile_input) != input_fields:
            raise ObservationError("optional manifest source digest contract is invalid")
        for name, value in manifest["inputs"][profile].items():
            _sha(value, f"manifest {profile} {name}")
    if _sha(manifest["operation_witnesses_sha256"], "manifest CSV digest") != _sha_file(witness_path):
        raise ObservationError("optional manifest does not match witness CSV bytes")


def check(witness_path: Path, trials_path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    if manifest_path is not None:
        _validate_manifest(manifest_path, witness_path)
    witness_rows = _read_csv(witness_path, WITNESS_FIELDS, "witness CSV")
    trial_rows = _read_csv(trials_path, TRIAL_FIELDS, "operation_trials.csv")
    witness: dict[tuple[str, str, str], dict[str, str]] = {}
    counts: Counter[tuple[str, str, str]] = Counter()
    for index, row in enumerate(witness_rows, start=2):
        key, recomputed = _validate_witness(row, index)
        if key in witness:
            raise ObservationError("witness CSV contains a duplicate holdout identity")
        witness[key] = row
        counts[(key[0], key[1], recomputed)] += 1
    expected = {(profile, design, str(seed)) for profile in PROFILES for design in DESIGNS for seed in SEEDS}
    if len(witness) != 400 or set(witness) != expected:
        raise ObservationError("witness CSV does not contain the exact 400-row holdout denominator")

    source: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in trial_rows:
        if row["phase"] != "holdout":
            continue
        key = _identity(row, "operation_trials.csv row")
        if key in source:
            raise ObservationError("operation_trials.csv contains a duplicate holdout identity")
        valid_outcomes = {"PASS", "FAIL", "UNKNOWN", "CRASH"}
        valid_legacy = {"true", "false", "unknown"}
        if row["outcome"] not in valid_outcomes or row["legacy_success"] not in valid_legacy:
            raise ObservationError("operation_trials.csv has an unsupported status")
        _sha(row["model_sha256"], "source model_sha256")
        _sha(row["protocol_sha256"], "source protocol_sha256")
        duration = row["attempt_duration_s"].strip()
        if duration:
            _finite_number(duration, "source attempt_duration_s")
        elif row["outcome"] not in {"UNKNOWN", "CRASH"}:
            raise ObservationError("source resolved row has no attempt duration")
        source[key] = row
    if len(source) != 400 or set(source) != expected:
        raise ObservationError("operation_trials.csv holdout rows do not match the exact 400-row denominator")
    for key in sorted(expected):
        source_row = source[key]
        # UNKNOWN and CRASH remain exactly as recorded; only PASS/FAIL arithmetic is recomputed.
        witness_row = witness[key]
        witness_row_outcome = witness_row["outcome"]
        if (witness_row_outcome != source_row["outcome"]
                or source_row["legacy_success"] != witness_row["legacy_success"]
                or source_row["model_sha256"] != witness_row["model_sha256"]
                or source_row["protocol_sha256"] != witness_row["protocol_sha256"]):
            raise ObservationError("witness classification or identity differs from canonical trial row")
    source_counts = Counter((key[0], key[1], row["outcome"]) for key, row in source.items())
    if any(counts[key] != source_counts[key] for key in set(counts) | set(source_counts)):
        raise ObservationError("witness classification counts differ from canonical source keys")
    return {
        "witness_rows": len(witness_rows),
        "joined_holdout_rows": len(source),
        "recomputed_classifications": {
            f"{profile}/{design}/{outcome}": counts[(profile, design, outcome)]
            for profile in PROFILES for design in DESIGNS for outcome in ("PASS", "FAIL", "UNKNOWN", "CRASH")
            if counts[(profile, design, outcome)]
        },
        "scope": ("arithmetic reanalysis of frozen aggregate observations; not new physics, "
                  "minimum proof, or full native-grid coverage"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness_csv", type=Path)
    parser.add_argument("operation_trials_csv", type=Path)
    parser.add_argument("--manifest", type=Path, help="optional operation_witnesses.manifest.json")
    args = parser.parse_args(argv)
    try:
        result = check(args.witness_csv, args.operation_trials_csv, args.manifest)
    except (ObservationError, OSError, ValueError) as exc:
        print(f"observation check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
