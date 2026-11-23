"""Render final paper result macros from a verified standalone analysis summary."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

E0_SCOPES = {
    "axis": "exact_centerline_depth",
    "local_wall": "constructed_local_minimum_wall",
    "evidence": "evidence_assessability",
    "replay": "replay_identity_binding",
    "claim": "production_claim_pointer_consistency",
}
E0_CHECKERS = {
    "legacy": "legacy_sampling", "new": "new_exact_brep",
    "evidence": "engineering_evidence", "replay": "replay_binding",
    "claim": "production_claim_consistency",
}
E1_PLANNED = 54
E1_SOURCE_COMMIT = "1e8e388afda1c98eb2c6ccf2de52d59d7b23b259"
E1_SOURCE_MANIFEST_SHA256 = "72968ebb370eb941c7e5cf971acdadcc6161effb6d3f1a0ccf6cfd8a351f2deb"
E1_PROTOCOL_SHA256 = "54cded11ebdb1b67ff3db7f454166161413af62d7ae59d2c6e0b04a5635d370c"
E1_AMENDMENT_SHA256 = "39a6bc9450b4185478fa87db98708346617c69e500ca0796ee46a1fa3468ae3f"
E2_EXPECTED = {
    ("development", "reference"): 12,
    ("development", "historical_fast"): 6,
    ("qualification", "reference"): 40,
    ("qualification", "historical_fast"): 40,
    ("holdout", "reference"): 100,
    ("holdout", "historical_fast"): 100,
}
E3_CONTRASTS = ("mechanical", "schedule", "combined", "interaction")
E3_METRIC_LABELS = {
    "good_units": "Good units",
    "flow_time_s": "Flow time",
    "horizon_s": "Horizon",
}
E3_CONTRAST_LABELS = {
    "mechanical": "Mechanical",
    "schedule": "Schedule",
    "combined": "Combined",
    "interaction": "Interaction",
}
# Protocol identities, independently read from the pinned selection receipt;
# these are eligibility cases, never expected outcomes.
FROZEN_REFINEMENT_SEEDS = {
    "reference": (630100, 630101, 630102, 630103, 630104, 630144),
    "historical_fast": (630100, 630101, 630102, 630103, 630104, 630131, 630169),
}
REFINEMENT_STATUSES = ("PASS", "FAIL", "UNKNOWN", "CRASH")
SHA256_HEX = set("0123456789abcdef")
PINNED_REFINEMENT_SELECTION_SHA256 = "86af8b7eedd2a7a769676d69421546db1461fc3b962494a8f5767dc1b1dce5ab"
PINNED_REFINEMENT_INTERPRETATION_SHA256 = "105d5a1094af2047ec6fe3394f31725e1318408fecccbc640a9d55f8515bd9dc"
PINNED_REFINEMENT_SOURCE_MANIFEST_SHA256 = "f76dbf46c77426ab3da9060c9b9fa48ee27e7e5099346c8a58840ba786fbf5ca"


def _escape(value: Any) -> str:
    text = str(value)
    substitutions = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(substitutions.get(char, char) for char in text)


def _number(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected a finite numeric result, found {value!r}")
    return f"{value:.{digits}f}"


def _stratum(e0: dict[str, Any], truth: str, checker: str, scope: str) -> dict[str, Any]:
    matches = [row for row in e0.get("strata", [])
               if row.get("truth") == truth and row.get("checker") == checker
               and row.get("evidence_scope") == scope]
    if len(matches) != 1:
        raise ValueError(f"expected one E0 stratum for truth={truth}, checker={checker}, scope={scope}")
    return matches[0]


def _required_case_count(e0: dict[str, Any], truth: str, checker: str, scope: str,
                         expected: int) -> dict[str, Any]:
    row = _stratum(e0, truth, checker, scope)
    if row.get("case_count") != expected:
        raise ValueError(f"E0 {truth}/{checker}/{scope}: expected {expected} cases, found {row.get('case_count')}")
    return row


def _macro(name: str, value: Any) -> str:
    # Ordinary TeX control-sequence names contain letters, not metric digits.
    words = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
    name = "".join(words[int(char)] if char in "0123456789" else char for char in name)
    if not name.isascii() or not name.isalpha():
        raise ValueError("paper macro name must resolve to ASCII letters")
    return f"\\newcommand{{\\{name}}}{{{_escape(value)}}}"


def _table_row(values: list[Any]) -> str:
    return " & ".join(_escape(value) for value in values) + r" \\"


def _e0_summary(e0: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    if e0.get("row_count") != 300 or e0.get("status") != "ANALYZED":
        raise ValueError("E0 requires 300 checker-assessment rows for 200 unique fixtures")
    expected = (
        ("invalid", E0_CHECKERS["new"], E0_SCOPES["axis"], 35),
        ("valid", E0_CHECKERS["new"], E0_SCOPES["axis"], 35),
        ("invalid", E0_CHECKERS["new"], E0_SCOPES["local_wall"], 15),
        ("valid", E0_CHECKERS["new"], E0_SCOPES["local_wall"], 15),
        ("invalid", E0_CHECKERS["evidence"], E0_SCOPES["evidence"], 40),
        ("valid", E0_CHECKERS["evidence"], E0_SCOPES["evidence"], 40),
        ("invalid", E0_CHECKERS["replay"], E0_SCOPES["replay"], 5),
        ("valid", E0_CHECKERS["replay"], E0_SCOPES["replay"], 5),
        ("invalid", E0_CHECKERS["claim"], E0_SCOPES["claim"], 5),
        ("valid", E0_CHECKERS["claim"], E0_SCOPES["claim"], 5),
    )
    canonical = [_required_case_count(e0, *item) for item in expected]
    if (sum(row["case_count"] for row in canonical),
        sum(row["case_count"] for row in canonical if row["truth"] == "invalid"),
        sum(row["case_count"] for row in canonical if row["truth"] == "valid")) != (200, 100, 100):
        raise ValueError("E0 strata do not reconcile to 200 unique fixtures (100 per truth class)")
    macros = [_macro("EZeroCaseCount", 200), _macro("EZeroInvalidCount", 100),
              _macro("EZeroValidCount", 100)]
    named = {
        "EZeroAxisThinLegacyAcceptance": ("invalid", "legacy", "axis", "PASS"),
        "EZeroAxisThinLegacyRejection": ("invalid", "legacy", "axis", "FAIL"),
        "EZeroAxisThinNewRejection": ("invalid", "new", "axis", "FAIL"),
        "EZeroAxisValidLegacyAcceptance": ("valid", "legacy", "axis", "PASS"),
        "EZeroAxisValidNewUnknown": ("valid", "new", "axis", "UNKNOWN"),
        "EZeroWebInvalidLegacyAcceptance": ("invalid", "legacy", "local_wall", "PASS"),
        "EZeroWebInvalidNewUnknown": ("invalid", "new", "local_wall", "UNKNOWN"),
        "EZeroWebValidLegacyAcceptance": ("valid", "legacy", "local_wall", "PASS"),
        "EZeroWebValidNewUnknown": ("valid", "new", "local_wall", "UNKNOWN"),
        "EZeroEvidenceInvalidUnknown": ("invalid", "evidence", "evidence", "UNKNOWN"),
        "EZeroEvidenceValidPass": ("valid", "evidence", "evidence", "PASS"),
    }
    for name, (truth, checker, scope, status) in named.items():
        row = _stratum(e0, truth, E0_CHECKERS[checker], E0_SCOPES[scope])
        macros.append(_macro(name, row["status_counts"].get(status, 0)))
    rows = []
    geometry_rows = []
    seen = set()
    for row in sorted(e0["strata"], key=lambda item: (item["evidence_scope"], item["truth"], item["checker"])):
        key = row["truth"], row["checker"], row["evidence_scope"]
        if key in seen:
            raise ValueError(f"duplicate E0 stratum key {key}")
        seen.add(key)
        counts = row["status_counts"]
        short_scope = {E0_SCOPES["axis"]: "axis depth", E0_SCOPES["local_wall"]: "local web",
                       E0_SCOPES["evidence"]: "evidence", E0_SCOPES["replay"]: "replay",
                       E0_SCOPES["claim"]: "claim"}.get(row["evidence_scope"], row["evidence_scope"])
        short_checker = {
            E0_CHECKERS["legacy"]: "legacy", E0_CHECKERS["new"]: "exact witness",
            E0_CHECKERS["evidence"]: "evidence", E0_CHECKERS["replay"]: "replay",
            E0_CHECKERS["claim"]: "claim",
        }.get(row["checker"], row["checker"])
        values = [short_scope, row["truth"], short_checker, row["case_count"], counts.get("PASS", 0),
                  counts.get("FAIL", 0), counts.get("UNKNOWN", 0), counts.get("ERROR", 0)]
        rows.append(_table_row(values))
        if row["evidence_scope"] in {E0_SCOPES["axis"], E0_SCOPES["local_wall"]}:
            geometry_rows.append(_table_row(values[:-1]))
    return macros, rows, geometry_rows


def _require_complete(studies: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    e1, e2, e3 = (studies.get(key) for key in ("E1", "E2", "E3"))
    if not all(isinstance(item, dict) for item in (e1, e2, e3)):
        raise ValueError("final paper generation requires complete E1, E2, and E3 summaries")
    if e1.get("status") != "ANALYZED" or e1.get("row_count") != E1_PLANNED:
        raise ValueError("final paper requires all 54 planned E1 episodes")
    arms = e1.get("arm_counts", {})
    if set(arms) != {"A0", "A1", "A2"} or any(arms[arm].get("episode_count") != 18 for arm in ("A0", "A1", "A2")):
        raise ValueError("final paper requires 18 observed E1 episodes in each feedback arm")
    for arm in arms.values():
        counts = arm.get("outcome_counts", {})
        if sum(counts.get(key, 0) for key in ("PASS", "FAIL", "UNKNOWN", "CRASH")) != arm["episode_count"]:
            raise ValueError("E1 outcome counts do not reconcile to the episode denominator")
    contrast = e1.get("primary_contrast", {})
    if (contrast.get("paired_task_count") != 9 or len(contrast.get("per_task", [])) != 9
            or contrast.get("estimate") is None or not contrast.get("bootstrap_95")):
        raise ValueError("final paper requires all nine paired E1 task contrasts and bootstrap interval")
    first = studies.get("E1_first_candidate")
    if (not isinstance(first, dict) or first.get("status") != "ANALYZED"
            or first.get("role") != "SECONDARY_EXPLORATORY"
            or first.get("row_count") != E1_PLANNED or first.get("task_count") != 9):
        raise ValueError("final paper requires all 54 first-candidate E1 records and nine task matches")
    identity = first.get("identity", {})
    if any(identity.get(field) != expected for field, expected in (
        ("source_commit", E1_SOURCE_COMMIT),
        ("source_manifest_sha256", E1_SOURCE_MANIFEST_SHA256),
        ("primary_protocol_sha256", E1_PROTOCOL_SHA256),
        ("amendment_sha256", E1_AMENDMENT_SHA256),
    )):
        raise ValueError("final paper first-candidate results do not match the pinned E1 source amendment")
    for field in (
        "primary_cohort_sha256", "first_cohort_sha256", "assessment_script_sha256",
        "primary_results_seal_sha256",
    ):
        digest = identity.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError(f"final paper first-candidate results lack a valid {field}")
    if set(first.get("arms", {})) != {"A0", "A1", "A2"}:
        raise ValueError("final paper first-candidate results require all three E1 arms")
    if any(first["arms"][arm].get("episode_count") != 18
           or first["arms"][arm].get("task_count") != 9
           or first["arms"][arm].get("task_cluster_bootstrap_95") is None
           for arm in ("A0", "A1", "A2")):
        raise ValueError("final paper first-candidate results require 18 episodes across nine tasks per arm")
    if (first.get("all_arms", {}).get("episode_count") != 54
            or first.get("all_arms", {}).get("task_count") != 9
            or first.get("all_arms", {}).get("task_cluster_bootstrap_95") is None):
        raise ValueError("final paper first-candidate overall contrast is incomplete")

    if e2.get("status") != "ANALYZED" or e2.get("row_count") != 2 * sum(E2_EXPECTED.values()):
        raise ValueError("final paper requires all planned E2 development, qualification, and holdout rows")
    cohorts = e2.get("cohorts", [])
    by_key = {(row.get("phase"), row.get("control_profile"), row.get("design_id")): row for row in cohorts}
    if len(by_key) != 12:
        raise ValueError("final paper requires all 12 E2 phase/profile/geometry cohorts")
    for (phase, profile), expected in E2_EXPECTED.items():
        for design in ("original", "historical_v4_lead_in"):
            row = by_key.get((phase, profile, design))
            if row is None or row.get("observed_count") != expected:
                raise ValueError(f"final paper requires {expected} E2 rows for {phase}/{profile}/{design}")
            counts = row.get("outcome_counts", {})
            if (sum(counts.get(key, 0) for key in ("PASS", "FAIL", "UNKNOWN", "CRASH")) != expected
                    or row.get("missing_count") not in (None, 0)):
                raise ValueError(f"E2 status counts do not reconcile for {phase}/{profile}/{design}")
    pairs = {(row.get("phase"), row.get("control_profile")): row for row in e2.get("paired_comparisons", [])}
    for (phase, profile), count in E2_EXPECTED.items():
        if pairs.get((phase, profile), {}).get("paired_count") != count:
            raise ValueError(f"final paper requires {count} paired E2 {phase}/{profile} trials")
    if not any(row.get("phase") == "holdout" and row.get("paired_count") == 100
               for row in e2.get("geometry_by_control_interaction", [])):
        raise ValueError("final paper requires all 100 complete E2 held-out four-way pairs")

    scenarios = e3.get("scenarios", [])
    if (e3.get("status") != "ANALYZED" or e3.get("primary_status") != "COMPLETE"
            or e3.get("scenario_count") != 6 or len(scenarios) != 6
            or any(row.get("primary_status") != "COMPLETE" for row in scenarios)):
        raise ValueError("final paper requires complete primary E3 results for all six scenarios")
    if {row.get("scenario_id") for row in scenarios} != {f"D{index}" for index in range(1, 7)}:
        raise ValueError("final paper requires the prespecified E3 scenarios D1 through D6")
    planned = {row.get("expected_quartet_count") for row in scenarios}
    if len(planned) != 1 or next(iter(planned)) not in (30, 100):
        raise ValueError("E3 scenarios must share the frozen 30- or 100-replication plan")
    for scenario in scenarios:
        if scenario.get("quartet_count") != scenario.get("expected_quartet_count"):
            raise ValueError(f"incomplete E3 quartets for {scenario.get('scenario_id')}")
        if (scenario.get("missing_planned_quartet_count") != 0
                or scenario.get("invalid_or_incomplete_quartet_count") != 0):
            raise ValueError(f"E3 invalid or missing quartets for {scenario.get('scenario_id')}")
        for metric in ("good_units", "flow_time_s", "horizon_s"):
            contrasts = scenario.get("metrics", {}).get(metric, {})
            if set(contrasts) != set(E3_CONTRASTS):
                raise ValueError(f"missing primary E3 contrasts for {metric}/{scenario.get('scenario_id')}")
            for name in E3_CONTRASTS:
                result = contrasts[name]
                interval = result.get("conditional_normal_95")
                if (result.get("estimate") is None or not isinstance(interval, list) or len(interval) != 2
                        or any(isinstance(bound, bool) or not isinstance(bound, (int, float))
                               or not math.isfinite(bound) for bound in interval)):
                    raise ValueError(
                        f"missing finite primary E3 estimate or conditional 95% interval "
                        f"for {metric}/{name}/{scenario.get('scenario_id')}"
                    )
    return e1, e2, e3


def _require_refinement(studies: dict[str, Any]) -> dict[str, Any]:
    refinement = studies.get("E2_refinement")
    if not isinstance(refinement, dict) or refinement.get("schema") != "forge.v6.refinement-summary/1":
        raise ValueError("final paper requires the reanalyzed E2 refinement summary")
    trial_count = refinement.get("trial_count")
    if refinement.get("status") != "ANALYZED" or isinstance(trial_count, bool) or not isinstance(trial_count, int):
        raise ValueError("final paper requires nonempty analyzed E2 refinement trials")
    if trial_count <= 0 or refinement.get("interval") is not None:
        raise ValueError("E2 refinement must have observed trials and no IID interval")
    if (refinement.get("verified_selection_receipt") is not True
            or refinement.get("selection_receipt_sha256") != PINNED_REFINEMENT_SELECTION_SHA256
            or refinement.get("interpretation_receipt_sha256") != PINNED_REFINEMENT_INTERPRETATION_SHA256
            or refinement.get("source_manifest_sha256") != PINNED_REFINEMENT_SOURCE_MANIFEST_SHA256):
        raise ValueError("final paper requires the independently pinned refinement-selection receipt")
    expected_count = refinement.get("expected_case_count")
    expected_cases = refinement.get("expected_cases")
    if (refinement.get("selection_complete") is not True
            or isinstance(expected_count, bool) or not isinstance(expected_count, int)
            or expected_count <= 0 or expected_count != trial_count
            or not isinstance(expected_cases, list) or len(expected_cases) != expected_count):
        raise ValueError("final paper requires a complete observed match to the frozen refinement selection")
    expected_keys = set()
    for case in expected_cases:
        if (not isinstance(case, dict) or set(case) != {"profile", "design_id", "disturbance_id"}
                or any(not isinstance(case[field], str) or not case[field]
                       for field in ("profile", "design_id", "disturbance_id"))):
            raise ValueError("E2 refinement contains an invalid expected selection identity")
        key = case["profile"], case["design_id"], case["disturbance_id"]
        if key in expected_keys:
            raise ValueError("E2 refinement contains a duplicate expected selection identity")
        expected_keys.add(key)
    if len(expected_keys) != expected_count:
        raise ValueError("E2 refinement expected identities do not match the frozen denominator")
    frozen_keys = {(profile, design, str(seed))
                   for profile, seeds in FROZEN_REFINEMENT_SEEDS.items()
                   for design in ("original", "historical_v4_lead_in") for seed in seeds}
    if expected_keys != frozen_keys or trial_count != len(frozen_keys):
        raise ValueError("final paper requires the pinned 26-case refinement selection")
    for field in ("interpretation_receipt_sha256", "selection_receipt_sha256"):
        digest = refinement.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - SHA256_HEX:
            raise ValueError(f"E2 refinement lacks a valid {field}")
    selection_note = refinement.get("selection_note")
    interval_note = refinement.get("interval_note")
    if (not isinstance(selection_note, str) or "selected" not in selection_note.lower()
            or "not a replacement" not in selection_note.lower()
            or not isinstance(interval_note, str) or "no iid confidence interval" not in interval_note.lower()):
        raise ValueError("E2 refinement must retain its selected-subset and no-IID interpretation notes")
    hashes = refinement.get("source_file_hashes")
    required_sources = {
        "operation_trials.csv", "refinement_manifest.json", "refinement_trials.csv",
        "refinement_selection.json",
    }
    if not isinstance(hashes, dict) or not required_sources.issubset(hashes):
        raise ValueError("E2 refinement lacks source hashes for both canonical and refinement CSVs")
    for name in required_sources:
        digest = hashes[name]
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - SHA256_HEX:
            raise ValueError(f"E2 refinement has an invalid {name} source hash")
    if hashes["refinement_selection.json"] != PINNED_REFINEMENT_SELECTION_SHA256:
        raise ValueError("final paper refinement summary is not bound to the pinned selection bytes")
    counts = refinement.get("status_counts")
    if not isinstance(counts, dict):
        raise ValueError("E2 refinement lacks status counts")  # noqa: TRY004 -- persisted result contract
    for phase in ("original", "refined"):
        phase_counts = counts.get(phase)
        if not isinstance(phase_counts, dict) or set(phase_counts) != set(REFINEMENT_STATUSES):
            raise ValueError(f"E2 refinement has incomplete {phase} status counts")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in phase_counts.values()):
            raise ValueError(f"E2 refinement has invalid {phase} status counts")
        if sum(phase_counts.values()) != trial_count:
            raise ValueError(f"E2 refinement {phase} counts do not match the trial denominator")
    flips = refinement.get("flips")
    flip_count = refinement.get("flip_count")
    if (not isinstance(flips, list) or isinstance(flip_count, bool)
            or not isinstance(flip_count, int) or flip_count != len(flips)):
        raise ValueError("E2 refinement flip count does not match its flip identities")
    flip_keys = set()
    for row in flips:
        if (not isinstance(row, dict) or any(not isinstance(row.get(field), str) or not row[field]
                for field in ("profile", "design_id", "disturbance_id"))
                or row.get("original_status") not in REFINEMENT_STATUSES
                or row.get("refined_status") not in REFINEMENT_STATUSES
                or row["original_status"] == row["refined_status"]):
            raise ValueError("E2 refinement contains an invalid decision-flip identity")
        key = row["profile"], row["design_id"], row["disturbance_id"]
        if key not in expected_keys:
            raise ValueError("E2 refinement flip identity is outside the frozen expected selection")
        if key in flip_keys:
            raise ValueError("E2 refinement contains a duplicate decision-flip identity")
        flip_keys.add(key)
    if not isinstance(refinement.get("profiles"), dict):
        raise ValueError("E2 refinement lacks per-profile summaries")  # noqa: TRY004 -- persisted result contract
    profiles = refinement["profiles"]
    for values in profiles.values():
        if (not isinstance(values, dict) or isinstance(values.get("trial_count"), bool)
                or not isinstance(values.get("trial_count"), int) or values["trial_count"] < 0
                or isinstance(values.get("flip_count"), bool)
                or not isinstance(values.get("flip_count"), int) or values["flip_count"] < 0):
            raise ValueError("E2 refinement has invalid per-profile counts")
        for field in ("original_status_counts", "refined_status_counts"):
            profile_counts = values.get(field)
            if (not isinstance(profile_counts, dict) or set(profile_counts) != set(REFINEMENT_STATUSES)
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                           for value in profile_counts.values())
                    or sum(profile_counts.values()) != values["trial_count"]):
                raise ValueError("E2 refinement has invalid per-profile status counts")
    if (sum(row["trial_count"] for row in profiles.values()) != trial_count
            or sum(row["flip_count"] for row in profiles.values()) != flip_count):
        raise ValueError("E2 refinement profile counts do not match the total denominator")
    facets48 = refinement.get("exploratory_facets48")
    facets_hash = hashes.get("refinement_facets48.csv")
    if (facets48 is None) != (facets_hash is None):
        raise ValueError("E2 48-facet summary and manifest source hash must be admitted together")
    if facets48 is not None:
        if not isinstance(facets48, dict):
            raise ValueError("E2 48-facet summary must be an object")
        if (facets48.get("status") != "ANALYZED"
                or facets48.get("selection_complete") is not True
                or facets48.get("expected_case_count") != expected_count
                or facets48.get("trial_count") != expected_count
                or facets48.get("expected_cases") != expected_cases
                or not isinstance(facets48.get("role"), str)
                or "exploratory" not in facets48["role"].lower()
                or "excluded from primary" not in facets48["role"].lower()):
            raise ValueError("E2 48-facet records must complete the pinned 26-case exploratory selection")
        if not isinstance(facets_hash, str) or len(facets_hash) != 64 or set(facets_hash) - SHA256_HEX:
            raise ValueError("E2 48-facet records lack a valid manifest source hash")
        facet_counts = facets48.get("status_counts")
        if not isinstance(facet_counts, dict):
            raise ValueError("E2 48-facet summary lacks status counts")
        for phase in ("original", "refined"):
            phase_counts = facet_counts.get(phase)
            if (not isinstance(phase_counts, dict) or set(phase_counts) != set(REFINEMENT_STATUSES)
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                           for value in phase_counts.values())
                    or sum(phase_counts.values()) != expected_count):
                raise ValueError(f"E2 48-facet {phase} counts do not match the pinned denominator")
        cavity_counts = facets48.get("cavity_witness_status_counts")
        if (not isinstance(cavity_counts, dict) or set(cavity_counts) != set(REFINEMENT_STATUSES)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in cavity_counts.values())
                or sum(cavity_counts.values()) != expected_count):
            raise ValueError("E2 48-facet cavity-witness counts do not match the pinned denominator")
        facet_flips = facets48.get("flips")
        facet_flip_count = facets48.get("flip_count")
        if (not isinstance(facet_flips, list) or isinstance(facet_flip_count, bool)
                or not isinstance(facet_flip_count, int) or facet_flip_count != len(facet_flips)):
            raise ValueError("E2 48-facet flip count does not match its identities")
        facet_flip_keys = set()
        for row in facet_flips:
            if (not isinstance(row, dict)
                    or any(not isinstance(row.get(field), str) or not row[field]
                           for field in ("profile", "design_id", "disturbance_id"))
                    or row.get("original_status") not in REFINEMENT_STATUSES
                    or row.get("refined_status") not in REFINEMENT_STATUSES
                    or row["original_status"] == row["refined_status"]):
                raise ValueError("E2 48-facet summary contains an invalid decision-flip identity")
            key = row["profile"], row["design_id"], row["disturbance_id"]
            if key not in expected_keys or key in facet_flip_keys:
                raise ValueError("E2 48-facet decision flip is outside or duplicated in the pinned selection")
            facet_flip_keys.add(key)
        facet_profiles = facets48.get("profiles")
        if (not isinstance(facet_profiles, dict) or set(facet_profiles) != {"reference", "historical_fast"}
                or any(not isinstance(values, dict)
                       or isinstance(values.get("trial_count"), bool)
                       or not isinstance(values.get("trial_count"), int)
                       or isinstance(values.get("flip_count"), bool)
                       or not isinstance(values.get("flip_count"), int)
                       or values["trial_count"] < 0 or values["flip_count"] < 0
                       for values in facet_profiles.values())
                or sum(values["trial_count"] for values in facet_profiles.values()) != expected_count
                or facet_flip_count != sum(values["flip_count"] for values in facet_profiles.values())):
            raise ValueError("E2 48-facet per-profile flip counts do not reconcile")
        comparison = facets48.get("comparison_to_refinement24")
        if (not isinstance(comparison, dict)
                or comparison.get("matched_case_count") != expected_count
                or not isinstance(comparison.get("changed_rows"), list)
                or comparison.get("changed_case_count") != len(comparison["changed_rows"])):
            raise ValueError("E2 48-vs-24 comparison does not reconcile to the pinned denominator")
        comparison_keys = set()
        for row in comparison["changed_rows"]:
            if (not isinstance(row, dict)
                    or set(row) != {"profile", "design_id", "disturbance_id",
                                    "refinement24_status", "facets48_status"}
                    or any(not isinstance(row.get(field), str) or not row[field]
                           for field in ("profile", "design_id", "disturbance_id"))
                    or row.get("refinement24_status") not in REFINEMENT_STATUSES
                    or row.get("facets48_status") not in REFINEMENT_STATUSES
                    or row["refinement24_status"] == row["facets48_status"]):
                raise ValueError("E2 48-vs-24 comparison contains an invalid changed row")
            key = row["profile"], row["design_id"], row["disturbance_id"]
            if key not in expected_keys or key in comparison_keys:
                raise ValueError("E2 48-vs-24 changed row is outside or duplicated in the pinned selection")
            comparison_keys.add(key)
    return refinement


def _bar_chart(items: list[tuple[str, float]], *, centered: bool = False) -> str:
    """Create a compact vector chart from summary values without chart dependencies."""
    height = len(items) + 1
    max_abs = max((abs(value) for _, value in items), default=0.0)
    scale = (4.0 / (max_abs or 1.0)) if centered else 10.0
    lines = [r"\begin{tikzpicture}[x=0.55cm,y=0.30cm]", rf"\draw[black!30] (0,0) -- (0,{height});"]
    if centered:
        lines.append(r"\draw[black!30] (-4.2,0) -- (-4.2," + str(height) + ");")
        lines.append(r"\draw[black!30] (-4,0.35) -- (4,0.35);")
        lines.append(r"\node[font=\scriptsize] at (-4,0.05) {" + _number(-4 / scale) + "};")
        lines.append(r"\node[font=\scriptsize] at (0,0.05) {0};")
        lines.append(r"\node[font=\scriptsize] at (4,0.05) {" + _number(4 / scale) + "};")
    else:
        lines.append(r"\draw[black!30] (0,0.35) -- (10,0.35);")
        for tick in (0, 5, 10):
            lines.append(r"\node[font=\scriptsize] at (" + str(tick) + ",0.05) {" + _number(tick / 10) + "};")
    for index, (label, value) in enumerate(items):
        y = len(items) - index
        x = value * scale
        left, right = (min(0.0, x), max(0.0, x)) if centered else (0.0, x)
        lines.append(f"\\fill[blue!55] ({left:.4f},{y - .16}) rectangle ({right:.4f},{y + .16});")
        label_x = -4.35 if centered else -0.15
        lines.append(r"\node[anchor=east,font=\scriptsize] at (" + f"{label_x:.2f},{y}" + ") {" + _escape(label) + "};")
        value_x = right + .12 if value >= 0 else left - .12
        anchor = "west" if value >= 0 else "east"
        lines.append(r"\node[anchor=" + anchor + r",font=\scriptsize] at (" + f"{value_x:.3f},{y}" + ") {" + _number(value) + "};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _e1_family_outcome_png(episodes: list[dict[str, str]]) -> bytes:
    outcomes = ("PASS", "FAIL", "UNKNOWN", "CRASH")
    colors = {"PASS": "#39866a", "FAIL": "#b94747",
              "UNKNOWN": "#e4b84d", "CRASH": "#687586"}
    families = sorted({row["family"] for row in episodes})
    arms = sorted({row["method"] for row in episodes})
    counts = Counter((row["family"], row["method"], row["outcome"]) for row in episodes)
    groups = [(family, arm) for family in families for arm in arms]
    totals = [sum(counts[(family, arm, outcome)] for outcome in outcomes) for family, arm in groups]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor="white")
    fig.subplots_adjust(left=0.22, right=0.91, top=0.84, bottom=0.24)
    positions = list(range(len(groups)))
    lefts = [0] * len(groups)
    for outcome in outcomes:
        widths = [counts[(family, arm, outcome)] for family, arm in groups]
        bars = ax.barh(positions, widths, left=lefts, height=0.66, color=colors[outcome],
                       edgecolor="white", linewidth=0.8, label=outcome)
        text_color = "#263238" if outcome == "UNKNOWN" else "white"
        for bar, count in zip(bars, widths):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        str(count), ha="center", va="center", color=text_color,
                        fontsize=20, fontweight="bold")
        lefts = [left + width for left, width in zip(lefts, widths)]

    for position, (total, (family, arm)) in enumerate(zip(totals, groups)):
        ax.text(total + 0.12, position, f"n={total}", va="center", fontsize=20, color="#334155")
    ax.set_yticks(positions, [f"{family.replace('_', ' ')}  ·  {arm}" for family, arm in groups],
                  fontsize=20)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals, default=0) + 1.5)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("Episode count", fontsize=22)
    ax.set_title(f"E1 · Episode outcomes by family and feedback arm (N={len(episodes)})",
                 fontsize=26, pad=24)
    ax.grid(axis="x", color="#d9e1e8", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=20)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return buffer.getvalue()


def _write_paper_figures(summary: dict[str, Any], figure_dir: Path,
                         episodes: list[dict[str, str]]) -> Path:
    e2 = summary["studies"]["E2"]
    e3 = summary["studies"]["E3"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    e2_pairs = {(row["phase"], row["control_profile"]): row for row in e2["paired_comparisons"]}
    e2_phases = (("development", "DEV"), ("qualification", "QUAL"), ("holdout", "HOLDOUT"))
    e2_profiles = (("reference", "REF · primary", "#236b8e", "o"),
                   ("historical_fast", "FAST · secondary", "#d17a22", "D"))
    e2_rows = [(phase, phase_label, profile, legend, color, marker,
                int(e2_pairs[(phase, profile)]["paired_count"]),
                100 * float(e2_pairs[(phase, profile)]["success_rate_delta_original_minus_lead_in"]))
               for phase, phase_label in e2_phases
               for profile, legend, color, marker in e2_profiles]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor="white")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.86, bottom=0.30)
    positions = list(range(len(e2_rows)))
    for profile, legend, color, marker in e2_profiles:
        selected = [(position, row[7]) for position, row in enumerate(e2_rows) if row[2] == profile]
        ax.scatter([value for _, value in selected], [position for position, _ in selected],
                   s=115, color=color, marker=marker, edgecolor="white", linewidth=0.8,
                   label=legend, zorder=3)
    ax.axvline(0, color="#334155", linewidth=1.2, zorder=1)
    ax.set_yticks(positions, [f"{phase_label}  ·  {'REF' if profile == 'reference' else 'FAST'}  ·  n={count}"
                              for _, phase_label, profile, _, _, _, count, _ in e2_rows], fontsize=20)
    ax.invert_yaxis()
    largest_delta = max((abs(row[7]) for row in e2_rows), default=0.0)
    extent = max(1.0, largest_delta * 1.3)
    ax.set_xlim(-extent, extent)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("Paired pass-rate difference (pp; original − lead-in)", fontsize=22)
    ax.set_title("E2 · Paired pass-rate differences by phase and control profile", fontsize=26, pad=24)
    ax.grid(axis="x", color="#d9e1e8", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=20)
    fig.text(0.09, 0.035, "Cohort-specific paired estimates; paired intervals are not reported.",
             fontsize=20, color="#475569")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    payloads = [("e2_pairs", "e2_pairs.png", buffer.getvalue())]

    scenarios = sorted(e3["scenarios"], key=lambda item: item["scenario_id"])
    e3_values = [scenario["metrics"]["good_units"] for scenario in scenarios]
    units = e3_values[0][E3_CONTRASTS[0]]["units"]
    labels = {"mechanical": "Mechanical", "schedule": "Schedule",
              "combined": "Combined", "interaction": "Interaction"}
    colors = {"mechanical": "#236b8e", "schedule": "#d17a22",
              "combined": "#39866a", "interaction": "#8762a5"}
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.31)
    scenario_positions = list(range(len(scenarios)))
    width = 0.18
    for index, contrast in enumerate(E3_CONTRASTS):
        estimates = [float(values[contrast]["estimate"]) for values in e3_values]
        intervals = [values[contrast]["conditional_normal_95"] for values in e3_values]
        lower = [max(0.0, estimate - float(interval[0]))
                 for estimate, interval in zip(estimates, intervals)]
        upper = [max(0.0, float(interval[1]) - estimate)
                 for estimate, interval in zip(estimates, intervals)]
        offset = (index - (len(E3_CONTRASTS) - 1) / 2) * width
        ax.bar([position + offset for position in scenario_positions], estimates, width,
               yerr=[lower, upper], capsize=4, color=colors[contrast], edgecolor="white",
               linewidth=0.7, label=labels[contrast], error_kw={"elinewidth": 1.2, "capthick": 1.2})
    ax.axhline(0, color="#334155", linewidth=1.2)
    ax.set_xticks(scenario_positions,
                  [f'{scenario["scenario_id"]}\n(n={scenario["quartet_count"]})' for scenario in scenarios],
                  fontsize=20)
    ax.set_xlabel("Fixed scenario", fontsize=22)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel(f"Good-unit contrast ({units})", fontsize=22)
    ax.set_title("E3 · Good-unit factorial contrasts by fixed scenario", fontsize=26, pad=24)
    ax.grid(axis="y", color="#d9e1e8", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=20)
    fig.text(0.09, 0.035, "Error bars show conditional normal 95% intervals.", fontsize=20, color="#475569")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    payloads.append(("e3_scenarios", "e3_scenarios.png", buffer.getvalue()))
    payloads.append(("e1_family_outcomes", "e1_family_outcomes.png", _e1_family_outcome_png(episodes)))

    for ident, filename, payload in payloads:
        destination = figure_dir / filename
        fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", dir=figure_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        figures.append({"id": ident, "file": filename,
                        "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {"schema": "forge.v6.paper-figures/1", "figures": figures}
    manifest_path = figure_dir / "paper-figures.json"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".paper-figures.", dir=figure_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest_path


def render(summary: dict[str, Any]) -> str:
    if summary.get("schema_version") != 1 or summary.get("source_manifest") != "manifest.json":
        raise ValueError("input must be a schema-version-1 standalone analysis summary")
    studies = summary.get("studies")
    if not isinstance(studies, dict) or "E0" not in studies:
        raise ValueError("analysis summary lacks required E0 results")
    e0_macros, e0_rows, e0_geometry_rows = _e0_summary(studies["E0"])
    e1, e2, e3 = _require_complete(studies)
    refinement = _require_refinement(studies)

    contrast = e1["primary_contrast"]
    first = studies["E1_first_candidate"]
    first_all = first["all_arms"]
    arm_counts = e1["arm_counts"]
    selected_count = sum(row.get("selected_artifact_count", 0) for row in arm_counts.values())
    count_words = []
    for arm in ("A0", "A1", "A2"):
        row = arm_counts[arm]
        counts = row["outcome_counts"]
        count_words.append(f'{arm}: {counts["PASS"]} PASS, {counts["UNKNOWN"]} UNKNOWN, '
                           f'{counts["CRASH"]} CRASH out of {row["episode_count"]}')
    arm_signatures = {(row["episode_count"], tuple(row["outcome_counts"][key] for key in ("PASS", "UNKNOWN", "CRASH")))
                      for row in arm_counts.values()}
    if len(arm_signatures) == 1:
        count_words = [(f'Each feedback arm recorded {counts["PASS"]} PASS, {counts["UNKNOWN"]} UNKNOWN, '
                        f'and {counts["CRASH"]} CRASH out of {row["episode_count"]} episodes')]
    loop_summary = (f'{selected_count}/{e1["row_count"]} episodes produced a selected build-valid artifact; '
                    f'{first_all["reused_primary_count"]} selections were the original ordinal-one candidate.')
    if selected_count and selected_count == first_all["reused_primary_count"]:
        loop_summary += ' No completed candidate-to-candidate repair was observed.'
    resource_rows = [_table_row(["Arm", "Submitted", "Completed", "Timeouts", "Minutes", "Token records"])]
    for arm, row in sorted(e1.get("resources", {}).items()):
        resource_rows.append(_table_row([arm, row["submitted_count"], row["completed_graded_count"],
                                        row["worker_timeout_count"], _number(row["wall_minutes"], 1),
                                        f'{row["token_observed_episodes"]}/{row["episode_count"]}']))
    macros = [*e0_macros,
              _macro("E1ObservedCounts", "; ".join(count_words) + "."),
              _macro("E1ObservedLoopSummary", loop_summary),
              _macro("E1A2MinusA1", _number(contrast["estimate"])),
              _macro("E1BootstrapLower", _number(contrast["bootstrap_95"][0])),
              _macro("E1BootstrapUpper", _number(contrast["bootstrap_95"][1])),
              _macro("E1PairedTasks", 9),
              _macro("E1FirstCandidateSummary", (
                  f'Across all 54 E1 episodes, ordinal-one candidates passed '
                  f'{first_all["first_pass_count"]}/54 compared with '
                  f'{first_all["final_pass_count"]}/54 for the selected candidates; '
                  f'the mean task-clustered final-minus-first PASS-fraction difference was '
                  f'{_number(first_all["final_minus_first_pass_fraction"])} '
                  f'(descriptive 95% bootstrap interval '
                  f'{_number(first_all["task_cluster_bootstrap_95"][0])} to '
                  f'{_number(first_all["task_cluster_bootstrap_95"][1])}).'
              )),
              _macro("VRefinementTrials", refinement["trial_count"]),
              _macro("VRefinementFlips", refinement["flip_count"])]
    macros.append(r"\newcommand{\GeneratedEOneResourceTable}{%" + "\n" +
                  r"\begin{tabular}{lrrrrr}\toprule" + "\n" +
                  "\n".join(resource_rows) + "\n" + r"\bottomrule\end{tabular}}")
    facets48 = refinement.get("exploratory_facets48")
    if facets48 is None:
        facets48_summary = "No 48-facet exploratory amendment is admitted in the manifest."
        facets48_trials = facets48_flips = facets48_cavity_fails = "not admitted"
        facets48_matched = facets48_changed = "not admitted"
        facets48_flip_rows = [_table_row(["No 48-facet amendment admitted", "--", "--"])]
        facets48_comparison_rows = [_table_row(["No 48-facet amendment admitted", "--", "--"])]
    else:
        comparison = facets48["comparison_to_refinement24"]
        facets48_summary = (
            f'The post-primary 48-facet amendment includes {facets48["trial_count"]} matched cases, '
            f'{facets48["flip_count"]} status changes versus canonical original outcomes, '
            f'{comparison["matched_case_count"]} cases compared with the 24-facet refinement, '
            f'{comparison["changed_case_count"]} changed outcomes versus that refinement, and '
            f'{facets48["cavity_witness_status_counts"]["FAIL"]} local cavity-witness failures.'
        )
        facets48_trials = facets48["trial_count"]
        facets48_flips = facets48["flip_count"]
        facets48_cavity_fails = facets48["cavity_witness_status_counts"]["FAIL"]
        facets48_matched = comparison["matched_case_count"]
        facets48_changed = comparison["changed_case_count"]
        facets48_flip_rows = [
            _table_row(["/".join((row["profile"], row["design_id"], row["disturbance_id"])),
                        row["original_status"], row["refined_status"]])
            for row in sorted(facets48["flips"],
                              key=lambda item: (item["profile"], item["design_id"], item["disturbance_id"]))
        ]
        facets48_comparison_rows = [
            _table_row(["/".join((row["profile"], row["design_id"], row["disturbance_id"])),
                        row["refinement24_status"], row["facets48_status"]])
            for row in sorted(comparison["changed_rows"],
                              key=lambda item: (item["profile"], item["design_id"], item["disturbance_id"]))
        ] or [_table_row(["No changed outcomes", "--", "--"])]
    macros.extend([
        _macro("VRefinement48Summary", facets48_summary),
        _macro("VRefinement48Trials", facets48_trials),
        _macro("VRefinement48Flips", facets48_flips),
        _macro("VRefinement48CavityWitnessFails", facets48_cavity_fails),
        _macro("VRefinement48ComparedCases", facets48_matched),
        _macro("VRefinement48ChangedCases", facets48_changed),
    ])
    e1_rows, e1_task_rows, e1_first_rows = [], [], []
    for method in ("A0", "A1", "A2"):
        arm = e1["arm_counts"][method]
        counts = arm["outcome_counts"]
        e1_rows.append(_table_row([method, arm["episode_count"], counts["PASS"], counts["FAIL"],
                                   counts["UNKNOWN"], counts["CRASH"], _number(arm["pass_fraction"])]))
    for task in contrast["per_task"]:
        e1_task_rows.append(_table_row([task["task_id"], _number(task["a1_pass_fraction"]),
                                        _number(task["a2_pass_fraction"]), _number(task["a2_minus_a1"]),
                                        len(task.get("source_row_ids", []))]))
    for method in ("A0", "A1", "A2"):
        row = first["arms"][method]
        lower, upper = row["task_cluster_bootstrap_95"]
        e1_first_rows.append(_table_row([
            method, f'{row["first_pass_count"]}/{row["episode_count"]}',
            f'{row["final_pass_count"]}/{row["episode_count"]}',
            _number(row["final_minus_first_pass_fraction"]),
            f'[{_number(lower)}, {_number(upper)}]',
            row["changed_outcome_count"], row["regressed_count"],
        ]))

    cohorts = sorted(e2["cohorts"], key=lambda row: (row["phase"], row["control_profile"], row["design_id"]))
    e2_holdout_rows, e2_full_rows, e2_pair_rows = [], [], []
    for row in cohorts:
        fixture = "lead-in" if row["design_id"] == "historical_v4_lead_in" else "original"
        counts = row["outcome_counts"]
        rate = _number(row["success_rate"])
        if row["phase"] == "holdout":
            e2_holdout_rows.append(_table_row([row["control_profile"], fixture,
                                               f'{counts["PASS"]}/{row["denominator"]}', rate]))
        e2_full_rows.append(_table_row([row["phase"], row["control_profile"], fixture,
                                        row["observed_count"],
                                        f'{counts["PASS"]}/{counts["FAIL"]}/{counts["UNKNOWN"]}/{counts["CRASH"]}',
                                        row["missing_count"] if row["missing_count"] is not None else 0, rate]))
    for row in e2["paired_comparisons"]:
        e2_pair_rows.append(_table_row([row["phase"], row["control_profile"], row["paired_count"],
                                        _number(row["success_rate_delta_original_minus_lead_in"])]))
    interaction = next(row for row in e2["geometry_by_control_interaction"] if row["phase"] == "holdout")
    e2_pair_rows.append(_table_row(["holdout", "geometry x profile", interaction["paired_count"],
                                    _number(interaction["estimate"])]))
    ref_holdout = next(row for row in e2["paired_comparisons"]
                       if row["phase"] == "holdout" and row["control_profile"] == "reference")
    macros.append(_macro("E2ReferenceHoldoutDifference",
                         _number(ref_holdout["success_rate_delta_original_minus_lead_in"])))

    refinement_profile_rows = []
    refinement_case_rows = []
    refinement_flip_rows = []
    for profile, values in sorted(refinement["profiles"].items()):
        refinement_profile_rows.append(_table_row([
            profile, values["trial_count"], values["flip_count"],
            "/".join(str(values["original_status_counts"][status]) for status in REFINEMENT_STATUSES),
            "/".join(str(values["refined_status_counts"][status]) for status in REFINEMENT_STATUSES),
        ]))
    for row in sorted(refinement["flips"], key=lambda item: (item["profile"], item["design_id"], item["disturbance_id"])):
        identity = "/".join((row["profile"], row["design_id"], row["disturbance_id"]))
        rendered = _table_row([identity, row["original_status"], row["refined_status"]])
        refinement_case_rows.append(rendered)
        refinement_flip_rows.append(rendered)

    scenarios = sorted(e3["scenarios"], key=lambda row: row["scenario_id"])
    e3_scenario_rows, e3_metric_rows = [], []
    for scenario in scenarios:
        e3_scenario_rows.append(_table_row([scenario["scenario_id"], scenario["quartet_count"],
                                            scenario["expected_quartet_count"],
                                            scenario["invalid_or_incomplete_quartet_count"]]))
        for metric, metric_label in E3_METRIC_LABELS.items():
            values = scenario["metrics"][metric]
            for name, contrast_label in E3_CONTRAST_LABELS.items():
                value = values[name]
                lower, upper = value["conditional_normal_95"]
                interval = f'[{_number(lower)}, {_number(upper)}]'
                e3_metric_rows.append(_table_row([scenario["scenario_id"], metric_label, contrast_label,
                                                  _number(value["estimate"]), interval,
                                                  value["denominator"], value["units"]]))
    e3_macro = [(name, sum(row["metrics"]["good_units"][name]["estimate"] for row in scenarios) / len(scenarios))
                for name in E3_CONTRASTS]
    macros.extend(_macro("E3SixScenario" + name.title(), _number(value)) for name, value in e3_macro)

    e0_plot = []
    for label, truth, checker, scope in (
        ("axis legacy", "invalid", "legacy", "axis"), ("axis witness", "invalid", "new", "axis"),
        ("web legacy", "invalid", "legacy", "local_wall"), ("web witness", "invalid", "new", "local_wall"),
    ):
        row = _stratum(studies["E0"], truth, E0_CHECKERS[checker], E0_SCOPES[scope])
        e0_plot.append((label, row["status_counts"].get("PASS", 0) / row["case_count"]))
    e1_plot = [(row["task_id"], row["a2_minus_a1"]) for row in contrast["per_task"]]
    e2_plot = [(f'{row["control_profile"][:1]} {"L" if row["design_id"] == "historical_v4_lead_in" else "O"}',
                row["success_rate"])
               for row in cohorts if row["phase"] == "holdout"]

    lines = [
        "% Generated from a manifest-verified standalone analysis summary. Do not edit.",
        "% Final-paper gate: complete E1, E2 development/qualification/holdout, and all planned E3 quartets.",
        *macros,
        r"\newcommand{\VRefinementFlipRows}{" + " ".join(refinement_flip_rows) + "}",
        r"\newcommand{\VRefinementFourEightFlipRows}{" + " ".join(facets48_flip_rows) + "}",
        r"\newcommand{\VRefinementFourEightComparisonRows}{" + " ".join(facets48_comparison_rows) + "}",
        r"\newcommand{\GeneratedEZeroPlot}{%" + "\n" + _bar_chart(e0_plot) + "}",
        r"\newcommand{\GeneratedEOnePlot}{%" + "\n" + _bar_chart(e1_plot, centered=True) + "}",
        r"\newcommand{\GeneratedETwoPlot}{%" + "\n" + _bar_chart(e2_plot) + "}",
        r"\newcommand{\GeneratedEThreePlot}{%" + "\n" + _bar_chart(e3_macro, centered=True) + "}",
        r"\newcommand{\GeneratedEZeroTable}{%", r"\begin{tabular}{llrrrrrr}",
        r"\toprule Scope & Truth & Checker & $n$ & PASS & FAIL & UNKNOWN & ERROR \\",
        r"\midrule", *e0_rows, r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedEZeroGeometryTable}{%", r"\begin{tabular}{llrrrrr}",
        r"\toprule Scope & Truth & Checker & $n$ & PASS & FAIL & UNKNOWN \\",
        r"\midrule", *e0_geometry_rows, r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedEOneTable}{%", r"\begin{tabular}{lrrrrrr}",
        r"\toprule Arm & $n$ & PASS & FAIL & UNKNOWN & CRASH & PASS fraction \\",
        r"\midrule", *e1_rows, r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedEOneTaskTable}{%", r"\begin{tabular}{lrrrr}",
        r"\toprule Task & A1 PASS & A2 PASS & A2$-$A1 & Source rows \\", r"\midrule", *e1_task_rows,
        r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedEOneFirstCandidateTable}{%", r"\begin{tabular}{lrrrrrrr}",
        r"\toprule Arm & First PASS & Final PASS & Final$-$first & Task-cluster bootstrap 95\% & Changed & Regressed \\",
        r"\midrule", *e1_first_rows, r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoTable}{%", r"\begin{tabular}{llrr}",
        r"\toprule Profile & Fixture & PASS / planned & Rate \\", r"\midrule", *e2_holdout_rows,
        r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoFullTable}{%", r"\begin{tabular}{lllrlrr}",
        r"\toprule Phase & Profile & Fixture & Seen & P/F/U/C & Missing & Rate \\",
        r"\midrule", *e2_full_rows, r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoPairTable}{%", r"\begin{tabular}{llrr}",
        r"\toprule Phase & Profile & Pairs & Original$-$lead-in \\", r"\midrule", *e2_pair_rows,
        r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoRefinementProfileTable}{%", r"\begin{tabular}{lrrll}",
        r"\toprule Profile & Trials & Flips & Original P/F/U/C & Refined P/F/U/C \\",
        r"\midrule", *refinement_profile_rows, r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoRefinementFlipTable}{%", r"\begin{tabular}{lll}",
        r"\toprule Case identity & Original & Refined \\", r"\midrule", *refinement_case_rows,
        r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoRefinementFourEightFlipTable}{%", r"\begin{tabular}{lll}",
        r"\toprule Case identity & Canonical original & 48-facet refined \\", r"\midrule",
        r"\VRefinementFourEightFlipRows", r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedETwoRefinementFourEightComparisonTable}{%", r"\begin{tabular}{lll}",
        r"\toprule Case identity & 24-facet refined & 48-facet refined \\", r"\midrule",
        r"\VRefinementFourEightComparisonRows", r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedEThreeTable}{%", r"\begin{tabular}{lrrr}",
        r"\toprule Scenario & Quartets & Planned & Invalid \\", r"\midrule", *e3_scenario_rows,
        r"\bottomrule\end{tabular}}",
        r"\newcommand{\GeneratedEThreeMetricsTable}{%", r"\small",
        r"\begin{longtable}{@{}lllrlrl@{}}",
        r"\caption{E3 primary contrasts with conditional paired-seed normal 95\% intervals.}\label{tab:supp-e3-estimates}\\",
        r"\toprule Scenario & Metric & Contrast & Estimate & Conditional 95\% interval & $n$ & Units \\",
        r"\midrule", r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename~\thetable\ (continued)}\\",
        r"\toprule Scenario & Metric & Contrast & Estimate & Conditional 95\% interval & $n$ & Units \\",
        r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{7}{r}{Continued on next page}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot", *e3_metric_rows,
        r"\end{longtable}\normalsize}", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path, help="summary.json emitted by the standalone analyzer")
    parser.add_argument("output_tex", type=Path, help="generated TeX destination, normally build/results.generated.tex")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="manifest-verified data directory used to independently recompute this summary")
    args = parser.parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.summarize import build_summary
    from analysis.verify import load_artifact

    artifact = load_artifact(args.artifact_dir)
    actual = build_summary(artifact)
    if json.dumps(summary, sort_keys=True, allow_nan=False) != json.dumps(actual, sort_keys=True, allow_nan=False):
        raise ValueError("paper summary differs from independent reanalysis of the verified data")
    output = render(summary)
    _write_paper_figures(summary, args.output_tex.parent.parent / "figures",
                         artifact["datasets"]["episodes.csv"])
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output_tex.name}.", dir=args.output_tex.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output_tex)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
