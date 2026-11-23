"""Standalone known-answer tests for receipt-anchored refinement reanalysis."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.refinement import (
    PINNED_INTERPRETATION_SHA256,
    PINNED_SELECTION_SHA256,
    PINNED_SOURCE_MANIFEST_SHA256,
    RefinementError,
    analyze_refinement,
)
from paper.generate_results import _require_refinement

CANONICAL_FIELDS = (
    "phase", "control_profile", "design_id", "disturbance_id", "outcome",
    "legacy_success", "attempt_duration_s", "model_sha256", "protocol_sha256",
)
REFINEMENT_FIELDS = (
    "profile", "design_id", "disturbance_id", "original_status", "refined_status",
    "original_dt_s", "refined_dt_s", "original_solver_iterations",
    "refined_solver_iterations", "cavity_witness_status",
)


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "sha256": _digest(path), "bytes": path.stat().st_size}


def _canonical(profile: str, design: str, seed: int, outcome: str) -> dict[str, str]:
    return {
        "phase": "holdout", "control_profile": profile, "design_id": design,
        "disturbance_id": str(seed), "outcome": outcome, "legacy_success": "unknown",
        "attempt_duration_s": "1.0", "model_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
    }


def _trial(profile: str, design: str, seed: int, original: str, refined: str) -> dict[str, str]:
    return {
        "profile": profile, "design_id": design, "disturbance_id": str(seed),
        "original_status": original, "refined_status": refined, "original_dt_s": "0.002",
        "refined_dt_s": "0.001", "original_solver_iterations": "50",
        "refined_solver_iterations": "100", "cavity_witness_status": "PASS",
    }


def _identity(row: dict[str, str]) -> dict[str, str]:
    return {field: row[field] for field in ("profile", "design_id", "disturbance_id")}


class RefinementAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.records = self.data_dir / "records"
        self.canonical_path = self.data_dir / "operation_trials.csv"
        self.trials_path = self.records / "refinement_trials.csv"
        self.selection_path = self.records / "refinement_selection.json"
        self.selected_seeds = {"reference": [101], "historical_fast": [202]}
        self.receipt = {
            "schema": "forge.v6-e2-refinement-selection/1",
            "created_after_primary_results": True,
            "rule_frozen_before_primary": True,
            "interpretation_receipt_sha256": PINNED_INTERPRETATION_SHA256,
            "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
            "selected_seeds": self.selected_seeds,
            "change": {"timestep_multiplier": 0.5, "solver_iteration_multiplier": 2},
            "primary_results_replaced": False,
            "prespecified_receipt": {"kind": "test fixture"},
        }
        self.selection_path.parent.mkdir(parents=True, exist_ok=True)
        self.selection_path.write_text(json.dumps(self.receipt, sort_keys=True), encoding="utf-8")
        self.pinned_patch = patch(
            "analysis.refinement.PINNED_SELECTION_SHA256", _digest(self.selection_path)
        )
        self.pinned_patch.start()

        self.expected_cases = [
            {"profile": profile, "design_id": design, "disturbance_id": str(seed)}
            for profile, seeds in self.selected_seeds.items()
            for design in ("original", "historical_v4_lead_in")
            for seed in seeds
        ]
        statuses = {
            ("reference", "original", "101"): ("FAIL", "PASS"),
            ("reference", "historical_v4_lead_in", "101"): ("FAIL", "FAIL"),
            ("historical_fast", "original", "202"): ("PASS", "PASS"),
            ("historical_fast", "historical_v4_lead_in", "202"): ("PASS", "PASS"),
        }
        self.canonical_rows = [
            _canonical(profile, design, int(seed), values[0])
            for (profile, design, seed), values in statuses.items()
        ]
        self.trial_rows = [
            _trial(profile, design, int(seed), *values)
            for (profile, design, seed), values in statuses.items()
        ]
        _write(self.canonical_path, CANONICAL_FIELDS, self.canonical_rows)
        self._seal_inputs(self.trial_rows)

    def tearDown(self) -> None:
        self.pinned_patch.stop()
        self.temp.cleanup()

    def _seal_inputs(
        self,
        primary_rows: list[dict[str, str]],
        *,
        expected_cases: list[dict[str, str]] | None = None,
        include_primary: bool = True,
        facets_rows: list[dict[str, str]] | None = None,
    ) -> None:
        _write(self.trials_path, REFINEMENT_FIELDS, primary_rows)
        files: list[dict[str, object]] = []
        if include_primary:
            files.append(_entry(self.trials_path, "records/refinement_trials.csv"))
        files.append(_entry(self.selection_path, "records/refinement_selection.json"))
        facets_path = self.records / "refinement_facets48.csv"
        if facets_rows is not None:
            _write(facets_path, REFINEMENT_FIELDS, facets_rows)
            files.append(_entry(facets_path, "records/refinement_facets48.csv"))
        selection = expected_cases if expected_cases is not None else self.expected_cases
        nested = {
            "schema": "forge.v6.refinement-inputs/1",
            "files": files,
            "primary_selection": {
                "expected_cases": selection,
                "interpretation_receipt_sha256": PINNED_INTERPRETATION_SHA256,
                "selection_receipt_sha256": _digest(self.selection_path),
            },
        }
        nested_path = self.data_dir / "refinement_manifest.json"
        nested_path.write_text(json.dumps(nested, sort_keys=True), encoding="utf-8")
        outer = {
            "schema_version": 1,
            "files": [
                _entry(self.canonical_path, "operation_trials.csv"),
                _entry(nested_path, "refinement_manifest.json"),
            ],
        }
        (self.data_dir / "manifest.json").write_text(json.dumps(outer, sort_keys=True), encoding="utf-8")

    def _rewrite_nested_files(self, files: list[dict[str, object]]) -> None:
        nested_path = self.data_dir / "refinement_manifest.json"
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        nested["files"] = files
        nested_path.write_text(json.dumps(nested, sort_keys=True), encoding="utf-8")
        self._reseal_outer_manifest()

    def _reseal_outer_manifest(self) -> None:
        nested_path = self.data_dir / "refinement_manifest.json"
        outer = {
            "schema_version": 1,
            "files": [
                _entry(self.canonical_path, "operation_trials.csv"),
                _entry(nested_path, "refinement_manifest.json"),
            ],
        }
        (self.data_dir / "manifest.json").write_text(json.dumps(outer, sort_keys=True), encoding="utf-8")

    def test_recomputes_counts_from_sealed_independently_anchored_selection(self) -> None:
        result = analyze_refinement(self.root)
        self.assertEqual(result["trial_count"], 4)
        self.assertEqual(result["flip_count"], 1)
        self.assertEqual(result["status_counts"]["original"]["FAIL"], 2)
        self.assertEqual(result["status_counts"]["refined"]["PASS"], 3)
        self.assertTrue(result["selection_complete"])
        self.assertEqual(result["expected_case_count"], 4)
        self.assertTrue(result["verified_selection_receipt"])
        self.assertEqual(result["selection_receipt_sha256"], _digest(self.selection_path))
        self.assertEqual(result["interpretation_receipt_sha256"], PINNED_INTERPRETATION_SHA256)
        self.assertIsNone(result["interval"])
        self.assertNotIn(str(self.root), repr(result))

    def test_accepts_canonical_data_directory_directly(self) -> None:
        result = analyze_refinement(self.data_dir)
        self.assertEqual(result["trial_count"], 4)
        self.assertTrue(result["selection_complete"])

    def test_rejects_nested_manifest_that_omits_primary_csv(self) -> None:
        selection_entry = _entry(self.selection_path, "records/refinement_selection.json")
        self._rewrite_nested_files([selection_entry])
        with self.assertRaisesRegex(RefinementError, "must list both"):
            analyze_refinement(self.root)

    def test_rejects_unlisted_refinement_manifest_in_canonical_manifest(self) -> None:
        manifest_path = self.data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "refinement_manifest.json"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RefinementError, "must be listed in manifest.json"):
            analyze_refinement(self.root)

    def test_rejects_tampered_nested_csv_before_parsing(self) -> None:
        with self.trials_path.open("a", encoding="utf-8") as stream:
            stream.write("tampered\n")
        with self.assertRaisesRegex(RefinementError, "SHA-256 mismatch|byte count mismatch"):
            analyze_refinement(self.root)

    def test_rejects_truncated_refinement_subset(self) -> None:
        self._seal_inputs(self.trial_rows[:-1])
        with self.assertRaisesRegex(RefinementError, "keyset differs from frozen selection"):
            analyze_refinement(self.root)

    def test_rewriting_csv_and_manifest_expected_list_cannot_change_receipt_selection(self) -> None:
        altered = [dict(row) for row in self.trial_rows]
        altered[0]["disturbance_id"] = "999"
        alternate_key = {"profile": altered[0]["profile"], "design_id": altered[0]["design_id"],
                         "disturbance_id": "999"}
        self.canonical_rows.append(_canonical("reference", "original", 999, "FAIL"))
        _write(self.canonical_path, CANONICAL_FIELDS, self.canonical_rows)
        self._seal_inputs([*altered], expected_cases=[alternate_key, *self.expected_cases[1:]])
        with self.assertRaisesRegex(RefinementError, "differs from the pinned selection receipt"):
            analyze_refinement(self.root)

    def test_rejects_wrong_selection_bytes_even_when_manifest_hashes_are_rewritten(self) -> None:
        altered_receipt = dict(self.receipt)
        altered_receipt["selected_seeds"] = {"reference": [999], "historical_fast": [202]}
        self.selection_path.write_text(json.dumps(altered_receipt, sort_keys=True), encoding="utf-8")
        self._seal_inputs(self.trial_rows)
        with self.assertRaisesRegex(RefinementError, "independently pinned receipt"):
            analyze_refinement(self.root)

    def test_rejects_duplicate_refinement_identity(self) -> None:
        duplicate = [self.trial_rows[0], self.trial_rows[0], *self.trial_rows[2:]]
        with self.assertRaisesRegex(RefinementError, "duplicate refinement key"):
            self._seal_inputs(duplicate)
            analyze_refinement(self.root)

    def test_rejects_original_status_mismatch_with_canonical_holdout(self) -> None:
        mismatch = [dict(row) for row in self.trial_rows]
        mismatch[0]["original_status"] = "PASS"
        self._seal_inputs(mismatch)
        with self.assertRaisesRegex(RefinementError, "disagrees with canonical status"):
            analyze_refinement(self.root)

    def test_rejects_invalid_numeric_values_and_contract_changes(self) -> None:
        invalid = [dict(row) for row in self.trial_rows]
        invalid[0]["refined_dt_s"] = "nan"
        self._seal_inputs(invalid)
        with self.assertRaisesRegex(RefinementError, "finite and > 0"):
            analyze_refinement(self.root)

        invalid[0]["refined_dt_s"] = "0.001"
        invalid[0]["refined_solver_iterations"] = "99"
        self._seal_inputs(invalid)
        with self.assertRaisesRegex(RefinementError, "must be double"):
            analyze_refinement(self.root)

    def test_optional_facets_must_share_selection_and_remain_exploratory(self) -> None:
        facets = [dict(row) for row in self.trial_rows]
        facets[2]["refined_status"] = "FAIL"
        self._seal_inputs(self.trial_rows, facets_rows=facets)
        result = analyze_refinement(self.root)
        self.assertEqual(result["exploratory_facets48"]["trial_count"], 4)
        self.assertTrue(result["exploratory_facets48"]["selection_complete"])
        self.assertIn("exploratory", result["exploratory_facets48"]["role"])
        self.assertEqual(result["exploratory_facets48"]["cavity_witness_status_counts"]["FAIL"], 0)
        self.assertEqual(
            result["exploratory_facets48"]["comparison_to_refinement24"],
            {
                "matched_case_count": 4,
                "changed_case_count": 1,
                "changed_rows": [{
                    "profile": "historical_fast", "design_id": "original", "disturbance_id": "202",
                    "refinement24_status": "PASS", "facets48_status": "FAIL",
                }],
            },
        )

        self._seal_inputs(self.trial_rows, facets_rows=[dict(row) for row in self.trial_rows[:-1]])
        with self.assertRaisesRegex(RefinementError, "facets48.csv keyset differs"):
            analyze_refinement(self.root)

    def test_final_paper_gate_requires_verified_pinned_selection_receipt(self) -> None:
        status_counts = {status: 0 for status in ("PASS", "FAIL", "UNKNOWN", "CRASH")}
        status_counts["PASS"] = 1
        summary = {
            "schema": "forge.v6.refinement-summary/1", "status": "ANALYZED",
            "trial_count": 1, "flip_count": 0, "selection_complete": True,
            "expected_case_count": 1,
            "expected_cases": [{"profile": "reference", "design_id": "original", "disturbance_id": "101"}],
            "verified_selection_receipt": True, "selection_receipt_sha256": PINNED_SELECTION_SHA256,
            "interpretation_receipt_sha256": PINNED_INTERPRETATION_SHA256,
            "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
            "selection_note": "Selected sensitivity subset; not a replacement for primary results.",
            "interval": None, "interval_note": "No IID confidence interval.",
            "source_file_hashes": {
                "operation_trials.csv": "a" * 64, "refinement_manifest.json": "b" * 64,
                "refinement_trials.csv": "c" * 64, "refinement_selection.json": PINNED_SELECTION_SHA256,
            },
            "status_counts": {"original": status_counts, "refined": status_counts},
            "profiles": {"reference": {
                "trial_count": 1, "flip_count": 0,
                "original_status_counts": status_counts, "refined_status_counts": status_counts,
            }},
            "flips": [],
        }
        with self.assertRaisesRegex(ValueError, "pinned 26-case refinement selection"):
            _require_refinement({"E2_refinement": summary})
        summary["verified_selection_receipt"] = False
        with self.assertRaisesRegex(ValueError, "independently pinned refinement-selection receipt"):
            _require_refinement({"E2_refinement": summary})


if __name__ == "__main__":
    unittest.main()
