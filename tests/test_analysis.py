"""Small known-answer contract fixtures; no Forge runtime is imported."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize import build_summary
from analysis.verify import ArtifactError, load_artifact

DIGEST = "a" * 64


def write_csv(root: Path, filename: str, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with (root / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seal(root: Path, names: list[str], *, planned: int | None = None,
         expected_seeds: dict[str, int] | None = None) -> None:
    if planned is not None or expected_seeds is not None:
        protocol = {"schema_version": 1}
        if planned is not None:
            protocol["planned_holdout_trials"] = {
                profile: {design: planned for design in ("original", "historical_v4_lead_in")}
                for profile in ("reference", "historical_fast")
            }
        if expected_seeds is not None:
            protocol["expected_production_seeds"] = expected_seeds
        (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        names = [*names, "protocol.json"]
    entries = []
    for name in names:
        payload = (root / name).read_bytes()
        entries.append({
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "files": entries,
    }), encoding="utf-8")


def write_auxiliary(root: Path, name: str, payload: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


class ArtifactAnalysisTests(unittest.TestCase):
    def test_auxiliary_records_are_admitted_as_integrity_bound_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "records/geometry_sensitivity.json",
                "records/operation_witnesses.csv",
            ]
            write_auxiliary(root, names[0], '{"case": "geometry"}')
            write_auxiliary(root, names[1], "witness_id,detail\nw1,retained\n")
            seal(root, names)

            artifact = load_artifact(root)

            self.assertEqual(artifact["datasets"], {})
            self.assertEqual([entry["path"] for entry in artifact["manifest"]["files"]], names)

    def test_auxiliary_record_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "records/operation_model_contracts.json"
            write_auxiliary(root, name, '{"contract": "sealed"}')
            seal(root, [name])
            write_auxiliary(root, name, '{"contract": "changed"}')

            with self.assertRaisesRegex(ArtifactError, "byte count mismatch|SHA-256 mismatch"):
                load_artifact(root)

    def test_auxiliary_records_reject_symlink_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "records").symlink_to(outside, target_is_directory=True)
            name = "records/e1_protocol.json"
            write_auxiliary(root, name, '{"protocol": "sealed"}')
            seal(root, [name])

            with self.assertRaisesRegex(ArtifactError, "records must be a real directory"):
                load_artifact(root)

    def test_unknown_auxiliary_record_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "records/unreviewed.json"
            write_auxiliary(root, name, "{}")
            seal(root, [name])

            with self.assertRaisesRegex(ArtifactError, "manifest path is not allowlisted"):
                load_artifact(root)

    def test_e0_classifies_false_accept_false_reject_correct_and_abstain_by_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "evaluator_cases.csv", (
                "case_id", "group_id", "fault_family", "truth", "checker", "status",
                "evidence_scope", "runtime_ms",
            ), [
                {"case_id": "v-pass", "group_id": "g", "fault_family": "f", "truth": "valid",
                 "checker": "c", "status": "PASS", "evidence_scope": "property-a", "runtime_ms": "1"},
                {"case_id": "v-reject", "group_id": "g", "fault_family": "f", "truth": "valid",
                 "checker": "c", "status": "FAIL", "evidence_scope": "property-a", "runtime_ms": "2"},
                {"case_id": "v-unknown", "group_id": "g", "fault_family": "f", "truth": "valid",
                 "checker": "c", "status": "UNKNOWN", "evidence_scope": "property-b", "runtime_ms": "3"},
                {"case_id": "i-accept", "group_id": "g", "fault_family": "f", "truth": "invalid",
                 "checker": "c", "status": "PASS", "evidence_scope": "property-a", "runtime_ms": "4"},
                {"case_id": "i-reject", "group_id": "g", "fault_family": "f", "truth": "invalid",
                 "checker": "c", "status": "FAIL", "evidence_scope": "property-a", "runtime_ms": "5"},
            ])
            seal(root, ["evaluator_cases.csv"])
            summary = build_summary(load_artifact(root))
            strata = summary["studies"]["E0"]["strata"]
            valid_a = next(row for row in strata if row["truth"] == "valid" and row["evidence_scope"] == "property-a")
            invalid_a = next(row for row in strata if row["truth"] == "invalid")
            valid_b = next(row for row in strata if row["evidence_scope"] == "property-b")
            self.assertEqual(valid_a["correct_count"], 1)
            self.assertEqual(valid_a["false_reject_count"], 1)
            self.assertEqual(invalid_a["false_accept_count"], 1)
            self.assertEqual(invalid_a["correct_count"], 1)
            self.assertEqual(valid_b["abstain_count"], 1)
            self.assertIsNone(valid_b["interval"])

    def test_e2_keeps_unknown_and_missing_holdout_trials_in_planned_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "phase", "control_profile", "design_id", "disturbance_id", "outcome", "legacy_success",
                "attempt_duration_s", "model_sha256", "protocol_sha256",
            )
            rows = []
            for profile in ("reference", "historical_fast"):
                for design, cases in (
                    ("original", [("d1", "PASS"), ("d2", "UNKNOWN")]),
                    ("historical_v4_lead_in", [("d1", "PASS"), ("d2", "FAIL")]),
                ):
                    rows.extend({
                        "phase": "holdout", "control_profile": profile, "design_id": design,
                        "disturbance_id": disturbance, "outcome": outcome,
                        "legacy_success": "unknown", "attempt_duration_s": "" if outcome == "UNKNOWN" else "1.5",
                        "model_sha256": DIGEST, "protocol_sha256": DIGEST,
                    } for disturbance, outcome in cases)
            write_csv(root, "operation_trials.csv", columns, rows)
            seal(root, ["operation_trials.csv"], planned=3)
            summary = build_summary(load_artifact(root))["studies"]["E2"]
            original = next(row for row in summary["cohorts"] if row["phase"] == "holdout" and row["control_profile"] == "reference" and row["design_id"] == "original")
            self.assertEqual(original["denominator"], 3)
            self.assertEqual(original["outcome_counts"]["UNKNOWN"], 1)
            self.assertEqual(original["missing_count"], 1)
            self.assertAlmostEqual(original["success_rate"], 1 / 3)
            paired = next(row for row in summary["paired_comparisons"] if row["phase"] == "holdout" and row["control_profile"] == "reference")
            self.assertEqual(paired["paired_count"], 2)
            self.assertAlmostEqual(paired["success_rate_delta_original_minus_lead_in"], 0.0)
            self.assertEqual(summary["geometry_by_control_interaction"][0]["paired_count"], 2)

    def test_e2_absent_cohort_is_explicit_without_a_fabricated_zero_rate_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "phase", "control_profile", "design_id", "disturbance_id", "outcome", "legacy_success",
                "attempt_duration_s", "model_sha256", "protocol_sha256",
            )
            write_csv(root, "operation_trials.csv", columns, [])
            seal(root, ["operation_trials.csv"], planned=100)
            summary = build_summary(load_artifact(root))
            result = summary["studies"]["E2"]
            absent = next(
                row for row in result["cohorts"]
                if row["phase"] == "holdout" and row["control_profile"] == "reference" and row["design_id"] == "original"
            )
            self.assertEqual(absent["status"], "MISSING_COHORT")
            self.assertEqual(absent["missing_count"], 100)
            self.assertIsNone(absent["success_rate"])
            self.assertFalse(any(
                c["metric_id"] == "e2.holdout.success_rate.reference.original"
                for c in summary["claims"]
            ))

    def test_e3_four_factorial_contrasts_match_known_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "scenario_id", "paired_seed", "condition", "good_units", "horizon_s",
                "flow_time_s", "check_status", "protocol_sha256",
            )
            offsets = {
                "M00": (0, 0, 0), "M10": (2, 0, -2),
                "M01": (3, 1, -3), "M11": (8, 2, -10),
            }
            rows = []
            for seed in ("seed-a", "seed-b"):
                for condition, (units, horizon, flow) in offsets.items():
                    rows.append({
                        "scenario_id": "scenario-fixed", "paired_seed": seed,
                        "condition": condition, "good_units": str(10 + units),
                        "horizon_s": str(100 + horizon), "flow_time_s": str(50 + flow),
                        "check_status": "PASS", "protocol_sha256": DIGEST,
                    })
            write_csv(root, "production_runs.csv", columns, rows)
            seal(root, ["production_runs.csv"], expected_seeds={"scenario-fixed": 2})
            summary = build_summary(load_artifact(root))["studies"]["E3"]
            scenario = summary["scenarios"][0]
            self.assertEqual(scenario["primary_status"], "COMPLETE")
            self.assertEqual(scenario["metrics"]["good_units"]["mechanical"]["estimate"], 2)
            self.assertEqual(scenario["metrics"]["good_units"]["schedule"]["estimate"], 3)
            self.assertEqual(scenario["metrics"]["good_units"]["combined"]["estimate"], 8)
            self.assertEqual(scenario["metrics"]["good_units"]["interaction"]["estimate"], 3)
            self.assertEqual(scenario["metrics"]["horizon_s"]["interaction"]["estimate"], 1)
            self.assertEqual(scenario["metrics"]["flow_time_s"]["interaction"]["estimate"], -5)

    def test_e1_task_cluster_contrast_uses_task_means_and_seeded_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "task_id", "family", "method", "repeat_id", "episode_id",
                "selected_candidate_id", "outcome", "wall_s", "provider_model", "protocol_sha256",
            )
            rows = []
            outcomes = {
                ("t1", "A1"): ["PASS", "PASS"], ("t1", "A2"): ["PASS", "FAIL"],
                ("t2", "A1"): ["FAIL", "FAIL"], ("t2", "A2"): ["PASS", "PASS"],
            }
            for (task, method), results in outcomes.items():
                for i, outcome in enumerate(results, start=1):
                    rows.append({
                        "task_id": task, "family": "small-known", "method": method,
                        "repeat_id": str(i), "episode_id": f"{task}-{method}-{i}",
                        "selected_candidate_id": f"candidate-{method}", "outcome": outcome,
                        "wall_s": "1.25", "provider_model": "fixture-model",
                        "protocol_sha256": DIGEST,
                    })
            rows.append({
                "task_id": "t0", "family": "small-known", "method": "A0",
                "repeat_id": "1", "episode_id": "t0-A0-1",
                "selected_candidate_id": "", "outcome": "CRASH", "wall_s": "",
                "provider_model": "fixture-model", "protocol_sha256": DIGEST,
            })
            write_csv(root, "episodes.csv", columns, rows)
            seal(root, ["episodes.csv"])
            e1 = build_summary(load_artifact(root))["studies"]["E1"]
            self.assertEqual(e1["primary_contrast"]["paired_task_count"], 2)
            self.assertAlmostEqual(e1["primary_contrast"]["estimate"], 0.25)
            self.assertEqual(e1["primary_contrast"]["bootstrap_seed"], 9999)
            self.assertEqual(len(e1["primary_contrast"]["source_row_ids"]), 8)
            self.assertEqual(e1["arm_counts"]["A0"]["episode_count"], 1)
            self.assertEqual(e1["arm_counts"]["A0"]["outcome_counts"]["CRASH"], 1)

    def test_duplicate_key_and_hash_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "phase", "control_profile", "design_id", "disturbance_id", "outcome", "legacy_success",
                "attempt_duration_s", "model_sha256", "protocol_sha256",
            )
            row = {
                "phase": "holdout", "control_profile": "reference", "design_id": "original", "disturbance_id": "d1",
                "outcome": "PASS", "legacy_success": "true", "attempt_duration_s": "0.2",
                "model_sha256": DIGEST, "protocol_sha256": DIGEST,
            }
            write_csv(root, "operation_trials.csv", columns, [row, row])
            seal(root, ["operation_trials.csv"])
            with self.assertRaisesRegex(ArtifactError, "duplicate operation trial key"):
                load_artifact(root)

            write_csv(root, "operation_trials.csv", columns, [row])
            seal(root, ["operation_trials.csv"])
            (root / "operation_trials.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "byte count mismatch|SHA-256 mismatch"):
                load_artifact(root)

    def test_incomplete_production_quartet_blocks_primary_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "scenario_id", "paired_seed", "condition", "good_units", "horizon_s",
                "flow_time_s", "check_status", "protocol_sha256",
            )
            rows = [{
                "scenario_id": "s", "paired_seed": "1", "condition": condition,
                "good_units": "1", "horizon_s": "10", "flow_time_s": "2",
                "check_status": "PASS", "protocol_sha256": DIGEST,
            } for condition in ("M00", "M01", "M10")]
            write_csv(root, "production_runs.csv", columns, rows)
            seal(root, ["production_runs.csv"], expected_seeds={"s": 100})
            summary = build_summary(load_artifact(root))
            e3 = summary["studies"]["E3"]
            scenario = e3["scenarios"][0]
            self.assertEqual(scenario["primary_status"], "INCOMPLETE")
            self.assertEqual(scenario["missing_planned_quartet_count"], 99)
            self.assertEqual(scenario["invalid_or_incomplete_quartet_count"], 1)
            self.assertIsNone(scenario["metrics"]["good_units"]["mechanical"]["estimate"])
            self.assertFalse(any(claim["metric_id"].startswith("e3.") for claim in summary["claims"]))

    def test_e3_nonpass_checker_quartet_blocks_primary_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = (
                "scenario_id", "paired_seed", "condition", "good_units", "horizon_s",
                "flow_time_s", "check_status", "protocol_sha256",
            )
            rows = [{
                "scenario_id": "s", "paired_seed": "1", "condition": condition,
                "good_units": str(value), "horizon_s": "10", "flow_time_s": "2",
                "check_status": "FAIL" if condition == "M11" else "PASS",
                "protocol_sha256": DIGEST,
            } for condition, value in (("M00", 1), ("M01", 2), ("M10", 3), ("M11", 4))]
            write_csv(root, "production_runs.csv", columns, rows)
            seal(root, ["production_runs.csv"])
            summary = build_summary(load_artifact(root))
            scenario = summary["studies"]["E3"]["scenarios"][0]
            self.assertEqual(scenario["invalid_or_incomplete_quartet_count"], 1)
            self.assertEqual(scenario["conditional_diagnostics"]["good_units"]["mechanical"]["denominator"], 0)
            self.assertIsNone(scenario["metrics"]["good_units"]["mechanical"]["estimate"])
            # An execution failure has no measured count or flow time. Preserve
            # missing values instead of inventing a zero-valued observation.
            rows[-1].update(check_status="ERROR", good_units="", flow_time_s="")
            write_csv(root, "production_runs.csv", columns, rows)
            seal(root, ["production_runs.csv"])
            incomplete = build_summary(load_artifact(root))
            self.assertIsNone(incomplete["studies"]["E3"]["scenarios"][0]["metrics"]["good_units"]["mechanical"]["estimate"])


if __name__ == "__main__":
    unittest.main()
