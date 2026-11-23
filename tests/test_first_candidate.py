"""Known-answer checks for the manifest-bound E1 first-candidate amendment."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize import build_summary, write_summary
from analysis.verify import PINNED_E1_PROTOCOL_SHA256, ArtifactError, load_artifact
from paper.generate_results import _require_complete

DIGEST = "a" * 64
EPISODE_FIELDS = (
    "task_id", "family", "method", "repeat_id", "episode_id",
    "selected_candidate_id", "outcome", "wall_s", "provider_model", "protocol_sha256",
)


def _fixture(root: Path, *, include_amendment: bool = True, null_same_candidate: bool = False) -> tuple[list[dict[str, str]], dict]:
    records_dir = root / "records"
    records_dir.mkdir(parents=True)
    episodes = []
    primary_rows = []
    primary_result_rows = []
    first_rows = []
    outcome_rows = []
    for task_index in range(9):
        for method in ("A0", "A1", "A2"):
            for repeat in ("1", "2"):
                episode_id = f"e{task_index}-{method}-{repeat}"
                final_id = f"final-{episode_id}"
                first_id = f"first-{episode_id}"
                outcome = "PASS" if repeat == "1" else "FAIL"
                no_candidate = null_same_candidate and task_index == 0 and method == "A0" and repeat == "1"
                if no_candidate:
                    final_id = None
                    first_id = None
                    outcome = "FAIL"
                first_outcome = "FAIL"
                if task_index == 0 and method == "A0" and repeat == "1":
                    first_outcome = "UNKNOWN"
                elif task_index == 0 and method == "A0" and repeat == "2":
                    first_outcome = "CRASH"
                identity = {
                    "episode_id": episode_id, "task_id": f"task-{task_index}",
                    "method": method, "repeat_id": repeat,
                }
                episodes.append({
                    **identity, "family": "arm6", "selected_candidate_id": final_id or "",
                    "outcome": outcome, "wall_s": "12", "provider_model": "fixture-model",
                    "protocol_sha256": PINNED_E1_PROTOCOL_SHA256,
                })
                primary_rows.append({
                    "episode_id": episode_id, "task_id": f"task-{task_index}",
                    "episode_sha256": "d" * 64,
                    "selected_sha256": None if no_candidate else "e" * 64,
                    "candidate_raw_sha256": None if no_candidate else "f" * 64,
                    "model_sha256": None if no_candidate else DIGEST, "candidate_id": final_id,
                })
                primary_result_rows.append({
                    "episode_id": episode_id, "task_id": f"task-{task_index}",
                    "episode_receipt_sha256": "d" * 64,
                    "selected_candidate_id": final_id,
                    "heldout_result_sha256": "b" * 64,
                    "heldout_sidecar_sha256": "1" * 64,
                })
                first_rows.append({
                    **identity, "first_ordinal": 1, "first_candidate_id": first_id,
                    "first_model_sha256": None if no_candidate else DIGEST,
                    "first_candidate_raw_sha256": None if no_candidate else DIGEST,
                    "eligible_for_heldout": not no_candidate,
                    "episode_receipt_sha256": "d" * 64,
                    "completion": "FAIL" if no_candidate else "COMPLETED",
                })
                outcome_rows.append({
                    **identity, "first_candidate_id": first_id, "final_candidate_id": final_id,
                    "first_outcome": first_outcome, "final_outcome": outcome, "reused_primary": False,
                    "first_result_sha256": DIGEST,
                    "final_result_sha256": "b" * 64,
                })

    with (root / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episodes)

    primary_payload = {
        "schema": "forge.v6-e1-cohort-freeze/1", "protocol_sha256": DIGEST,
        "state": "ALL_CANDIDATES_FROZEN", "episodes": primary_rows,
    }
    # The verifier pins the production protocol identity, so preserve it in fixtures.
    primary_payload["protocol_sha256"] = PINNED_E1_PROTOCOL_SHA256
    (records_dir / "primary_candidate_cohort.json").write_text(
        json.dumps(primary_payload, separators=(",", ":")), encoding="utf-8"
    )
    primary_digest = hashlib.sha256((records_dir / "primary_candidate_cohort.json").read_bytes()).hexdigest()
    result_identities = {
        "schema": "forge.v6-e1-result-identities/1",
        "protocol_sha256": PINNED_E1_PROTOCOL_SHA256,
        "cohort_freeze_sha256": primary_digest,
        "episodes": [
            {field: row[field] for field in (
                "episode_id", "episode_receipt_sha256", "heldout_result_sha256",
            )}
            for row in primary_result_rows
        ],
    }
    (records_dir / "result_identities.json").write_text(
        json.dumps(result_identities), encoding="utf-8"
    )

    source_identity = {
        "source_commit": "1e8e388afda1c98eb2c6ccf2de52d59d7b23b259",
        "source_manifest_sha256": "72968ebb370eb941c7e5cf971acdadcc6161effb6d3f1a0ccf6cfd8a351f2deb",
        "primary_protocol_sha256": PINNED_E1_PROTOCOL_SHA256,
        "primary_cohort_sha256": primary_digest,
    }
    primary_results_payload = {
        "schema": "forge.v6.e1-primary-results-seal/1", "state": "ALL_PRIMARY_RESULTS_FROZEN",
        "created_at": "2026-11-16T00:00:00Z", **source_identity,
        "episodes": primary_result_rows,
    }
    (records_dir / "primary_results_seal.json").write_text(
        json.dumps(primary_results_payload, separators=(",", ":")), encoding="utf-8"
    )
    primary_results_seal_digest = hashlib.sha256(
        (records_dir / "primary_results_seal.json").read_bytes()
    ).hexdigest()

    names = [
        "episodes.csv", "records/result_identities.json", "records/primary_candidate_cohort.json",
    ]
    if include_amendment:
        amendment_identity = {
            "amendment_sha256": "39a6bc9450b4185478fa87db98708346617c69e500ca0796ee46a1fa3468ae3f",
            "source_commit": "1e8e388afda1c98eb2c6ccf2de52d59d7b23b259",
            "source_manifest_sha256": "72968ebb370eb941c7e5cf971acdadcc6161effb6d3f1a0ccf6cfd8a351f2deb",
            "primary_protocol_sha256": "54cded11ebdb1b67ff3db7f454166161413af62d7ae59d2c6e0b04a5635d370c",
            "primary_cohort_sha256": primary_digest,
            "assessment_script_sha256": "c" * 64,
        }
        first_payload = {
            "schema": "forge.v6.e1-first-cohort/1", **amendment_identity,
            "rows": first_rows,
        }
        (records_dir / "first_candidate_cohort.json").write_text(
            json.dumps(first_payload, separators=(",", ":")), encoding="utf-8"
        )
        first_digest = hashlib.sha256((records_dir / "first_candidate_cohort.json").read_bytes()).hexdigest()
        result_payload = {
            "schema": "forge.v6.e1-first-results/1", **amendment_identity,
            "first_cohort_sha256": first_digest,
            "primary_results_seal_sha256": primary_results_seal_digest, "rows": outcome_rows,
        }
        (records_dir / "first_candidate_outcomes.json").write_text(
            json.dumps(result_payload, separators=(",", ":")), encoding="utf-8"
        )
        names.extend((
            "records/first_candidate_cohort.json", "records/first_candidate_outcomes.json",
            "records/primary_results_seal.json",
        ))

    entries = []
    for name in names:
        payload = (root / name).read_bytes()
        entries.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": entries}), encoding="utf-8")
    return outcome_rows, {"first": first_rows}


class FirstCandidateAnalysisTests(unittest.TestCase):
    def test_known_fixture_recomputes_same_episode_secondary_contrast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)

            full_summary = build_summary(load_artifact(root))
            summary = full_summary["studies"]["E1_first_candidate"]

            self.assertEqual(summary["row_count"], 54)
            self.assertEqual(summary["task_count"], 9)
            self.assertEqual(summary["all_arms"]["episode_count"], 54)
            self.assertEqual(summary["all_arms"]["first_pass_count"], 0)
            self.assertEqual(summary["all_arms"]["first_outcome_counts"]["UNKNOWN"], 1)
            self.assertEqual(summary["all_arms"]["first_outcome_counts"]["CRASH"], 1)
            self.assertEqual(summary["all_arms"]["final_pass_count"], 27)
            self.assertEqual(summary["all_arms"]["final_minus_first_pass_fraction"], 0.5)
            self.assertEqual(summary["all_arms"]["source_row_refs"][0], {
                "json_index": 0, "csv_row_id": "episodes.csv:2",
            })
            claim = next(
                row for row in full_summary["claims"]
                if row["metric_id"] == "e1.first_candidate.all.final_minus_first_pass_fraction"
            )
            self.assertIn("episodes.csv:2", claim["source_row_ids"])
            self.assertIn(0, claim["source_json_indices"])
            output_dir = root / "derived"
            write_summary(root, output_dir)
            with (output_dir / "claims.csv").open(encoding="utf-8", newline="") as stream:
                claims = list(csv.DictReader(stream))
            sidecar_claim = next(row for row in claims if row["metric_id"] == claim["metric_id"])
            self.assertEqual(json.loads(sidecar_claim["source_json_indices"]), list(range(54)))
            self.assertEqual(len(json.loads(sidecar_claim["source_row_ids"])), 54)

    def test_bad_identity_join_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            result_path = root / "records/first_candidate_outcomes.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["rows"][0]["task_id"] = "wrong-task"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "does not match canonical E1 identity"):
                load_artifact_and_summary(root)

    def test_canonical_episode_protocol_must_match_the_pinned_e1_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            episodes_path = root / "episodes.csv"
            with episodes_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["protocol_sha256"] = DIGEST
            with episodes_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "does not match the pinned E1 protocol"):
                load_artifact_and_summary(root)

    def test_result_identity_link_must_match_both_frozen_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            path = root / "records/result_identities.json"
            identities = json.loads(path.read_text(encoding="utf-8"))
            identities["episodes"][0]["heldout_result_sha256"] = "9" * 64
            path.write_text(json.dumps(identities), encoding="utf-8")
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "disagrees with frozen result seals"):
                load_artifact_and_summary(root)

    def test_result_identity_record_requires_pinned_schema_and_protocol(self) -> None:
        for field in ("schema", "protocol_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _fixture(root)
                path = root / "records/result_identities.json"
                identities = json.loads(path.read_text(encoding="utf-8"))
                identities[field] = "unsupported"
                path.write_text(json.dumps(identities), encoding="utf-8")
                _reseal(root)

                with self.assertRaisesRegex(ArtifactError, "pinned primary result-identities record"):
                    load_artifact_and_summary(root)

    def test_result_identity_links_must_have_unique_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            path = root / "records/result_identities.json"
            identities = json.loads(path.read_text(encoding="utf-8"))
            identities["episodes"][1]["episode_id"] = identities["episodes"][0]["episode_id"]
            path.write_text(json.dumps(identities), encoding="utf-8")
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "duplicate or invalid episode_id"):
                load_artifact_and_summary(root)

    def test_reused_primary_requires_identical_nonnull_identity_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            result_path = root / "records/first_candidate_outcomes.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["rows"][0]["reused_primary"] = True
            result_path.write_text(json.dumps(result), encoding="utf-8")
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "invalid primary-result reuse"):
                load_artifact_and_summary(root)

    def test_final_result_digest_must_match_the_primary_results_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            result_path = root / "records/first_candidate_outcomes.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["rows"][0]["final_result_sha256"] = "9" * 64
            result_path.write_text(json.dumps(result), encoding="utf-8")
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "canonical final E1 result"):
                load_artifact_and_summary(root)

    def test_missing_secondary_is_allowed_in_development_but_missing_row_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, include_amendment=False)
            summary = build_summary(load_artifact(root))["studies"]["E1_first_candidate"]
            self.assertEqual(summary["status"], "NOT_AVAILABLE")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            result_path = root / "records/first_candidate_outcomes.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["rows"].pop()
            result_path.write_text(json.dumps(result), encoding="utf-8")
            _reseal(root)
            with self.assertRaisesRegex(ArtifactError, "all 54 E1 episode rows"):
                load_artifact_and_summary(root)

    def test_invalid_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            result_path = root / "records/first_candidate_outcomes.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["rows"][0]["first_outcome"] = "ERROR"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            _reseal(root)

            with self.assertRaisesRegex(ArtifactError, "invalid first_outcome"):
                load_artifact_and_summary(root)

    def test_null_candidate_identity_does_not_mark_primary_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, null_same_candidate=True)

            summary = build_summary(load_artifact(root))["studies"]["E1_first_candidate"]

            self.assertEqual(summary["all_arms"]["reused_primary_count"], 0)
            self.assertEqual(summary["all_arms"]["episode_count"], 54)

    def test_final_paper_gate_requires_complete_pinned_secondary_summary(self) -> None:
        studies = _paper_gate_fixture()
        with self.assertRaisesRegex(ValueError, "all 54 first-candidate E1 records"):
            _require_complete(studies)

        first = studies["E1_first_candidate"]
        first.update({
            "status": "ANALYZED", "role": "SECONDARY_EXPLORATORY", "row_count": 54,
            "task_count": 9,
        })
        from analysis.verify import (
            PINNED_E1_AMENDMENT_SHA256,
            PINNED_E1_PROTOCOL_SHA256,
            PINNED_E1_SOURCE_COMMIT,
            PINNED_E1_SOURCE_MANIFEST_SHA256,
        )
        first["identity"] = {
            "amendment_sha256": PINNED_E1_AMENDMENT_SHA256,
            "source_commit": PINNED_E1_SOURCE_COMMIT,
            "source_manifest_sha256": PINNED_E1_SOURCE_MANIFEST_SHA256,
            "primary_protocol_sha256": PINNED_E1_PROTOCOL_SHA256,
            "primary_cohort_sha256": "a" * 64,
            "first_cohort_sha256": "b" * 64,
            "assessment_script_sha256": "c" * 64,
            "primary_results_seal_sha256": "e" * 64,
        }
        first["arms"] = {
            method: {"episode_count": 18, "task_count": 9, "task_cluster_bootstrap_95": [0, 0]}
            for method in ("A0", "A1", "A2")
        }
        first["all_arms"] = {
            "episode_count": 54, "task_count": 9, "task_cluster_bootstrap_95": [0, 0]
        }
        studies["E2"] = {}
        studies["E3"] = {}
        first["identity"]["amendment_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "pinned E1 source amendment"):
            _require_complete(studies)


def _reseal(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        payload = (root / entry["path"]).read_bytes()
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["bytes"] = len(payload)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def load_artifact_and_summary(root: Path) -> None:
    build_summary(load_artifact(root))


def _paper_gate_fixture() -> dict:
    return {
        "E2": {},
        "E3": {},
        "E1": {
            "status": "ANALYZED", "row_count": 54,
            "arm_counts": {
                method: {"episode_count": 18, "outcome_counts": {"PASS": 0, "FAIL": 18, "UNKNOWN": 0, "CRASH": 0}}
                for method in ("A0", "A1", "A2")
            },
            "primary_contrast": {"paired_task_count": 9, "per_task": [{} for _ in range(9)],
                                 "estimate": 0.0, "bootstrap_95": [0.0, 0.0]},
        },
        "E1_first_candidate": {},
    }


if __name__ == "__main__":
    unittest.main()
