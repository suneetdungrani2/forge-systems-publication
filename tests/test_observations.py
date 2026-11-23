"""Known-answer checks for arithmetic reanalysis of frozen E2 aggregates."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analysis.observations import (
    LIMITS,
    TRIAL_FIELDS,
    WITNESS_FIELDS,
    ObservationError,
    check,
    classify,
)

DIGESTS = {
    "observation_sha256": "a" * 64,
    "replay_sha256": "b" * 64,
    "model_sha256": "c" * 64,
    "protocol_sha256": "d" * 64,
}


def _metric_fields() -> dict[str, str]:
    return {
        "worst_abs_z_error_mm": "0.2", "limit_abs_z_error_mm": "0.4",
        "worst_xy_error_mm": "0.5", "limit_xy_error_mm": "1",
        "worst_abs_yaw_error_deg": "1.0", "limit_abs_yaw_error_deg": "3.3",
        "worst_tilt_deg": "1.0", "limit_tilt_deg": "2",
    }


def _rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    witnesses, trials = [], []
    for profile in ("reference", "historical_fast"):
        for design in ("original", "historical_v4_lead_in"):
            for seed in range(630100, 630200):
                outcome, legacy = "PASS", "true"
                witness = {
                    "phase": "holdout", "control_profile": profile, "design_id": design,
                    "disturbance_id": str(seed), "outcome": outcome, "legacy_success": legacy,
                    "n_hold_samples": "12", "reason_code": "", **_metric_fields(), **DIGESTS,
                }
                if seed == 630101 and profile == "reference" and design == "original":
                    outcome, witness["outcome"] = "FAIL", "FAIL"
                    witness["worst_xy_error_mm"] = "1.01"
                    witness["reason_code"] = "finite_hold_out_of_limits"
                elif seed == 630102 and profile == "reference" and design == "original":
                    outcome, legacy = "FAIL", "false"
                    witness["outcome"], witness["legacy_success"] = outcome, legacy
                    witness["reason_code"] = "legacy_terminal_failure"
                elif seed == 630103 and profile == "reference" and design == "original":
                    outcome, legacy = "UNKNOWN", "unknown"
                    witness.update(outcome=outcome, legacy_success=legacy, n_hold_samples="",
                                   reason_code="missing_observation", worst_abs_z_error_mm="", worst_xy_error_mm="",
                                   worst_abs_yaw_error_deg="", worst_tilt_deg="")
                    witness["observation_sha256"] = witness["replay_sha256"] = ""
                elif seed == 630104 and profile == "reference" and design == "original":
                    outcome, legacy = "CRASH", "unknown"
                    witness.update(outcome=outcome, legacy_success=legacy, n_hold_samples="",
                                   reason_code="crash", worst_abs_z_error_mm="", worst_xy_error_mm="",
                                   worst_abs_yaw_error_deg="", worst_tilt_deg="")
                    witness["observation_sha256"] = witness["replay_sha256"] = ""
                witnesses.append(witness)
                trials.append({
                    "phase": "holdout", "control_profile": profile, "design_id": design,
                    "disturbance_id": str(seed), "outcome": outcome,
                    "legacy_success": legacy,
                    "attempt_duration_s": "" if outcome in {"UNKNOWN", "CRASH"} else "1.5",
                    "model_sha256": DIGESTS["model_sha256"],
                    "protocol_sha256": DIGESTS["protocol_sha256"],
                })
    return witnesses, trials


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ObservationTests(unittest.TestCase):
    def test_classification_known_answers_at_and_around_frozen_limits(self) -> None:
        at_limits = dict(LIMITS)
        self.assertEqual(classify("PASS", "true", at_limits, dict(LIMITS)), "PASS")
        over = dict(at_limits, xy_error_mm=1.000001)
        self.assertEqual(classify("FAIL", "true", over, dict(LIMITS)), "FAIL")
        self.assertEqual(classify("FAIL", "false", at_limits, dict(LIMITS)), "FAIL")
        self.assertEqual(classify("UNKNOWN", "true", None, None), "UNKNOWN")
        self.assertEqual(classify("CRASH", "true", None, None), "CRASH")

    def test_joins_all_400_and_keeps_unknown_crash_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            witness, source = _rows()
            _write(root / "witness.csv", WITNESS_FIELDS, witness)
            _write(root / "trials.csv", TRIAL_FIELDS, source)
            result = check(root / "witness.csv", root / "trials.csv")
            self.assertEqual(result["witness_rows"], 400)
            self.assertEqual(result["joined_holdout_rows"], 400)
            counts = result["recomputed_classifications"]
            self.assertEqual(counts["reference/original/UNKNOWN"], 1)
            self.assertEqual(counts["reference/original/CRASH"], 1)
            self.assertEqual(counts["reference/original/FAIL"], 2)

    def test_rejects_missing_denominator_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            witness, source = _rows()
            _write(root / "witness.csv", WITNESS_FIELDS, witness[:-1])
            _write(root / "trials.csv", TRIAL_FIELDS, source)
            with self.assertRaises(ObservationError):
                check(root / "witness.csv", root / "trials.csv")

    def test_rejects_nonfinite_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            witness, source = _rows()
            witness[0]["worst_abs_z_error_mm"] = "nan"
            _write(root / "witness.csv", WITNESS_FIELDS, witness)
            _write(root / "trials.csv", TRIAL_FIELDS, source)
            with self.assertRaises(ObservationError):
                check(root / "witness.csv", root / "trials.csv")

    def test_rejects_changed_limit_and_changed_unit_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            witness, source = _rows()
            witness[0]["limit_abs_z_error_mm"] = "0.5"
            _write(root / "witness.csv", WITNESS_FIELDS, witness)
            _write(root / "trials.csv", TRIAL_FIELDS, source)
            with self.assertRaises(ObservationError):
                check(root / "witness.csv", root / "trials.csv")
            witness, source = _rows()
            _write(root / "witness.csv", WITNESS_FIELDS, witness)
            path = root / "witness.csv"
            changed_header = path.read_text(encoding="utf-8").replace(
                "worst_abs_yaw_error_deg", "worst_abs_yaw_error_rad", 1)
            path.write_text(changed_header, encoding="utf-8")
            with self.assertRaises(ObservationError):
                check(path, root / "trials.csv")

    def test_rejects_source_status_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            witness, source = _rows()
            source[0]["outcome"] = "FAIL"
            _write(root / "witness.csv", WITNESS_FIELDS, witness)
            _write(root / "trials.csv", TRIAL_FIELDS, source)
            with self.assertRaises(ObservationError):
                check(root / "witness.csv", root / "trials.csv")


if __name__ == "__main__":
    unittest.main()
