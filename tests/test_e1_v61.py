"""Known-answer tests for the standalone V6.1 E1 analysis (synthetic rows; no Forge needed)."""
from __future__ import annotations

import unittest

from analysis import e1_v61 as e1


def row(task, arm, final, *, sel=1, cands=None, transition=None, first_final=None, receipt=True):
    return {"episode_id": f"{task}-{arm}-{final}", "task_id": task, "family": task.split("-")[0], "arm": arm,
            "repeat_id": "1", "block": "b", "episode_receipt_present": receipt, "heldout_selected_final": final,
            "selected_ordinal": sel, "first_build_valid_ordinal": 1 if sel else None,
            "completed_repair_transition": transition, "heldout_first_final": first_final or final,
            "candidates": cands or [], "submitted_revisions": len(cands or []),
            "build_valid_revisions": sum(1 for c in (cands or []) if c.get("build_valid")), "wall_s": 100.0}


class E1V61AnalysisTests(unittest.TestCase):
    def test_outcome_mapping_keeps_every_planned_episode(self):
        assert e1.outcome(row("t", "A0", "PASS")) == "PASS"
        assert e1.outcome({"heldout_selected_final": None, "selected_ordinal": None, "episode_receipt_present": True}) == "NO_CANDIDATE"
        assert e1.outcome({"heldout_selected_final": None, "selected_ordinal": None, "episode_receipt_present": False}) == "CRASH"
        assert e1.outcome({"heldout_selected_final": None, "selected_ordinal": 2, "episode_receipt_present": True}) == "CRASH"


    def test_cluster_bootstrap_known_difference(self):
        rows = []
        for task in ("a-1", "a-2", "a-3", "a-4"):
            rows += [row(task, "A2", "PASS"), row(task, "A1", "FAIL")]
        result = e1.cluster_bootstrap(rows, "A2", "A1", resamples=999)
        assert result["estimate"] == 1.0 and result["interval"] == [1.0, 1.0] and result["degenerate"] is True


    def test_cluster_bootstrap_is_task_level_not_episode_level(self):
        # Task x has two sessions per arm; the per-task mean, not the episode count, enters the contrast.
        rows = [row("x-1", "A2", "PASS"), row("x-1", "A2", "FAIL"), row("x-1", "A1", "FAIL"), row("x-1", "A1", "FAIL"),
                row("y-1", "A2", "FAIL"), row("y-1", "A1", "FAIL")]
        result = e1.cluster_bootstrap(rows, "A2", "A1", resamples=999)
        assert result["tasks"] == 2
        self.assertAlmostEqual(result["estimate"], (0.5 + 0.0) / 2)


    def test_wilson_bounds(self):
        assert e1.wilson(0, 0) is None
        lo, hi = e1.wilson(6, 18)
        assert 0.15 < lo < 0.17 and 0.56 < hi < 0.59


    def test_summary_counts_transitions_and_first_vs_final(self):
        cands = [{"ordinal": 1, "build_valid": True, "development_task_status": "FAIL", "constraint_V": 0.5,
                  "constraint_assessed": ["window"], "request_seen_s": 40.0, "worker_elapsed_s": 70.0},
                 {"ordinal": 2, "build_valid": True, "development_task_status": "PASS", "constraint_V": 0.0,
                  "constraint_assessed": ["window"], "request_seen_s": 200.0, "worker_elapsed_s": 15.0}]
        rows = [row("arm6-a", "A2", "PASS", sel=2, cands=cands, transition=True, first_final="FAIL"),
                row("arm6-a", "A1", "FAIL", sel=1, cands=cands[:1], transition=False),
                row("arm6-a", "A0", "NO_CANDIDATE", sel=None)]
        export = {"label": "t", "episodes": rows,
                  "baselines": [{"task_id": "arm6-a", "c0": {"constraint_V": 1.0, "constraint_assessed": ["window"],
                                                             "development_task_status": "FAIL"},
                                 "heldout_c0_final": "FAIL"}]}
        summary = e1.summarize(export)
        assert summary["per_arm_outcomes"]["A2"]["PASS"] == 1
        assert summary["per_arm_outcomes"]["A0"]["NO_CANDIDATE"] == 1
        assert summary["completed_repair_transitions"] == {"A0": 0, "A1": 0, "A2": 1}
        assert summary["first_vs_final"]["A2"] == {"FAIL->PASS": 1}
        rhos = [p["rho"] for p in summary["continuous_progress_rows"] if p["arm"] == "A2"]
        assert rhos == [0.5, 1.0]
        assert summary["timing"]["A2"]["first_build_valid_request_s_median"] == 40.0


    def test_rho_is_undefined_when_assessed_sets_differ_or_baseline_is_zero(self):
        cands = [{"ordinal": 1, "build_valid": True, "constraint_V": 0.2, "constraint_assessed": ["window", "object_at:x@dev-1"]}]
        export = {"label": "t", "episodes": [row("g-a", "A1", "FAIL", cands=cands)],
                  "baselines": [{"task_id": "g-a", "c0": {"constraint_V": 0.4, "constraint_assessed": ["window"]}}]}
        assert e1.summarize(export)["continuous_progress_rows"][0]["rho"] is None
        export["baselines"][0]["c0"] = {"constraint_V": 0.0, "constraint_assessed": cands[0]["constraint_assessed"]}
        assert e1.summarize(export)["continuous_progress_rows"][0]["rho"] is None


    def test_protocol_literal_transition_counts_an_unbuildable_first_revision(self):
        cands = [{"ordinal": 1, "build_valid": False, "geometry_sha256": None, "candidate_case_sha256": None},
                 {"ordinal": 2, "build_valid": True, "geometry_sha256": "g", "candidate_case_sha256": "c"}]
        episode = row("arm6-a", "A0", "FAIL", sel=2, cands=cands, transition=False)
        summary = e1.summarize({"label": "t", "episodes": [episode], "baselines": []})
        assert summary["completed_repair_transitions"]["A0"] == 0
        assert summary["completed_repair_transitions_protocol_literal"]["A0"] == 1


if __name__ == "__main__":
    unittest.main()
