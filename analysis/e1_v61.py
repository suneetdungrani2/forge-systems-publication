"""Standalone V6.1 E1 analysis (stdlib only): counts, repair transitions, first-vs-final, cluster bootstrap.

Input: the sanitized export ``data/v6_1/e1/study_rows.json`` (schema forge.v6_1-e1-export/1). Every planned episode
stays in its denominator. Outcomes: PASS, FAIL, UNKNOWN, CRASH, NO_CANDIDATE (non-PASS in the primary).
Primary contrast: A2 minus A1 held-out success fraction, computed per task as the mean over its sessions and then
averaged over tasks; uncertainty by a task-cluster percentile bootstrap (9999 resamples, seed 61101). With nine
clusters this interval is descriptive; a zero observed difference is not evidence of equivalence.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from pathlib import Path

OUTCOMES = ("PASS", "FAIL", "UNKNOWN", "CRASH", "NO_CANDIDATE")
ARMS = ("A0", "A1", "A2")
BOOTSTRAP_RESAMPLES = 9999
BOOTSTRAP_SEED = 61101


def outcome(row: dict) -> str:
    value = row.get("heldout_selected_final")
    if value in OUTCOMES:
        return value
    if row.get("selected_ordinal") is None:
        return "NO_CANDIDATE" if row.get("episode_receipt_present") else "CRASH"
    return "CRASH" if value in (None, "CRASH") else "UNKNOWN"


def task_success(rows: list[dict], arm: str) -> dict[str, float]:
    per_task: dict[str, list[int]] = collections.defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            per_task[row["task_id"]].append(1 if outcome(row) == "PASS" else 0)
    return {task: sum(values) / len(values) for task, values in per_task.items()}


def cluster_bootstrap(rows: list[dict], arm_a: str, arm_b: str, *, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    a, b = task_success(rows, arm_a), task_success(rows, arm_b)
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return {"estimate": None, "tasks": 0}
    diffs = [a[t] - b[t] for t in tasks]
    estimate = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(diffs) for _ in tasks) / len(tasks) for _ in range(resamples))
    lo, hi = draws[int(0.025 * (resamples + 1)) - 1], draws[int(0.975 * (resamples + 1)) - 1]
    return {"contrast": f"{arm_a}-{arm_b}", "estimate": round(estimate, 6), "tasks": len(tasks),
            "per_task_difference": {t: round(d, 6) for t, d in zip(tasks, diffs)},
            "interval": [round(lo, 6), round(hi, 6)], "interval_type": "task-cluster percentile bootstrap 95%",
            "resamples": resamples, "seed": seed,
            "degenerate": len(set(diffs)) == 1,
            "note": "descriptive; few clusters; identical per-task differences give a zero-width interval that is not "
                    "evidence of equivalence"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 6), round(centre + half, 6)]


def protocol_transition(row: dict) -> bool:
    """Frozen protocol wording: selected ordinal >= 2 whose geometry or case hash differs from ordinal 1
    (an unbuildable ordinal 1 has no hashes and therefore differs)."""
    sel = row.get("selected_ordinal")
    if not sel or sel < 2:
        return False
    cands = {c.get("ordinal"): c for c in row.get("candidates", [])}
    first, chosen = cands.get(1, {}), cands.get(sel, {})
    return (first.get("geometry_sha256") != chosen.get("geometry_sha256")
            or first.get("candidate_case_sha256") != chosen.get("candidate_case_sha256"))


def summarize(export: dict) -> dict:
    rows = export["episodes"]
    by_arm = {arm: collections.Counter(outcome(r) for r in rows if r["arm"] == arm) for arm in ARMS}
    n_arm = {arm: sum(1 for r in rows if r["arm"] == arm) for arm in ARMS}
    families = sorted({r["family"] for r in rows})
    by_family = {f: {arm: dict(collections.Counter(outcome(r) for r in rows if r["arm"] == arm and r["family"] == f))
                     for arm in ARMS} for f in families}
    transitions = {arm: sum(1 for r in rows if r["arm"] == arm and r.get("completed_repair_transition")) for arm in ARMS}

    transitions_literal = {arm: sum(1 for r in rows if r["arm"] == arm and protocol_transition(r)) for arm in ARMS}
    c0_repairs = {arm: sum(1 for r in rows if r["arm"] == arm and outcome(r) == "PASS" and r.get("selected_differs_from_c0"))
                  for arm in ARMS}
    second_or_later = {arm: sum(1 for r in rows if r["arm"] == arm and (r.get("build_valid_revisions") or 0) >= 2)
                       for arm in ARMS}
    revisions = {arm: collections.Counter(r.get("submitted_revisions", 0) for r in rows if r["arm"] == arm)
                 for arm in ARMS}
    first_final = {}
    for arm in ARMS:
        pairs = collections.Counter()
        for r in rows:
            if r["arm"] != arm or r.get("first_build_valid_ordinal") is None:
                continue
            pairs[f"{r.get('heldout_first_final')}->{outcome(r)}"] += 1
        first_final[arm] = dict(pairs)
    regressions = {arm: 0 for arm in ARMS}
    for r in rows:
        dev_pass_seen = False
        for c in r.get("candidates", []):
            if c.get("development_task_status") == "PASS":
                dev_pass_seen = True
            if dev_pass_seen and c.get("ordinal") == r.get("selected_ordinal") and outcome(r) == "FAIL" \
                    and c.get("development_task_status") != "PASS":
                regressions[r["arm"]] += 1
    timing = {}
    for arm in ARMS:
        first_valid = [next((c.get("request_seen_s") for c in r.get("candidates", []) if c.get("build_valid")), None)
                       for r in rows if r["arm"] == arm]
        first_valid = [v for v in first_valid if isinstance(v, (int, float))]
        worker = [c.get("worker_elapsed_s") for r in rows if r["arm"] == arm for c in r.get("candidates", [])
                  if isinstance(c.get("worker_elapsed_s"), (int, float))]
        wall = [r.get("wall_s") for r in rows if r["arm"] == arm and isinstance(r.get("wall_s"), (int, float))]
        timing[arm] = {"first_build_valid_request_s_median": round(statistics.median(first_valid), 1) if first_valid else None,
                       "worker_elapsed_s_median": round(statistics.median(worker), 1) if worker else None,
                       "worker_elapsed_s_max": round(max(worker), 1) if worker else None,
                       "episode_wall_s_median": round(statistics.median(wall), 1) if wall else None,
                       "requests_with_platform_reply": sum(1 for r in rows if r["arm"] == arm
                                                           for c in r.get("candidates", []) if c.get("platform_reply"))}
    end_reasons = {arm: dict(collections.Counter(r.get("end_reason") for r in rows if r["arm"] == arm)) for arm in ARMS}
    progress = []
    base = {b["task_id"]: b for b in export.get("baselines", [])}
    for r in rows:
        c0 = (base.get(r["task_id"]) or {}).get("c0") or {}
        v0 = c0.get("constraint_V")
        for c in r.get("candidates", []):
            vk = c.get("constraint_V")
            same = c0.get("constraint_assessed") == c.get("constraint_assessed")
            rho = (round(1 - vk / v0, 6) if (isinstance(v0, (int, float)) and v0 > 0 and isinstance(vk, (int, float))
                                              and same) else None)
            progress.append({"episode_id": r["episode_id"], "arm": r["arm"], "ordinal": c.get("ordinal"),
                             "V_c0": v0, "V": vk, "rho": rho, "same_assessed_set": same})
    usage = {arm: {"episodes_with_reported_usage": sum(1 for r in rows if r["arm"] == arm and (r.get("provider") or {}).get("usage_reported")),
                   "input_tokens_sum_where_reported": sum(u.get("input_tokens", 0) for r in rows if r["arm"] == arm
                                                          for u in (r.get("provider") or {}).get("turn_usage", [])),
                   "output_tokens_sum_where_reported": sum(u.get("output_tokens", 0) for r in rows if r["arm"] == arm
                                                           for u in (r.get("provider") or {}).get("turn_usage", []))}
             for arm in ARMS}
    baselines = {b["task_id"]: {"development": (b.get("c0") or {}).get("development_task_status"),
                                "heldout": b.get("heldout_c0_final"), "V": (b.get("c0") or {}).get("constraint_V")}
                 for b in export.get("baselines", [])}
    return {"schema": "forge.v6_1-e1-summary/1", "label": export.get("label"),
            "protocol_sha256": export.get("protocol_sha256"), "cohort_freeze_sha256": export.get("cohort_freeze_sha256"),
            "episodes": len(rows), "per_arm_n": n_arm,
            "per_arm_outcomes": {arm: {o: by_arm[arm].get(o, 0) for o in OUTCOMES} for arm in ARMS},
            "per_arm_pass_wilson95_unclustered": {arm: wilson(by_arm[arm].get("PASS", 0), n_arm[arm]) for arm in ARMS},
            "per_family_outcomes": by_family,
            "primary_A2_minus_A1": cluster_bootstrap(rows, "A2", "A1"),
            "secondary_A2_minus_A0": cluster_bootstrap(rows, "A2", "A0"),
            "secondary_A1_minus_A0": cluster_bootstrap(rows, "A1", "A0"),
            "completed_repair_transitions": transitions,
            "completed_repair_transitions_protocol_literal": transitions_literal,
            "transition_definition_note": ("completed_repair_transitions compares the selected revision with the first "
                                           "BUILD-VALID revision (analysis code committed before outcomes); "
                                           "completed_repair_transitions_protocol_literal applies the frozen protocol "
                                           "wording (compare with ordinal 1, an unbuildable ordinal 1 differs). The "
                                           "protocol wording governs; both are reported."),
            "starting_design_repairs": c0_repairs,
            "episodes_with_two_or_more_build_valid_revisions": second_or_later,
            "submitted_revision_histogram": {arm: dict(sorted(v.items())) for arm, v in revisions.items()},
            "first_vs_final": first_final, "regressions": regressions, "timing": timing,
            "end_reasons": end_reasons, "provider_usage": usage, "starting_designs": baselines,
            "continuous_progress_rows": progress}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize(json.loads(args.export.read_text()))
    args.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "per_arm_outcomes", "completed_repair_transitions")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
