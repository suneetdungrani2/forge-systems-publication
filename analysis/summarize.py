"""Recompute descriptive study summaries and provenance-bearing claim rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from analysis.verify import (
    CONDITIONS,
    CONTROL_PROFILES,
    DESIGNS,
    PHASES,
    ArtifactError,
    first_candidate_records,
    load_artifact,
)

WILSON_Z = 1.959963984540054
E3_CONTRASTS = {
    "mechanical": {"M10": 1.0, "M00": -1.0},
    "schedule": {"M01": 1.0, "M00": -1.0},
    "combined": {"M11": 1.0, "M00": -1.0},
    "interaction": {"M11": 1.0, "M10": -1.0, "M01": -1.0, "M00": 1.0},
}
E3_METRICS = {
    "good_units": ("good_units", "units"),
    "horizon_s": ("horizon_s", "s"),
    "flow_time_s": ("flow_time_s", "s"),
}


def _value(value: str) -> float:
    return float(value)


def _wilson(successes: int, denominator: int) -> tuple[float, float] | None:
    if denominator <= 0:
        return None
    proportion = successes / denominator
    z2 = WILSON_Z**2
    center = (proportion + z2 / (2 * denominator)) / (1 + z2 / denominator)
    half = WILSON_Z * math.sqrt(
        proportion * (1 - proportion) / denominator + z2 / (4 * denominator**2)
    ) / (1 + z2 / denominator)
    return max(0.0, center - half), min(1.0, center + half)


def _claim(
    metric_id: str,
    numerator: float,
    denominator: int,
    units: str,
    scope: str,
    source_row_ids: Iterable[str],
    *,
    estimate: float | None = None,
    source_json_indices: list[int] | None = None,
) -> dict[str, Any]:
    claim = {
        "metric_id": metric_id,
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate if estimate is not None else (
            numerator / denominator if denominator else None
        ),
        "units": units,
        "scope": scope,
        "source_row_ids": sorted(source_row_ids),
    }
    if source_json_indices is not None:
        claim["source_json_indices"] = sorted(source_json_indices)
    return claim


def _task_cluster_bootstrap(task_deltas: list[float]) -> list[float] | None:
    if not task_deltas:
        return None
    rng = random.Random(E1_BOOTSTRAP_SEED)
    sampled_means = sorted(
        sum(rng.choices(task_deltas, k=len(task_deltas))) / len(task_deltas)
        for _ in range(E1_BOOTSTRAP_REPLICATES)
    )
    return [sampled_means[int(0.025 * (len(sampled_means) - 1))],
            sampled_means[int(0.975 * (len(sampled_means) - 1))]]


def _summarize_e0(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["truth"], row["checker"], row["evidence_scope"])].append(row)

    result = []
    claims = []
    for (truth, checker, evidence_scope), group in sorted(groups.items()):
        n = len(group)
        status_counts = Counter(row["status"] for row in group)
        correct = status_counts["PASS"] if truth == "valid" else status_counts["FAIL"]
        false_accept = status_counts["PASS"] if truth == "invalid" else 0
        false_reject = status_counts["FAIL"] if truth == "valid" else 0
        abstain = status_counts["UNKNOWN"] + status_counts["ERROR"]
        scope = f"truth={truth};checker={checker};evidence_scope={evidence_scope}"
        item = {
            "truth": truth,
            "checker": checker,
            "evidence_scope": evidence_scope,
            "case_count": n,
            "status_counts": {key: status_counts[key] for key in ("PASS", "FAIL", "UNKNOWN", "ERROR")},
            "correct_count": correct,
            "false_accept_count": false_accept,
            "false_reject_count": false_reject,
            "abstain_count": abstain,
            "correct_fraction": correct / n,
            "false_accept_fraction": false_accept / n,
            "false_reject_fraction": false_reject / n,
            "abstain_fraction": abstain / n,
            "interval": None,
            "interval_note": "No IID confidence interval: evaluator cases may be related fixtures.",
        }
        result.append(item)
        source_ids = [row["_row_id"] for row in group]
        for label, count, rate in (
            ("correct", correct, correct / n),
            ("false_accept", false_accept, false_accept / n),
            ("false_reject", false_reject, false_reject / n),
            ("abstain", abstain, abstain / n),
        ):
            claims.append(_claim(
                f"e0.{label}", count, n, "case_fraction", scope, source_ids, estimate=rate
            ))
    return {"status": "ANALYZED" if rows else "PRESENT_EMPTY", "strata": result}, claims


E1_BOOTSTRAP_SEED = 9999
E1_BOOTSTRAP_REPLICATES = 9999


def _summarize_e1(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_task_method: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_task_method[(row["family"], row["task_id"], row["method"])].append(row)

    arm_counts = {}
    claims = []
    for method in ("A0", "A1", "A2"):
        group = by_method.get(method, [])
        counts = Counter(row["outcome"] for row in group)
        arm_counts[method] = {
            "episode_count": len(group),
            "selected_artifact_count": sum(bool(row["selected_candidate_id"]) for row in group),
            "outcome_counts": {outcome: counts[outcome] for outcome in ("PASS", "FAIL", "UNKNOWN", "CRASH")},
            "pass_fraction": counts["PASS"] / len(group) if group else None,
            "source_row_ids": sorted(row["_row_id"] for row in group),
        }

    tasks: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(dict)
    for (family, task, method), episodes in by_task_method.items():
        tasks[(family, task)][method] = episodes
    paired = []
    task_deltas = []
    paired_source_ids = []
    for (family, task), methods in sorted(tasks.items()):
        if "A1" not in methods or "A2" not in methods:
            continue
        a1 = sum(row["outcome"] == "PASS" for row in methods["A1"]) / len(methods["A1"])
        a2 = sum(row["outcome"] == "PASS" for row in methods["A2"]) / len(methods["A2"])
        delta = a2 - a1
        row_ids = sorted(row["_row_id"] for method in ("A1", "A2") for row in methods[method])
        paired.append({"family": family, "task_id": task, "a1_pass_fraction": a1,
                       "a2_pass_fraction": a2, "a2_minus_a1": delta,
                       "source_row_ids": row_ids})
        task_deltas.append(delta)
        paired_source_ids.extend(row_ids)

    estimate = sum(task_deltas) / len(task_deltas) if task_deltas else None
    interval = None
    if task_deltas:
        interval = _task_cluster_bootstrap(task_deltas)
        claims.append(_claim("e1.paired_task_pass_fraction.a2_minus_a1", sum(task_deltas),
                             len(task_deltas), "task_pass_fraction_delta",
                             f"paired_tasks={len(task_deltas)};task_cluster_bootstrap_seed={E1_BOOTSTRAP_SEED}",
                             paired_source_ids, estimate=estimate))
    return {
        "status": "ANALYZED" if rows else "PRESENT_EMPTY",
        "arm_counts": arm_counts,
        "primary_contrast": {
            "method": "A2-A1", "estimand": "mean of task-level PASS-fraction differences",
            "paired_task_count": len(task_deltas), "estimate": estimate,
            "bootstrap_95": interval,
            "bootstrap_replicates": E1_BOOTSTRAP_REPLICATES if task_deltas else 0,
            "bootstrap_seed": E1_BOOTSTRAP_SEED,
            "warning": "Task-cluster bootstrap is descriptive; small task counts make interval estimates unstable.",
            "per_task": paired, "source_row_ids": sorted(paired_source_ids),
        },
    }, claims


def _summarize_first_candidate(
    records: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = records["rows"]
    by_method_task: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_key = (row["family"], row["task_id"])
        by_method_task[(*task_key, row["method"])].append(row)
        by_method[row["method"]].append(row)
        by_task[task_key].append(row)

    arm_summaries: dict[str, dict[str, Any]] = {}
    for method in ("A0", "A1", "A2"):
        arm_rows = by_method.get(method, [])
        task_deltas = []
        task_clusters = sorted({(row["family"], row["task_id"]) for row in arm_rows})
        for family, task_id in task_clusters:
            task_rows = by_method_task[(family, task_id, method)]
            first_rate = sum(row["first_outcome"] == "PASS" for row in task_rows) / len(task_rows)
            final_rate = sum(row["final_outcome"] == "PASS" for row in task_rows) / len(task_rows)
            task_deltas.append(final_rate - first_rate)
        arm_summaries[method] = _first_candidate_counts(arm_rows, task_deltas)

    all_task_deltas = []
    for task_cluster in sorted(by_task):
        task_rows = by_task[task_cluster]
        first_rate = sum(row["first_outcome"] == "PASS" for row in task_rows) / len(task_rows)
        final_rate = sum(row["final_outcome"] == "PASS" for row in task_rows) / len(task_rows)
        all_task_deltas.append(final_rate - first_rate)
    overall = _first_candidate_counts(rows, all_task_deltas)
    summary = {
        "status": "ANALYZED",
        "role": "SECONDARY_EXPLORATORY",
        "row_count": len(rows),
        "task_count": len(all_task_deltas),
        "identity": records["identity"],
        "arms": arm_summaries,
        "all_arms": overall,
        "bootstrap_seed": E1_BOOTSTRAP_SEED,
        "bootstrap_replicates": E1_BOOTSTRAP_REPLICATES,
        "warning": "Secondary exploratory analysis; amendment was fixed after E1 episode execution began and before hidden evaluation. Task-cluster intervals are descriptive, not new confirmatory tests.",
    }
    claims = []
    for label, result in [*arm_summaries.items(), ("all", overall)]:
        references = result["source_row_refs"]
        claims.append(_claim(
            f"e1.first_candidate.{label}.final_minus_first_pass_fraction",
            result["final_minus_first_pass_fraction"] * result["task_count"],
            result["task_count"], "task_pass_fraction_delta",
            f"secondary_exploratory;paired_tasks={result['task_count']};task_cluster_bootstrap_seed={E1_BOOTSTRAP_SEED}",
            (reference["csv_row_id"] for reference in references),
            estimate=result["final_minus_first_pass_fraction"],
            source_json_indices=[reference["json_index"] for reference in references],
        ))
    return summary, claims


def _first_candidate_counts(rows: list[dict[str, Any]], task_deltas: list[float]) -> dict[str, Any]:
    first_pass = sum(row["first_outcome"] == "PASS" for row in rows)
    final_pass = sum(row["final_outcome"] == "PASS" for row in rows)
    first_outcomes = Counter(row["first_outcome"] for row in rows)
    final_outcomes = Counter(row["final_outcome"] for row in rows)
    return {
        "episode_count": len(rows),
        "first_pass_count": first_pass,
        "final_pass_count": final_pass,
        "first_outcome_counts": {outcome: first_outcomes[outcome] for outcome in ("PASS", "FAIL", "UNKNOWN", "CRASH")},
        "final_outcome_counts": {outcome: final_outcomes[outcome] for outcome in ("PASS", "FAIL", "UNKNOWN", "CRASH")},
        "first_pass_fraction": first_pass / len(rows) if rows else None,
        "final_pass_fraction": final_pass / len(rows) if rows else None,
        "final_minus_first_pass_fraction": sum(task_deltas) / len(task_deltas) if task_deltas else None,
        "task_count": len(task_deltas),
        "task_cluster_bootstrap_95": _task_cluster_bootstrap(task_deltas),
        "changed_outcome_count": sum(row["first_outcome"] != row["final_outcome"] for row in rows),
        "regressed_count": sum(row["first_outcome"] == "PASS" and row["final_outcome"] != "PASS" for row in rows),
        "improved_count": sum(row["first_outcome"] != "PASS" and row["final_outcome"] == "PASS" for row in rows),
        "reused_primary_count": sum(row["reused_primary"] for row in rows),
        "source_row_refs": [
            {"json_index": row["_json_index"], "csv_row_id": row["_row_id"]}
            for row in sorted(rows, key=lambda item: item["_json_index"])
        ],
    }


def _summarize_e2(
    rows: list[dict[str, str]], protocol: dict[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["control_profile"], row["design_id"])].append(row)

    cohorts = []
    claims = []
    planned = protocol.get("planned_holdout_trials") if protocol else None
    for phase, profile, design in sorted(
        (phase, profile, design)
        for phase in PHASES for profile in CONTROL_PROFILES for design in DESIGNS
    ):
        group = groups.get((phase, profile, design), [])
        counts = Counter(row["outcome"] for row in group)
        planned_count = planned[profile][design] if phase == "holdout" and planned else None
        denominator = planned_count if planned_count is not None else len(group)
        if planned_count is not None and len(group) > planned_count:
            raise ArtifactError(
                f"holdout {profile}/{design}: observed {len(group)} trials exceeds planned denominator {planned_count}"
            )
        missing = max(0, denominator - len(group)) if planned_count is not None else None
        successes = counts["PASS"]
        interval = _wilson(successes, denominator) if group and denominator else None
        cohort = {
            "phase": phase,
            "control_profile": profile,
            "role": "primary" if profile == "reference" else "secondary",
            "design_id": design,
            "status": "ANALYZED" if group else "MISSING_COHORT",
            "observed_count": len(group),
            "planned_count": planned_count,
            "denominator": denominator,
            "missing_count": missing,
            "outcome_counts": {key: counts[key] for key in ("PASS", "FAIL", "UNKNOWN", "CRASH")},
            "legacy_success_counts": {
                key: sum(row["legacy_success"] == key for row in group)
                for key in ("true", "false", "unknown")
            },
            "success_rate": successes / denominator if group and denominator else None,
            "wilson_95": list(interval) if interval is not None else None,
            "wilson_note": (
                "Wilson score interval; descriptive binomial interval conditional on independent trials. "
                "It does not account for pairing or protocol-level dependence."
            ),
        }
        cohorts.append(cohort)
        if phase == "holdout" and group and profile == "reference":
            source_ids = [row["_row_id"] for row in group]
            scope = (
                f"phase=holdout;role=primary;control_profile=reference;design_id={design};"
                f"denominator={denominator};planned={'yes' if planned_count is not None else 'no'}"
            )
            claims.append(_claim(
                f"e2.holdout.success_rate.reference.{design}", successes, denominator,
                "trial_fraction", scope, source_ids,
            ))

    pair_results = []
    for phase, profile in sorted((phase, profile) for phase in PHASES for profile in CONTROL_PROFILES):
        left = {row["disturbance_id"]: row for row in groups.get((phase, profile, "original"), [])}
        right = {row["disturbance_id"]: row for row in groups.get((phase, profile, "historical_v4_lead_in"), [])}
        common = sorted(left.keys() & right.keys())
        original_pass = sum(left[key]["outcome"] == "PASS" for key in common)
        historical_pass = sum(right[key]["outcome"] == "PASS" for key in common)
        delta = (original_pass - historical_pass) / len(common) if common else None
        source_ids = [left[key]["_row_id"] for key in common] + [right[key]["_row_id"] for key in common]
        pair_results.append({
            "phase": phase,
            "control_profile": profile,
            "role": "primary" if profile == "reference" else "secondary",
            "paired_count": len(common),
            "original_only_count": len(left.keys() - right.keys()),
            "lead_in_only_count": len(right.keys() - left.keys()),
            "paired_original_pass_count": original_pass,
            "paired_lead_in_pass_count": historical_pass,
            "success_rate_delta_original_minus_lead_in": delta,
            "interpretation": "Paired descriptive contrast; no independent-design claim.",
        })
        if common:
            claims.append(_claim(
                f"e2.paired_success_delta.{profile}.{phase}", original_pass - historical_pass,
                len(common), "paired_trial_fraction_delta",
                f"phase={phase};role={'primary' if profile == 'reference' else 'secondary'};"
                f"control_profile={profile};paired_disturbance_ids={len(common)}",
                source_ids, estimate=delta,
            ))
    interaction_rows = []
    if rows:
        profile_maps = {
            (profile, design): {
                row["disturbance_id"]: row
                for row in groups.get(("holdout", profile, design), [])
            }
            for profile in CONTROL_PROFILES for design in DESIGNS
        }
        all_keys = set.intersection(*(set(mapping) for mapping in profile_maps.values()))
        differences = []
        source_ids = []
        for key in sorted(all_keys):
            ref_geometry = (
                int(profile_maps[("reference", "original")][key]["outcome"] == "PASS")
                - int(profile_maps[("reference", "historical_v4_lead_in")][key]["outcome"] == "PASS")
            )
            fast_geometry = (
                int(profile_maps[("historical_fast", "original")][key]["outcome"] == "PASS")
                - int(profile_maps[("historical_fast", "historical_v4_lead_in")][key]["outcome"] == "PASS")
            )
            differences.append(ref_geometry - fast_geometry)
            source_ids.extend(profile_maps[(profile, design)][key]["_row_id"]
                              for profile in CONTROL_PROFILES for design in DESIGNS)
        interaction = sum(differences) / len(differences) if differences else None
        interaction_rows.append({
            "phase": "holdout", "paired_count": len(differences),
            "estimate": interaction, "units": "paired_trial_fraction_difference",
            "definition": "(original-lead_in geometry delta under reference) minus the same delta under historical_fast control",
            "status": "ANALYZED" if differences else "UNAVAILABLE_INCOMPLETE_PAIRS",
            "source_row_ids": sorted(source_ids),
        })
        if differences:
            claims.append(_claim("e2.geometry_by_control_interaction.holdout", sum(differences),
                                 len(differences), "paired_trial_fraction_difference",
                                 f"phase=holdout;complete_four_way_disturbance_pairs={len(differences)}",
                                 source_ids, estimate=interaction))
    status = "ANALYZED" if rows else "MISSING_STUDY"
    return {"status": status, "cohorts": cohorts, "paired_comparisons": pair_results,
            "geometry_by_control_interaction": interaction_rows}, claims


def _summarize_e3(
    rows: list[dict[str, str]], protocol: dict[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_pair[(row["scenario_id"], row["paired_seed"])][row["condition"]] = row
    scenarios: dict[str, list[tuple[str, dict[str, dict[str, str]]]]] = defaultdict(list)
    for (scenario, seed), condition_rows in sorted(by_pair.items()):
        scenarios[scenario].append((seed, condition_rows))

    expected_seeds = protocol.get("expected_production_seeds", {}) if protocol else {}
    for scenario in expected_seeds:
        scenarios.setdefault(scenario, [])
    result = []
    claims = []
    for scenario, seed_sets in sorted(scenarios.items()):
        valid = [(seed, rows_by_condition) for seed, rows_by_condition in seed_sets
                 if set(rows_by_condition) == CONDITIONS
                 and all(rows_by_condition[c]["check_status"] == "PASS" for c in CONDITIONS)]
        invalid = [(seed, rows_by_condition) for seed, rows_by_condition in seed_sets
                   if set(rows_by_condition) != CONDITIONS
                   or any(row["check_status"] != "PASS" for row in rows_by_condition.values())]
        expected = expected_seeds.get(scenario)
        missing = max(0, expected - len(seed_sets)) if expected is not None else 0
        primary_complete = not invalid and missing == 0
        status_counts = Counter()
        for _, rows_by_condition in invalid:
            states = [rows_by_condition[c]["check_status"] for c in sorted(rows_by_condition)]
            if set(rows_by_condition) != CONDITIONS:
                states.append("MISSING_CONDITION")
            status_counts["+".join(sorted(set(states)))] += 1

        primary_metrics = {}
        conditional_metrics = {}
        for metric, (field, unit) in E3_METRICS.items():
            primary_metrics[metric] = {}
            conditional_metrics[metric] = {}
            for contrast, weights in E3_CONTRASTS.items():
                deltas = []
                source_ids = []
                for _, rows_by_condition in valid:
                    deltas.append(sum(weight * _value(rows_by_condition[c][field])
                                      for c, weight in weights.items()))
                    source_ids.extend(rows_by_condition[c]["_row_id"] for c in weights)
                n = len(deltas)
                total = sum(deltas)
                estimate = total / n if n else None
                if n >= 2:
                    variance = sum((value - estimate) ** 2 for value in deltas) / (n - 1)
                    mcse = math.sqrt(variance / n)
                    interval = [estimate - 1.959963984540054 * mcse,
                                estimate + 1.959963984540054 * mcse]
                else:
                    mcse = None
                    interval = None
                primary_metrics[metric][contrast] = {
                    "numerator": total if primary_complete else None,
                    "denominator": expected if expected is not None else len(seed_sets),
                    "estimate": estimate if primary_complete else None,
                    "units": unit,
                    "status": "PRIMARY_COMPLETE" if primary_complete else "PRIMARY_INCOMPLETE",
                    "source_row_ids": sorted(source_ids) if primary_complete else [],
                    "conditional_normal_95": interval if primary_complete else None,
                }
                conditional_metrics[metric][contrast] = {
                    "numerator": total, "denominator": n, "estimate": estimate,
                    "units": unit, "seed_level_deltas": deltas,
                    "conditional_mcse": mcse, "conditional_normal_95": interval,
                    "source_row_ids": sorted(source_ids),
                    "interpretation": "Secondary diagnostic conditional on complete PASS-check quartets within this fixed scenario.",
                }
                if primary_complete and n:
                    claims.append(_claim(f"e3.{metric}.{contrast}.{scenario}", total, n, unit,
                                         f"scenario_id={scenario};all_planned_quartets_pass_check={n};fixed_scenario",
                                         source_ids, estimate=estimate))
        result.append({
            "scenario_id": scenario,
            "quartet_count": len(seed_sets),
            "expected_quartet_count": expected,
            "missing_planned_quartet_count": missing,
            "valid_pass_check_quartet_count": len(valid),
            "invalid_or_incomplete_quartet_count": len(invalid),
            "primary_status": "COMPLETE" if primary_complete else "INCOMPLETE",
            "invalid_quartet_status_counts": dict(sorted(status_counts.items())),
            "metrics": primary_metrics,
            "conditional_diagnostics": conditional_metrics,
        })
    all_complete = bool(result) and all(item["primary_status"] == "COMPLETE" for item in result)
    if not all_complete:
        claims = []
        for scenario in result:
            for contrasts in scenario["metrics"].values():
                for metric in contrasts.values():
                    metric.update({
                        "numerator": None, "estimate": None,
                        "status": "PRIMARY_INCOMPLETE", "source_row_ids": [],
                        "conditional_normal_95": None,
                    })
            scenario["primary_status"] = "INCOMPLETE"
    return {
        "status": "ANALYZED" if rows else ("INCOMPLETE" if expected_seeds else "MISSING_STUDY"),
        "primary_status": "COMPLETE" if all_complete else "INCOMPLETE",
        "scenario_count": len(scenarios),
        "scenario_seed_pair_count": len(by_pair),
        "contrast_definitions": E3_CONTRASTS,
        "scenarios": result,
    }, claims


def build_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    """Compute all available analyses directly from validated source rows."""
    datasets = artifact["datasets"]
    studies: dict[str, Any] = {}
    claims: list[dict[str, Any]] = []
    for filename, key in (
        ("evaluator_cases.csv", "E0"),
        ("operation_trials.csv", "E2"),
        ("production_runs.csv", "E3"),
        ("episodes.csv", "E1"),
    ):
        if filename not in datasets:
            missing_study: dict[str, Any] = {"status": "MISSING_STUDY", "row_count": None}
            if key == "E2":
                planned = (
                    artifact["protocol"].get("planned_holdout_trials")
                    if artifact["protocol"] else None
                )
                missing_study.update({
                    "planned_holdout_trials": planned,
                    "missing_holdout_trials": planned,
                })
            elif key == "E3":
                expected = artifact["protocol"].get("expected_production_seeds", {}) if artifact["protocol"] else {}
                missing_study.update({
                    "primary_status": "INCOMPLETE" if expected else "NOT_ASSESSABLE",
                    "missing_planned_quartets_by_scenario": expected,
                })
            studies[key] = missing_study
            continue
        rows = datasets[filename]
        if key == "E0":
            study, study_claims = _summarize_e0(rows)
        elif key == "E2":
            study, study_claims = _summarize_e2(rows, artifact["protocol"])
        elif key == "E1":
            study, study_claims = _summarize_e1(rows)
        else:
            study, study_claims = _summarize_e3(rows, artifact["protocol"])
        studies[key] = {"row_count": len(rows), **study}
        claims.extend(study_claims)
    accounting = artifact.get("auxiliary_json", {}).get("records/resource_accounting.json")
    if accounting is not None:
        if not isinstance(accounting, dict) or accounting.get("schema") != "forge.v6-e1-resources/2":
            raise ArtifactError("unsupported E1 resource observations")
        resource_rows = accounting.get("episodes")
        canonical = {row["episode_id"]: row for row in datasets.get("episodes.csv", [])}
        if (not isinstance(resource_rows, list) or len(resource_rows) != len(canonical)
                or len({row["episode_id"] for row in resource_rows}) != len(resource_rows)
                or {row["episode_id"] for row in resource_rows} != set(canonical)):
            raise ArtifactError("E1 resource observations do not match canonical episodes")
        resources = {}
        for row in resource_rows:
            if row["arm"] != canonical[row["episode_id"]]["method"]:
                raise ArtifactError("E1 resource arm differs from canonical episode")
            usage = row["provider_usage"]
            if usage["status"] != ("available" if usage["usage_records"] else "unavailable"):
                raise ArtifactError("E1 token availability contradicts observed usage records")
        for arm in ("A0", "A1", "A2"):
            group = [(index, row) for index, row in enumerate(resource_rows) if row["arm"] == arm]
            resources[arm] = {
                "episode_count": len(group),
                **{key: sum(row[key] for _, row in group) for key in (
                    "submitted_count", "completed_graded_count", "worker_timeout_count",
                    "protected_task_feedback_published_count")},
                "wall_minutes": sum(row["wall_s"] for _, row in group) / 60,
                "token_observed_episodes": sum(bool(row["provider_usage"]["usage_records"]) for _, row in group),
                "source_json_indices": [index for index, _ in group],
            }
        studies["E1"]["resources"] = resources
    first_candidate = first_candidate_records(artifact)
    if first_candidate is not None:
        studies["E1_first_candidate"], first_candidate_claims = _summarize_first_candidate(first_candidate)
        claims.extend(first_candidate_claims)
    else:
        studies["E1_first_candidate"] = {
            "status": "NOT_AVAILABLE", "role": "SECONDARY_EXPLORATORY", "row_count": 0
        }
    source_root = artifact.get("root")
    if source_root is not None:
        refinement_file = Path(source_root) / "records" / "refinement_trials.csv"
        refinement_manifest = Path(source_root) / "refinement_manifest.json"
        if (refinement_file.exists() or refinement_file.is_symlink()
                or refinement_manifest.exists() or refinement_manifest.is_symlink()):
            from analysis.refinement import analyze_refinement

            studies["E2_refinement"] = analyze_refinement(source_root)
    return {
            "schema_version": 1,
        "source_manifest": "manifest.json",
        "studies": studies,
        "claims": claims,
        "limitations": [
            "Summaries are recomputed from the allowlisted CSV rows; no expected result values are embedded.",
            "E0 fixture strata have no IID confidence intervals.",
            "E2 development and qualification cohorts are descriptive; holdout is the only primary cohort.",
            "E3 primary claims require every observed and protocol-planned quartet to be complete with four PASS checks; conditional complete-quartet diagnostics are secondary.",
        ],
    }


def write_summary(artifact_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    artifact = load_artifact(artifact_dir)
    summary = build_summary(artifact)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    claims = summary["claims"]
    with (destination / "claims.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "metric_id", "numerator", "denominator", "estimate", "units", "scope",
            "source_row_ids", "source_json_indices",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for claim in claims:
            row = dict(claim)
            row["source_row_ids"] = json.dumps(row["source_row_ids"], separators=(",", ":"))
            if "source_json_indices" in row:
                row["source_json_indices"] = json.dumps(row["source_json_indices"], separators=(",", ":"))
            writer.writerow(row)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = write_summary(args.artifact_dir, args.output_dir)
    except ArtifactError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "SUMMARIZED",
        "output_dir": str(args.output_dir),
        "claim_count": len(summary["claims"]),
        "studies": {key: value["status"] for key, value in summary["studies"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
