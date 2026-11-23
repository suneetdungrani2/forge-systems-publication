#!/usr/bin/env python3
"""Check frozen checker counts and recorded runtimes without running CAD evaluators."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

CHECKERS = ("old_sampled", "v6_centerline", "v61_scoped", "v61_global")
CATEGORIES = ("correct_accept", "correct_reject", "false_accept", "false_reject", "abstain", "crash")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empty_table():
    return {"invalid": dict.fromkeys(CATEGORIES, 0), "valid": dict.fromkeys(CATEGORIES, 0),
            "total": dict.fromkeys(CATEGORIES, 0), "n": 0}


def classify(truth, status):
    if status == "CRASH":
        return "crash"
    if status not in ("PASS", "FAIL"):
        return "abstain"
    if truth == "invalid":
        return "correct_reject" if status == "FAIL" else "false_accept"
    return "correct_accept" if status == "PASS" else "false_reject"


def runtime_stats(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(ordered[math.ceil(.95 * len(ordered)) - 1], 3),
            "max_ms": round(max(values), 3), "total_ms": round(sum(values), 3)}


def recompute(rows):
    totals = {group: {checker: empty_table() for checker in CHECKERS}
              for group in ("v6_geometry", "v61_new", "all_geometry")}
    families, non_geometry = {}, {}
    statuses = {checker: Counter() for checker in CHECKERS}
    runtimes = {key: [] for key in ("old_sampled", "v6_centerline", "v61", "case_total")}
    false_accepts, false_rejects, mismatches, errors = [], [], [], []
    identifiers = set()
    for row in rows:
        identifier = row["case_id"]
        if identifier in identifiers:
            raise ValueError("duplicate case identifier")
        identifiers.add(identifier)
        if type(row["geometry"]) is not bool:
            raise ValueError("geometry flag must be boolean")
        actual = row["actual"]
        if not row["geometry"]:
            counts = non_geometry.setdefault(row["family"], Counter())
            counts[actual["v6_checker"].get("status", "UNKNOWN")] += 1
            if actual["v6_checker"].get("execution_status") == "ERROR":
                errors.append(identifier)
            continue
        if row["truth"] not in ("valid", "invalid") or row["origin"] not in ("v6", "v61"):
            raise ValueError("unrecognized geometric truth or origin")
        old, center, revised = (actual[key] for key in ("old_sampled", "v6_centerline", "v61"))
        verdicts = {
            "old_sampled": "CRASH" if old.get("execution_status") == "ERROR" else old.get("gate_status", "UNKNOWN"),
            "v6_centerline": "CRASH" if center.get("execution_status") == "ERROR" else center.get("status", "UNKNOWN"),
            "v61_scoped": revised.get("status", "UNKNOWN"),
            "v61_global": revised.get("global_status", "UNKNOWN"),
        }
        if any(record.get("execution_status") == "ERROR" for record in (old, center, revised)):
            errors.append(identifier)
        family = families.setdefault(row["family"], {checker: empty_table() for checker in CHECKERS})
        group = "v6_geometry" if row["origin"] == "v6" else "v61_new"
        for checker, verdict in verdicts.items():
            local = row["truth_feature_local"]
            truth = local if checker == "v61_scoped" and local in ("valid", "invalid") else row["truth"]
            category = classify(truth, verdict)
            statuses[checker][verdict] += 1
            for table in (family[checker], totals[group][checker], totals["all_geometry"][checker]):
                table[truth][category] += 1
                table["total"][category] += 1
                table["n"] += 1
            if checker.startswith("v61") and category in ("false_accept", "false_reject"):
                target = false_accepts if category == "false_accept" else false_rejects
                target.append({"case_id": identifier, "checker": checker})
        if verdicts["v61_scoped"] != row["expected_v61"]:
            mismatches.append({"case_id": identifier, "expected": row["expected_v61"], "actual": verdicts["v61_scoped"]})
        for key in runtimes:
            value = row["runtime_ms"] if key == "case_total" else actual[key].get("runtime_ms")
            if value is None and key != "case_total":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid recorded runtime")
            runtimes[key].append(float(value))
    geometry_count = sum(row["geometry"] for row in rows)
    return {"case_count": len(rows), "geometry_case_count": geometry_count,
            "non_geometry_v6_case_count": len(rows) - geometry_count,
            "confusion_by_family": families, "confusion_totals": totals,
            "status_counts_by_checker": statuses, "non_geometry_v6_status_counts": non_geometry,
            "runtime": {key: runtime_stats(values) for key, values in runtimes.items()},
            "v61_false_acceptances": false_accepts, "v61_false_rejections": false_rejects,
            "v61_expectation_mismatches": mismatches, "execution_error_cases": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/v6_1")
    args = parser.parse_args()
    root = args.data_root
    inputs = (root / "raw_sanitized/e0_results.jsonl", root / "derived/e0_summary.json")
    try:
        before = {str(path): digest(path) for path in inputs}
        rows = [json.loads(line) for line in inputs[0].read_text().splitlines()]
        expected = json.loads(inputs[1].read_text())
        actual = recompute(rows)
        problems = [key for key, value in actual.items() if value != expected.get(key)]
        if tuple(actual[key] for key in ("case_count", "geometry_case_count", "non_geometry_v6_case_count")) != (360, 260, 100):
            problems.append("required_360_260_100_counts")
        if {str(path): digest(path) for path in inputs} != before:
            problems.append("input_bytes_changed")
        report = {"status": "FAIL" if problems else "PASS", "mismatches": problems,
                  "input_sha256": before, "recomputed": actual,
                  "scope": "Recorded-row arithmetic; feature-local truth for scoped checker, global truth otherwise; no evaluator or physics rerun"}
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return int(bool(problems))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
