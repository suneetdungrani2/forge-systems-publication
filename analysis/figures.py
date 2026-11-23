"""Render publication-ready figures from a generated summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

GRAPHITE = "#30363B"
ORANGE = "#C94F12"
PALE = "#D9DEE2"


def _save(figure: Any, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", dpi=180 if extension == "png" else None)
        paths.append(path.name)
    return paths


def render_figures(summary_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Render only studies with observations; report every intentionally skipped study."""
    import matplotlib.pyplot as plt

    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    studies = summary.get("studies")
    if not isinstance(studies, dict):
        raise ValueError("summary.json must contain a studies object")  # noqa: TRY004 -- malformed artifacts retain the public ValueError contract
    status: dict[str, Any] = {}

    e0 = studies.get("E0", {})
    strata = e0.get("strata", [])
    if e0.get("status") == "ANALYZED" and strata:
        labels = [f"{row['truth']} · {row['checker']} · {row['evidence_scope']}" for row in strata]
        values = [row["correct_fraction"] for row in strata]
        colors = [GRAPHITE if row["truth"] == "valid" else ORANGE for row in strata]
        fig, ax = plt.subplots(figsize=(max(8, min(16, 0.38 * len(labels) + 5)), 5.2))
        positions = list(range(len(labels)))
        ax.bar(positions, values, color=colors, edgecolor=GRAPHITE, linewidth=0.6)
        ax.set_ylabel("Correct classification / all cases")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(positions, labels, rotation=45, ha="right")
        ax.set_title("E0 · Evaluator classification by truth and evidence scope")
        ax.grid(axis="y", color=PALE, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        files = _save(fig, destination, "e0-classification")
        plt.close(fig)
        status["E0"] = {"status": "GENERATED", "files": files}
    else:
        status["E0"] = {"status": "SKIPPED_MISSING_STUDY_DATA"}

    e2 = studies.get("E2", {})
    e2_status = {}
    if e2.get("status") == "ANALYZED":
        for profile, role in (("reference", "PRIMARY"), ("historical_fast", "SECONDARY")):
            holdout = [
                row for row in e2.get("cohorts", [])
                if row.get("phase") == "holdout" and row.get("control_profile") == profile
                and row.get("observed_count", 0) > 0 and row.get("denominator", 0) > 0
                and row.get("success_rate") is not None
            ]
            if not holdout:
                e2_status[profile] = {"status": "SKIPPED_MISSING_STUDY_DATA"}
                continue
            labels = [row["design_id"].replace("_", " ") for row in holdout]
            rates = [row["success_rate"] for row in holdout]
            lower, upper = [], []
            for row, rate in zip(holdout, rates, strict=True):
                interval = row["wilson_95"]
                if interval is None:
                    lower.append(0.0)
                    upper.append(0.0)
                else:
                    lower.append(max(0.0, rate - interval[0]))
                    upper.append(max(0.0, interval[1] - rate))
            colors = [GRAPHITE if label == "original" else ORANGE for label in labels]
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            positions = list(range(len(labels)))
            ax.bar(positions, rates, yerr=[lower, upper], capsize=5, color=colors,
                   edgecolor=GRAPHITE, linewidth=0.6)
            ax.set_ylabel("PASS / planned or observed denominator")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(positions, labels)
            ax.set_title(f"E2 · {role.title()} {profile} control profile · holdout success")
            ax.grid(axis="y", color=PALE, linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()
            files = _save(fig, destination, f"e2-holdout-success-{profile.replace('_', '-')}")
            plt.close(fig)
            e2_status[profile] = {"status": "GENERATED", "files": files, "role": role}
    else:
        e2_status = {profile: {"status": "SKIPPED_MISSING_STUDY_DATA"}
                     for profile in ("reference", "historical_fast")}
    status["E2"] = e2_status

    e1 = studies.get("E1", {})
    arms = e1.get("arm_counts", {})
    e1_methods = [method for method in ("A0", "A1", "A2")
                  if arms.get(method, {}).get("pass_fraction") is not None]
    if e1.get("status") == "ANALYZED" and e1_methods:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.bar(e1_methods, [arms[method]["pass_fraction"] for method in e1_methods],
               color=[GRAPHITE, PALE, ORANGE][:len(e1_methods)], edgecolor=GRAPHITE, linewidth=0.6)
        ax.set_ylabel("PASS / all observed episodes")
        ax.set_ylim(0, 1.05)
        ax.set_title("E1 · Episode pass fractions by method")
        ax.grid(axis="y", color=PALE, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        files = _save(fig, destination, "e1-episode-pass-fraction")
        plt.close(fig)
        status["E1"] = {"status": "GENERATED", "files": files}
    else:
        status["E1"] = {"status": "SKIPPED_MISSING_STUDY_DATA"}

    e3 = studies.get("E3", {})
    scenarios = e3.get("scenarios", [])
    e3_files: list[str] = []
    if e3.get("status") == "ANALYZED" and e3.get("primary_status") == "COMPLETE" and scenarios:
        contrasts = ("mechanical", "schedule", "combined", "interaction")
        for metric in ("good_units", "horizon_s", "flow_time_s"):
            selected = []
            for scenario in scenarios:
                metrics = scenario.get("metrics", {}).get(metric, {})
                for contrast in contrasts:
                    value = metrics.get(contrast, {})
                    if value.get("estimate") is not None:
                        selected.append((scenario["scenario_id"], contrast, value))
            if not selected:
                continue
            labels = [f"{scenario} · {contrast}" for scenario, contrast, _ in selected]
            estimates = [value["estimate"] for _, _, value in selected]
            yerr = []
            for _, _, value in selected:
                interval = value["conditional_normal_95"]
                yerr.append(
                    [max(0.0, value["estimate"] - interval[0]),
                     max(0.0, interval[1] - value["estimate"])]
                    if interval is not None else [0.0, 0.0]
                )
            colors = [ORANGE if contrast == "interaction" else GRAPHITE for _, contrast, _ in selected]
            fig, ax = plt.subplots(figsize=(max(8.5, min(16, 0.4 * len(labels) + 5)), 5.3))
            positions = list(range(len(labels)))
            ax.bar(positions, estimates, yerr=list(zip(*yerr)), capsize=4, color=colors,
                   edgecolor=GRAPHITE, linewidth=0.6)
            ax.axhline(0, color=GRAPHITE, linewidth=0.9)
            unit = selected[0][2]["units"]
            ax.set_ylabel(f"Paired contrast ({unit})")
            ax.set_xticks(positions, labels, rotation=45, ha="right")
            ax.set_title(f"E3 · {metric.replace('_', ' ')} factorial contrasts by fixed scenario")
            ax.grid(axis="y", color=PALE, linewidth=0.8)
            ax.set_axisbelow(True)
            fig.tight_layout()
            e3_files.extend(_save(fig, destination, f"e3-{metric.replace('_', '-')}-contrasts"))
            plt.close(fig)
        status["E3"] = (
            {"status": "GENERATED", "files": e3_files}
            if e3_files else {"status": "SKIPPED_MISSING_STUDY_DATA"}
        )
    else:
        status["E3"] = {
            "status": "SKIPPED_INCOMPLETE_PRIMARY" if e3.get("status") == "ANALYZED"
            else "SKIPPED_MISSING_STUDY_DATA"
        }

    report = {"status": status}
    (destination / "figures_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        report = render_figures(args.summary_json, args.output_dir)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"FIGURE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
