"""V6.1 figures generated only from frozen data files (matplotlib; PNG 300 dpi + SVG).

Palette: validated categorical order (dataviz reference palette, light surface #fcfcfb) mapped to outcomes in a fixed
stack order; every segment carries its count as a direct label and every figure has a legend, so identity is never
color-alone. Inputs: E1 summary (analysis/e1_v61.py output) and E0 V6.1 summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "forge-figures"   # deterministic SVG element ids

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STACK = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#6250d6"]  # validated adjacent order
E1_ORDER = [("PASS", "PASS"), ("NO_CANDIDATE", "No candidate"), ("UNKNOWN", "UNKNOWN"), ("FAIL", "FAIL"),
            ("CRASH", "CRASH")]
E0_ORDER = [("correct_accept", "Correct accept"), ("correct_reject", "Correct reject"), ("abstain", "Abstain (UNKNOWN)"),
            ("false_accept", "False accept"), ("false_reject", "False reject")]
ARMS = ["A0", "A1", "A2"]
ARM_LABEL = {"A0": "A0 build diagnostics", "A1": "A1 + aggregate verdict", "A2": "A2 + typed residuals"}


def _style(ax):
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out: Path, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "svg"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches="tight",
                    metadata={"Date": None} if ext == "svg" else {"Software": None})
        paths.append(str(path))
    plt.close(fig)
    return paths


def _stacked_row(ax, y: float, counts: list[int], height: float = 0.62):
    left = 0
    for (key, _), color, value in zip(E1_ORDER if len(counts) == 5 and ax.get_label() == "e1" else E0_ORDER,
                                      STACK, counts):
        if value:
            ax.barh(y, value, left=left, color=color, height=height, edgecolor=SURFACE, linewidth=1.0)
            if value / max(1, sum(counts)) > 0.06:
                ax.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=7,
                        color="white" if color in ("#2a78d6", "#e34948", "#6250d6") else INK)
        left += value


# Reader-facing names of the task families (the data keep their identifiers).
FAMILY_LABEL = {"arm6": "Six-axis arm", "arm6_cell": "Six-axis arm in cell", "gantry_v4b": "Gantry"}


def e1_outcomes(summary: dict, out: Path) -> list[str]:
    families = list(summary["per_family_outcomes"])
    fig, axes = plt.subplots(1, len(families) + 1, figsize=(7.2, 2.2), sharey=True)
    panels = [("All 9 tasks", summary["per_arm_outcomes"])] + [
        (FAMILY_LABEL.get(f, f), {arm: summary["per_family_outcomes"][f].get(arm, {}) for arm in ARMS}) for f in families]
    for ax, (title, table) in zip(axes, panels):
        ax.set_label("e1")
        _style(ax)
        for i, arm in enumerate(ARMS):
            _stacked_row(ax, len(ARMS) - 1 - i, [table[arm].get(k, 0) for k, _ in E1_ORDER])
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("episodes", fontsize=7, color=INK2)
    axes[0].set_yticks(range(len(ARMS)))
    axes[0].set_yticklabels([ARM_LABEL[a] for a in reversed(ARMS)], fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STACK]
    fig.legend(handles, [label for _, label in E1_ORDER], loc="lower center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.24))
    fig.patch.set_facecolor(SURFACE)
    return _save(fig, out, "fig_v61_e1_outcomes_by_family")


def e1_revisions(summary: dict, out: Path) -> list[str]:
    hist = summary["submitted_revision_histogram"]
    counts = sorted({int(k) for arm in ARMS for k in hist[arm]} | {0, 1, 2, 3, 4})
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    _style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    width = 0.26
    colors = {"A0": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a"}
    for i, arm in enumerate(ARMS):
        xs = [c + (i - 1) * width for c in counts]
        ys = [hist[arm].get(str(c), hist[arm].get(c, 0)) for c in counts]
        ax.bar(xs, ys, width=width * 0.92, color=colors[arm], label=ARM_LABEL[arm], edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(counts)
    ax.set_xlabel("revisions submitted in the episode (0–4)", fontsize=7, color=INK2)
    ax.set_ylabel("episodes", fontsize=7, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    fig.patch.set_facecolor(SURFACE)
    return _save(fig, out, "fig_v61_e1_revisions_submitted")


def e0_confusion(e0: dict, out: Path) -> list[str]:
    fams = e0["confusion_by_family"]
    checkers = [("old_sampled", "Earlier sampled screen"), ("v6_centerline", "Centerline witness (earlier)"),
                ("v61_scoped", "Feature-scoped checker"), ("v61_global", "Whole-part checker")]
    groups = [("Previously recorded geometry cases (100)", [f for f in fams if f in
                                                           ("pilot_axis", "off_axis_local_web", "off_axis_control")]),
              ("New analytic controls (160)", [f for f in fams if f not in
                                                    ("pilot_axis", "off_axis_local_web", "off_axis_control")
                                                    and "v61_scoped" in fams[f]])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0), sharey=True)
    for ax, (title, members) in zip(axes, groups):
        ax.set_label("e0")
        _style(ax)
        for i, (key, _) in enumerate(checkers):
            totals = {k: 0 for k, _ in E0_ORDER}
            for f in members:
                tot = fams[f].get(key, {}).get("total", {})
                for k in totals:
                    totals[k] += int(tot.get(k, 0))
            _stacked_row(ax, len(checkers) - 1 - i, [totals[k] for k, _ in E0_ORDER])
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("geometry cases", fontsize=7, color=INK2)
    axes[0].set_yticks(range(len(checkers)))
    axes[0].set_yticklabels([label for _, label in reversed(checkers)], fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STACK]
    fig.legend(handles, [label for _, label in E0_ORDER], loc="lower center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.26))
    fig.patch.set_facecolor(SURFACE)
    return _save(fig, out, "fig_v61_e0_checker_outcomes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-summary", type=Path)
    parser.add_argument("--e0-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if args.e1_summary:
        summary = json.loads(args.e1_summary.read_text())
        written += e1_outcomes(summary, args.out) + e1_revisions(summary, args.out)
    if args.e0_summary:
        written += e0_confusion(json.loads(args.e0_summary.read_text()), args.out)
    print(json.dumps({"written": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
