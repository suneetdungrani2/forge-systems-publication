#!/usr/bin/env python3
"""Reanalyse the frozen study and compile both manuscripts outside the release tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers/current"


def run(out, name, args, cwd=ROOT):
    with (out / f"{name}.log").open("w") as log:
        result = subprocess.run([str(a) for a in args], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {name}.log")
    print(f"{name}: PASS", flush=True)


def matches(generated, accepted):
    assert generated.read_bytes() == accepted.read_bytes(), accepted.name


def main():
    if not __debug__:
        raise RuntimeError("Disable Python optimization for verification.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("--out must be outside the source tree")
    out.mkdir(parents=True, exist_ok=False)
    python = sys.executable
    run(out, "no-proprietary-forge", [python, "-c", "import importlib.util; assert importlib.util.find_spec('forge') is None"])
    run(out, "data-manifest", [python, "-m", "analysis.verify", "data"])
    run(out, "unit-tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run(out, "archived-summary", [python, "-m", "analysis.summarize", "data", out / "summary"])
    run(out, "fixture-witnesses", [python, "-m", "analysis.observations", "data/records/operation_witnesses.csv",
         "data/operation_trials.csv", "--manifest", "data/records/operation_witnesses.manifest.json"])
    run(out, "fixture-sensitivity", [python, "-m", "analysis.refinement", "data", out / "refinement.json"])
    run(out, "repair-summary", [python, "-m", "analysis.e1_v61", "data/v6_1/raw_sanitized/e1_study_rows.json", "--out", out / "e1-summary.json"])
    assert json.loads((out / "e1-summary.json").read_text()) == json.loads((ROOT / "data/v6_1/derived/e1_summary.json").read_text())
    run(out, "checker-counts", [python, "scripts/check_checker_counts.py"])
    run(out, "analysis-figures", [python, "-m", "analysis.figures_v61", "--e1-summary", out / "e1-summary.json",
         "--e0-summary", "data/v6_1/derived/e0_summary.json", "--out", out / "figures"])
    for generated in sorted((out / "figures").glob("*.svg")):
        matches(generated, ROOT / "figures" / generated.name)
    run(out, "claim-sources", [python, "scripts/verify_claim_sources.py"])
    claims = json.loads((out / "claim-sources.log").read_text())
    assert claims["ok"]
    paper = out / "paper"
    shutil.copytree(PAPERS / "paper", paper)
    run(out, "derive-tables-and-figures", [python, paper / "scripts/derive_paper.py", "--evidence", ROOT / "evidence",
         "--artifact", ROOT, "--paper", paper])
    for folder in ("build", "figures"):
        for accepted in sorted((PAPERS / "paper" / folder).iterdir()):
            matches(paper / folder / accepted.name, accepted)
    matches(out / "PAPER_CLAIM_EVIDENCE_MAP.csv", PAPERS / "PAPER_CLAIM_EVIDENCE_MAP.csv")
    run(out, "latex-build", ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex", "supplement.tex"], paper)
    pdfs = {}
    for compiled, filename, pages in (("main", "FORGE_Research_Paper.pdf", 12),
                                      ("supplement", "FORGE_Technical_Supplement.pdf", 23)):
        rebuilt, original = paper / f"{compiled}.pdf", PAPERS / filename
        assert subprocess.check_output(["pdftotext", "-layout", str(rebuilt), "-"]) == subprocess.check_output(
            ["pdftotext", "-layout", str(original), "-"])
        info = subprocess.check_output(["pdfinfo", str(rebuilt)], text=True)
        assert re.search(r"^Author:\s+Suneet Dungrani$", info, re.M)
        assert re.search(rf"^Pages:\s+{pages}$", info, re.M)
        log = (paper / f"{compiled}.log").read_text(errors="replace")
        assert not any(x in log for x in ("Overfull", "undefined", "Missing character"))
        pdfs[filename] = {"pages": pages, "text_match": True, "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
    result = {"status": "PASS", "scope": "Tier A frozen-data reanalysis and manuscript rebuild",
              "claims": claims, "pdfs": pdfs}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
