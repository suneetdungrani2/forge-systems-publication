#!/usr/bin/env python3
"""Verify that each registered claim cites the included source bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("evidence/04_CLAIM_EVIDENCE_REGISTRY.jsonl")


def main() -> int:
    problems = []
    verified = 0
    for line_number, line in enumerate((ROOT / REGISTRY).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        claim = json.loads(line)
        source = claim.get("source") or {}
        cited, expected = source.get("path"), source.get("sha256")
        if not isinstance(cited, str) or not cited or not isinstance(expected, str):
            problems.append(f"registry line {line_number}: source path or SHA-256 is missing")
            continue
        relative = PurePosixPath(cited)
        if relative.is_absolute() or ".." in relative.parts or chr(92) in cited or ":" in cited:
            problems.append(f"registry line {line_number}: source path is not package-relative")
            continue
        path = ROOT.joinpath(*relative.parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(ROOT.resolve())
        except (OSError, ValueError):
            problems.append(f"registry line {line_number}: cited source is missing or outside the package")
            continue
        if not resolved.is_file() or hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
            problems.append(f"registry line {line_number}: cited source bytes do not match the registered SHA-256")
            continue
        verified += 1
    if verified == 0:
        problems.append("claim registry contains no verifiable source rows")
    print(json.dumps({"ok": not problems, "verified": verified, "problems": problems[:20]}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
