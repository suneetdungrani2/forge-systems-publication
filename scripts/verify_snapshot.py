#!/usr/bin/env python3
"""Verify the sealed release ZIP or the source-only Git checkout after extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    entries = manifest["files"]
    media = {rel for rel, row in entries.items() if row["git"] is False}
    present = {rel for rel in media if (root / rel).is_file()}
    if present and present != media:
        raise ValueError("Reviewer media is incomplete")
    for rel, row in entries.items():
        pure = Path(rel)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != rel:
            raise ValueError(f"Unsafe release path: {rel}")
        if rel in media and not present:
            continue
        path = root / pure
        if path.is_symlink() or root not in path.resolve().parents or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Release file differs: {rel}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError(f"Release hash differs: {rel}")
    print(json.dumps({"status": "PASS", "files": len(entries) - (len(media) if not present else 0),
                      "media_present": bool(present)}))


if __name__ == "__main__":
    main()
