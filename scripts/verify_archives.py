#!/usr/bin/env python3.11
"""Verify split archive manifests without reconstructing full archives on disk."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_PART_BYTES = 100 * 1024 * 1024


def sha256_stream(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024 * 16), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_stream([path])


def main() -> int:
    index_path = REPO_ROOT / "manifests/release_index.json"
    if not index_path.exists():
        print(f"missing {index_path}", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text())
    failures: list[str] = []
    summary = []

    for archive in index["archives"]:
        part_paths = []
        total_size = 0
        for part in archive["parts"]:
            path = REPO_ROOT / part["path"]
            if not path.exists():
                failures.append(f"missing part: {path}")
                continue
            actual_size = path.stat().st_size
            actual_sha = sha256_file(path)
            if actual_size != part["size_bytes"]:
                failures.append(f"size mismatch: {path}")
            if actual_sha != part["sha256"]:
                failures.append(f"part sha mismatch: {path}")
            if actual_size > MAX_PART_BYTES:
                failures.append(f"part exceeds 100 MiB: {path} ({actual_size})")
            total_size += actual_size
            part_paths.append(path)

        combined_sha = sha256_stream(part_paths)
        if combined_sha != archive["combined_archive_sha256"]:
            failures.append(f"combined sha mismatch: {archive['archive']}")
        if total_size != archive["combined_archive_size_bytes"]:
            failures.append(f"combined size mismatch: {archive['archive']}")
        summary.append(
            {
                "archive": archive["archive"],
                "parts": len(part_paths),
                "bytes": total_size,
                "sha256": combined_sha,
            }
        )

    print(json.dumps({"archive_count": len(summary), "archives": summary}, indent=2))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
