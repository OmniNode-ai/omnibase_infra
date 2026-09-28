# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail closed unless a disposable sim corpus has the declared source shape."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _profile(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("profile") != "sim-preflight-source-compatible-v1"
        or raw.get("lane") != "sim-preflight"
        or not isinstance(raw.get("required_migrations"), list)
    ):
        raise ValueError("invalid sim preflight migration profile")
    return raw


def verify(profile_path: Path, migrations_dir: Path) -> tuple[str, ...]:
    """Verify the four audited replay prerequisites, without applying anything."""
    profile = _profile(profile_path)
    required = profile["required_migrations"]
    if len(required) != 4:
        raise ValueError("sim profile must name exactly four migrations")
    seen: set[str] = set()
    verified: list[str] = []
    for item in required:
        if not isinstance(item, dict):
            raise ValueError("sim profile migration must be an object")
        identifier = item.get("id")
        relative_path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(identifier, str)
            or not isinstance(relative_path, str)
            or not isinstance(digest, str)
            or identifier in seen
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("sim profile migration declaration is invalid")
        candidate = migrations_dir / relative_path
        if not candidate.is_file():
            raise ValueError(f"required sim migration is absent: {identifier}")
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != digest:
            raise ValueError(f"required sim migration checksum differs: {identifier}")
        seen.add(identifier)
        verified.append(identifier)
    return tuple(verified)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--migrations-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"verified": verify(args.profile, args.migrations_dir)}))


if __name__ == "__main__":
    main()
