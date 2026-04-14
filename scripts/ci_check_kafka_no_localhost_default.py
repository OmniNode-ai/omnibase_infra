#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI gate: assert no localhost:19092 default in container-facing source (OMN-8783).

Docker containers must always receive KAFKA_BOOTSTRAP_SERVERS via catalog
hardcoded_env (redpanda:9092). Any localhost:19092 fallback in production
handler code means the container will connect to the wrong broker when the
overlay is absent.

This script scans src/ for:
  - os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092") patterns
  - os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092") patterns
  - DEFAULT_BOOTSTRAP_SERVERS = "localhost:19092" (in non-CLI modules)

Allowlisted paths (host-only CLI tools where localhost:19092 is valid):
  - src/omnibase_infra/cli/artifact_reconcile.py
  - src/omnibase_infra/cli/infra_test/
  - src/omnibase_infra/diagnostics/

Usage::

    uv run python scripts/ci_check_kafka_no_localhost_default.py
    uv run python scripts/ci_check_kafka_no_localhost_default.py --src src/

Exit codes:
    0 -- no violations found
    1 -- violations found (stderr has details)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns that indicate a silent localhost fallback in container-facing code
_VIOLATION_PATTERNS = [
    re.compile(r'os\.environ\.get\s*\(\s*"KAFKA_BOOTSTRAP_SERVERS"\s*,\s*"localhost:\d+"\s*\)'),
    re.compile(r'os\.getenv\s*\(\s*"KAFKA_BOOTSTRAP_SERVERS"\s*,\s*"localhost:\d+"\s*\)'),
    re.compile(r'DEFAULT_BOOTSTRAP_SERVERS\s*=\s*"localhost:\d+"'),
]

# Host-side CLI tools where localhost fallback is allowlisted (bootstrap_only bucket)
_ALLOWLIST = [
    "cli/artifact_reconcile.py",
    "cli/infra_test/",
    "diagnostics/",
]


def _is_allowlisted(path: Path, src_root: Path) -> bool:
    rel = str(path.relative_to(src_root))
    return any(rel.replace("\\", "/").startswith(a) for a in _ALLOWLIST)


def check(src_root: Path) -> list[str]:
    violations: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        if _is_allowlisted(py_file, src_root):
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in _VIOLATION_PATTERNS:
                if pattern.search(line):
                    rel = py_file.relative_to(src_root)
                    violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="src/omnibase_infra",
        help="Source directory to scan (default: src/omnibase_infra)",
    )
    args = parser.parse_args()
    src_root = Path(args.src).resolve()

    if not src_root.is_dir():
        print(f"ERROR: src directory not found: {src_root}", file=sys.stderr)
        return 1

    violations = check(src_root)
    if violations:
        print(
            "ERROR: localhost:19092 fallback detected in container-facing source (OMN-8783).",
            file=sys.stderr,
        )
        print(
            "Containers must receive KAFKA_BOOTSTRAP_SERVERS via catalog hardcoded_env.",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print(f"OK: No localhost:19092 fallback in container-facing source ({src_root}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
