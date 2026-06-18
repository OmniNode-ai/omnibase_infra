# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""check_lane_attestation.py — Pre-commit gate: reject lane-boundary attestations
that lack a same-session "verified: <date> via <command>" annotation (OMN-13034).

THE CLASS FIX (retro B-6): The 2026-06-11 post-mortem identified that phantom
lane entries (documented as present when absent) and running-but-undocumented
lanes (judge lane was live but absent from the map) shared a common root: lane
state was copied from memory, not from a same-session probe. This gate makes
phantom-lane boilerplate unwritable.

RULE: Any change to a file that contains lane-boundary attestation language
(patterns like "Lane | Compose project" table headers, "compose_project:" in
lane-manifest.yaml, or the GENERATED_LANE_TABLE block in CLAUDE.md) MUST carry
a `verified: <date> via <command>` annotation within the file. A change that
adds or modifies lane attestation language WITHOUT the annotation is rejected.

Specifically checked patterns:
  - Markdown lane table rows: lines matching "| <lane> | `omnibase-infra" patterns
  - lane-manifest.yaml edits: changes to lane names or compose_project values
  - CLAUDE.md GENERATED_LANE_TABLE block edits

The check is lenient on *deletions* (removing a phantom lane is correct action).
It is strict on *additions and modifications* to lane attestation content.

Exit codes:
  0 — all staged lane-attestation changes carry a verified: annotation
  1 — one or more staged changes violate the attestation rule

Usage (pre-commit):
  python scripts/check_lane_attestation.py --staged
  python scripts/check_lane_attestation.py --file path/to/file.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Patterns that identify lane-boundary attestation language in ANY file.
# A "lane attestation" is content that declares or documents a lane boundary.
# These are path-independent because their shape is specific to lane attestation
# language (a lane table row referencing the omnibase-infra compose family, or the
# generated lane-table block marker) and cannot collide with ordinary content.
_LANE_ATTESTATION_PATTERNS = [
    # Markdown lane table rows — e.g. "| dev | `omnibase-infra` |"
    re.compile(r"^\|[^|]+\|\s*`omnibase-infra[^`]*`", re.MULTILINE),
    # Generated lane table block header (CLAUDE.md)
    re.compile(r"<!-- GENERATED_LANE_TABLE", re.MULTILINE),
]

# Patterns that are ONLY meaningful inside a lane-census manifest file. A bare
# 2-space-indented YAML key (``^  name:``) and a ``compose_project:`` line occur
# in countless ordinary node contracts (``  handlers:``, ``  tags:``), so applying
# them everywhere mis-fired the gate on every edited node contract (OMN-13215).
# Scope them to the lane-manifest path so the gate guards real lane attestations
# without flagging unrelated YAML.
_LANE_MANIFEST_ONLY_PATTERNS = [
    # Lane manifest compose_project declarations
    re.compile(r"^\s+compose_project:\s+\S+", re.MULTILINE),
    # Lane manifest lane keys (top-level lane names)
    re.compile(r"^  [a-z][a-z0-9-]+:\s*$", re.MULTILINE),
]

# Path marker identifying a lane-census manifest/snapshot file. The lane-key /
# compose_project patterns only apply to these files.
_LANE_MANIFEST_PATH_PATTERN = re.compile(r"lane-census/|lane-manifest")

# Pattern for the required verified annotation. Accepts:
#   verified: 2026-06-12 via <command>
#   verified: 2026-06-12T09:15Z via <command>
#   <!-- verified: 2026-06-12 via <command> -->
_VERIFIED_PATTERN = re.compile(
    r"verified:\s+\d{4}-\d{2}-\d{2}[T\d:Z]*\s+via\s+\S",
    re.IGNORECASE,
)

# Files that are exempt from the attestation check (CI / test fixtures that
# intentionally contain lane patterns without operational attestation).
_EXEMPT_PATH_PATTERNS = [
    re.compile(r"^tests/"),
    re.compile(r"^\.github/workflows/"),
    re.compile(r"scripts/check_lane_attestation\.py$"),
    re.compile(r"scripts/generate_claude_lane_block\.py$"),
]


def _is_exempt(path: str) -> bool:
    return any(p.search(path) for p in _EXEMPT_PATH_PATTERNS)


def _has_lane_attestation(content: str, path: str) -> bool:
    if any(p.search(content) for p in _LANE_ATTESTATION_PATTERNS):
        return True
    if _LANE_MANIFEST_PATH_PATTERN.search(path):
        return any(p.search(content) for p in _LANE_MANIFEST_ONLY_PATTERNS)
    return False


def _has_verified_annotation(content: str) -> bool:
    return bool(_VERIFIED_PATTERN.search(content))


def check_file(path: Path) -> list[str]:
    """Check a single file. Returns a list of violation messages."""
    violations: list[str] = []
    rel = str(path)

    if _is_exempt(rel):
        return violations

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations  # file disappeared between staging and check

    if not _has_lane_attestation(content, rel):
        return violations

    if not _has_verified_annotation(content):
        violations.append(
            f"{rel}: contains lane-boundary attestation language but lacks a "
            f"'verified: <date> via <command>' annotation. "
            f"Run the probe command on .201 and add the verified: line in the "
            f"same session before committing. "
            f"Example: 'verified: 2026-06-17T10:00Z via "
            f"ssh jonah@192.168.86.201 docker ps' "
            f"(retro B-6 / OMN-13034 — phantom-lane boilerplate is unwritable)"
        )

    return violations


def check_staged() -> list[str]:
    """Check all staged files for attestation violations."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: git diff failed: {exc}", file=sys.stderr)
        return []

    staged_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    violations: list[str] = []

    for rel_path in staged_files:
        path = Path(rel_path)
        if not path.exists():
            continue
        violations.extend(check_file(path))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-commit gate: reject lane-boundary attestations that lack "
            "a verified: annotation (OMN-13034)"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--staged",
        action="store_true",
        help="Check all staged files (pre-commit mode)",
    )
    group.add_argument(
        "--file",
        type=Path,
        dest="file_path",
        metavar="FILE",
        help="Check a single file",
    )
    args = parser.parse_args(argv)

    if args.staged:
        violations = check_staged()
    else:
        violations = check_file(args.file_path)

    if violations:
        print(
            "LANE-ATTESTATION GATE: lane-boundary attestation without "
            "same-session verified: annotation",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nAdd 'verified: <date> via <command>' to the file (retro B-6 / OMN-13034).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
