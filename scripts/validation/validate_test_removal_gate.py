#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-8732: Test removal gate — blocks PRs that delete integration tests
without adding replacements.

Parses `git diff --name-status -M` output and counts integration test
deletions vs genuine additions. Hardened against two counting bypasses
flagged by CodeRabbit on PR #1336:

  1. `M` (modified) was previously counted as a replacement. Modifying
     any existing integration test could offset a deletion. Now `M` is
     ignored — only `A` (added) and qualifying renames count as new tests.

  2. Renames (`R*`) were not handled. Moving a file OUT of
     `tests/integration/` bypassed the gate because the file did not
     show as deleted. Now renames are decomposed:
       - renamed OUT of tests/integration/ → counts as deletion
       - renamed INTO tests/integration/   → counts as addition
     Copies (`C*`) are handled symmetrically on the addition side.

Usage:
    python scripts/validation/validate_test_removal_gate.py
    python scripts/validation/validate_test_removal_gate.py --base origin/main

Reads diff from `git diff --name-status -M <base>...HEAD` by default, or
from stdin if `--stdin` is passed (used in unit tests).

Exit codes:
    0 — no unmatched deletions
    1 — unmatched deletion(s) present; PR body must carry
        `[skip-test-removal-gate: <reason>]` to bypass in CI
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

INTEGRATION_TEST_RE = re.compile(r"^tests/integration/[^/]+\.py$")

RENAME_STATUS_RE = re.compile(r"^R[0-9]+$")
COPY_STATUS_RE = re.compile(r"^C[0-9]+$")


def _is_integration_test(path: str) -> bool:
    return bool(INTEGRATION_TEST_RE.match(path))


def count_deletions_and_additions(diff_output: str) -> tuple[int, int]:
    """Count integration-test deletions and additions from `git diff --name-status -M` output.

    Rules:
      - `D <path>` where path matches integration test → +1 deletion
      - `A <path>` where path matches integration test → +1 addition
      - `M <path>` — ignored (modification is not a replacement)
      - `R<score> <old> <new>`:
          * old matches and new does not → +1 deletion (moved out)
          * new matches and old does not → +1 addition (moved in)
          * both match — rename within tests/integration/, counted as
            neither deletion nor addition (net zero file count)
          * neither matches — irrelevant
      - `C<score> <old> <new>` where new matches → +1 addition (copied in).
        Copies never count as deletions because the source file is preserved.

    Lines are tab-separated as emitted by `git diff --name-status`.
    """
    deleted = 0
    added = 0

    for raw_line in diff_output.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]

        if status == "D" and len(parts) >= 2:
            if _is_integration_test(parts[1]):
                deleted += 1
            continue

        if status == "A" and len(parts) >= 2:
            if _is_integration_test(parts[1]):
                added += 1
            continue

        if status == "M":
            continue

        if RENAME_STATUS_RE.match(status) and len(parts) >= 3:
            old_path, new_path = parts[1], parts[2]
            old_match = _is_integration_test(old_path)
            new_match = _is_integration_test(new_path)
            if old_match and not new_match:
                deleted += 1
            elif new_match and not old_match:
                added += 1
            continue

        if COPY_STATUS_RE.match(status) and len(parts) >= 3:
            new_path = parts[2]
            if _is_integration_test(new_path):
                added += 1
            continue

    return deleted, added


def _run_git_diff(base: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--name-status", "-M", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref to diff against (default: origin/main)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read `git diff --name-status` output from stdin instead of invoking git",
    )
    args = parser.parse_args(argv)

    if args.stdin:
        diff_output = sys.stdin.read()
    else:
        diff_output = _run_git_diff(args.base)

    deleted, added = count_deletions_and_additions(diff_output)

    print(f"Integration test files deleted: {deleted}")
    print(f"Integration test files added: {added}")

    if deleted == 0:
        print("No integration test files deleted — gate passed")
        return 0

    if added >= deleted:
        print(f"Deleted tests covered by {added} replacement(s) — gate passed")
        return 0

    unmatched = deleted - added
    print(
        f"::error::PR deletes {deleted} integration test file(s) with only "
        f"{added} replacement(s). {unmatched} deletion(s) are unmatched. "
        "Either add replacement tests in tests/integration/ or include "
        "[skip-test-removal-gate: <reason>] in the PR body for legitimate "
        "consolidation.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
