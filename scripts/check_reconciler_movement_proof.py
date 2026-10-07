#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Gate: a workspace reconciler may not judge a surface by an exit code (OMN-17307).

CLAUDE.md rule 5 -- "detection tools that aren't wired as pre-merge CI gates are
advisory and get ignored" -- so the movement-proof rule ships as a gate in the
same PR that introduces it, not as a convention in a docstring.

WHAT IT ENFORCES, IN TWO PARTS OF VERY DIFFERENT STRENGTH
Both parts are stated plainly because overclaiming a gate's strength is its own
failure mode.

**Part 1 is structural and cannot be satisfied by editing a comment.**

  * ``scripts/reconcile-host.sh`` -- the orchestrator every scheduler calls --
    must actually invoke ``reconcile_verify_movement.py``. If someone deletes
    the readback, this fails.
  * ``verdict()`` in ``reconcile_verify_movement.py`` must take exactly
    ``(before, after, target)``. Adding an exit-status parameter is the single
    change that would quietly re-open the whole defect class, so the signature
    is pinned here as well as in the unit tests.

**Part 2 is a declaration gate, in the same family as this repo's existing**
``# raw-prod-bypass-ok:`` **and** ``# canonical-inference-ok:`` **ratchets.**

  Any script whose *name* marks it a reconciler (``reconcile*`` under
  ``scripts/``) must carry one of three explicit markers:

    1. it invokes ``reconcile_verify_movement.py``; or
    2. ``# movement-proof-delegated-to: <path>`` -- it is a delegate, and the
       named caller does the readback; or
    3. ``# movement-proof: <how>`` -- it performs its own readback, described.

  Discovery is by filesystem glob, not by a hand-maintained manifest, precisely
  so a new reconciler cannot be added without the gate seeing it. That is the
  OMN-15525 lesson: an artifact absent from a manifest is unguarded and nobody
  finds out.

  What this part does NOT do: it cannot tell a real readback from a marker
  someone typed to make the gate quiet. Its value is that it converts an
  invisible omission into a reviewable, greppable line in the diff -- the same
  value the two existing ratchets provide, and the same limit they have.

Usage:
    python scripts/check_reconciler_movement_proof.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VERIFIER_NAME = "reconcile_verify_movement.py"
ORCHESTRATOR = "scripts/reconcile-host.sh"

# `reconcile*` under scripts/, at any depth. Deliberately a glob rather than a
# list: a list is a thing you forget to update.
RECONCILER_GLOBS = ("scripts/reconcile*.sh", "scripts/**/reconcile*.sh")

_DELEGATION_MARKER = re.compile(r"#\s*movement-proof-delegated-to:\s*(\S+)")
_SELF_PROOF_MARKER = re.compile(r"#\s*movement-proof:\s*(\S.*)")
# `[^),]+` on the last parameter, not `[^)]+`: a trailing comma would mean a
# FOURTH parameter, and a fourth parameter is the whole thing being forbidden.
_VERDICT_SIGNATURE = re.compile(
    r"def\s+verdict\s*\(\s*before\s*:[^,]+,\s*after\s*:[^,]+,\s*target\s*:[^),]+\)"
)


def discover_reconcilers(repo_root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in RECONCILER_GLOBS:
        found.update(p for p in repo_root.glob(pattern) if p.is_file())
    return sorted(found)


def check(repo_root: Path) -> list[str]:
    failures: list[str] = []

    # -- Part 1: structural ------------------------------------------------- #
    orchestrator = repo_root / ORCHESTRATOR
    if not orchestrator.is_file():
        failures.append(
            f"{ORCHESTRATOR} is missing. It is the single scheduled entry point "
            "on every host; without it nothing proves any surface moved."
        )
    elif VERIFIER_NAME not in orchestrator.read_text(encoding="utf-8"):
        failures.append(
            f"{ORCHESTRATOR} no longer invokes {VERIFIER_NAME}. The orchestrator's "
            "whole job is the readback; without it a delegate that exits 0 "
            "without moving anything reports success (OMN-17291)."
        )

    verifier = repo_root / "scripts" / VERIFIER_NAME
    if not verifier.is_file():
        failures.append(f"scripts/{VERIFIER_NAME} is missing.")
    else:
        source = verifier.read_text(encoding="utf-8")
        if not _VERDICT_SIGNATURE.search(source):
            failures.append(
                f"scripts/{VERIFIER_NAME}: verdict() must take exactly "
                "(before, after, target). A parameter carrying an exit status "
                "would let any caller turn 'the command succeeded' into 'the "
                "surface moved', which is the defect OMN-17307 closes."
            )

    # -- Part 2: declaration ratchet ---------------------------------------- #
    for script in discover_reconcilers(repo_root):
        rel = script.relative_to(repo_root)
        text = script.read_text(encoding="utf-8")
        if VERIFIER_NAME in text:
            continue
        delegated = _DELEGATION_MARKER.search(text)
        if delegated:
            target = repo_root / delegated.group(1)
            if not target.is_file():
                failures.append(
                    f"{rel}: declares movement-proof-delegated-to "
                    f"'{delegated.group(1)}', which does not exist. A delegation "
                    "to a missing file is an unproven surface with a comment on it."
                )
            continue
        if _SELF_PROOF_MARKER.search(text):
            continue
        failures.append(
            f"{rel}: a reconciler with no movement proof. It must either invoke "
            f"scripts/{VERIFIER_NAME}, or declare one of:\n"
            f"    # movement-proof-delegated-to: <path that does the readback>\n"
            f"    # movement-proof: <how this script reads the surface back>\n"
            "  Judging a reconcile by the exit status of the command that was "
            "supposed to move the surface is the OMN-17307 defect class: it makes "
            "a repair and a no-op the same observation."
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root to scan (default: this script's repo)",
    )
    # pre-commit passes staged filenames; the check is whole-repo by design
    # (a reconciler can be made non-compliant by DELETING a line elsewhere), so
    # they are accepted and ignored rather than rejected.
    parser.add_argument("filenames", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    failures = check(Path(args.repo_root))
    if not failures:
        print("reconciler movement-proof gate: OK")
        return 0
    print("reconciler movement-proof gate: FAILED", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
