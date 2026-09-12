# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail a job when a declared evidence artifact is absent or empty (OMN-18247).

WHY THIS EXISTS
    ``actions/upload-artifact``'s ``if-no-files-found`` has three settings and
    none of them is sufficient on its own:

    * ``warn`` (the action's DEFAULT) makes absence indistinguishable from a
      measured empty result -- the run stays green and the annotation scrolls
      past. The lab-load probe was dead from the day it landed, failing
      ``ModuleNotFoundError: No module named 'yaml'`` on 10 of 12 runs and
      producing a zero-byte artifact, and no run ever went red (OMN-18031).
    * ``ignore`` is the same hole with the annotation removed.
    * ``error`` fires only when NO path matches at all. A file that exists and
      is **zero bytes** satisfies it. So does a directory upload where the one
      file that carries the claim is missing but a sibling is present.

    So ``error`` alone still cannot tell "the producer failed" from "the
    producer measured nothing". This script is the missing half: it asserts, on
    the bytes actually about to be uploaded, that every path the policy
    declares REQUIRED exists and is non-empty.

WHERE IT RUNS
    Immediately before the uploader, in the same job, so it reads the bytes the
    uploader is about to read rather than a file a later step could replace.
    ``tests/ci/test_ci_evidence_policy.py`` asserts that placement structurally
    for every artifact declared in ``config/ci_evidence_policy.yaml``.

WHAT IT DELIBERATELY DOES NOT DO
    It does not parse or validate the artifact's content. "Non-empty" is the
    weakest honest claim and the only one that generalises across a receipt, a
    JSON record and a log bundle. Content shape is the emitting model's job
    (e.g. ``ModelLabPassReceipt`` derives its own verdict from its checks).

    It also does not decide whether a producer SHOULD have failed. A probe that
    legitimately measures "the lab is busy" writes that measurement and passes
    here; a probe that crashed writes nothing and fails here. Distinguishing
    those two is the whole point.

EXIT CODES
    ``0``  every required path exists and is non-empty.
    ``1``  at least one required path is absent, empty, or unreadable.
    ``2``  the invocation itself is malformed (no ``--require`` given).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Keep in one place so the failure text and the tests agree on the vocabulary.
_ABSENT = "ABSENT"
_EMPTY = "EMPTY (zero bytes)"
_BLANK = "BLANK (whitespace only)"
_NOT_A_FILE = "NOT A FILE"
_UNREADABLE = "UNREADABLE"


@dataclass(frozen=True)
class Classification:
    reason: str | None
    size: int = 0


def classify(path: Path) -> Classification:
    """Return the path's failure reason and observed size.

    ``reason is None`` means: the path exists, is a regular file, and holds at
    least one non-whitespace byte. Every other outcome is a reason string
    reported verbatim.
    """
    try:
        if not path.exists():
            return Classification(_ABSENT)
        if not path.is_file():
            return Classification(_NOT_A_FILE)
        data = path.read_bytes()
    except OSError as exc:  # pragma: no cover - surfaced, never swallowed
        return Classification(f"{_UNREADABLE}: {exc}")
    if len(data) == 0:
        return Classification(_EMPTY)
    if not data.strip():
        return Classification(_BLANK, len(data))
    return Classification(None, len(data))


def assert_required(artifact: str, required: list[str]) -> int:
    """Check every required path, printing one line per path. Returns an exit code."""
    failures: list[tuple[str, str]] = []
    for raw in required:
        path = Path(raw)
        result = classify(path)
        if result.reason is None:
            print(f"  OK      {raw} ({result.size} bytes)")
        else:
            print(f"  FAIL    {raw} -- {result.reason}")
            failures.append((raw, result.reason))

    if failures:
        print(
            f"\nEvidence artifact '{artifact}' is NOT honest: "
            f"{len(failures)} of {len(required)} required path(s) absent or empty.",
            file=sys.stderr,
        )
        print(
            "An artifact that cannot distinguish 'the producer failed' from "
            "'the producer measured nothing' is the defect this gate closes "
            "(OMN-18247). Fix the producer, or make it write an explicit "
            "record of its own failure -- do not relax this assertion.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nEvidence artifact '{artifact}': {len(required)} required path(s) present and non-empty."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert a declared evidence artifact is present and non-empty (OMN-18247).",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        help=(
            "The artifact's stable id as declared in config/ci_evidence_policy.yaml. "
            "This is the id, not the rendered upload-artifact name, so it stays "
            "matchable when the name carries a ${{ }} expression."
        ),
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="PATH",
        help="A path that must exist and be non-empty. Repeatable.",
    )
    args = parser.parse_args(argv)

    if not args.require:
        print(
            "assert_evidence_artifact.py: no --require path given. An assertion "
            "with nothing to assert is worse than none, because it reads as proof.",
            file=sys.stderr,
        )
        return 2

    print(f"Evidence artifact assertion (OMN-18247): {args.artifact}")
    return assert_required(args.artifact, args.require)


if __name__ == "__main__":
    raise SystemExit(main())
