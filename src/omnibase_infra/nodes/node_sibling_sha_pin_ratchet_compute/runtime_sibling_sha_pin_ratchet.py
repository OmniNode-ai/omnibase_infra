# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Git EFFECT boundary and CLI for the sibling sha-pin ratchet (OMN-18594).

    uv run python -m omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.runtime_sibling_sha_pin_ratchet --base origin/dev
    uv run python -m omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.runtime_sibling_sha_pin_ratchet --count

Exit 0 for PASS, 1 when a file gained a pin against the base, 2 when the check
cannot decide. ``--count`` prints the per-file counts of the working tree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.handler import (
    SHA_TOKEN,
    NodeSiblingShaPinRatchetCompute,
    count_pins,
)
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.models import (
    ModelSiblingShaPinRatchetInput,
)


class ShaPinRatchetError(Exception):
    """The ratchet cannot read the tree or the base."""


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def sha_lines(repo_root: Path, ref: str | None) -> tuple[tuple[str, str], ...]:
    """(path, line) for every tracked line holding a 40-hex token, at ``ref`` or in the tree."""
    args = ["grep", "-I", "-z", "--full-name", "--no-color", "-E", SHA_TOKEN]
    if ref:
        args.append(ref)
    args += ["--", "."]
    result = _git(repo_root, *args)
    if result.returncode not in (0, 1):
        raise ShaPinRatchetError(
            f"git grep failed ({result.returncode}): {result.stderr.strip()}"
        )
    prefix = f"{ref}:" if ref else ""
    out: list[tuple[str, str]] = []
    for raw in result.stdout.splitlines():
        path, sep, line = raw.partition("\0")
        if prefix and path.startswith(prefix):
            path = path[len(prefix) :]
        if sep:
            out.append((path, line))
    return tuple(out)


def collect_request(repo_root: Path, base: str) -> ModelSiblingShaPinRatchetInput:
    if (
        _git(
            repo_root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"
        ).returncode
        != 0
    ):
        raise ShaPinRatchetError(f"base ref {base} does not resolve")
    return ModelSiblingShaPinRatchetInput(
        base_ref=base,
        base_lines=sha_lines(repo_root, base),
        lines=sha_lines(repo_root, None),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Shrink-only ratchet on internal-sibling commit-sha pins (OMN-18594)."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--base", default="origin/dev", help="the base ref; its pins are the baseline"
    )
    parser.add_argument(
        "--count", action="store_true", help="print the working tree's per-file counts"
    )
    args = parser.parse_args(argv)
    try:
        if args.count:
            counts = count_pins(sha_lines(args.repo_root, None))
            for path in sorted(counts):
                sys.stdout.write(f"{counts[path]:4d} {path}\n")
            sys.stdout.write(
                f"{sum(counts.values()):4d} total in {len(counts)} files\n"
            )
            return 0
        report = NodeSiblingShaPinRatchetCompute().handle(
            collect_request(args.repo_root, args.base)
        )
    except ShaPinRatchetError as exc:
        sys.stderr.write(f"FAIL: {exc}\n")
        return 2
    if report.overall_status == "FAIL":
        sys.stderr.write(
            "FAIL: internal-sibling commit-sha pins may only fall (OMN-18594):\n"
        )
        for finding in report.findings:
            sys.stderr.write(f"  - {finding.message} Fix: {finding.remediation}\n")
        return 1
    sys.stdout.write(
        f"PASS: no file gained an internal-sibling commit-sha pin against {args.base}.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
