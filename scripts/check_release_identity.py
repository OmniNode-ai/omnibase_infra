#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fresh-deploy fitness gate: release identity / no-merge-onto-published (OMN-13412).

Thin CLI collector/shim over the canonical ``node_release_identity_compute`` COMPUTE
node (OMN-14471). This file performs ONLY the I/O — reading ``pyproject.toml``,
listing published git tags, and diffing changed files against a base — then hands
the collected facts to the pure ``HandlerReleaseIdentity`` and renders its decision
(exit code + message). All gate LOGIC lives in the node/handler; this shim exists so
the two existing wiring points (fresh-deploy-fitness CI + the pre-commit hook) keep
invoking the same command path with identical behavior.

A runtime image stamps ``org.opencontainers.image.version`` from ``pyproject.toml``.
If source under ``src/`` changes but the version is NOT bumped past the most recently
*published* version (the latest ``vX.Y.Z`` git tag), then two distinct builds ship
the SAME version string and every downstream proof packet that cites the runtime
version can no longer distinguish them. This gate makes the version bump mandatory so
the human-facing version never silently aliases two code states.

What it enforces
----------------
When the diff under inspection touches packaged source (``src/**``) AND any published
tag exists, ``pyproject.toml``'s ``project.version`` MUST be strictly greater than the
highest published version (latest ``v*`` / bare-semver tag).

Modes
-----
* ``--base <ref>``    Compare the working tree against ``<ref>`` (e.g. ``origin/dev``)
                      to decide whether packaged source changed. CI passes the PR
                      base. If omitted, the gate assumes source MAY have changed and
                      always enforces the version-ahead invariant.
* ``--changed-file``  Explicit changed-file list (repeatable / newline list on stdin
                      via ``-``). Overrides ``--base`` diffing.

A docs-only / tests-only / CI-only diff (no ``src/**`` change) is exempt: the
published image is unaffected, so no bump is required.

Usage::

    uv run python scripts/check_release_identity.py --base origin/dev
    uv run python scripts/check_release_identity.py   # strict: always require ahead

Exit codes:
    0 — version is correctly ahead of the latest published tag (or exempt diff)
    1 — packaged source changed without bumping past the latest published version
    2 — configuration error (no pyproject version, malformed version/tag)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from omnibase_infra.nodes.node_release_identity_compute import (
    HandlerReleaseIdentity,
    ModelReleaseIdentityRequest,
)


def _git(args: list[str]) -> str:
    """Run a git command in the repo root and return trimmed stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _read_pyproject_version_raw(pyproject: Path) -> str | None:
    """Read the raw ``project.version`` string from ``pyproject.toml``.

    Returns the value as a string, or ``None`` when the key is absent. Parsing and
    validation (empty/malformed) are the handler's job, so the shim stays pure-I/O.
    """
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    raw = data.get("project", {}).get("version")
    return None if raw is None else str(raw)


def _collect_changed_files(
    base: str | None, explicit: list[str]
) -> tuple[str, ...] | None:
    """Collect the changed-file set for the diff.

    Returns an explicit list when provided; otherwise diffs against ``base``
    (three-dot, falling back to two-dot when the merge-base form is empty). Returns
    ``None`` when neither a base nor an explicit list is available — the handler
    then enforces the invariant (cannot prove the diff is exempt).
    """
    if explicit:
        return tuple(explicit)
    if base:
        diff = _git(["diff", "--name-only", f"{base}...HEAD"])
        files = [f for f in diff.splitlines() if f.strip()]
        if not files:
            # Fall back to a two-dot diff if the merge-base form yielded nothing.
            diff = _git(["diff", "--name-only", base])
            files = [f for f in diff.splitlines() if f.strip()]
        return tuple(files)
    return None


def collect_request(
    base: str | None, explicit: list[str]
) -> ModelReleaseIdentityRequest:
    """Gather all I/O facts into the handler's typed request."""
    out = _git(["tag", "--list"])
    published_tags = tuple(out.splitlines()) if out else ()
    return ModelReleaseIdentityRequest(
        pyproject_version_raw=_read_pyproject_version_raw(_PYPROJECT),
        pyproject_path=str(_PYPROJECT),
        published_tags=published_tags,
        changed_files=_collect_changed_files(base, explicit),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help="Git ref to diff against (e.g. origin/dev) to detect src/ changes.",
    )
    parser.add_argument(
        "--changed-file",
        dest="changed_files",
        action="append",
        default=[],
        help="Explicit changed file (repeatable). Overrides --base diffing.",
    )
    args = parser.parse_args(argv)

    explicit = list(args.changed_files)
    if explicit == ["-"]:
        explicit = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]

    request = collect_request(args.base, explicit)
    decision = HandlerReleaseIdentity().handle(request)

    stream = sys.stderr if decision.stream == "stderr" else sys.stdout
    print(decision.message, file=stream)
    return decision.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
