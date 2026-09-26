# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Choose the file scope for a per-file pre-commit hook run in CI (OMN-19613).

Prints the ``pre-commit run`` scope arguments on stdout, one per line: either
``--all-files``, or ``--from-ref <base>`` and ``--to-ref <head>``. The reason
for the choice goes to stderr so the job log says why it scanned what it did.

Why this exists
---------------
The ``Lint`` job runs ``no-plugin-daemon-classes`` (an AST check on each file's
own classes, ``omnibase_core.validators.no_plugin_daemon_classes``) over the
whole tree on every PR: 24 s on the critical path, run 35616533588. A per-file
check cannot be broken by a file the PR did not touch, so on a pull request
only the PR's own files need scanning, with these exceptions, each of which
scans every file:

* Any event other than ``pull_request``: push (which is what proves the union
  of merged PRs on ``dev``, OMN-18835), merge_group and workflow_dispatch.
* The PR changes the hook's configuration (``.pre-commit-config.yaml``) or the
  workflow that runs it (``.github/workflows/ci.yml``).
* The PR moves the ``omnibase-core`` package in ``uv.lock``. The validator's
  code lives in ``omnibase_core``, so a new pin is a new validator, and a new
  validator has to see the whole tree once.
* The base cannot be resolved. On ``pull_request`` the checkout is GitHub's
  merge commit, whose first parent is the base-branch tip it was merged onto
  and whose second parent is the PR head, so ``HEAD^1..HEAD`` is exactly what
  the PR changes on top of its base. The Lint checkout fetches depth 2 so the
  first parent is present. If HEAD is not a two-parent merge commit, or git
  cannot answer, the scope fails closed to every file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ALL_FILES = ("--all-files",)

# A change to any of these re-validates the whole tree.
FULL_RUN_PATHS = frozenset(
    {
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
    }
)

LOCKFILE = "uv.lock"
PINNED_PACKAGE = "omnibase-core"

GitRunner = Callable[[Sequence[str]], str]


class GitError(RuntimeError):
    """git could not answer; the caller falls back to every file."""


def run_git(args: Sequence[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


@dataclass(frozen=True)
class Scope:
    args: tuple[str, ...]
    reason: str


def _locked_package(lock_text: str, name: str) -> dict[str, object] | None:
    data = tomllib.loads(lock_text)
    for package in data.get("package", []):
        if isinstance(package, dict) and package.get("name") == name:
            return {k: package.get(k) for k in ("version", "source")}
    return None


def core_pin_moved(base: str, head: str, git: GitRunner) -> bool:
    """True when ``omnibase-core``'s locked version or source differs."""
    before = _locked_package(git(["show", f"{base}:{LOCKFILE}"]), PINNED_PACKAGE)
    after = _locked_package(git(["show", f"{head}:{LOCKFILE}"]), PINNED_PACKAGE)
    return before != after


def decide(event_name: str, git: GitRunner = run_git) -> Scope:
    if event_name != "pull_request":
        return Scope(ALL_FILES, f"event {event_name!r} is not pull_request: full tree")
    try:
        head = git(["rev-parse", "--verify", "HEAD"]).strip()
        base = git(["rev-parse", "--verify", "HEAD^1"]).strip()
        git(["rev-parse", "--verify", "HEAD^2"])
    except GitError as exc:
        return Scope(
            ALL_FILES, f"base unresolvable, failing closed to the full tree: {exc}"
        )
    try:
        changed = {
            p for p in git(["diff", "--name-only", base, head]).splitlines() if p
        }
        if changed & FULL_RUN_PATHS:
            hit = ", ".join(sorted(changed & FULL_RUN_PATHS))
            return Scope(ALL_FILES, f"PR changes {hit}: full tree")
        if LOCKFILE in changed and core_pin_moved(base, head, git):
            return Scope(
                ALL_FILES, f"PR moves the {PINNED_PACKAGE} pin in {LOCKFILE}: full tree"
            )
    except (GitError, tomllib.TOMLDecodeError) as exc:
        return Scope(
            ALL_FILES,
            f"could not read the PR diff, failing closed to the full tree: {exc}",
        )
    return Scope(
        ("--from-ref", base, "--to-ref", head),
        f"pull_request: only files changed between base {base[:12]} and merge {head[:12]}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-name", required=True, help="github.event_name")
    ns = parser.parse_args(argv)
    scope = decide(ns.event_name)
    print(f"pre-commit scope: {scope.reason}", file=sys.stderr)
    print("\n".join(scope.args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
