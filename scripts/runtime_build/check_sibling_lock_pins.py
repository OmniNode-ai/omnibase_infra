#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-fast preflight: vendored sibling versions/SHAs must match the lock pin.

Surfaced by the 2026-06-11 resolver-crash diagnosis (OMN-12987). A workspace-mode
``--no-cache`` rebuild vendored omnibase_infra ``0.37.0-dev`` (~``2c1d672f``) +
omnibase_core ``0.42.0`` even though omnimarket dev's ``uv.lock`` pinned infra
``0.38.1 @ e2dbdc95`` + core ``0.44.0 @ c97c2c9a``. The build silently shipped a
13-day-stale sibling that predated the OMN-12501 Protocol-quarantine guard,
turning a latent contract defect into a fatal bootstrap crash.

This preflight resolves the *expected* sibling pins from the consuming repo's lock
(the authority) and compares them against the *actual* version/SHA of each repo
the build vendors. Any mismatch aborts the build — there is no silent fallback.

The comparison is also serialised so the deploy verifier (and
``build-provenance.json``) can record expected-vs-actual per sibling.

Run on the host before staging/building (it needs the canonical clones' git SHAs):

    OMNI_HOME=/data/omninode/omni_home \\
      uv run python scripts/runtime_build/check_sibling_lock_pins.py \\
        --consuming-repo omnimarket \\
        --build-context-repo omnibase_infra

Exit codes:
  0  all vendored siblings match the consuming repo's lock pin
  1  one or more siblings mismatch (version and/or SHA)
  2  malformed inputs (missing lock, missing pyproject, unreadable git SHA)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

# Distribution name (as it appears in uv.lock) -> repo directory name under OMNI_HOME.
# These are the siblings the workspace build either vendors directly
# (compat/onex_change_control/omnimarket) or builds *from* as the build context
# (omnibase_infra). omnibase_core is pinned transitively and vendored via the
# infra wheel, so it is included too: the 06-11 crash shipped a stale core.
PACKAGE_TO_REPO: dict[str, str] = {
    "omnibase-compat": "omnibase_compat",
    "omnibase-core": "omnibase_core",
    "omnibase-infra": "omnibase_infra",
    "onex-change-control": "onex_change_control",
}


@dataclass(frozen=True)
class ExpectedPin:
    """Expected sibling pin resolved from the consuming repo's lock."""

    package: str
    version: str
    git_rev: str  # empty string when the lock pins a non-git (registry) source


@dataclass(frozen=True)
class PinComparison:
    """Expected-vs-actual comparison for a single vendored sibling."""

    package: str
    repo: str
    expected_version: str
    actual_version: str
    expected_git_rev: str
    actual_git_rev: str
    version_match: bool
    sha_match: bool

    @property
    def ok(self) -> bool:
        # SHA is the authoritative pin; version is a secondary signal. When the
        # lock pins a git source we require the SHA to match. When the lock has
        # no git rev (registry source) we fall back to the version match.
        if self.expected_git_rev:
            return self.sha_match
        return self.version_match


def parse_lock_pins(lock_path: Path) -> dict[str, ExpectedPin]:
    """Extract expected sibling pins from a uv.lock TOML file.

    Returns a mapping of distribution name -> ExpectedPin for every sibling in
    PACKAGE_TO_REPO that the lock declares. Siblings absent from the lock are
    simply omitted (the caller decides whether that is an error).
    """
    if not lock_path.is_file():
        raise FileNotFoundError(f"consuming-repo lock not found: {lock_path}")

    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    pins: dict[str, ExpectedPin] = {}
    for pkg in data.get("package", []):
        name = pkg.get("name", "")
        if name not in PACKAGE_TO_REPO:
            continue
        version = str(pkg.get("version", ""))
        git_rev = _extract_git_rev(pkg.get("source", {}))
        pins[name] = ExpectedPin(package=name, version=version, git_rev=git_rev)
    return pins


def _extract_git_rev(source: dict[str, object]) -> str:
    """Return the resolved git commit SHA from a uv.lock source table.

    uv records git sources as ``{"git": "<url>?rev=<sha>#<sha>"}``; the resolved
    commit is the fragment after ``#``. Registry/path sources yield an empty rev.
    """
    git = source.get("git")
    if not isinstance(git, str):
        return ""
    _, _, fragment = git.partition("#")
    return fragment.strip()


def read_actual_version(repo_path: Path) -> str:
    """Read the installed version from a repo's pyproject.toml ``[project]``."""
    pyproject = repo_path / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"pyproject.toml not found for repo: {repo_path}")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version", "")
    if not version:
        raise ValueError(f"no [project].version in {pyproject}")
    return str(version)


def read_actual_git_rev(repo_path: Path) -> str:
    """Return the full HEAD SHA of a repo.

    Prefers a committed ``.build-sha`` marker (staged trees rsync without
    ``.git``); falls back to ``git rev-parse HEAD`` for live canonical clones.
    """
    marker = repo_path / ".build-sha"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot resolve git SHA for {repo_path}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def compare_pin(expected: ExpectedPin, repo_path: Path) -> PinComparison:
    """Build the expected-vs-actual comparison for one sibling."""
    actual_version = read_actual_version(repo_path)
    actual_rev = read_actual_git_rev(repo_path)
    # SHA equality is prefix-tolerant: lock records full 40-char rev, markers or
    # short SHAs may be abbreviated. Compare on the shorter length.
    n = min(len(expected.git_rev), len(actual_rev)) if expected.git_rev else 0
    sha_match = (
        bool(expected.git_rev) and n > 0 and (expected.git_rev[:n] == actual_rev[:n])
    )
    return PinComparison(
        package=expected.package,
        repo=PACKAGE_TO_REPO[expected.package],
        expected_version=expected.version,
        actual_version=actual_version,
        expected_git_rev=expected.git_rev,
        actual_git_rev=actual_rev,
        version_match=expected.version == actual_version,
        sha_match=sha_match,
    )


def evaluate(
    pins: dict[str, ExpectedPin],
    repo_root: Path,
    build_context_repo: str,
    repo_path_overrides: dict[str, Path] | None = None,
) -> list[PinComparison]:
    """Compare every locked sibling pin against the repo the build will vendor.

    ``repo_root`` is OMNI_HOME (canonical clones live directly beneath it).
    ``repo_path_overrides`` lets callers point a package at a staged tree.
    """
    overrides = repo_path_overrides or {}
    comparisons: list[PinComparison] = []
    for package, expected in sorted(pins.items()):
        repo_dir = PACKAGE_TO_REPO[package]
        repo_path = overrides.get(package, repo_root / repo_dir)
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"sibling repo for '{package}' not found at {repo_path}"
            )
        comparisons.append(compare_pin(expected, repo_path))
    # build_context_repo is documented for the operator; the comparison itself
    # already covers it because the infra package is in PACKAGE_TO_REPO.
    _ = build_context_repo
    return comparisons


def render_report(comparisons: list[PinComparison]) -> str:
    """Human-readable summary, one line per sibling."""
    lines: list[str] = []
    for c in comparisons:
        flag = "OK " if c.ok else "MISMATCH"
        lines.append(
            f"  [{flag}] {c.package}: "
            f"expected {c.expected_version}@{c.expected_git_rev[:12] or '(no-rev)'} "
            f"actual {c.actual_version}@{c.actual_git_rev[:12] or '(no-rev)'}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consuming-repo",
        default="omnimarket",
        help="repo whose uv.lock is the pin authority (default: omnimarket)",
    )
    parser.add_argument(
        "--build-context-repo",
        default="omnibase_infra",
        help="repo the image is built FROM (default: omnibase_infra)",
    )
    parser.add_argument(
        "--omni-home",
        default=os.environ.get("OMNI_HOME", ""),
        help="path to OMNI_HOME (canonical clones root). Defaults to $OMNI_HOME.",
    )
    parser.add_argument(
        "--provenance-out",
        default="",
        help="optional path to write the expected-vs-actual JSON report.",
    )
    args = parser.parse_args()

    if not args.omni_home:
        print(
            "ERROR: OMNI_HOME not set; pass --omni-home or export OMNI_HOME.",
            file=sys.stderr,
        )
        return 2
    repo_root = Path(args.omni_home)
    lock_path = repo_root / args.consuming_repo / "uv.lock"

    try:
        pins = parse_lock_pins(lock_path)
        comparisons = evaluate(pins, repo_root, args.build_context_repo)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: sibling lock-pin preflight failed: {exc}", file=sys.stderr)
        return 2

    print(f"Sibling lock-pin preflight (authority: {args.consuming_repo}/uv.lock):")
    print(render_report(comparisons))

    if args.provenance_out:
        Path(args.provenance_out).write_text(
            json.dumps([asdict(c) for c in comparisons], indent=2),
            encoding="utf-8",
        )

    mismatches = [c for c in comparisons if not c.ok]
    if mismatches:
        print(
            f"\nFATAL: {len(mismatches)} vendored sibling(s) do not match the "
            f"{args.consuming_repo} lock pin. Refusing to build a stale image.",
            file=sys.stderr,
        )
        for c in mismatches:
            print(
                f"  - {c.package}: lock pins "
                f"{c.expected_version}@{c.expected_git_rev[:12] or '(no-rev)'} but "
                f"vendored tree is "
                f"{c.actual_version}@{c.actual_git_rev[:12] or '(no-rev)'}",
                file=sys.stderr,
            )
        return 1

    print(f"\nAll {len(comparisons)} vendored siblings match the lock pin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
