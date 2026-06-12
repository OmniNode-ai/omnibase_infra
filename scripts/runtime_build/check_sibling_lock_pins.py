#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workspace-build sibling-pin preflight + recurrence ratchet [OMN-12977].

The workspace-mode image build (BUILD_SOURCE=workspace) vendors sibling repos
from whatever the canonical clones under OMNI_HOME happen to be checked out at.
On 2026-06-11 that shipped a 13-day-stale omnibase_infra 0.37.0-dev (pre-OMN-12501
Protocol-quarantine guard) + omnibase_core 0.42.0 even though omnimarket dev's
``uv.lock`` pinned omnibase-infra 0.38.1 @ e2dbdc95 / omnibase-core 0.44.0 @
c97c2c9a. The stale image dropped the guard and crashed bootstrap fatally.

This module makes the consuming repo's ``uv.lock`` the pin authority:

  - ``parse_lock_pins`` extracts the expected version (+ git rev) of the
    foundation/sibling packages from the consuming repo's lock.
  - ``resolve_actual_pin`` reads the version (from pyproject) + HEAD git SHA of
    a canonical clone that the build is about to vendor.
  - ``compare_pins`` reports match / mismatch with an identifying reason.
  - ``check_pins`` orchestrates a fail-fast preflight and writes the
    expected-vs-actual comparison so deploy verifiers and build-provenance.json
    can assert on it.

There are NO silent fallbacks: a requested package missing from the lock, an
unparseable source, or any version / SHA drift is a hard error. Run BEFORE the
docker build (preflight) and again inside the build to enrich provenance.

Usage:
    uv run python scripts/runtime_build/check_sibling_lock_pins.py \\
        --lock "$OMNI_HOME/omnimarket/uv.lock" \\
        --repo omnibase-infra="$OMNI_HOME/omnibase_infra" \\
        --repo omnibase-core="$OMNI_HOME/omnibase_core" \\
        --output workspace/sibling-pin-comparison.json

Exit codes:
    0  every requested pin matches the lock
    1  one or more pins drifted from the lock (build must abort)
    2  malformed input (missing lock, unparseable source, missing repo)

OMN-12977
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Foundation + sibling packages whose pins the workspace build must honor.
# Keys are uv.lock distribution names; values are the canonical clone dir names
# under OMNI_HOME that the build vendors / installs from.
DEFAULT_PACKAGE_REPO_DIRS: dict[str, str] = {
    "omnibase-infra": "omnibase_infra",
    "omnibase-core": "omnibase_core",
    "omnibase-spi": "omnibase_spi",
    "omnibase-compat": "omnibase_compat",
    "onex-change-control": "onex_change_control",
    "omnimarket": "omnimarket",
}

# Matches a package's own ``source = { ... }`` line in a uv.lock block. The rev
# (when the source is a git source) is captured from the ``?rev=<sha>`` query.
# Scoping to the source line is required: a ``dependencies = [...]`` entry may
# also carry a ``?rev=`` for a transitive git dep, and a block-wide search would
# wrongly attribute that rev to an editable/registry package (e.g. omnimarket
# itself is ``source = { editable = "." }`` with no rev of its own).
_SOURCE_LINE_RE = re.compile(r"^source = \{(?P<body>.*)\}\s*$", re.MULTILINE)
_GIT_REV_RE = re.compile(r"[?&]rev=(?P<rev>[0-9a-fA-F]{7,40})")


class ModelLockPin(BaseModel):
    """Expected pin for a package as declared in the consuming repo's uv.lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package: str = Field(..., description="uv.lock distribution name")
    version: str = Field(..., description="Locked version string")
    git_rev: str | None = Field(
        default=None,
        description="Locked git rev (full/short SHA) or None for registry pins",
    )


class ModelActualPin(BaseModel):
    """Actual pin of a canonical clone the build is about to vendor/install."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package: str = Field(..., description="uv.lock distribution name")
    version: str = Field(..., description="Version from the clone's pyproject.toml")
    git_sha: str = Field(..., description="HEAD commit SHA of the canonical clone")


class ModelPinComparison(BaseModel):
    """Expected-vs-actual comparison for one package (provenance-serializable)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package: str = Field(...)
    expected_version: str = Field(...)
    actual_version: str = Field(...)
    expected_git_rev: str | None = Field(default=None)
    actual_git_sha: str = Field(...)
    matches: bool = Field(...)
    drift_direction: str = Field(
        default="none",
        description=(
            "'none' (match), 'backward' (clone older than lock -- the stale-"
            "sibling crash), 'forward' (clone newer than lock), or 'unknown' "
            "(non-comparable versions)"
        ),
    )
    mismatch_reason: str = Field(default="", description="Empty when matches is True")


def _classify_drift(expected_version: str, actual_version: str) -> str:
    """Classify version drift direction (backward is the stale-sibling crash)."""
    if expected_version == actual_version:
        return "none"
    try:
        exp = tuple(int(p) for p in expected_version.split(".")[:3])
        act = tuple(int(p) for p in actual_version.split(".")[:3])
    except ValueError:
        return "unknown"
    if act < exp:
        return "backward"
    if act > exp:
        return "forward"
    return "unknown"


def parse_lock_pins(lock_text: str, packages: list[str]) -> dict[str, ModelLockPin]:
    """Parse a uv.lock and return expected pins for the requested packages.

    Each ``[[package]]`` block carries a ``name``, ``version`` and ``source``.
    Git sources embed the resolved rev in the source URL; registry sources have
    no rev (version-only pin). A requested package absent from the lock is a
    hard error -- workspace builds must never silently skip a foundation pin.
    """
    blocks = lock_text.split("[[package]]")
    found: dict[str, ModelLockPin] = {}
    requested = set(packages)

    for block in blocks:
        name_match = re.search(r'^name = "(?P<name>[^"]+)"', block, re.MULTILINE)
        if name_match is None:
            continue
        name = name_match.group("name")
        if name not in requested:
            continue

        version_match = re.search(
            r'^version = "(?P<version>[^"]+)"', block, re.MULTILINE
        )
        if version_match is None:
            raise ValueError(f"uv.lock block for {name!r} has no version field")
        version = version_match.group("version")

        # Only the package's OWN source line may contribute a git rev. Editable
        # ("." path) and registry (PyPI) sources have no rev.
        git_rev: str | None = None
        source_match = _SOURCE_LINE_RE.search(block)
        if source_match is not None:
            rev_match = _GIT_REV_RE.search(source_match.group("body"))
            if rev_match is not None:
                git_rev = rev_match.group("rev")

        found[name] = ModelLockPin(package=name, version=version, git_rev=git_rev)

    missing = requested - set(found)
    if missing:
        raise KeyError(
            f"packages absent from uv.lock: {sorted(missing)}. "
            "Workspace build cannot resolve a pin for them."
        )
    return found


def resolve_actual_pin(package: str, repo_root: Path) -> ModelActualPin:
    """Read the canonical clone's version (pyproject) + HEAD SHA.

    No defaults: a missing pyproject, a missing version, or an unreadable git
    HEAD is a hard error (exit 2 at the CLI boundary).
    """
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise FileNotFoundError(
            f"missing pyproject.toml for {package}: {pyproject_path}"
        )
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        raise ValueError(f"no project.version in {pyproject_path}")

    head_sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head_sha:
        raise ValueError(f"could not resolve git HEAD for {repo_root}")

    return ModelActualPin(package=package, version=version, git_sha=head_sha)


def compare_pins(expected: ModelLockPin, actual: ModelActualPin) -> ModelPinComparison:
    """Compare an expected lock pin against the actual clone pin.

    Version must always match. When the lock pin is a git source, the resolved
    rev must be a prefix of the clone HEAD (or vice-versa) -- uv records the full
    rev while a short SHA may appear elsewhere, so prefix-equality covers both.
    Registry pins (git_rev is None) are version-only.
    """
    reasons: list[str] = []

    if expected.version != actual.version:
        reasons.append(
            f"version drift: lock pins {expected.version!r} but clone is "
            f"{actual.version!r}"
        )

    if expected.git_rev is not None:
        exp = expected.git_rev.lower()
        act = actual.git_sha.lower()
        if not (act.startswith(exp) or exp.startswith(act)):
            reasons.append(
                f"git SHA drift: lock pins {expected.git_rev} but clone HEAD is "
                f"{actual.git_sha}"
            )

    matches = not reasons
    return ModelPinComparison(
        package=expected.package,
        expected_version=expected.version,
        actual_version=actual.version,
        expected_git_rev=expected.git_rev,
        actual_git_sha=actual.git_sha,
        matches=matches,
        drift_direction=_classify_drift(expected.version, actual.version),
        mismatch_reason="; ".join(reasons),
    )


def check_pins(
    lock_path: Path,
    repo_roots: dict[str, Path],
    output_path: Path | None,
    allow_drift: bool = False,
) -> int:
    """Preflight: compare every requested clone against the lock; fail fast.

    When ``allow_drift`` is True the drift is recorded in the provenance artifact
    and a warning is printed, but the build is not aborted -- reserved for an
    explicit operator decision (e.g. an intentional forward rebuild ahead of a
    lock bump). The override is always durable in the artifact, never silent.

    Returns a process exit code (0 pass-or-allowed / 1 drift / 2 malformed input).
    """
    if not lock_path.is_file():
        print(f"ERROR: lock file not found: {lock_path}", file=sys.stderr)
        return 2

    packages = list(repo_roots)
    try:
        expected_pins = parse_lock_pins(
            lock_path.read_text(encoding="utf-8"), packages=packages
        )
    except (KeyError, ValueError) as exc:
        print(f"ERROR: cannot resolve lock pins: {exc}", file=sys.stderr)
        return 2

    comparisons: list[ModelPinComparison] = []
    for package, repo_root in repo_roots.items():
        try:
            actual = resolve_actual_pin(package, repo_root)
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
            print(
                f"ERROR: cannot resolve clone pin for {package}: {exc}", file=sys.stderr
            )
            return 2
        comparisons.append(compare_pins(expected_pins[package], actual))

    drifted = [c for c in comparisons if not c.matches]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "lock_source": str(lock_path),
                    "allow_drift": allow_drift,
                    "drift_count": len(drifted),
                    "comparisons": [c.model_dump() for c in comparisons],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"sibling-pin comparison written to {output_path}")

    for c in comparisons:
        status = "OK" if c.matches else f"DRIFT/{c.drift_direction}"
        line = f"  [{status}] {c.package}: lock={c.expected_version} clone={c.actual_version}"
        print(
            line if c.matches else f"{line}  -- {c.mismatch_reason}",
            file=sys.stderr if not c.matches else sys.stdout,
        )

    if drifted:
        if allow_drift:
            print(
                f"\nWARNING: {len(drifted)} sibling pin(s) drifted from "
                f"{lock_path.name}; proceeding under explicit operator override "
                "(ALLOW drift) -- recorded in provenance (OMN-12977).",
                file=sys.stderr,
            )
            return 0
        print(
            f"\nFAIL: {len(drifted)} sibling pin(s) drifted from {lock_path.name}. "
            "Workspace build would vendor stale code -- aborting (OMN-12977).",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(comparisons)} sibling pins match {lock_path.name}.")
    return 0


def _parse_repo_arg(value: str) -> tuple[str, Path]:
    """Parse a ``package=path`` --repo argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--repo must be package=path, got {value!r}")
    package, _, path = value.partition("=")
    package = package.strip()
    if package not in DEFAULT_PACKAGE_REPO_DIRS:
        raise argparse.ArgumentTypeError(
            f"unknown package {package!r}; expected one of "
            f"{sorted(DEFAULT_PACKAGE_REPO_DIRS)}"
        )
    return package, Path(path).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        required=True,
        type=Path,
        help="Path to the consuming repo's uv.lock (the pin authority)",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        type=_parse_repo_arg,
        metavar="PACKAGE=PATH",
        help="Canonical clone the build will vendor (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the expected-vs-actual comparison JSON",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help=(
            "Record drift in the provenance artifact and proceed instead of "
            "aborting. Explicit operator override only -- never the default."
        ),
    )
    args = parser.parse_args(argv)

    if not args.repo:
        print("ERROR: at least one --repo PACKAGE=PATH is required", file=sys.stderr)
        return 2

    repo_roots = dict(args.repo)
    return check_pins(args.lock, repo_roots, args.output, allow_drift=args.allow_drift)


if __name__ == "__main__":
    sys.exit(main())
