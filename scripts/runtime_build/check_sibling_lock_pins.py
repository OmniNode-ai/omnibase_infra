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

There are NO silent fallbacks: a requested package missing from the lock or an
unparseable source is a hard error. SHA drift is FATAL by default, with ONE
narrow exception (OMN-13403): a sibling whose clone version EXACTLY equals the
locked version and whose clone HEAD is a git DESCENDANT of the locked SHA -- i.e.
the clone is strictly AHEAD of the lock -- is recorded as a non-fatal
``clone_ahead`` note rather than aborting the build. This is the unavoidable
steady state for a real runtime sibling like ``onex_change_control`` whose dev
branch advances on every receipt PR (including the receipt PRs that pin-bump PRs
themselves require), so an exact lock pin can never durably converge. The
2026-06-11 crash this guard prevents was the OPPOSITE direction -- a STALE clone
that was BEHIND the lock with a MISMATCHED version -- which stays FATAL.

Registry-sourced forward-drift exception (OMN-13902 / OMN-14138): a WORKSPACE-mode
build vendors the sibling *clone*, not the lock-pinned wheel, so a clone that is
NEWER than the lock is the intended state -- not drift. OMN-13929's post-publish
release-identity disarm bumps a sibling's DEV ``pyproject`` version to N+1 the
instant version N is published to PyPI, so every registry-sourced sibling's dev
clone is PERMANENTLY one version ahead of the last-published pin any consumer's
``uv.lock`` can carry. Blocking that ``[DRIFT/forward]`` state aborted every
workspace runtime build from dev. In workspace mode a registry-sourced pin
(``git_rev`` is None, so there is no SHA to prove ancestry with) whose clone
version is strictly FORWARD of the lock is therefore recorded as a non-fatal
``forward_drift_allowed`` note -- ``matches`` stays True, the build proceeds. This
is the registry-sourced analogue of the git-rev ``clone_ahead`` exemption above;
version ordering is the only "clone is ahead" signal a registry pin exposes.
BACKWARD drift (stale clone -- the 2026-06-11 crash shape) and any non-comparable
("unknown") version stay FATAL, and git-sourced pins keep the stronger SHA-
ancestry proof (a forward VERSION bump on a git pin stays fatal: it means the
immutable rev pin is stale and must be re-bumped, not steady-state noise). The
exception applies ONLY in workspace mode: a RELEASE build (``--build-source
release``, the strict default) installs the published wheel, so forward drift
there is a real skew and stays FATAL. Run BEFORE the docker build (preflight) and
again inside the build to enrich provenance.

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
    clone_ahead: bool = Field(
        default=False,
        description=(
            "True (OMN-13403) when the version matches exactly but the clone HEAD "
            "is a strict git DESCENDANT of the locked SHA. This is a non-fatal "
            "note, not drift: ``matches`` stays True and the build proceeds."
        ),
    )
    forward_drift_allowed: bool = Field(
        default=False,
        description=(
            "True (OMN-13902) when a WORKSPACE-mode build sees a REGISTRY-sourced "
            "pin whose clone version is strictly FORWARD of the lock (the "
            "OMN-13929 disarm-bump / unreleased-dev steady state). The workspace "
            "build vendors the newer clone, so this is a non-fatal note, not "
            "drift: ``matches`` stays True and the build proceeds. Registry-sourced "
            "analogue of ``clone_ahead``; never set in release mode or for a "
            "git-sourced pin."
        ),
    )
    note: str = Field(
        default="",
        description="Non-fatal advisory (e.g. the clone-ahead descendant note)",
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


def _is_clone_strictly_ahead(
    locked_sha: str, clone_head_sha: str, repo_root: Path
) -> bool:
    """Return True iff the clone HEAD is a strict git DESCENDANT of the locked SHA.

    "Strictly ahead" (OMN-13403) means: the locked SHA is reachable from the
    clone HEAD AND the two are not the same commit. Implemented with
    ``git merge-base --is-ancestor <locked_sha> <clone_head>`` evaluated against
    the ACTUAL clone repo (not the lock or any other tree).

    Returns False -- the conservative answer that keeps the comparison FATAL --
    in every non-ahead case:
      * the locked SHA is not present in the clone (unrelated history / not
        fetched): ``git`` errors with a non-zero, non-1 exit code;
      * the clone is BEHIND the lock (clone HEAD is an ancestor of the locked
        SHA, or unrelated): ``--is-ancestor`` returns exit code 1;
      * identical SHAs (handled by the caller as a plain match, never here).
    """
    locked = locked_sha.strip().lower()
    head = clone_head_sha.strip().lower()
    if not locked or not head:
        return False
    # Identical commits are an exact match, not "ahead". Guard prefix-equality
    # too: uv records the full rev while a short SHA may appear elsewhere.
    if head.startswith(locked) or locked.startswith(head):
        return False

    # The locked SHA must be a real object in this clone; an unknown rev means we
    # cannot prove ancestry and must stay fatal.
    # OMN-14900: safe.directory scoping so a uid-mismatched invoker (deploy
    # runner) is never rejected with "detected dubious ownership".
    cat = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "cat-file",
            "-e",
            f"{locked}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if cat.returncode != 0:
        return False

    # exit 0 => locked is an ancestor of head (clone ahead); exit 1 => it is not
    # (clone behind / unrelated). Any other exit code is an error -> not ahead.
    ancestry = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            locked,
            head,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return ancestry.returncode == 0


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
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head_sha:
        raise ValueError(f"could not resolve git HEAD for {repo_root}")

    return ModelActualPin(package=package, version=version, git_sha=head_sha)


def compare_pins(
    expected: ModelLockPin,
    actual: ModelActualPin,
    repo_root: Path | None = None,
    workspace_mode: bool = False,
) -> ModelPinComparison:
    """Compare an expected lock pin against the actual clone pin.

    A BACKWARD version mismatch (clone older than the lock) is fatal and is never
    softened (that was the 2026-06-11 stale-sibling crash). A non-comparable
    ("unknown") version mismatch is also fatal.

    When the lock pin is a git source, the resolved rev must be a prefix of the
    clone HEAD (or vice-versa) -- uv records the full rev while a short SHA may
    appear elsewhere, so prefix-equality covers both. Registry pins (git_rev is
    None) are version-only.

    OMN-13403 clone-ahead exception: when the version matches EXACTLY and the
    clone HEAD is a strict git DESCENDANT of the locked SHA (clone strictly
    AHEAD), the SHA difference is recorded as a non-fatal ``clone_ahead`` note --
    ``matches`` stays True, the build proceeds. This requires ``repo_root`` so
    ancestry can be checked against the actual clone; without it, or when the
    locked SHA is absent / the clone is behind / histories are unrelated, the SHA
    difference stays FATAL.

    OMN-13902 registry forward-drift exception: when ``workspace_mode`` is True
    and a REGISTRY-sourced pin (``git_rev`` is None) has a clone version strictly
    FORWARD of the lock, the difference is recorded as a non-fatal
    ``forward_drift_allowed`` note -- ``matches`` stays True, the build proceeds.
    A workspace build vendors the newer clone, and the OMN-13929 disarm-bump makes
    every registry-sourced dev clone permanently one version ahead of the last-
    published pin, so this forward state is the intended steady state, not drift.
    The exception applies ONLY in workspace mode (a release build installs the
    published wheel, so forward drift there is a real skew) and ONLY to registry-
    sourced pins (git-sourced pins keep the stronger SHA-ancestry proof above).
    """
    reasons: list[str] = []

    drift_direction = _classify_drift(expected.version, actual.version)
    version_matches = expected.version == actual.version
    forward_drift_allowed = False

    clone_ahead = False
    note = ""

    if not version_matches:
        # Registry-sourced (git_rev is None) forward drift in a workspace build is
        # the OMN-13929 disarm-bump steady state, not a stale-sibling crash: the
        # build vendors the newer clone, so allow it. Backward / unknown drift, and
        # any drift on a git-sourced pin, stay fatal.
        if workspace_mode and drift_direction == "forward" and expected.git_rev is None:
            forward_drift_allowed = True
            note = (
                f"forward-drift allowed (OMN-13902): workspace build vendors clone "
                f"{actual.version}, which is forward of the registry-sourced lock "
                f"pin {expected.version} (OMN-13929 disarm-bump / unreleased-dev "
                "steady state). Recorded, non-fatal."
            )
        else:
            reasons.append(
                f"version drift: lock pins {expected.version!r} but clone is "
                f"{actual.version!r}"
            )

    if expected.git_rev is not None:
        exp = expected.git_rev.lower()
        act = actual.git_sha.lower()
        sha_matches = act.startswith(exp) or exp.startswith(act)
        if not sha_matches:
            # The clone-ahead exception ONLY applies when the version is an exact
            # match -- a version mismatch is fatal regardless of SHA ancestry.
            if (
                version_matches
                and repo_root is not None
                and _is_clone_strictly_ahead(exp, act, repo_root)
            ):
                clone_ahead = True
                note = (
                    f"clone-ahead (OMN-13403): version {actual.version} matches the "
                    f"lock; clone HEAD {actual.git_sha} is a descendant of locked "
                    f"{expected.git_rev}. Recorded, non-fatal."
                )
            else:
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
        drift_direction=drift_direction,
        clone_ahead=clone_ahead,
        forward_drift_allowed=forward_drift_allowed,
        note=note,
        mismatch_reason="; ".join(reasons),
    )


def check_pins(
    lock_path: Path,
    repo_roots: dict[str, Path],
    output_path: Path | None,
    allow_drift: bool = False,
    workspace_mode: bool = False,
) -> int:
    """Preflight: compare every requested clone against the lock; fail fast.

    When ``workspace_mode`` is True a registry-sourced pin whose clone version is
    strictly FORWARD of the lock is a non-fatal ``forward_drift_allowed`` note
    (OMN-13902): the workspace build vendors the newer clone and the OMN-13929
    disarm-bump makes that the steady state, so it is not aborting drift. Backward
    drift and git-sourced drift stay fatal. In release mode (the default) any
    version mismatch is fatal.

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
        comparisons.append(
            compare_pins(
                expected_pins[package],
                actual,
                repo_root=repo_root,
                workspace_mode=workspace_mode,
            )
        )

    drifted = [c for c in comparisons if not c.matches]
    clone_ahead = [c for c in comparisons if c.clone_ahead]
    forward_allowed = [c for c in comparisons if c.forward_drift_allowed]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "lock_source": str(lock_path),
                    "workspace_mode": workspace_mode,
                    "allow_drift": allow_drift,
                    "drift_count": len(drifted),
                    "clone_ahead_count": len(clone_ahead),
                    "forward_drift_allowed_count": len(forward_allowed),
                    "comparisons": [c.model_dump() for c in comparisons],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"sibling-pin comparison written to {output_path}")

    for c in comparisons:
        if not c.matches:
            status = f"DRIFT/{c.drift_direction}"
        elif c.clone_ahead:
            status = "AHEAD"
        elif c.forward_drift_allowed:
            status = "FWD-OK"
        else:
            status = "OK"
        line = f"  [{status}] {c.package}: lock={c.expected_version} clone={c.actual_version}"
        if not c.matches:
            print(f"{line}  -- {c.mismatch_reason}", file=sys.stderr)
        elif c.clone_ahead or c.forward_drift_allowed:
            print(f"{line}  -- {c.note}")
        else:
            print(line)

    if clone_ahead:
        print(
            f"\nNOTE: {len(clone_ahead)} sibling pin(s) are clone-AHEAD of "
            f"{lock_path.name} (same version, clone HEAD descends from the locked "
            "SHA); non-fatal, recorded in provenance (OMN-13403).",
            file=sys.stderr,
        )

    if forward_allowed:
        print(
            f"\nNOTE: {len(forward_allowed)} registry-sourced sibling pin(s) are "
            f"forward of {lock_path.name} (clone version newer than the lock); "
            "non-fatal in workspace mode, recorded in provenance (OMN-13902).",
            file=sys.stderr,
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
    parser.add_argument(
        "--build-source",
        choices=("workspace", "release"),
        default="release",
        help=(
            "Build mode. 'workspace' vendors the sibling clones, so a registry-"
            "sourced pin whose clone version is FORWARD of the lock is the "
            "OMN-13929 disarm-bump steady state and is non-fatal (OMN-13902). "
            "'release' (default, strict) installs the published wheel, so any "
            "version mismatch stays fatal. Backward / git-sourced drift is fatal "
            "in both modes."
        ),
    )
    args = parser.parse_args(argv)

    if not args.repo:
        print("ERROR: at least one --repo PACKAGE=PATH is required", file=sys.stderr)
        return 2

    repo_roots = dict(args.repo)
    return check_pins(
        args.lock,
        repo_roots,
        args.output,
        allow_drift=args.allow_drift,
        workspace_mode=(args.build_source == "workspace"),
    )


if __name__ == "__main__":
    sys.exit(main())
