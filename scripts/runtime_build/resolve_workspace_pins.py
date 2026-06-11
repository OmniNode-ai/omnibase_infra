#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve + enforce workspace-mode sibling lock pins (OMN-12989).

Background — the 2026-06-11 stability bootstrap crash
-----------------------------------------------------
A workspace-mode ``--no-cache`` rebuild staged its sibling repos from a stale
MERGE-SWEEP deploy worktree instead of the canonical clone, vendoring
omnibase_infra 0.37.0-dev (~2c1d672f, 13 days stale) even though omnimarket
dev's ``uv.lock`` pins infra 0.38.1 @ e2dbdc95. The downgraded sibling
predated the OMN-12501 Protocol-quarantine guard, so a latent contract defect
crashed bootstrap fatally and took the stability lane down.

This module is the fail-fast ratchet for that hole. It:

  * parses the consuming repo's ``uv.lock`` to extract the expected sibling
    pins (version + git rev),
  * compares each staged/installed sibling against its lock pin,
  * classifies the comparison as ``exact`` / ``ahead`` / ``regression`` /
    ``unpinned``,
  * raises ``WorkspacePinError`` (fail-fast, no silent fallback) when any
    sibling has regressed below its lock pin, and
  * exposes a serializable comparison block for ``build-provenance.json`` so
    deploy verifiers can mechanically check it.

Policy: the canonical clones track moving dev branches, so a staged sibling may
legitimately be **at or ahead of** the lock pin (dev advanced past the last
omnimarket lock). What must never happen is a sibling **below** the lock pin —
that is the downgrade that caused the crash and is the only violation enforced.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# `packaging` is the preferred PEP 440 comparator, but the minimal runtime build
# venv is not guaranteed to carry it. Fall back to a stdlib release-tuple
# comparator so this ratchet never fails-open on a missing dependency.
try:
    from packaging.version import InvalidVersion as _InvalidVersion
    from packaging.version import Version as _Version

    def _version_key(value: str) -> tuple[int, ...]:
        return _Version(value).release

    _VersionError: type[Exception] = _InvalidVersion
except ModuleNotFoundError:  # pragma: no cover - exercised only without packaging

    class _VersionError(Exception):  # type: ignore[no-redef]
        pass

    def _version_key(value: str) -> tuple[int, ...]:
        # Parse the leading numeric release segment (e.g. "0.38.1-dev" -> (0,38,1)).
        match = re.match(r"^\s*(\d+(?:\.\d+)*)", value)
        if not match:
            raise _VersionError(value)
        return tuple(int(part) for part in match.group(1).split("."))


# Directory-name -> uv.lock distribution name for every sibling that workspace
# mode provisions. Mirrors WORKSPACE_PACKAGES in compute_workspace_provenance.py.
SIBLING_DISTRIBUTIONS: dict[str, str] = {
    "omnibase_compat": "omnibase-compat",
    "onex_change_control": "onex-change-control",
    "omnimarket": "omnimarket",
}


class WorkspacePinError(RuntimeError):
    """Raised when a staged sibling has regressed below its lock pin."""


@dataclass(frozen=True)
class LockPin:
    """Expected version + git rev for one package, parsed from a uv.lock."""

    package: str
    version: str
    rev: str | None


@dataclass(frozen=True)
class PinComparison:
    """Expected-vs-actual comparison for one sibling, for the provenance block."""

    package: str
    expected_version: str
    expected_rev: str | None
    actual_version: str
    actual_rev: str | None
    match: bool
    status: str  # one of: exact | ahead | regression | unpinned

    def to_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "expected_version": self.expected_version,
            "expected_rev": self.expected_rev,
            "actual_version": self.actual_version,
            "actual_rev": self.actual_rev,
            "match": self.match,
            "status": self.status,
        }


def _rev_from_git_source(git_url: str) -> str | None:
    """Extract the resolved commit SHA from a uv.lock git source URL.

    uv encodes git pins as ``...repo.git?rev=<full-sha>#<full-sha>`` (the
    fragment is the resolved commit). Prefer the fragment; fall back to the
    ``rev=`` query value.
    """
    if "#" in git_url:
        fragment = git_url.rsplit("#", 1)[1].strip()
        if fragment:
            return fragment
    if "rev=" in git_url:
        rev = git_url.split("rev=", 1)[1]
        # rev runs until the next URL delimiter.
        for delim in ("#", "&", " "):
            rev = rev.split(delim, 1)[0]
        rev = rev.strip()
        if rev:
            return rev
    return None


def parse_lock_pins(lock_path: Path) -> dict[str, LockPin]:
    """Parse a uv.lock into ``{distribution_name: LockPin}``.

    Raises FileNotFoundError if the lock file is absent — a missing lock is a
    hard error, never a silent skip.
    """
    if not lock_path.exists():
        raise FileNotFoundError(f"uv.lock not found: {lock_path}")

    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    pins: dict[str, LockPin] = {}
    for package in data.get("package", []):
        name = package.get("name")
        if name is None:
            continue
        version = package.get("version", "")
        rev: str | None = None
        source = package.get("source", {})
        git_url = source.get("git")
        if isinstance(git_url, str):
            rev = _rev_from_git_source(git_url)
        pins[name] = LockPin(package=name, version=str(version), rev=rev)
    return pins


def read_repo_version(repo_dir: Path) -> str:
    """Read the ``[project].version`` from a repo's pyproject.toml.

    Raises FileNotFoundError if the pyproject is absent.
    """
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject}")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def read_repo_head_sha(repo_dir: Path) -> str | None:
    """Return the staged repo's git HEAD SHA, or None if not a git tree.

    Staged sibling trees are rsync'd with ``--exclude=.git`` so HEAD is usually
    unavailable inside the build context; this is best-effort provenance only
    and never gates the build (version regression is the enforced signal).
    """
    head = repo_dir / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        ref = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if ref.startswith("ref:"):
        ref_path = repo_dir / ".git" / ref.split(" ", 1)[1].strip()
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        return None
    return ref


def compare_pin(
    *,
    package: str,
    expected_version: str,
    expected_rev: str | None,
    actual_version: str,
    actual_rev: str | None,
) -> PinComparison:
    """Classify one sibling against its lock pin.

    status:
      * ``exact``      — actual version == expected version
      * ``ahead``      — actual version > expected (dev advanced; allowed)
      * ``regression`` — actual version < expected (the downgrade; violation)
      * ``unpinned``   — lock pin carried no comparable version (advisory)

    ``match`` is True for exact/ahead/unpinned and False only for regression.
    """
    if not expected_version:
        return PinComparison(
            package=package,
            expected_version=expected_version,
            expected_rev=expected_rev,
            actual_version=actual_version,
            actual_rev=actual_rev,
            match=True,
            status="unpinned",
        )

    try:
        expected = _version_key(expected_version)
        actual = _version_key(actual_version)
    except _VersionError:
        # Unorderable versions cannot be ranked; fall back to exact-string match.
        status = "exact" if actual_version == expected_version else "regression"
        return PinComparison(
            package=package,
            expected_version=expected_version,
            expected_rev=expected_rev,
            actual_version=actual_version,
            actual_rev=actual_rev,
            match=status != "regression",
            status=status,
        )

    if actual < expected:
        status = "regression"
    elif actual > expected:
        status = "ahead"
    else:
        status = "exact"

    return PinComparison(
        package=package,
        expected_version=expected_version,
        expected_rev=expected_rev,
        actual_version=actual_version,
        actual_rev=actual_rev,
        match=status != "regression",
        status=status,
    )


def build_comparisons(
    *,
    lock_path: Path,
    siblings: dict[str, Path],
) -> list[PinComparison]:
    """Build expected-vs-actual comparisons for staged ``siblings``.

    ``siblings`` maps distribution name -> staged repo directory. The expected
    pin comes from ``lock_path``; the actual version comes from the staged
    tree's pyproject. A sibling absent from the lock is reported ``unpinned``.
    """
    pins = parse_lock_pins(lock_path)
    comparisons: list[PinComparison] = []
    for dist_name, repo_dir in siblings.items():
        actual_version = read_repo_version(repo_dir)
        actual_rev = read_repo_head_sha(repo_dir)
        pin = pins.get(dist_name)
        if pin is None:
            comparisons.append(
                PinComparison(
                    package=dist_name,
                    expected_version="",
                    expected_rev=None,
                    actual_version=actual_version,
                    actual_rev=actual_rev,
                    match=True,
                    status="unpinned",
                )
            )
            continue
        comparisons.append(
            compare_pin(
                package=dist_name,
                expected_version=pin.version,
                expected_rev=pin.rev,
                actual_version=actual_version,
                actual_rev=actual_rev,
            )
        )
    return comparisons


def assert_pins_satisfied(comparisons: list[PinComparison]) -> None:
    """Fail-fast if any comparison is a regression. No silent fallback.

    Raises WorkspacePinError naming every regressed sibling with its
    expected pin and the stale version actually staged.
    """
    violations = [c for c in comparisons if c.status == "regression"]
    if not violations:
        return
    lines = [
        "Workspace-mode build aborted: vendored sibling(s) regressed below the "
        "consuming repo's uv.lock pin (this is the 2026-06-11 stability-crash "
        "condition — a stale sibling was staged instead of the lock-pinned one).",
    ]
    for c in violations:
        lines.append(
            f"  - {c.package}: lock pins >= {c.expected_version}"
            f" (rev {c.expected_rev}), but staged tree is {c.actual_version}"
            f" (rev {c.actual_rev}). Re-stage from the canonical clone at or "
            "above the lock pin."
        )
    raise WorkspacePinError("\n".join(lines))
