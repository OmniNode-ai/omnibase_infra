# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23843: a2a-sdk is held below 1.0.0 in pyproject.toml and uv.lock.

The 1.x line of a2a-sdk is unchecked against handler_a2a_task.py, so the
dependency is pinned ``>=0.3.4,<1.0.0``. These tests read the real
pyproject.toml and uv.lock and fail while either drifts past that bound or
the lock stops matching the declared specifier.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
UPPER_BOUND = Version("1.0.0")
LOCKED_VERSION = Version("0.3.26")


def _pyproject_a2a_sdk_requirement() -> Requirement:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    entries = [
        entry
        for entry in pyproject["project"]["dependencies"]
        if entry.startswith("a2a-sdk")
    ]
    assert len(entries) == 1, (
        f"expected exactly one a2a-sdk dependency in pyproject.toml, got {entries!r}"
    )
    return Requirement(entries[0])


def _locked_package(name: str) -> dict[str, object]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    matches = [package for package in lock["package"] if package["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name} entry, got {len(matches)}"
    return matches[0]


def _locked_requires_dist(name: str) -> dict[str, object]:
    requires_dist = _locked_package(name).get("metadata", {}).get("requires-dist")
    entries = [
        entry for entry in (requires_dist or []) if entry.get("name") == "a2a-sdk"
    ]
    assert len(entries) == 1, (
        f"expected exactly one a2a-sdk requires-dist entry for {name}, got {entries!r}"
    )
    return entries[0]


def test_pyproject_holds_a2a_sdk_below_one() -> None:
    requirement = _pyproject_a2a_sdk_requirement()
    assert UPPER_BOUND not in requirement.specifier, (
        f"pyproject.toml specifier {requirement.specifier} admits a2a-sdk "
        f"{UPPER_BOUND}; expected it excluded"
    )
    assert LOCKED_VERSION in requirement.specifier, (
        f"pyproject.toml specifier {requirement.specifier} excludes the locked "
        f"a2a-sdk {LOCKED_VERSION}; expected it admitted"
    )


def test_lock_resolves_a2a_sdk_below_one() -> None:
    package = _locked_package("a2a-sdk")
    version = Version(str(package["version"]))
    assert version < UPPER_BOUND, (
        f"uv.lock resolves a2a-sdk to {version}, which is at or above {UPPER_BOUND}"
    )


def test_lock_metadata_specifier_matches_pyproject() -> None:
    requires_dist = _locked_requires_dist("omnibase-infra")
    lock_specifier = SpecifierSet(str(requires_dist["specifier"]))
    pyproject_specifier = _pyproject_a2a_sdk_requirement().specifier
    assert lock_specifier == pyproject_specifier, (
        f"uv.lock omnibase-infra requires-dist specifier {lock_specifier!r} does "
        f"not equal the pyproject.toml specifier {pyproject_specifier!r}"
    )
