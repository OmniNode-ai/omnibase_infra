# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20174: uv.lock resolves msgpack and dnspython past their advisories.

The Lockfile CVE Scan blocks on msgpack 1.2.1 (GHSA-j586-36cw-2gc2, fixed in
1.2.3) and dnspython 2.8.0 (GHSA-45c3-73f7-pv4m, fixed in 2.9.0). These tests
read the real uv.lock and fail while either package is locked below its fixed
floor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXED_FLOORS = {
    "msgpack": (1, 2, 3),
    "dnspython": (2, 9, 0),
}


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse dotted integer versions, refusing components we cannot compare."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        pytest.fail(f"version has a non-integer component: {version!r}")


def _locked_package(name: str) -> dict[str, object]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    matches = [package for package in lock["package"] if package["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name} entry, got {len(matches)}"
    return matches[0]


@pytest.mark.parametrize(("name", "floor"), FIXED_FLOORS.items())
def test_fixed_versions_locked(name: str, floor: tuple[int, ...]) -> None:
    package = _locked_package(name)
    version = str(package["version"])
    assert _version_tuple(version) >= floor, (
        f"{name} {version} is below fixed floor {floor}"
    )
