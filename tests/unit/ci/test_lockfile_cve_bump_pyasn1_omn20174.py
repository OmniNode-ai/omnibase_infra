# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20174: uv.lock resolves pyasn1 past its HIGH advisories.

The Lockfile CVE Scan blocks on pyasn1 0.6.4 (GHSA-2qm8-x737-gmw5,
GHSA-9m75-vc3g-4x39, GHSA-fm9c-5rcf-2q5p, GHSA-mm57-4v3p-rchx, fixed in
0.6.5). This test reads the real uv.lock and fails while pyasn1 is locked
below that floor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PYASN1_FIXED_FLOOR = Version("0.6.5")


def _assert_fixed_version(version: str) -> None:
    assert Version(version) >= PYASN1_FIXED_FLOOR, (
        f"pyasn1 {version} is below fixed floor {PYASN1_FIXED_FLOOR}"
    )


def test_pyasn1_fixed_version_locked() -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    matches = [p for p in lock["package"] if p["name"] == "pyasn1"]
    assert len(matches) == 1, f"expected exactly one pyasn1 entry, got {len(matches)}"
    version = str(matches[0]["version"])
    _assert_fixed_version(version)


@pytest.mark.parametrize(
    "version",
    ["0.6.5", "0.6.5.post1", "0.7.0rc1", "0.7.0.dev1", "0.6.5+local.1", "1!0.1.0"],
)
def test_fixed_version_accepts_pep440_releases(version: str) -> None:
    _assert_fixed_version(version)


@pytest.mark.parametrize(
    "version", ["0.6.4", "0.6.4.post1", "0.6.5rc1", "0.6.5.dev1", "0.6.4+local.1"]
)
def test_fixed_version_rejects_versions_below_floor(version: str) -> None:
    with pytest.raises(AssertionError, match=r"below fixed floor 0\.6\.5"):
        _assert_fixed_version(version)


def test_fixed_version_rejects_invalid_version() -> None:
    with pytest.raises(InvalidVersion, match="Invalid version"):
        _assert_fixed_version("not-a-version")
