# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: ``uv.lock`` may not pin certifi below the fixed release.

Incident: ``uv.lock`` pinned certifi 2026.1.4, which carries PYSEC-2026-4320
(GHSA-6q48-6c7w-9gw2, the emSign root CAs with compromised intermediate CA
keys). The Lockfile CVE Scan failed every merge group, so nothing merged in
this repository, and the bump pull request could not be committed because a
pre-commit hook refused every commit until a pull request that the same scan
was blocking landed. The first fixed release, per OSV, is 2026.10.19.

The floor check is a pure function over a version string. It runs on the
recorded vulnerable version as the positive control, where it must refuse, and
on the entry this repository's ``uv.lock`` holds today.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

UV_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"

FIXED_VERSION = "2026.10.19"
VULNERABLE_VERSION = "2026.1.4"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def below_floor(version: str) -> bool:
    return _version_tuple(version) < _version_tuple(FIXED_VERSION)


def _locked_certifi_versions() -> list[str]:
    with UV_LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    return [
        package["version"]
        for package in lock["package"]
        if package["name"] == "certifi"
    ]


def test_recorded_vulnerable_version_is_refused() -> None:
    assert below_floor(VULNERABLE_VERSION)


def test_fixed_version_and_later_are_accepted() -> None:
    assert not below_floor(FIXED_VERSION)
    assert not below_floor("2026.11.2")
    assert not below_floor("2027.1.1")


def test_uv_lock_pins_certifi_at_or_above_the_fixed_release() -> None:
    versions = _locked_certifi_versions()
    assert versions, "uv.lock has no certifi entry"
    too_old = [version for version in versions if below_floor(version)]
    assert not too_old, (
        f"uv.lock pins certifi {too_old}, below the fixed release {FIXED_VERSION} "
        "(PYSEC-2026-4320, GHSA-6q48-6c7w-9gw2)"
    )
