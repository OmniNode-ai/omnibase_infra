# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: ``uv.lock`` may not pin virtualenv below the release the scan cleared.

Incident: four HIGH advisories against virtualenv's activation scripts
(GHSA-5vjq-rrrf-7h2q, GHSA-8rjx-v5ww-45pp, GHSA-c947-3pg5-gm8q,
GHSA-vxj4-q4g9-crj7) were published against the locked virtualenv 21.7.13.
The Lockfile CVE Scan failed every dev merge group under an unchanged lock.
The lock now holds 21.14.12, the release the Lockfile CVE Scan reads clean.

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

FIXED_VERSION = "21.14.12"
VULNERABLE_VERSION = "21.7.13"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def below_floor(version: str) -> bool:
    return _version_tuple(version) < _version_tuple(FIXED_VERSION)


def _locked_virtualenv_versions() -> list[str]:
    with UV_LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    return [
        package["version"]
        for package in lock["package"]
        if package["name"] == "virtualenv"
    ]


def test_recorded_vulnerable_version_is_refused() -> None:
    assert below_floor(VULNERABLE_VERSION)


def test_fixed_version_and_later_are_accepted() -> None:
    assert not below_floor(FIXED_VERSION)
    assert not below_floor("21.15.0")
    assert not below_floor("22.0.1")


def test_uv_lock_pins_virtualenv_at_or_above_the_fixed_release() -> None:
    versions = _locked_virtualenv_versions()
    assert versions, "uv.lock has no virtualenv entry"
    too_old = [version for version in versions if below_floor(version)]
    assert not too_old, (
        f"uv.lock pins virtualenv {too_old}, below {FIXED_VERSION} "
        "(GHSA-5vjq-rrrf-7h2q, GHSA-8rjx-v5ww-45pp, GHSA-c947-3pg5-gm8q, "
        "GHSA-vxj4-q4g9-crj7)"
    )
