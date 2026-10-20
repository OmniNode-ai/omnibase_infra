# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: uv.lock carries certifi at or above the GHSA-6q48-6c7w-9gw2 fix.

The Lockfile CVE Scan (OMN-16228) blocked dev at 54c57239d on [CRITICAL]
GHSA-6q48-6c7w-9gw2 in certifi==2026.1.4 (certifi bundles emSign root CAs with
compromised intermediate CA keys); the fixed version is 2026.10.19.

Falsifier: lock certifi back to 2026.1.4 and
test_locked_certifi_is_at_or_above_the_fix fails.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED = Version("2026.10.19")


def _locked_versions(name: str) -> list[Version]:
    with (REPO_ROOT / "uv.lock").open("rb") as lockfile:
        lock = tomllib.load(lockfile)
    return [Version(p["version"]) for p in lock["package"] if p["name"] == name]


def test_locked_certifi_is_at_or_above_the_fix() -> None:
    versions = _locked_versions("certifi")
    assert versions, "certifi must be locked in uv.lock"
    for version in versions:
        assert version >= FIXED, (
            f"certifi=={version} is below {FIXED}, the fix for GHSA-6q48-6c7w-9gw2"
        )
