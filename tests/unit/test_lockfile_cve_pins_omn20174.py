# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20174: uv.lock resolves msgpack and dnspython at advisory-fixed releases.

The lockfile resolved msgpack 1.2.1 (GHSA-j586-36cw-2gc2) and dnspython 2.8.0
(GHSA-45c3-73f7-pv4m). Each test reads every resolved ``[[package]]`` entry for
its package and fails when any is below the fixed version, as at the merge
base. These tests do not run the CVE scanner.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.unit

UV_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"


def _resolved_versions(name: str) -> list[Version]:
    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    return [
        Version(package["version"])
        for package in lock.get("package", [])
        if package.get("name") == name
    ]


@pytest.mark.parametrize(
    ("name", "fixed"),
    [
        pytest.param("msgpack", "1.2.3", id="msgpack"),
        pytest.param("dnspython", "2.9.0", id="dnspython"),
    ],
)
def test_lockfile_resolves_package_at_or_above_fixed_version(
    name: str, fixed: str
) -> None:
    versions = _resolved_versions(name)
    assert versions, f"uv.lock resolves no {name} entry"
    below = [str(v) for v in versions if v < Version(fixed)]
    assert not below, f"uv.lock resolves {name} {below}, below fixed {fixed}"
