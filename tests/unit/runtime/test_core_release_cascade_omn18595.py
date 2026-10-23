# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The core 0.47.53 cascade reaches the resolved runtime (OMN-18595)."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from omnibase_infra.runtime.version_compatibility import check_version_compatibility

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_FLOOR = Version("0.47.53")


def test_core_cascade_resolves_at_least_the_triggering_release() -> None:
    """A consistent lock for the previous release still misses this cascade."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    with (REPO_ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)

    core_packages = [p for p in lock["package"] if p["name"] == "omnibase-core"]
    assert len(core_packages) == 1
    locked_version = Version(core_packages[0]["version"])
    assert locked_version >= RELEASE_FLOOR, (
        f"core cascade requires >= {RELEASE_FLOOR}; uv.lock resolves {locked_version}"
    )
    assert (
        f"omnibase-core=={locked_version}"
        in manifest["tool"]["uv"]["override-dependencies"]
    )
    core_requirements = [
        Requirement(spec)
        for spec in manifest["project"]["dependencies"]
        if Requirement(spec).name.replace("_", "-") == "omnibase-core"
    ]
    assert len(core_requirements) == 1
    assert locked_version in core_requirements[0].specifier


def test_installed_core_matches_the_cascade_and_passes_runtime_admission() -> None:
    """Read the real installed distribution and invoke the runtime startup guard."""
    with (REPO_ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    locked_version = next(
        p["version"] for p in lock["package"] if p["name"] == "omnibase-core"
    )

    assert importlib.metadata.version("omnibase-core") == locked_version
    assert check_version_compatibility() == []
