# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The locked runtime core supports omnimarket's binding-kind imports (OMN-19998)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


def test_locked_core_meets_the_binding_kind_release_floor() -> None:
    """The image's locked core must satisfy both its override and the plugin floor."""
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((_REPO / "uv.lock").read_text(encoding="utf-8"))
    overrides = [
        requirement
        for value in project["tool"]["uv"]["override-dependencies"]
        if (requirement := Requirement(value)).name == "omnibase-core"
    ]
    packages = [item for item in lock["package"] if item["name"] == "omnibase-core"]

    assert len(overrides) == 1, overrides
    assert len(packages) == 1, packages
    locked = Version(packages[0]["version"])
    assert locked in overrides[0].specifier, (locked, str(overrides[0]))
    assert locked >= Version("0.47.46"), (
        f"core {locked} predates the binding-kind enum required by omnimarket"
    )


def test_runtime_core_exposes_binding_kinds() -> None:
    """Import the same module the two omnimarket binding handlers need at boot."""
    from omnibase_core.enums.enum_binding_kind import EnumBindingKind

    assert EnumBindingKind("replacing") is EnumBindingKind.REPLACING
    assert EnumBindingKind("additive") is EnumBindingKind.ADDITIVE
