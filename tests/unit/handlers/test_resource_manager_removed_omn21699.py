# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The unused resource manager stays removed (OMN-21699)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_resources_package_is_not_importable() -> None:
    assert importlib.util.find_spec("omnibase_infra.resources") is None


@pytest.mark.unit
def test_no_source_or_test_names_the_resource_manager() -> None:
    offenders = [
        str(path.relative_to(_REPO))
        for root in ("src", "tests")
        for path in (_REPO / root).rglob("*.py")
        if path.name != Path(__file__).name
        and "HandlerResourceManager" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
