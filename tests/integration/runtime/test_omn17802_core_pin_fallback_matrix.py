# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17802 integration proof for the source-tree-free version fallback."""

from __future__ import annotations

from importlib.metadata import version

import pytest

from omnibase_infra.runtime import version_compatibility as vc

pytestmark = pytest.mark.integration


def test_fallback_matrix_rejects_the_previous_core_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_versions = {
        "omnibase_core": "0.47.57",
        "omnibase_spi": version("omnibase-spi"),
    }
    monkeypatch.setattr(vc, "_get_installed_version", installed_versions.get)

    errors = vc.check_version_compatibility(vc._FALLBACK_MATRIX)

    assert errors == [
        "omnibase_core: 0.47.57 is incompatible (required >=0.47.58,<0.48.0)"
    ]


def test_fallback_matrix_accepts_the_dev_tag_core_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_versions = {
        "omnibase_core": "0.47.58",
        "omnibase_spi": version("omnibase-spi"),
    }
    monkeypatch.setattr(vc, "_get_installed_version", installed_versions.get)

    assert vc.check_version_compatibility(vc._FALLBACK_MATRIX) == []
