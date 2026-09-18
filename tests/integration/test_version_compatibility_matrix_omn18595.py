# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for OMN-18595 fallback version matrix bumps."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime import version_compatibility as vc

pytestmark = pytest.mark.integration


def test_fallback_matrix_accepts_the_installed_runtime_stack() -> None:
    assert vc.check_version_compatibility(vc._FALLBACK_MATRIX) == []


def test_pyproject_matrix_and_fallback_matrix_agree_on_core_floor() -> None:
    pyproject_matrix = vc._build_matrix_from_pyproject()
    assert pyproject_matrix is not None

    fallback_by_package = {
        constraint.package: constraint for constraint in vc._FALLBACK_MATRIX
    }
    pyproject_by_package = {
        constraint.package: constraint for constraint in pyproject_matrix
    }

    assert pyproject_by_package["omnibase_core"] == fallback_by_package["omnibase_core"]
