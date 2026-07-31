# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for Git-hook environment isolation (OMN-15555)."""

from __future__ import annotations

import os

import pytest

_GIT_KEYS_AT_COLLECTION = tuple(
    sorted(key for key in os.environ if key.startswith("GIT_"))
)


@pytest.mark.unit
def test_git_hook_environment_is_scrubbed_before_collection() -> None:
    """Caller repository authority must be gone before test modules import."""
    assert _GIT_KEYS_AT_COLLECTION == ()


@pytest.mark.unit
def test_git_environment_is_scrubbed_at_test_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared helper removes Git state introduced during a test session."""
    monkeypatch.setenv("GIT_DIR", "/must/not/be/used")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'user.name'='must-not-leak'")

    from tests.conftest import _strip_inherited_git_environment

    _strip_inherited_git_environment()

    assert not any(key.startswith("GIT_") for key in os.environ)
