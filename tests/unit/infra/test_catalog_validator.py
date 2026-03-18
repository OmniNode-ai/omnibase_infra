# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the catalog env validator."""

from __future__ import annotations

import pytest

from omnibase_infra.docker.catalog.validator import validate_env


@pytest.mark.unit
def test_validator_passes_when_all_required_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    monkeypatch.setenv("VALKEY_PASSWORD", "test")
    result = validate_env(required={"POSTGRES_PASSWORD", "VALKEY_PASSWORD"})
    assert result.ok


@pytest.mark.unit
def test_validator_fails_when_required_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALKEY_PASSWORD", raising=False)
    result = validate_env(required={"VALKEY_PASSWORD"})
    assert not result.ok
    assert "VALKEY_PASSWORD" in result.missing
