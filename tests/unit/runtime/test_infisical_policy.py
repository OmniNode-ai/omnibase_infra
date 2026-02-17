# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelInfisicalPolicy."""

from __future__ import annotations

import pytest

from omnibase_infra.models.security.model_infisical_policy import ModelInfisicalPolicy


class TestModelInfisicalPolicy:
    """Test INFISICAL_REQUIRED policy model."""

    def test_default_policy(self) -> None:
        """Test default policy values."""
        policy = ModelInfisicalPolicy()
        assert policy.policy_name == "INFISICAL_REQUIRED"
        assert policy.enforce is True
        assert "infisical" in policy.allowed_source_types
        assert policy.allowed_source_types == frozenset({"infisical"})

    def test_policy_immutable(self) -> None:
        """Test policy is frozen/immutable."""
        policy = ModelInfisicalPolicy()
        with pytest.raises(Exception):
            policy.enforce = False  # type: ignore[misc]

    def test_policy_warn_mode(self) -> None:
        """Test policy in warning mode (non-enforcing)."""
        policy = ModelInfisicalPolicy(enforce=False)
        assert not policy.enforce

    def test_policy_custom_allowed_types(self) -> None:
        """Test policy with custom allowed source types."""
        policy = ModelInfisicalPolicy(
            allowed_source_types=frozenset({"infisical"}),
        )
        assert policy.allowed_source_types == frozenset({"infisical"})
        assert "env" not in policy.allowed_source_types

    def test_policy_no_extra_fields(self) -> None:
        """Test policy rejects extra fields."""
        with pytest.raises(Exception):
            ModelInfisicalPolicy(unknown_field="value")  # type: ignore[call-arg]
