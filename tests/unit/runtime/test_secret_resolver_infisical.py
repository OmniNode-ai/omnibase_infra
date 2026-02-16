# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for Infisical source type support in SecretResolver.

Tests the 'infisical' source_type in secret mappings and resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
    SecretSourceType,
)


class TestSecretSourceTypeInfisical:
    """Test that 'infisical' is a valid SecretSourceType."""

    def test_infisical_literal_accepted(self) -> None:
        """Test creating a spec with source_type='infisical'."""
        spec = ModelSecretSourceSpec(
            source_type="infisical",
            source_path="DB_PASSWORD",
        )
        assert spec.source_type == "infisical"
        assert spec.source_path == "DB_PASSWORD"

    def test_env_still_works(self) -> None:
        """Test that 'env' source type still works."""
        spec = ModelSecretSourceSpec(
            source_type="env",
            source_path="MY_VAR",
        )
        assert spec.source_type == "env"

    def test_vault_still_works(self) -> None:
        """Test that 'vault' source type still works."""
        spec = ModelSecretSourceSpec(
            source_type="vault",
            source_path="secret/data/db#pass",
        )
        assert spec.source_type == "vault"

    def test_file_still_works(self) -> None:
        """Test that 'file' source type still works."""
        spec = ModelSecretSourceSpec(
            source_type="file",
            source_path="/run/secrets/pass",
        )
        assert spec.source_type == "file"

    def test_invalid_source_type_rejected(self) -> None:
        """Test that invalid source types are rejected."""
        with pytest.raises(Exception):
            ModelSecretSourceSpec(
                source_type="redis",  # type: ignore[arg-type]
                source_path="key",
            )


class TestSecretResolverInfisicalHandler:
    """Test SecretResolver with infisical_handler parameter."""

    def test_constructor_accepts_infisical_handler(self) -> None:
        """Test SecretResolver accepts infisical_handler parameter."""
        from omnibase_infra.runtime.models.model_secret_resolver_config import (
            ModelSecretResolverConfig,
        )
        from omnibase_infra.runtime.secret_resolver import SecretResolver

        config = ModelSecretResolverConfig(mappings=[])
        mock_handler = MagicMock()

        resolver = SecretResolver(
            config=config,
            infisical_handler=mock_handler,
        )

        assert resolver._infisical_handler is mock_handler

    def test_constructor_backward_compatible(self) -> None:
        """Test SecretResolver still works without infisical_handler."""
        from omnibase_infra.runtime.models.model_secret_resolver_config import (
            ModelSecretResolverConfig,
        )
        from omnibase_infra.runtime.secret_resolver import SecretResolver

        config = ModelSecretResolverConfig(mappings=[])
        resolver = SecretResolver(config=config)

        assert resolver._infisical_handler is None
        assert resolver._vault_handler is None
