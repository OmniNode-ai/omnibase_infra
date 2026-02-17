# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for BindingConfigResolver.

These tests verify the BindingConfigResolver's behavior with mocked sources,
including file-based configs, environment variables, Vault secrets, caching,
and thread safety.

Test Coverage:
- Basic resolution from inline config
- File-based config resolution (YAML/JSON)
- Environment variable config resolution
- Environment variable overrides
- Vault-based config resolution
- Cache hit/miss behavior
- TTL and expiration
- Thread safety under concurrent access
- Async API support
- Validation and error handling
- Security (path traversal, sanitization)
- Container-based dependency injection
- Recursion depth limits for nested config resolution
- Async lock lifecycle (acquisition, release on success/error, leak prevention)

Related:
- OMN-765: BindingConfigResolver implementation
- PR #168: Async lock cleanup test coverage
- docs/milestones/BETA_v0.2.0_HARDENING.md
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml
from pydantic import SecretStr

from omnibase_infra.errors import ProtocolConfigurationError, SecretResolutionError
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models.model_binding_config import (
    ModelBindingConfig,
    ModelRetryPolicy,
)
from omnibase_infra.runtime.models.model_binding_config_resolver_config import (
    ModelBindingConfigResolverConfig,
)
from omnibase_infra.runtime.models.model_config_ref import (
    EnumConfigRefScheme,
    ModelConfigRef,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


def create_mock_container(
    config: ModelBindingConfigResolverConfig,
    secret_resolver: MagicMock | None = None,
) -> MagicMock:
    """Create a mock container with config registered in service registry.

    This helper creates a mock ModelONEXContainer with the required
    service_registry.resolve_service() behavior for BindingConfigResolver.

    Args:
        config: The resolver config to register.
        secret_resolver: Optional mock SecretResolver to register.

    Returns:
        Mock container with service registry configured.
    """
    from omnibase_infra.runtime.secret_resolver import SecretResolver

    container = MagicMock()

    # Map of types to instances for resolve_service
    service_map: dict[type, object] = {
        ModelBindingConfigResolverConfig: config,
    }
    if secret_resolver is not None:
        service_map[SecretResolver] = secret_resolver

    def resolve_service_side_effect(service_type: type) -> object:
        if service_type in service_map:
            return service_map[service_type]
        raise KeyError(f"Service {service_type} not registered")

    container.service_registry.resolve_service.side_effect = resolve_service_side_effect
    return container


def create_resolver(
    config: ModelBindingConfigResolverConfig,
    secret_resolver: MagicMock | None = None,
) -> tuple[BindingConfigResolver, MagicMock]:
    """Create a BindingConfigResolver with mock container for testing.

    This helper creates both the mock container and the resolver instance,
    passing _config directly to avoid async factory method requirements.

    Args:
        config: The resolver config to use.
        secret_resolver: Optional mock SecretResolver to inject.

    Returns:
        Tuple of (resolver, container) for testing.
    """
    container = create_mock_container(config, secret_resolver)
    resolver = BindingConfigResolver(
        container,
        _config=config,
        _secret_resolver=secret_resolver,
    )
    return resolver, container


class TestBindingConfigResolverBasic:
    """Basic resolution functionality tests."""

    def test_resolve_inline_config(self) -> None:
        """Resolve with inline config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(
                handler_type="db",
                inline_config={"timeout_ms": 5000, "priority": 10},
            )

            assert result is not None
            assert result.handler_type == "db"
            assert result.timeout_ms == 5000
            assert result.priority == 10

    def test_resolve_minimal_config(self) -> None:
        """Resolve with only required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(handler_type="vault")

            assert result is not None
            assert result.handler_type == "vault"
            # Defaults should be applied
            assert result.enabled is True
            assert result.priority == 0
            assert result.timeout_ms == 30000

    def test_resolve_full_config(self) -> None:
        """Resolve with all fields populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            retry_policy = ModelRetryPolicy(
                max_retries=5,
                backoff_strategy="exponential",
                base_delay_ms=200,
                max_delay_ms=10000,
            )

            result = resolver.resolve(
                handler_type="consul",
                inline_config={
                    "name": "primary-consul",
                    "enabled": True,
                    "priority": 50,
                    "timeout_ms": 10000,
                    "rate_limit_per_second": 100.0,
                    "retry_policy": retry_policy.model_dump(),
                },
            )

            assert result is not None
            assert result.handler_type == "consul"
            assert result.name == "primary-consul"
            assert result.enabled is True
            assert result.priority == 50
            assert result.timeout_ms == 10000
            assert result.rate_limit_per_second == 100.0
            assert result.retry_policy is not None
            assert result.retry_policy.max_retries == 5
            assert result.retry_policy.backoff_strategy == "exponential"

    def test_handler_type_validation(self) -> None:
        """Handler type is required and validated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Empty handler_type should fail validation
            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="",
                    inline_config={"timeout_ms": 5000},
                )

    def test_resolve_many_basic(self) -> None:
        """Resolve multiple configurations at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            bindings = [
                {"handler_type": "db", "config": {"timeout_ms": 5000}},
                {"handler_type": "vault", "config": {"timeout_ms": 10000}},
            ]

            results = resolver.resolve_many(bindings)

            assert len(results) == 2
            assert results[0].handler_type == "db"
            assert results[0].timeout_ms == 5000
            assert results[1].handler_type == "vault"
            assert results[1].timeout_ms == 10000

    def test_resolve_many_missing_handler_type(self) -> None:
        """resolve_many raises when handler_type is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            bindings = [
                {"config": {"timeout_ms": 5000}},  # Missing handler_type
            ]

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve_many(bindings)

            assert "handler_type" in str(exc_info.value)


class TestBindingConfigResolverFileSource:
    """File-based config resolution tests."""

    def test_load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(
                yaml.dump(
                    {
                        "timeout_ms": 15000,
                        "priority": 20,
                        "enabled": True,
                    }
                )
            )

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            assert result is not None
            assert result.handler_type == "db"
            assert result.timeout_ms == 15000
            assert result.priority == 20

    def test_load_json_config(self) -> None:
        """Load configuration from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "vault.json"
            config_file.write_text(
                json.dumps(
                    {
                        "timeout_ms": 20000,
                        "priority": 30,
                    }
                )
            )

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(
                handler_type="vault",
                config_ref="file:vault.json",
            )

            assert result is not None
            assert result.handler_type == "vault"
            assert result.timeout_ms == 20000
            assert result.priority == 30

    def test_file_not_found_error(self) -> None:
        """Appropriate error when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:nonexistent.yaml",
                )

            assert "not found" in str(exc_info.value).lower()

    def test_file_size_limit(self) -> None:
        """Reject files exceeding size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Create a file larger than 1MB (the limit)
            large_file = config_dir / "large.yaml"
            # Write just over 1MB of content
            large_file.write_text("x" * (1024 * 1024 + 100))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:large.yaml",
                )

            assert "size limit" in str(exc_info.value).lower()

    def test_path_traversal_blocked(self) -> None:
        """Path traversal attempts are blocked at resolution layer.

        Single parent directory references (../) are allowed at the parsing layer
        as legitimate use cases. Security enforcement happens at the resolution layer
        where resolved paths are validated against the config_dir boundary.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()

            # Create a file outside configs dir
            outside_file = Path(tmpdir) / "secret.yaml"
            outside_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            # Attempt path traversal - parsing succeeds but resolution blocks
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:../secret.yaml",
                )

            # Resolution layer catches path traversal with a safe error message
            error_msg = str(exc_info.value).lower()
            assert "path traversal not allowed" in error_msg
            # Ensure the actual file path is NOT exposed in the error message
            assert "../secret.yaml" not in error_msg
            # The resolved path should not be exposed either
            assert tmpdir not in error_msg

    def test_relative_path_resolution(self) -> None:
        """Relative paths resolved against config_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            subdir = config_dir / "handlers"
            subdir.mkdir()

            config_file = subdir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 7000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(
                handler_type="db",
                config_ref="file:handlers/db.yaml",
            )

            assert result.timeout_ms == 7000

    def test_absolute_path_resolution(self) -> None:
        """Absolute paths work when file is within allowed directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "absolute.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 8000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            # Use file:// with absolute path
            result = resolver.resolve(
                handler_type="db",
                config_ref=f"file://{config_file}",
            )

            assert result.timeout_ms == 8000

    def test_invalid_yaml_error(self) -> None:
        """Handle invalid YAML in configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "invalid.yaml"
            config_file.write_text("{ invalid yaml: [")

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:invalid.yaml",
                )

            assert "yaml" in str(exc_info.value).lower()

    def test_invalid_json_error(self) -> None:
        """Handle invalid JSON in configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "invalid.json"
            config_file.write_text("{ invalid json:")

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:invalid.json",
                )

            assert "json" in str(exc_info.value).lower()

    def test_config_must_be_dict(self) -> None:
        """Configuration file must contain a dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "list.yaml"
            config_file.write_text(yaml.dump(["item1", "item2"]))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:list.yaml",
                )

            assert "dictionary" in str(exc_info.value).lower()

    def test_relative_path_without_config_dir(self) -> None:
        """Relative path provided but no config_dir configured."""
        config = ModelBindingConfigResolverConfig(
            config_dir=None,  # No config_dir
        )
        resolver, _container = create_resolver(config)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="db",
                config_ref="file:relative/path.yaml",
            )

        assert "config_dir" in str(exc_info.value).lower()

    def test_unsupported_file_format(self) -> None:
        """Unsupported file format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "config.toml"
            config_file.write_text("[section]\nkey = 'value'")

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:config.toml",
                )

            assert "unsupported" in str(exc_info.value).lower()


class TestBindingConfigResolverEnvSource:
    """Environment variable config resolution tests."""

    def test_load_json_from_env(self) -> None:
        """Load JSON config from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            env_config = json.dumps({"timeout_ms": 12000, "priority": 15})
            with patch.dict(os.environ, {"DB_HANDLER_CONFIG": env_config}):
                result = resolver.resolve(
                    handler_type="db",
                    config_ref="env:DB_HANDLER_CONFIG",
                )

            assert result.timeout_ms == 12000
            assert result.priority == 15

    def test_load_yaml_from_env(self) -> None:
        """Load YAML config from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            env_config = yaml.dump({"timeout_ms": 13000, "enabled": False})
            with patch.dict(os.environ, {"VAULT_HANDLER_CONFIG": env_config}):
                result = resolver.resolve(
                    handler_type="vault",
                    config_ref="env:VAULT_HANDLER_CONFIG",
                )

            assert result.timeout_ms == 13000
            assert result.enabled is False

    def test_env_var_not_found_error(self) -> None:
        """Appropriate error when env var doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Ensure env var is not set
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NONEXISTENT_CONFIG", None)

                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        config_ref="env:NONEXISTENT_CONFIG",
                    )

            assert "not set" in str(exc_info.value).lower()

    def test_invalid_json_in_env(self) -> None:
        """Handle invalid JSON/YAML in environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"INVALID_CONFIG": "{ invalid json:"}):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        config_ref="env:INVALID_CONFIG",
                    )

            assert "invalid" in str(exc_info.value).lower()

    def test_env_config_must_be_dict(self) -> None:
        """Environment variable config must be a dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"LIST_CONFIG": '["item1", "item2"]'}):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        config_ref="env:LIST_CONFIG",
                    )

            assert "dictionary" in str(exc_info.value).lower()


class TestBindingConfigResolverEnvOverrides:
    """Environment variable override tests."""

    def test_override_timeout_ms(self) -> None:
        """Override timeout_ms via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_TIMEOUT_MS": "25000"}):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={"timeout_ms": 5000},  # Will be overridden
                )

            assert result.timeout_ms == 25000

    def test_override_enabled(self) -> None:
        """Override enabled via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_VAULT_ENABLED": "false"}):
                result = resolver.resolve(
                    handler_type="vault",
                    inline_config={"enabled": True},  # Will be overridden
                )

            assert result.enabled is False

    def test_override_priority(self) -> None:
        """Override priority via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_CONSUL_PRIORITY": "75"}):
                result = resolver.resolve(
                    handler_type="consul",
                    inline_config={"priority": 10},  # Will be overridden
                )

            assert result.priority == 75

    def test_override_rate_limit(self) -> None:
        """Override rate_limit_per_second via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_RATE_LIMIT_PER_SECOND": "500.5"}):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={"rate_limit_per_second": 100.0},
                )

            assert result.rate_limit_per_second == 500.5

    def test_override_precedence(self) -> None:
        """Environment overrides take precedence over file config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000, "priority": 10}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_TIMEOUT_MS": "99000"}):
                result = resolver.resolve(
                    handler_type="db",
                    config_ref="file:db.yaml",
                )

            # Env override takes precedence
            assert result.timeout_ms == 99000
            # File config is used for non-overridden fields
            assert result.priority == 10

    def test_custom_env_prefix(self) -> None:
        """Custom env_prefix in config is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="ONEX_CUSTOM",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"ONEX_CUSTOM_DB_TIMEOUT_MS": "45000"}):
                result = resolver.resolve(handler_type="db")

            assert result.timeout_ms == 45000

    def test_override_name(self) -> None:
        """Override name via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_NAME": "override-name"}):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={"name": "original-name"},
                )

            assert result.name == "override-name"

    def test_override_retry_policy_fields(self) -> None:
        """Override retry policy fields via environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(
                os.environ,
                {
                    "HANDLER_DB_MAX_RETRIES": "7",
                    "HANDLER_DB_BACKOFF_STRATEGY": "fixed",
                    "HANDLER_DB_BASE_DELAY_MS": "500",
                    "HANDLER_DB_MAX_DELAY_MS": "15000",
                },
            ):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={
                        "retry_policy": {
                            "max_retries": 3,
                            "backoff_strategy": "exponential",
                        }
                    },
                )

            assert result.retry_policy is not None
            assert result.retry_policy.max_retries == 7
            assert result.retry_policy.backoff_strategy == "fixed"
            assert result.retry_policy.base_delay_ms == 500
            assert result.retry_policy.max_delay_ms == 15000

    def test_invalid_env_value_ignored(self) -> None:
        """Invalid environment variable values are ignored with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_TIMEOUT_MS": "not_a_number"}):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={"timeout_ms": 5000},
                )

            # Should fall back to inline config value
            assert result.timeout_ms == 5000

    def test_invalid_boolean_env_value_strict_mode_raises(self) -> None:
        """Invalid boolean value in strict mode raises ProtocolConfigurationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
                strict_env_coercion=True,  # Enable strict mode
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_ENABLED": "banana"}):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        inline_config={"enabled": True},
                    )

            # Verify error message mentions boolean and the expected values
            assert "boolean" in str(exc_info.value).lower()
            assert "enabled" in str(exc_info.value).lower()

    def test_env_coercion_error_includes_correlation_id(self) -> None:
        """Env coercion errors include correlation_id in error context.

        Verifies that when strict_env_coercion=True and an invalid env value
        is provided, the raised ProtocolConfigurationError includes the
        correlation_id for traceability.

        Related: PR #168 review - correlation_id propagation in env coercion errors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
                strict_env_coercion=True,
            )
            resolver, _container = create_resolver(config)

            test_correlation_id = uuid4()

            with patch.dict(os.environ, {"HANDLER_DB_TIMEOUT_MS": "not_a_number"}):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        inline_config={"timeout_ms": 5000},
                        correlation_id=test_correlation_id,
                    )

            # Verify correlation_id is in error context
            assert exc_info.value.model.correlation_id == test_correlation_id

    def test_env_coercion_error_auto_generates_correlation_id(self) -> None:
        """Env coercion errors auto-generate correlation_id when not provided.

        Verifies that when no correlation_id is passed to resolve() and an
        env coercion error occurs, a correlation_id is still generated and
        included in the error context.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
                strict_env_coercion=True,
            )
            resolver, _container = create_resolver(config)

            with patch.dict(
                os.environ, {"HANDLER_DB_RATE_LIMIT_PER_SECOND": "invalid"}
            ):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        inline_config={"rate_limit_per_second": 100.0},
                        # No correlation_id provided
                    )

            # Verify correlation_id is auto-generated and present
            assert exc_info.value.model.correlation_id is not None
            # Verify it's a valid UUID (version 4)
            from uuid import UUID

            assert isinstance(exc_info.value.model.correlation_id, UUID)

    def test_invalid_boolean_env_value_non_strict_mode_skips_override(self) -> None:
        """Invalid boolean value in non-strict mode skips override with warning.

        Unlike the old behavior that coerced to False, the new behavior
        skips the override entirely (returns None from _convert_env_value),
        preserving the original config value. This is consistent with how
        other types (integer, float) handle invalid values.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="HANDLER",
                strict_env_coercion=False,  # Non-strict mode (default)
            )
            resolver, _container = create_resolver(config)

            with patch.dict(os.environ, {"HANDLER_DB_ENABLED": "invalid_value"}):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={"enabled": True},  # Original value preserved
                )

            # Invalid value is skipped, original config value preserved
            assert result.enabled is True

    def test_valid_boolean_env_values_truthy(self) -> None:
        """Valid truthy boolean environment variable values are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="TEST",
            )
            # Test true values - each with unique handler type and matching env var
            true_values = ["true", "1", "yes", "on", "TRUE", "YES", "ON"]
            for idx, val in enumerate(true_values):
                resolver, _ = create_resolver(config)
                handler_type = f"truthy{idx}"
                env_var = f"TEST_{handler_type.upper()}_ENABLED"
                with patch.dict(os.environ, {env_var: val}):
                    result = resolver.resolve(
                        handler_type=handler_type,
                        inline_config={"enabled": False},
                    )
                assert result.enabled is True, f"Failed for truthy value '{val}'"

    def test_valid_boolean_env_values_falsy(self) -> None:
        """Valid falsy boolean environment variable values are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                env_prefix="TEST",
            )
            # Test false values - each with unique handler type and matching env var
            false_values = ["false", "0", "no", "off", "FALSE", "NO", "OFF"]
            for idx, val in enumerate(false_values):
                resolver, _ = create_resolver(config)
                handler_type = f"falsy{idx}"
                env_var = f"TEST_{handler_type.upper()}_ENABLED"
                with patch.dict(os.environ, {env_var: val}):
                    result = resolver.resolve(
                        handler_type=handler_type,
                        inline_config={"enabled": True},
                    )
                assert result.enabled is False, f"Failed for falsy value '{val}'"


class TestBindingConfigResolverVaultSource:
    """Vault-based config resolution tests."""

    def test_vault_config_resolution(self) -> None:
        """Resolve config from Vault via SecretResolver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SecretResolver
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = json.dumps(
                {"timeout_ms": 60000, "priority": 100}
            )
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                result = resolver.resolve(
                    handler_type="db",
                    config_ref="vault:secret/data/handlers/db",
                )

            assert result.timeout_ms == 60000
            assert result.priority == 100
            mock_resolver.get_secret.assert_called_once()

    def test_vault_with_fragment(self) -> None:
        """Resolve specific field from Vault secret."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = json.dumps(
                {"timeout_ms": 70000}
            )
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                result = resolver.resolve(
                    handler_type="db",
                    config_ref="vault:secret/data/handlers/db#config",
                )

            assert result.timeout_ms == 70000
            # Verify fragment was passed
            call_args = mock_resolver.get_secret.call_args
            assert "config" in call_args[0][0]

    def test_vault_resolver_not_configured(self) -> None:
        """Error when vault: used but no SecretResolver registered in container."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            # Create container without SecretResolver registered
            resolver, _container = create_resolver(config, secret_resolver=None)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="vault:secret/data/db",
                )

            assert "secretresolver" in str(exc_info.value).lower()

    def test_vault_secret_not_found(self) -> None:
        """Handle missing Vault secret."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_resolver.get_secret.return_value = None  # Secret not found

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        config_ref="vault:secret/data/missing",
                    )

            assert "not found" in str(exc_info.value).lower()

    def test_vault_secret_exception(self) -> None:
        """Handle exception from SecretResolver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            # Use SecretResolutionError (specific exception) instead of generic Exception
            mock_resolver.get_secret.side_effect = SecretResolutionError(
                "Vault connection failed"
            )

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver.resolve(
                        handler_type="db",
                        config_ref="vault:secret/data/db",
                    )

            assert "vault" in str(exc_info.value).lower()

    def test_vault_inline_reference_resolution(self) -> None:
        """Resolve vault: references within config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "secret_value"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Config with vault reference in a value
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={
                        "timeout_ms": 5000,
                        "config": {"password": "vault:secret/db#password"},
                    },
                )

            # Config should have the secret resolved (in the nested dict)
            assert result.config is not None

    def test_fail_on_secret_error_true_raises_when_resolver_absent(self) -> None:
        """Raise error when fail_on_secret_error=True and SecretResolver absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                fail_on_secret_error=True,
            )
            # Create container without SecretResolver registered
            resolver, _container = create_resolver(config, secret_resolver=None)

            # Config with vault reference - should raise because no SecretResolver
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    inline_config={
                        "timeout_ms": 5000,
                        "config": {"password": "vault:secret/db#password"},
                    },
                )

            assert "secretresolver" in str(exc_info.value).lower()

    def test_fail_on_secret_error_false_skips_when_resolver_absent(self) -> None:
        """Skip vault resolution when fail_on_secret_error=False and SecretResolver absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                fail_on_secret_error=False,  # Default, but explicit for clarity
            )
            # Create container without SecretResolver registered
            resolver, _container = create_resolver(config, secret_resolver=None)

            # Config with vault reference - should NOT raise, just return config unchanged
            result = resolver.resolve(
                handler_type="db",
                inline_config={
                    "timeout_ms": 5000,
                    "config": {"password": "vault:secret/db#password"},
                },
            )

            # Config returned with original vault reference (not resolved)
            assert result.timeout_ms == 5000
            assert result.config is not None
            assert result.config.get("password") == "vault:secret/db#password"

    @pytest.mark.asyncio
    async def test_fail_on_secret_error_true_raises_async_when_resolver_absent(
        self,
    ) -> None:
        """Async: Raise error when fail_on_secret_error=True and SecretResolver absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                fail_on_secret_error=True,
            )
            resolver, _container = create_resolver(config, secret_resolver=None)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await resolver.resolve_async(
                    handler_type="db",
                    inline_config={
                        "timeout_ms": 5000,
                        "config": {"password": "vault:secret/db#password"},
                    },
                )

            assert "secretresolver" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_fail_on_secret_error_false_skips_async_when_resolver_absent(
        self,
    ) -> None:
        """Async: Skip vault resolution when fail_on_secret_error=False and SecretResolver absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                fail_on_secret_error=False,
            )
            resolver, _container = create_resolver(config, secret_resolver=None)

            result = await resolver.resolve_async(
                handler_type="db",
                inline_config={
                    "timeout_ms": 5000,
                    "config": {"password": "vault:secret/db#password"},
                },
            )

            assert result.timeout_ms == 5000
            assert result.config is not None
            assert result.config.get("password") == "vault:secret/db#password"


class TestBindingConfigResolverCaching:
    """Caching behavior tests."""

    def test_cache_hit(self) -> None:
        """Subsequent calls return cached value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=True,
                cache_ttl_seconds=300.0,
            )
            resolver, _container = create_resolver(config)

            # First call - cache miss
            result1 = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Modify file (shouldn't affect cached value)
            config_file.write_text(yaml.dump({"timeout_ms": 99999}))

            # Second call - cache hit
            result2 = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            assert result1.timeout_ms == 5000
            assert result2.timeout_ms == 5000  # Still cached value

            stats = resolver.get_cache_stats()
            assert stats.hits == 1
            assert stats.misses == 1

    def test_cache_miss(self) -> None:
        """First call is a cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            resolver.resolve(handler_type="db")

            stats = resolver.get_cache_stats()
            assert stats.misses == 1

    def test_cache_expiry(self) -> None:
        """Cached values expire after TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            # Very short TTL for testing
            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=True,
                cache_ttl_seconds=0.1,  # 100ms TTL
            )
            resolver, _container = create_resolver(config)

            # First call
            result1 = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Wait for TTL to expire
            time.sleep(0.15)

            # Modify file
            config_file.write_text(yaml.dump({"timeout_ms": 99999}))

            # Second call - should be cache miss due to expiry
            result2 = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            assert result1.timeout_ms == 5000
            assert result2.timeout_ms == 99999  # New value

            stats = resolver.get_cache_stats()
            assert stats.expired_evictions >= 1

    def test_refresh_invalidates_cache(self) -> None:
        """refresh() invalidates specific cache entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            # First call
            resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Modify file
            config_file.write_text(yaml.dump({"timeout_ms": 99999}))

            # Refresh cache
            resolver.refresh("db")

            # Next call should get new value
            result = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            assert result.timeout_ms == 99999

            stats = resolver.get_cache_stats()
            assert stats.refreshes == 1

    def test_refresh_all_clears_cache(self) -> None:
        """refresh_all() clears entire cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            # Cache multiple entries
            resolver.resolve(handler_type="db")
            resolver.resolve(handler_type="vault")
            resolver.resolve(handler_type="consul")

            stats_before = resolver.get_cache_stats()
            assert stats_before.total_entries == 3

            resolver.refresh_all()

            stats_after = resolver.get_cache_stats()
            assert stats_after.total_entries == 0
            assert stats_after.refreshes == 3

    def test_cache_disabled(self) -> None:
        """No caching when enable_caching=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=False,
            )
            resolver, _container = create_resolver(config)

            # First call
            result1 = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Modify file
            config_file.write_text(yaml.dump({"timeout_ms": 99999}))

            # Second call - should get new value
            result2 = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            assert result1.timeout_ms == 5000
            assert result2.timeout_ms == 99999  # New value, not cached

    def test_cache_stats(self) -> None:
        """Cache statistics are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            # Cache miss
            resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Cache hit
            resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Another cache hit
            resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            stats = resolver.get_cache_stats()
            assert stats.hits == 2
            assert stats.misses == 1
            assert stats.total_entries == 1
            assert stats.file_loads == 1


class TestBindingConfigResolverAsync:
    """Async operation tests."""

    @pytest.mark.asyncio
    async def test_resolve_async(self) -> None:
        """Basic async resolution works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            result = await resolver.resolve_async(
                handler_type="db",
                inline_config={"timeout_ms": 5000},
            )

            assert result.handler_type == "db"
            assert result.timeout_ms == 5000

    @pytest.mark.asyncio
    async def test_resolve_many_async_parallel(self) -> None:
        """Multiple async resolutions run in parallel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            bindings = [
                {"handler_type": "db", "config": {"timeout_ms": 5000}},
                {"handler_type": "vault", "config": {"timeout_ms": 10000}},
                {"handler_type": "consul", "config": {"timeout_ms": 15000}},
            ]

            results = await resolver.resolve_many_async(bindings)

            assert len(results) == 3
            assert results[0].handler_type == "db"
            assert results[1].handler_type == "vault"
            assert results[2].handler_type == "consul"

    @pytest.mark.asyncio
    async def test_async_caching(self) -> None:
        """Async operations use cache correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            # First call - cache miss
            result1 = await resolver.resolve_async(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            # Modify file
            config_file.write_text(yaml.dump({"timeout_ms": 99999}))

            # Second call - cache hit
            result2 = await resolver.resolve_async(
                handler_type="db",
                config_ref="file:db.yaml",
            )

            assert result1.timeout_ms == 5000
            assert result2.timeout_ms == 5000  # Still cached

            stats = resolver.get_cache_stats()
            assert stats.hits >= 1

    @pytest.mark.asyncio
    async def test_async_vault_resolution(self) -> None:
        """Async Vault resolution works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = json.dumps(
                {"timeout_ms": 60000}
            )

            # Mock async method
            async def mock_get_secret_async(name: str, required: bool = True) -> object:
                return mock_secret

            mock_resolver.get_secret_async = mock_get_secret_async

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                result = await resolver.resolve_async(
                    handler_type="db",
                    config_ref="vault:secret/data/db",
                )

            assert result.timeout_ms == 60000

    @pytest.mark.asyncio
    async def test_resolve_many_async_empty(self) -> None:
        """resolve_many_async with empty list returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            results = await resolver.resolve_many_async([])

            assert results == []


class TestBindingConfigResolverThreadSafety:
    """Thread safety tests."""

    def test_concurrent_resolve_same_handler(self) -> None:
        """Concurrent resolve calls for same handler are safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            results: list[ModelBindingConfig] = []
            errors: list[Exception] = []
            results_lock = threading.Lock()

            def resolve_handler() -> None:
                try:
                    result = resolver.resolve(
                        handler_type="db",
                        inline_config={"timeout_ms": 5000},
                    )
                    with results_lock:
                        results.append(result)
                except Exception as e:
                    with results_lock:
                        errors.append(e)

            threads = [threading.Thread(target=resolve_handler) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors encountered: {errors}"
            assert len(results) == 10
            assert all(r.timeout_ms == 5000 for r in results)

    def test_concurrent_resolve_different_handlers(self) -> None:
        """Concurrent resolve calls for different handlers are safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            handler_types = ["db", "vault", "consul", "kafka", "redis"]
            results: dict[str, ModelBindingConfig] = {}
            errors: list[Exception] = []
            results_lock = threading.Lock()

            def resolve_handler(handler_type: str) -> None:
                try:
                    result = resolver.resolve(
                        handler_type=handler_type,
                        inline_config={"timeout_ms": 5000},
                    )
                    with results_lock:
                        results[handler_type] = result
                except Exception as e:
                    with results_lock:
                        errors.append(e)

            threads = [
                threading.Thread(target=resolve_handler, args=(ht,))
                for ht in handler_types
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors encountered: {errors}"
            assert len(results) == 5
            for ht in handler_types:
                assert results[ht].handler_type == ht

    def test_cache_thread_safety(self) -> None:
        """Cache operations are thread-safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            errors: list[Exception] = []
            stop_event = threading.Event()

            def reader() -> None:
                while not stop_event.is_set():
                    try:
                        resolver.resolve(
                            handler_type="db",
                            inline_config={"timeout_ms": 5000},
                        )
                    except Exception as e:
                        errors.append(e)

            def refresher() -> None:
                while not stop_event.is_set():
                    try:
                        resolver.refresh("db")
                    except Exception as e:
                        errors.append(e)

            readers = [threading.Thread(target=reader) for _ in range(5)]
            refreshers = [threading.Thread(target=refresher) for _ in range(2)]

            for t in readers + refreshers:
                t.start()

            time.sleep(0.1)
            stop_event.set()

            for t in readers + refreshers:
                t.join()

            assert len(errors) == 0, f"Errors encountered: {errors}"


class TestBindingConfigResolverValidation:
    """Input validation tests."""

    def test_invalid_handler_type_empty(self) -> None:
        """Empty handler_type raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(handler_type="")

    def test_invalid_timeout_ms_too_low(self) -> None:
        """timeout_ms below minimum raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    inline_config={"timeout_ms": 50},  # Below minimum of 100
                )

    def test_invalid_timeout_ms_too_high(self) -> None:
        """timeout_ms above maximum raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    inline_config={"timeout_ms": 700000},  # Above maximum
                )

    def test_invalid_priority_out_of_range(self) -> None:
        """priority out of range raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    inline_config={"priority": 200},  # Above maximum of 100
                )

    def test_invalid_retry_policy(self) -> None:
        """Invalid retry_policy raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    inline_config={
                        "retry_policy": {
                            "max_retries": 20,  # Above maximum of 10
                        }
                    },
                )

    def test_retry_policy_max_delay_less_than_base(self) -> None:
        """max_delay_ms less than base_delay_ms raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    inline_config={
                        "retry_policy": {
                            "base_delay_ms": 1000,
                            "max_delay_ms": 500,  # Less than base
                        }
                    },
                )

    def test_strict_validation_extra_fields(self) -> None:
        """Strict validation fails on unknown fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                strict_validation=True,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    inline_config={
                        "timeout_ms": 5000,
                        "unknown_field": "value",  # Unknown field
                    },
                )

    def test_non_strict_validation_ignores_extra_fields(self) -> None:
        """Non-strict validation ignores unknown fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                strict_validation=False,
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(
                handler_type="db",
                inline_config={
                    "timeout_ms": 5000,
                    "unknown_field": "value",  # Should be ignored
                },
            )

            assert result.timeout_ms == 5000

    def test_unknown_config_ref_scheme(self) -> None:
        """Unknown config_ref scheme raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(
                    handler_type="db",
                    config_ref="unknown:path/to/config",
                )

    def test_scheme_not_in_allowed_schemes(self) -> None:
        """Scheme not in allowed_schemes raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                allowed_schemes=frozenset({"file"}),  # Only file allowed
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="env:DB_CONFIG",
                )

            assert "not in allowed schemes" in str(exc_info.value).lower()


class TestBindingConfigResolverSecurity:
    """Security-related tests."""

    def test_error_messages_sanitized(self) -> None:
        """Error messages don't contain sensitive data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:secret_passwords.yaml",
                )

            error_msg = str(exc_info.value)
            # Should not expose full path details
            assert (
                "secret_password" not in error_msg.lower()
                or "not found" in error_msg.lower()
            )

    def test_unknown_scheme_rejected(self) -> None:
        """Unknown config_ref schemes are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="http://example.com/config",
                )

            # Should raise a generic error that doesn't expose the invalid scheme
            # Security: detailed parse errors are logged at DEBUG level, not in exception
            error_msg = str(exc_info.value).lower()
            assert "invalid config reference format" in error_msg
            # Ensure the scheme and URL are NOT exposed in the error message
            assert "http" not in error_msg
            assert "example.com" not in error_msg

    def test_source_description_sanitized(self) -> None:
        """Source description in cache doesn't expose paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "secret.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=True,
            )
            resolver, _container = create_resolver(config)

            resolver.resolve(
                handler_type="db",
                config_ref="file:secret.yaml",
            )

            # Check that internal cache entry doesn't expose full path
            # (This is an internal detail but important for security)
            cache_entry = resolver._cache.get("db")
            if cache_entry:
                assert "secret.yaml" not in cache_entry.source
                assert "..." in cache_entry.source or "file://" in cache_entry.source


class TestModelConfigRef:
    """ModelConfigRef parsing tests."""

    def test_parse_file_absolute(self) -> None:
        """Parse file:///absolute/path."""
        result = ModelConfigRef.parse("file:///etc/onex/config.yaml")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.FILE
        assert result.config_ref.path == "/etc/onex/config.yaml"

    def test_parse_file_relative(self) -> None:
        """Parse file://relative/path."""
        result = ModelConfigRef.parse("file://relative/path/config.yaml")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.FILE
        assert result.config_ref.path == "relative/path/config.yaml"

    def test_parse_file_shorthand(self) -> None:
        """Parse file:path shorthand."""
        result = ModelConfigRef.parse("file:configs/db.yaml")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.FILE
        assert result.config_ref.path == "configs/db.yaml"

    def test_parse_env(self) -> None:
        """Parse env:VAR_NAME."""
        result = ModelConfigRef.parse("env:DB_CONFIG")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.ENV
        assert result.config_ref.path == "DB_CONFIG"

    def test_parse_vault(self) -> None:
        """Parse vault:path."""
        result = ModelConfigRef.parse("vault:secret/data/db")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.VAULT
        assert result.config_ref.path == "secret/data/db"
        assert result.config_ref.fragment is None

    def test_parse_vault_with_fragment(self) -> None:
        """Parse vault:path#field."""
        result = ModelConfigRef.parse("vault:secret/data/db#password")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.VAULT
        assert result.config_ref.path == "secret/data/db"
        assert result.config_ref.fragment == "password"

    def test_parse_invalid_empty(self) -> None:
        """Empty string returns error."""
        result = ModelConfigRef.parse("")

        assert not result.success
        assert result.config_ref is None
        assert result.error_message is not None
        assert "empty" in result.error_message.lower()

    def test_parse_invalid_scheme(self) -> None:
        """Unknown scheme returns error."""
        result = ModelConfigRef.parse("unknown:path")

        assert not result.success
        assert result.config_ref is None
        assert "unknown" in result.error_message.lower()

    def test_parse_path_traversal(self) -> None:
        """Multiple consecutive path traversal returns error."""
        result = ModelConfigRef.parse("file:../../../etc/passwd")

        assert not result.success
        assert result.config_ref is None
        assert "traversal" in result.error_message.lower()

    def test_parse_parent_directory_single(self) -> None:
        """Single parent directory reference is allowed."""
        result = ModelConfigRef.parse("file:../config.yaml")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.path == "../config.yaml"

    def test_parse_parent_directory_in_path(self) -> None:
        """Parent directory in middle of path is allowed."""
        result = ModelConfigRef.parse("file:a/b/../c/config.yaml")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.path == "a/b/../c/config.yaml"

    def test_parse_explicit_relative(self) -> None:
        """Explicit relative path (./path) is allowed."""
        result = ModelConfigRef.parse("file:./config.yaml")

        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.path == "./config.yaml"

    def test_parse_missing_path(self) -> None:
        """Missing path after scheme returns error."""
        result = ModelConfigRef.parse("file:")

        assert not result.success
        assert "missing" in result.error_message.lower()

    def test_parse_missing_separator(self) -> None:
        """Missing scheme separator returns error."""
        result = ModelConfigRef.parse("noscheme")

        assert not result.success
        assert ":" in result.error_message

    def test_to_uri_roundtrip_file(self) -> None:
        """parse() and to_uri() are inverse operations for file."""
        original = "file:configs/db.yaml"
        result = ModelConfigRef.parse(original)

        assert result.success
        assert result.config_ref.to_uri() == original

    def test_to_uri_roundtrip_vault_with_fragment(self) -> None:
        """parse() and to_uri() are inverse operations for vault with fragment."""
        original = "vault:secret/db#password"
        result = ModelConfigRef.parse(original)

        assert result.success
        assert result.config_ref.to_uri() == original

    def test_bool_context(self) -> None:
        """Result can be used in boolean context."""
        success_result = ModelConfigRef.parse("file:config.yaml")
        failure_result = ModelConfigRef.parse("")

        assert success_result  # Truthy
        assert not failure_result  # Falsy


class TestModelBindingConfig:
    """ModelBindingConfig validation tests."""

    def test_minimal_valid(self) -> None:
        """Minimal valid configuration."""
        config = ModelBindingConfig(handler_type="db")

        assert config.handler_type == "db"
        assert config.enabled is True
        assert config.priority == 0
        assert config.timeout_ms == 30000

    def test_full_valid(self) -> None:
        """Full configuration with all fields."""
        config = ModelBindingConfig(
            handler_type="db",
            name="primary-postgres",
            enabled=True,
            priority=50,
            timeout_ms=10000,
            rate_limit_per_second=100.0,
            retry_policy=ModelRetryPolicy(
                max_retries=5,
                backoff_strategy="exponential",
                base_delay_ms=200,
                max_delay_ms=10000,
            ),
        )

        assert config.handler_type == "db"
        assert config.name == "primary-postgres"
        assert config.priority == 50
        assert config.retry_policy.max_retries == 5

    def test_invalid_timeout_too_low(self) -> None:
        """Timeout below minimum rejected."""
        with pytest.raises(ValueError):
            ModelBindingConfig(
                handler_type="db",
                timeout_ms=50,  # Below 100
            )

    def test_invalid_timeout_too_high(self) -> None:
        """Timeout above maximum rejected."""
        with pytest.raises(ValueError):
            ModelBindingConfig(
                handler_type="db",
                timeout_ms=700000,  # Above 600000
            )

    def test_invalid_priority_out_of_range(self) -> None:
        """Priority out of range rejected."""
        with pytest.raises(ValueError):
            ModelBindingConfig(
                handler_type="db",
                priority=200,  # Above 100
            )

    def test_handler_type_required(self) -> None:
        """handler_type is required."""
        with pytest.raises(ValueError):
            ModelBindingConfig()  # type: ignore[call-arg]

    def test_frozen_immutability(self) -> None:
        """Config is immutable after creation."""
        config = ModelBindingConfig(handler_type="db")

        with pytest.raises(Exception):  # ValidationError or AttributeError
            config.timeout_ms = 5000  # type: ignore[misc]

    def test_config_ref_scheme_validation(self) -> None:
        """config_ref scheme is validated."""
        with pytest.raises(ValueError):
            ModelBindingConfig(
                handler_type="db",
                config_ref="http://example.com/config",  # Invalid scheme
            )

    def test_get_effective_name_with_name(self) -> None:
        """get_effective_name returns name when set."""
        config = ModelBindingConfig(
            handler_type="db",
            name="my-database",
        )

        assert config.get_effective_name() == "my-database"

    def test_get_effective_name_without_name(self) -> None:
        """get_effective_name returns handler_type when name not set."""
        config = ModelBindingConfig(handler_type="db")

        assert config.get_effective_name() == "db"


class TestModelRetryPolicy:
    """ModelRetryPolicy validation tests."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        policy = ModelRetryPolicy()

        assert policy.max_retries == 3
        assert policy.backoff_strategy == "exponential"
        assert policy.base_delay_ms == 100
        assert policy.max_delay_ms == 5000

    def test_max_delay_gte_base_delay(self) -> None:
        """max_delay_ms must be >= base_delay_ms."""
        with pytest.raises(ValueError) as exc_info:
            ModelRetryPolicy(
                base_delay_ms=1000,
                max_delay_ms=500,  # Less than base
            )

        assert "base_delay" in str(exc_info.value).lower()

    def test_backoff_strategy_literal(self) -> None:
        """backoff_strategy must be 'fixed' or 'exponential'."""
        # Valid values work
        policy_fixed = ModelRetryPolicy(backoff_strategy="fixed")
        policy_exp = ModelRetryPolicy(backoff_strategy="exponential")

        assert policy_fixed.backoff_strategy == "fixed"
        assert policy_exp.backoff_strategy == "exponential"

        # Invalid value fails
        with pytest.raises(ValueError):
            ModelRetryPolicy(backoff_strategy="linear")  # type: ignore[arg-type]

    def test_max_retries_bounds(self) -> None:
        """max_retries must be 0-10."""
        # Valid bounds
        ModelRetryPolicy(max_retries=0)
        ModelRetryPolicy(max_retries=10)

        # Invalid bounds
        with pytest.raises(ValueError):
            ModelRetryPolicy(max_retries=-1)

        with pytest.raises(ValueError):
            ModelRetryPolicy(max_retries=11)

    def test_base_delay_bounds(self) -> None:
        """base_delay_ms must be 10-60000."""
        # Valid bounds
        ModelRetryPolicy(base_delay_ms=10)
        ModelRetryPolicy(base_delay_ms=60000, max_delay_ms=60000)

        # Invalid bounds
        with pytest.raises(ValueError):
            ModelRetryPolicy(base_delay_ms=5)

        with pytest.raises(ValueError):
            ModelRetryPolicy(base_delay_ms=70000)

    def test_frozen_immutability(self) -> None:
        """RetryPolicy is immutable after creation."""
        policy = ModelRetryPolicy()

        with pytest.raises(Exception):
            policy.max_retries = 5  # type: ignore[misc]


class TestBindingConfigResolverInlinePrecedence:
    """Tests for inline config precedence over file config."""

    def test_inline_takes_precedence_over_file(self) -> None:
        """Inline config takes precedence over file config for overlapping keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "db.yaml"
            config_file.write_text(
                yaml.dump({"timeout_ms": 5000, "priority": 10, "enabled": True})
            )

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver, _container = create_resolver(config)

            result = resolver.resolve(
                handler_type="db",
                config_ref="file:db.yaml",
                inline_config={"timeout_ms": 99999},  # Override
            )

            # Inline takes precedence for overlapping key
            assert result.timeout_ms == 99999
            # File config used for non-overlapping keys
            assert result.priority == 10
            assert result.enabled is True


class TestBindingConfigResolverRecursionDepth:
    """Tests for recursion depth limits in nested config resolution."""

    def test_resolve_vault_refs_depth_limit(self) -> None:
        """Deeply nested configs hit recursion limit.

        Tests that _resolve_vault_refs raises ProtocolConfigurationError
        when nesting exceeds _MAX_NESTED_CONFIG_DEPTH (20).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SecretResolver
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "secret_value"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Create deeply nested config (21 levels deep to exceed limit of 20)
                nested: dict[str, object] = {"value": "vault:secret/test"}
                for _ in range(21):
                    nested = {"nested": nested}

                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver._resolve_vault_refs(nested, uuid4(), depth=0)

                assert "nesting exceeds maximum depth" in str(exc_info.value).lower()

    def test_resolve_vault_refs_at_depth_limit_succeeds(self) -> None:
        """Config at exactly the depth limit succeeds.

        Tests that configs with nesting at exactly _MAX_NESTED_CONFIG_DEPTH (20)
        are processed successfully.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SecretResolver
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "resolved_secret"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Create config at exactly 20 levels deep (should succeed)
                nested: dict[str, object] = {"value": "vault:secret/test"}
                for _ in range(20):
                    nested = {"nested": nested}

                # Should not raise - at the limit, not exceeding it
                result = resolver._resolve_vault_refs(nested, uuid4(), depth=0)

                # Verify the nested structure was processed
                assert result is not None
                assert "nested" in result

    @pytest.mark.asyncio
    async def test_resolve_vault_refs_async_depth_limit(self) -> None:
        """Async vault ref resolution also respects depth limit.

        Tests that _resolve_vault_refs_async raises ProtocolConfigurationError
        when nesting exceeds _MAX_NESTED_CONFIG_DEPTH (20).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SecretResolver with async support
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "secret_value"

            async def mock_get_secret_async(name: str, required: bool = True) -> object:
                return mock_secret

            mock_resolver.get_secret_async = mock_get_secret_async

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Create deeply nested config (21 levels deep to exceed limit of 20)
                nested: dict[str, object] = {"value": "vault:secret/test"}
                for _ in range(21):
                    nested = {"nested": nested}

                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    await resolver._resolve_vault_refs_async(nested, uuid4(), depth=0)

                assert "nesting exceeds maximum depth" in str(exc_info.value).lower()


class TestAsyncKeyLockCleanup:
    """Tests for _cleanup_stale_async_key_locks method.

    These tests verify that async key locks are properly cleaned up to prevent
    unbounded memory growth in long-running processes. The cleanup mechanism
    removes locks that are:
    1. Older than _ASYNC_KEY_LOCK_MAX_AGE_SECONDS (3600s)
    2. Not currently held (unlocked)

    Related:
    - OMN-765: BindingConfigResolver implementation
    - PR #168 review feedback
    """

    def test_cleanup_removes_stale_unlocked_locks(self) -> None:
        """Test that stale unlocked locks are removed during cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Create a lock and make it stale
            resolver._get_async_key_lock("test_handler")

            # Simulate lock being old by manipulating timestamp
            # _ASYNC_KEY_LOCK_MAX_AGE_SECONDS is 3600
            with resolver._lock:
                resolver._async_key_lock_timestamps["test_handler"] = (
                    time.monotonic() - 4000  # Older than 3600 seconds
                )
                resolver._cleanup_stale_async_key_locks()

            # Verify lock was removed
            assert "test_handler" not in resolver._async_key_locks
            assert "test_handler" not in resolver._async_key_lock_timestamps
            assert resolver._async_key_lock_cleanups == 1

    def test_cleanup_preserves_recent_locks(self) -> None:
        """Test that recent locks are preserved during cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Create a recent lock
            resolver._get_async_key_lock("recent_handler")

            with resolver._lock:
                resolver._cleanup_stale_async_key_locks()

            # Verify lock is preserved (timestamp is recent)
            assert "recent_handler" in resolver._async_key_locks
            assert "recent_handler" in resolver._async_key_lock_timestamps
            # No cleanup should have occurred since the lock is recent
            assert resolver._async_key_lock_cleanups == 0

    @pytest.mark.asyncio
    async def test_cleanup_preserves_held_locks(self) -> None:
        """Test that held locks are not removed even if stale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Create and acquire a lock
            lock = resolver._get_async_key_lock("held_handler")

            async with lock:
                # Make the timestamp stale while lock is held
                with resolver._lock:
                    resolver._async_key_lock_timestamps["held_handler"] = (
                        time.monotonic() - 4000  # Older than 3600 seconds
                    )
                    resolver._cleanup_stale_async_key_locks()

                # Verify held lock is preserved despite being stale
                assert "held_handler" in resolver._async_key_locks
                assert "held_handler" in resolver._async_key_lock_timestamps
                # No cleanup counter increment since no locks were actually cleaned
                assert resolver._async_key_lock_cleanups == 0

    def test_cleanup_counter_increments_only_when_locks_cleaned(self) -> None:
        """Test that _async_key_lock_cleanups stat increments only when locks are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            initial_cleanups = resolver._async_key_lock_cleanups
            assert initial_cleanups == 0

            # Create multiple stale locks
            for i in range(5):
                resolver._get_async_key_lock(f"handler_{i}")

            # Make all locks stale
            with resolver._lock:
                current_time = time.monotonic()
                for i in range(5):
                    resolver._async_key_lock_timestamps[f"handler_{i}"] = (
                        current_time - 4000  # Older than 3600 seconds
                    )
                resolver._cleanup_stale_async_key_locks()

            # Verify all locks were removed
            assert len(resolver._async_key_locks) == 0
            # Counter should increment once per cleanup call (not per lock)
            assert resolver._async_key_lock_cleanups == 1

            # Calling cleanup again with no stale locks should not increment
            with resolver._lock:
                resolver._cleanup_stale_async_key_locks()
            assert resolver._async_key_lock_cleanups == 1

    def test_cleanup_mixed_stale_and_recent_locks(self) -> None:
        """Test cleanup with a mix of stale and recent locks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Create multiple locks
            for i in range(3):
                resolver._get_async_key_lock(f"stale_handler_{i}")
            for i in range(2):
                resolver._get_async_key_lock(f"recent_handler_{i}")

            # Make only some locks stale
            with resolver._lock:
                current_time = time.monotonic()
                for i in range(3):
                    resolver._async_key_lock_timestamps[f"stale_handler_{i}"] = (
                        current_time - 4000  # Older than 3600 seconds
                    )
                # recent_handler_* timestamps remain current

                resolver._cleanup_stale_async_key_locks()

            # Verify only stale locks were removed
            for i in range(3):
                assert f"stale_handler_{i}" not in resolver._async_key_locks
            for i in range(2):
                assert f"recent_handler_{i}" in resolver._async_key_locks

            assert len(resolver._async_key_locks) == 2
            assert resolver._async_key_lock_cleanups == 1

    def test_threshold_triggered_cleanup(self) -> None:
        """Test that cleanup is triggered when lock count exceeds threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # _ASYNC_KEY_LOCK_CLEANUP_THRESHOLD is 1000
            # Manually create locks to exceed threshold (simulating many handlers)
            with resolver._lock:
                stale_time = time.monotonic() - 4000  # Older than 3600 seconds
                for i in range(1001):
                    lock = asyncio.Lock()
                    resolver._async_key_locks[f"handler_{i}"] = lock
                    resolver._async_key_lock_timestamps[f"handler_{i}"] = stale_time

            # Verify we have more than threshold
            assert len(resolver._async_key_locks) > 1000

            # Create one more lock via _get_async_key_lock to trigger cleanup
            resolver._get_async_key_lock("trigger_handler")

            # After cleanup, stale locks should be removed
            # Only the "trigger_handler" (which is recent) should remain
            assert len(resolver._async_key_locks) == 1
            assert "trigger_handler" in resolver._async_key_locks
            assert resolver._async_key_lock_cleanups == 1

    def test_cache_stats_includes_lock_cleanup_count(self) -> None:
        """Test that get_cache_stats() returns async_key_lock_cleanups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Initial stats
            stats = resolver.get_cache_stats()
            assert stats.async_key_lock_cleanups == 0

            # Create and make a lock stale, then clean up
            resolver._get_async_key_lock("test_handler")
            with resolver._lock:
                resolver._async_key_lock_timestamps["test_handler"] = (
                    time.monotonic() - 4000
                )
                resolver._cleanup_stale_async_key_locks()

            # Verify stats updated
            stats = resolver.get_cache_stats()
            assert stats.async_key_lock_cleanups == 1

    @pytest.mark.asyncio
    async def test_async_lock_released_after_successful_resolution(self) -> None:
        """Verify async lock is released after successful resolution completes.

        This test ensures that the per-key async lock used during resolve_async()
        is properly released after resolution completes successfully, preventing
        lock leaks that could cause deadlocks in subsequent calls.

        Related:
        - PR #168 review feedback: missing test for async lock cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=False,  # Disable caching to force resolution each time
            )
            resolver, _container = create_resolver(config)

            handler_type = "test_handler_success"

            # Perform successful async resolution
            result = await resolver.resolve_async(
                handler_type=handler_type,
                inline_config={"timeout_ms": 5000},
            )

            # Verify resolution succeeded
            assert result is not None
            assert result.handler_type == handler_type
            assert result.timeout_ms == 5000

            # Verify the lock exists (was created during resolution)
            assert handler_type in resolver._async_key_locks

            # Verify the lock is NOT held (released after resolution)
            lock = resolver._async_key_locks[handler_type]
            assert not lock.locked(), (
                "Async lock should be released after successful resolution"
            )

    @pytest.mark.asyncio
    async def test_async_lock_released_after_failed_resolution(self) -> None:
        """Verify async lock is released after resolution fails with error.

        This test ensures that the per-key async lock is properly released
        even when resolution raises an exception, preventing lock leaks.

        Related:
        - PR #168 review feedback: missing test for async lock cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=False,
            )
            resolver, _container = create_resolver(config)

            handler_type = "test_handler_failure"

            # Attempt resolution that will fail (empty handler_type causes validation error)
            # Note: We use an empty string as handler_type which fails validation
            with pytest.raises(ProtocolConfigurationError):
                await resolver.resolve_async(
                    handler_type="",  # Invalid: empty handler_type
                    inline_config={"timeout_ms": 5000},
                )

            # The empty string handler type should have had a lock created
            # Verify the lock exists (was created during resolution attempt)
            assert "" in resolver._async_key_locks

            # Verify the lock is NOT held (released despite error)
            lock = resolver._async_key_locks[""]
            assert not lock.locked(), (
                "Async lock should be released after failed resolution"
            )

    @pytest.mark.asyncio
    async def test_async_lock_released_after_file_not_found_error(self) -> None:
        """Verify async lock is released when config file is not found.

        This tests a more realistic error scenario where resolution fails
        due to a missing configuration file.

        Related:
        - PR #168 review feedback: missing test for async lock cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=False,
            )
            resolver, _container = create_resolver(config)

            handler_type = "test_handler_file_error"

            # Attempt resolution with non-existent file
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await resolver.resolve_async(
                    handler_type=handler_type,
                    config_ref="file:nonexistent.yaml",
                )

            # Verify appropriate error
            assert "not found" in str(exc_info.value).lower()

            # Verify the lock exists (was created during resolution)
            assert handler_type in resolver._async_key_locks

            # Verify the lock is NOT held (released despite error)
            lock = resolver._async_key_locks[handler_type]
            assert not lock.locked(), (
                "Async lock should be released after file not found error"
            )

    @pytest.mark.asyncio
    async def test_no_lock_leak_on_consecutive_operations(self) -> None:
        """Verify no lock leaks occur across consecutive async operations.

        This test ensures that multiple consecutive resolve_async() calls
        (both successful and failed) don't accumulate held locks.

        Related:
        - PR #168 review feedback: missing test for async lock cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=False,
            )
            resolver, _container = create_resolver(config)

            # Perform multiple consecutive operations
            handlers = ["handler_a", "handler_b", "handler_c"]

            for handler_type in handlers:
                # Successful resolution
                result = await resolver.resolve_async(
                    handler_type=handler_type,
                    inline_config={"timeout_ms": 1000},
                )
                assert result.handler_type == handler_type

            # Also perform some failing operations
            for i in range(3):
                try:
                    await resolver.resolve_async(
                        handler_type=f"fail_handler_{i}",
                        config_ref="file:nonexistent.yaml",
                    )
                except ProtocolConfigurationError:
                    pass  # Expected

            # Verify NO locks are held after all operations
            for handler_type in handlers:
                assert handler_type in resolver._async_key_locks
                lock = resolver._async_key_locks[handler_type]
                assert not lock.locked(), f"Lock for {handler_type} should be released"

            for i in range(3):
                handler_type = f"fail_handler_{i}"
                assert handler_type in resolver._async_key_locks
                lock = resolver._async_key_locks[handler_type]
                assert not lock.locked(), (
                    f"Lock for {handler_type} should be released despite error"
                )

            # Verify cache stats track lock count correctly
            stats = resolver.get_cache_stats()
            assert stats.async_key_lock_count == len(handlers) + 3

    @pytest.mark.asyncio
    async def test_lock_held_during_resolution(self) -> None:
        """Verify the async lock is held during resolution.

        This test confirms that the lock is properly acquired and held
        while resolution is in progress, preventing concurrent resolution
        for the same handler type.

        Related:
        - PR #168 review feedback: missing test for async lock cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Create a valid config file
            config_file = config_dir / "slow_handler.yaml"
            config_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                enable_caching=False,
            )
            resolver, _container = create_resolver(config)

            handler_type = "slow_handler"
            lock_was_held = False

            # Track whether lock is held during resolution by intercepting
            original_resolve_config = resolver._resolve_config_async

            async def intercepting_resolve(
                handler_type: str,
                config_ref: str | None,
                inline_config: dict[str, object] | None,
                correlation_id: object,
            ) -> ModelBindingConfig:
                nonlocal lock_was_held
                # Check if the lock is held during resolution
                if handler_type in resolver._async_key_locks:
                    lock = resolver._async_key_locks[handler_type]
                    lock_was_held = lock.locked()
                return await original_resolve_config(
                    handler_type, config_ref, inline_config, correlation_id
                )

            # Patch the internal method to check lock state during resolution
            resolver._resolve_config_async = intercepting_resolve  # type: ignore[method-assign]

            # Perform resolution
            result = await resolver.resolve_async(
                handler_type=handler_type,
                config_ref="file:slow_handler.yaml",
            )

            # Verify resolution succeeded
            assert result.timeout_ms == 5000

            # Verify lock was held during resolution
            assert lock_was_held, "Lock should be held during resolution"

            # Verify lock is released after resolution
            lock = resolver._async_key_locks[handler_type]
            assert not lock.locked(), "Lock should be released after resolution"

    @pytest.mark.asyncio
    async def test_async_key_lock_cleanup_removes_stale_locks(self) -> None:
        """Test that stale locks created through resolve_async are cleaned up.

        This is an end-to-end test that verifies the async lock cleanup mechanism
        when locks are created through actual resolve_async() calls, not through
        direct _get_async_key_lock() calls.

        The test:
        1. Creates a resolver with container
        2. Makes multiple async resolve calls to create locks
        3. Verifies locks are created
        4. Simulates time passage by manipulating timestamps
        5. Triggers cleanup via threshold
        6. Verifies stale locks are removed

        Related:
        - OMN-765: BindingConfigResolver implementation
        - PR #168 review feedback: missing test for async lock cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                enable_caching=False,  # Disable caching to force resolution each time
            )
            resolver, _container = create_resolver(config)

            # Make multiple async resolve calls to create locks
            handler_types = ["handler_a", "handler_b", "handler_c"]
            for handler_type in handler_types:
                await resolver.resolve_async(
                    handler_type=handler_type,
                    inline_config={"timeout_ms": 5000},
                )

            # Verify locks were created through resolve_async
            for handler_type in handler_types:
                assert handler_type in resolver._async_key_locks, (
                    f"Lock for {handler_type} should exist after resolve_async"
                )
                assert handler_type in resolver._async_key_lock_timestamps, (
                    f"Timestamp for {handler_type} should exist after resolve_async"
                )

            # Verify cache stats report correct lock count
            stats = resolver.get_cache_stats()
            assert stats.async_key_lock_count == len(handler_types)
            assert stats.async_key_lock_cleanups == 0

            # Simulate time passage by making locks appear stale
            # _ASYNC_KEY_LOCK_MAX_AGE_SECONDS is 3600 (1 hour)
            stale_time = time.monotonic() - 4000  # Older than 3600 seconds
            with resolver._lock:
                for handler_type in handler_types:
                    resolver._async_key_lock_timestamps[handler_type] = stale_time

            # To trigger cleanup, we need to exceed _ASYNC_KEY_LOCK_CLEANUP_THRESHOLD (1000)
            # Add enough locks to exceed threshold
            with resolver._lock:
                for i in range(1001):
                    lock = asyncio.Lock()
                    resolver._async_key_locks[f"bulk_handler_{i}"] = lock
                    resolver._async_key_lock_timestamps[f"bulk_handler_{i}"] = (
                        stale_time
                    )

            # Verify we have many locks now (original + bulk)
            assert len(resolver._async_key_locks) > 1000

            # Trigger cleanup by creating one more lock via resolve_async
            # This should call _get_async_key_lock which triggers cleanup when threshold exceeded
            await resolver.resolve_async(
                handler_type="trigger_handler",
                inline_config={"timeout_ms": 5000},
            )

            # Verify stale locks were cleaned up
            # Only the trigger_handler (which is recent) should remain
            for handler_type in handler_types:
                assert handler_type not in resolver._async_key_locks, (
                    f"Stale lock for {handler_type} should be removed after cleanup"
                )
                assert handler_type not in resolver._async_key_lock_timestamps, (
                    f"Stale timestamp for {handler_type} should be removed after cleanup"
                )

            # Bulk handlers should also be removed (they were also stale)
            for i in range(1001):
                assert f"bulk_handler_{i}" not in resolver._async_key_locks

            # Only the trigger_handler should remain
            assert "trigger_handler" in resolver._async_key_locks
            assert len(resolver._async_key_locks) == 1

            # Verify cleanup counter was incremented
            stats = resolver.get_cache_stats()
            assert stats.async_key_lock_cleanups == 1
            assert stats.async_key_lock_count == 1

    def test_configurable_cleanup_threshold(self) -> None:
        """Test that async_lock_cleanup_threshold config is respected.

        Verifies that the cleanup threshold can be configured via
        ModelBindingConfigResolverConfig and the resolver uses the configured
        value instead of the default (1000).

        Related:
        - PR #168 review feedback: make cleanup threshold configurable
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure with a low threshold (10 instead of default 200)
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                async_lock_cleanup_threshold=10,
                async_lock_max_age_seconds=3600.0,
            )
            resolver, _container = create_resolver(config)

            # Create locks just above the custom threshold
            stale_time = time.monotonic() - 4000  # Older than 3600 seconds
            with resolver._lock:
                for i in range(11):  # 11 > threshold of 10
                    lock = asyncio.Lock()
                    resolver._async_key_locks[f"handler_{i}"] = lock
                    resolver._async_key_lock_timestamps[f"handler_{i}"] = stale_time

            # Verify we have 11 locks
            assert len(resolver._async_key_locks) == 11

            # Create one more lock to trigger cleanup
            resolver._get_async_key_lock("trigger_handler")

            # After cleanup, only the trigger_handler (which is recent) should remain
            assert len(resolver._async_key_locks) == 1
            assert "trigger_handler" in resolver._async_key_locks
            assert resolver._async_key_lock_cleanups == 1

    def test_configurable_max_age_seconds(self) -> None:
        """Test that async_lock_max_age_seconds config is respected.

        Verifies that the max age can be configured via
        ModelBindingConfigResolverConfig and the resolver uses the configured
        value instead of the default (3600 seconds).

        Related:
        - PR #168 review feedback: make cleanup max age configurable
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure with a short max age (60 seconds instead of default 3600)
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                async_lock_cleanup_threshold=5,
                async_lock_max_age_seconds=60.0,  # 1 minute
            )
            resolver, _container = create_resolver(config)

            # Create some locks: some older than 60s, some newer
            current_time = time.monotonic()
            with resolver._lock:
                # These should be cleaned up (older than 60 seconds)
                for i in range(3):
                    lock = asyncio.Lock()
                    resolver._async_key_locks[f"old_handler_{i}"] = lock
                    resolver._async_key_lock_timestamps[f"old_handler_{i}"] = (
                        current_time - 100  # 100 seconds old, > 60s threshold
                    )
                # These should be preserved (newer than 60 seconds)
                for i in range(3):
                    lock = asyncio.Lock()
                    resolver._async_key_locks[f"new_handler_{i}"] = lock
                    resolver._async_key_lock_timestamps[f"new_handler_{i}"] = (
                        current_time - 30  # 30 seconds old, < 60s threshold
                    )

            # Verify we have 6 locks total
            assert len(resolver._async_key_locks) == 6

            # Create one more lock to trigger cleanup (exceeds threshold of 5)
            resolver._get_async_key_lock("trigger_handler")

            # After cleanup:
            # - old_handler_* (3 locks) should be removed (age > 60s)
            # - new_handler_* (3 locks) should be preserved (age < 60s)
            # - trigger_handler should be added (brand new)
            assert len(resolver._async_key_locks) == 4
            for i in range(3):
                assert f"old_handler_{i}" not in resolver._async_key_locks
                assert f"new_handler_{i}" in resolver._async_key_locks
            assert "trigger_handler" in resolver._async_key_locks
            assert resolver._async_key_lock_cleanups == 1


class TestAsyncKeyLockCleanupOnEviction:
    """Tests for async key lock cleanup during cache eviction.

    These tests verify that async key locks are properly cleaned up when their
    corresponding cache entries are evicted, either via LRU eviction or TTL
    expiration. This prevents memory leaks in long-running processes.

    Related:
    - OMN-765: BindingConfigResolver implementation
    - PR #168 review feedback: async lock memory leak on eviction
    """

    def test_lru_eviction_cleans_up_async_lock(self) -> None:
        """Test that LRU eviction also removes the associated async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure with max_cache_entries=2 to trigger LRU eviction
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                max_cache_entries=2,
            )
            resolver, _container = create_resolver(config)

            # Create async locks manually for testing
            with resolver._lock:
                resolver._async_key_locks["handler_a"] = asyncio.Lock()
                resolver._async_key_lock_timestamps["handler_a"] = time.monotonic()
                resolver._async_key_locks["handler_b"] = asyncio.Lock()
                resolver._async_key_lock_timestamps["handler_b"] = time.monotonic()

            # Resolve first two handlers (fills cache to capacity)
            resolver.resolve(
                handler_type="handler_a",
                inline_config={"timeout_ms": 5000},
            )
            resolver.resolve(
                handler_type="handler_b",
                inline_config={"timeout_ms": 5000},
            )

            # Verify cache has 2 entries and locks exist
            assert len(resolver._cache) == 2
            assert "handler_a" in resolver._async_key_locks
            assert "handler_b" in resolver._async_key_locks

            # Resolve a third handler - should evict handler_a (LRU)
            with resolver._lock:
                resolver._async_key_locks["handler_c"] = asyncio.Lock()
                resolver._async_key_lock_timestamps["handler_c"] = time.monotonic()

            resolver.resolve(
                handler_type="handler_c",
                inline_config={"timeout_ms": 5000},
            )

            # Verify handler_a was evicted and its lock was cleaned up
            assert "handler_a" not in resolver._cache
            assert "handler_a" not in resolver._async_key_locks
            assert "handler_a" not in resolver._async_key_lock_timestamps

            # Verify handler_b and handler_c remain
            assert "handler_b" in resolver._cache
            assert "handler_c" in resolver._cache

            # Verify LRU eviction count
            stats = resolver.get_cache_stats()
            assert stats.lru_evictions == 1

    def test_ttl_expiration_cleans_up_async_lock(self) -> None:
        """Test that TTL expiration also removes the associated async lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                cache_ttl_seconds=1.0,  # Very short TTL
            )
            resolver, _container = create_resolver(config)

            # Create an async lock for testing
            with resolver._lock:
                resolver._async_key_locks["handler_expire"] = asyncio.Lock()
                resolver._async_key_lock_timestamps["handler_expire"] = time.monotonic()

            # Resolve a handler to cache it
            resolver.resolve(
                handler_type="handler_expire",
                inline_config={"timeout_ms": 5000},
            )

            # Verify cache entry and lock exist
            assert "handler_expire" in resolver._cache
            assert "handler_expire" in resolver._async_key_locks

            # Manually expire the cache entry by manipulating expires_at
            from datetime import UTC, datetime, timedelta

            with resolver._lock:
                cached = resolver._cache["handler_expire"]
                # Create an expired entry (expired 10 seconds ago)
                expired_entry = cached.__class__(
                    config=cached.config,
                    expires_at=datetime.now(UTC) - timedelta(seconds=10),
                    source=cached.source,
                )
                resolver._cache["handler_expire"] = expired_entry

            # Access the cache - this should trigger expiration cleanup
            result = resolver._get_from_cache("handler_expire")

            # Verify entry was evicted and lock was cleaned up
            assert result is None
            assert "handler_expire" not in resolver._cache
            assert "handler_expire" not in resolver._async_key_locks
            assert "handler_expire" not in resolver._async_key_lock_timestamps

            # Verify expiration eviction count
            stats = resolver.get_cache_stats()
            assert stats.expired_evictions == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_preserves_held_lock(self) -> None:
        """Test that LRU eviction does not remove a lock that is currently held."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                max_cache_entries=1,  # Small cache to force eviction
            )
            resolver, _container = create_resolver(config)

            # Create and hold a lock
            with resolver._lock:
                resolver._async_key_locks["held_handler"] = asyncio.Lock()
                resolver._async_key_lock_timestamps["held_handler"] = time.monotonic()

            lock = resolver._async_key_locks["held_handler"]

            # Acquire the lock
            async with lock:
                # Resolve to cache the handler
                resolver.resolve(
                    handler_type="held_handler",
                    inline_config={"timeout_ms": 5000},
                )

                # Resolve another handler to trigger LRU eviction of held_handler
                with resolver._lock:
                    resolver._async_key_locks["new_handler"] = asyncio.Lock()
                    resolver._async_key_lock_timestamps["new_handler"] = (
                        time.monotonic()
                    )

                resolver.resolve(
                    handler_type="new_handler",
                    inline_config={"timeout_ms": 5000},
                )

                # Verify cache entry was evicted
                assert "held_handler" not in resolver._cache

                # But the lock should be preserved because it's held
                assert "held_handler" in resolver._async_key_locks
                assert "held_handler" in resolver._async_key_lock_timestamps

    def test_multiple_lru_evictions_clean_up_locks(self) -> None:
        """Test that multiple consecutive LRU evictions clean up locks correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                max_cache_entries=2,
            )
            resolver, _container = create_resolver(config)

            # Create locks for multiple handlers
            handlers = ["handler_1", "handler_2", "handler_3", "handler_4"]
            for handler in handlers:
                with resolver._lock:
                    resolver._async_key_locks[handler] = asyncio.Lock()
                    resolver._async_key_lock_timestamps[handler] = time.monotonic()

            # Resolve handlers one by one - first 2 fill cache, then evictions begin
            for handler in handlers:
                resolver.resolve(
                    handler_type=handler,
                    inline_config={"timeout_ms": 5000},
                )

            # Only the last 2 handlers should remain in cache
            assert len(resolver._cache) == 2
            assert "handler_3" in resolver._cache
            assert "handler_4" in resolver._cache

            # Evicted handlers should have their locks cleaned up
            assert "handler_1" not in resolver._async_key_locks
            assert "handler_2" not in resolver._async_key_locks

            # Remaining handlers should keep their locks
            assert "handler_3" in resolver._async_key_locks
            assert "handler_4" in resolver._async_key_locks

            # Verify LRU eviction count
            stats = resolver.get_cache_stats()
            assert stats.lru_evictions == 2


class TestVaultReferencesInLists:
    """Tests for vault reference resolution inside list values.

    These tests verify that vault: references inside list values are properly
    resolved, including nested lists and mixed structures.

    Related:
    - OMN-765: BindingConfigResolver implementation
    - PR #168 review feedback: vault reference resolution skips list values
    """

    def test_vault_refs_in_simple_list(self) -> None:
        """Test resolving vault references in a simple list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock SecretResolver
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "resolved_secret"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Patch the _get_secret_resolver method to return our mock
            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Config with vault refs in a list
                test_config: dict[str, object] = {
                    "secrets": [
                        "vault:secret/key1",
                        "vault:secret/key2#field",
                        "plain_value",
                    ]
                }

                result = resolver._resolve_vault_refs(test_config, uuid4())

                # Verify list items were resolved
                assert isinstance(result["secrets"], list)
                secrets_list = result["secrets"]
                assert secrets_list[0] == "resolved_secret"
                assert secrets_list[1] == "resolved_secret"
                assert secrets_list[2] == "plain_value"

    def test_vault_refs_in_nested_list(self) -> None:
        """Test resolving vault references in nested lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "nested_secret"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Config with vault refs in nested list
                test_config: dict[str, object] = {
                    "nested": [
                        ["vault:secret/nested1", "plain"],
                        ["vault:secret/nested2"],
                    ]
                }

                result = resolver._resolve_vault_refs(test_config, uuid4())

                nested = result["nested"]
                assert isinstance(nested, list)
                assert nested[0][0] == "nested_secret"
                assert nested[0][1] == "plain"
                assert nested[1][0] == "nested_secret"

    def test_vault_refs_in_list_with_dict_items(self) -> None:
        """Test resolving vault references in list containing dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "dict_secret"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Config with list of dicts containing vault refs
                test_config: dict[str, object] = {
                    "databases": [
                        {"name": "db1", "password": "vault:secret/db1#password"},
                        {"name": "db2", "password": "vault:secret/db2#password"},
                    ]
                }

                result = resolver._resolve_vault_refs(test_config, uuid4())

                databases = result["databases"]
                assert isinstance(databases, list)
                assert databases[0]["name"] == "db1"
                assert databases[0]["password"] == "dict_secret"
                assert databases[1]["name"] == "db2"
                assert databases[1]["password"] == "dict_secret"

    def test_has_vault_references_in_list(self) -> None:
        """Test _has_vault_references detects vault refs in lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            # Config with vault ref in list
            with_ref: dict[str, object] = {"items": ["vault:secret/test"]}
            assert resolver._has_vault_references(with_ref) is True

            # Config with nested vault ref in list
            nested_ref: dict[str, object] = {"items": [{"key": "vault:secret/nested"}]}
            assert resolver._has_vault_references(nested_ref) is True

            # Config without vault refs
            without_ref: dict[str, object] = {"items": ["plain", "values"]}
            assert resolver._has_vault_references(without_ref) is False

    @pytest.mark.asyncio
    async def test_vault_refs_in_list_async(self) -> None:
        """Test async resolution of vault references in lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "async_secret"

            async def mock_get_secret_async(name: str, required: bool = True) -> object:
                return mock_secret

            mock_resolver.get_secret_async = mock_get_secret_async

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                test_config: dict[str, object] = {
                    "api_keys": [
                        "vault:secret/key1",
                        "vault:secret/key2",
                    ]
                }

                result = await resolver._resolve_vault_refs_async(test_config, uuid4())

                api_keys = result["api_keys"]
                assert isinstance(api_keys, list)
                assert api_keys[0] == "async_secret"
                assert api_keys[1] == "async_secret"

    def test_vault_ref_resolution_failure_in_list(self) -> None:
        """Test vault reference resolution failure in list with fail_on_secret_error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            # Use SecretResolutionError (specific exception) instead of generic Exception
            mock_resolver.get_secret.side_effect = SecretResolutionError("Vault error")

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                fail_on_secret_error=True,
            )
            resolver, _container = create_resolver(config)

            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                test_config: dict[str, object] = {"secrets": ["vault:secret/will_fail"]}

                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver._resolve_vault_refs(test_config, uuid4())

                assert "list index" in str(exc_info.value).lower()

    def test_vault_ref_in_list_depth_limit(self) -> None:
        """Test that deeply nested lists hit recursion limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "secret"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver, _container = create_resolver(config)

            with patch.object(
                resolver, "_get_secret_resolver", return_value=mock_resolver
            ):
                # Create deeply nested list (21 levels to exceed limit)
                nested: list[object] = ["vault:secret/deep"]
                for _ in range(21):
                    nested = [nested]

                test_config: dict[str, object] = {"deep_list": nested}

                with pytest.raises(ProtocolConfigurationError) as exc_info:
                    resolver._resolve_vault_refs(test_config, uuid4())

                assert "nesting exceeds maximum depth" in str(exc_info.value).lower()


class TestDeferredConfigDirValidation:
    """Test deferred config_dir validation (at use-time, not construction-time).

    Related:
    - OMN-765: BindingConfigResolver implementation
    - PR #168 review feedback: config_dir validation timing
    """

    def test_config_dir_nonexistent_at_construction_succeeds(self) -> None:
        """Config can be created with nonexistent config_dir (deferred validation)."""
        # This should succeed even though path doesn't exist
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")
        config = ModelBindingConfigResolverConfig(config_dir=nonexistent_path)

        # Config is created successfully
        assert config.config_dir == nonexistent_path

    def test_config_dir_nonexistent_at_use_time_raises(self) -> None:
        """Error raised at use-time when config_dir doesn't exist."""
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")
        config = ModelBindingConfigResolverConfig(config_dir=nonexistent_path)
        resolver, _container = create_resolver(config)

        # Attempting to load a relative path should fail at use-time
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            resolver.resolve(
                handler_type="test",
                config_ref="file:config.yaml",
            )

        assert "config_dir does not exist" in str(exc_info.value)

    def test_config_dir_file_not_directory_at_use_time_raises(self) -> None:
        """Error raised at use-time when config_dir is a file, not directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file instead of directory
            config_file = Path(tmpdir) / "not_a_dir.txt"
            config_file.write_text("I am a file")

            config = ModelBindingConfigResolverConfig(config_dir=config_file)
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="test",
                    config_ref="file:config.yaml",
                )

            assert "config_dir exists but is not a directory" in str(exc_info.value)

    def test_config_dir_created_after_config_construction(self) -> None:
        """Config works when directory is created after config object construction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path that doesn't exist yet
            config_dir = Path(tmpdir) / "will_be_created"

            # Create config with nonexistent path (this should succeed)
            config = ModelBindingConfigResolverConfig(config_dir=config_dir)
            resolver, _container = create_resolver(config)

            # Now create the directory and config file
            config_dir.mkdir()
            (config_dir / "test.yaml").write_text(
                "handler_type: db\ntimeout_ms: 5000\n"
            )

            # Should work now
            result = resolver.resolve(
                handler_type="db",
                config_ref="file:test.yaml",
            )

            assert result.handler_type == "db"
            assert result.timeout_ms == 5000

    def test_config_dir_null_byte_rejected_at_construction(self) -> None:
        """Null bytes in config_dir are rejected at construction time (security)."""
        with pytest.raises(ValueError) as exc_info:
            ModelBindingConfigResolverConfig(config_dir=Path("/path\x00with/null"))

        assert "null byte" in str(exc_info.value).lower()


class TestSymlinkValidation:
    """Test symlink handling in file loader.

    Related:
    - OMN-765: BindingConfigResolver implementation
    - PR #168 review feedback: symlink validation in file loader
    """

    def test_symlink_allowed_by_default(self) -> None:
        """Symlinks are allowed by default (within config_dir)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create actual config file inside config_dir
            real_config = config_dir / "real_config.yaml"
            real_config.write_text("handler_type: db\ntimeout_ms: 3000\n")

            # Create symlink to it (both inside config_dir, so path traversal is OK)
            symlink_config = config_dir / "config.yaml"
            symlink_config.symlink_to(real_config)

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                allow_symlinks=True,  # Explicit default
            )
            resolver, _container = create_resolver(config)

            # Should work with symlinks allowed (target within config_dir)
            result = resolver.resolve(
                handler_type="db",
                config_ref="file:config.yaml",
            )

            assert result.handler_type == "db"
            assert result.timeout_ms == 3000

    def test_symlink_rejected_when_disabled(self) -> None:
        """Symlinks are rejected when allow_symlinks=False (within config_dir)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve to real path to avoid macOS /var -> /private/var symlink
            # This ensures we test explicit symlink rejection, not system symlinks
            config_dir = Path(tmpdir).resolve()

            # Create actual config file inside config_dir
            real_config = config_dir / "real_config.yaml"
            real_config.write_text("handler_type: db\ntimeout_ms: 3000\n")

            # Create symlink to it (both inside config_dir to isolate symlink test)
            symlink_config = config_dir / "config.yaml"
            symlink_config.symlink_to(real_config)

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                allow_symlinks=False,  # Disable symlinks
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:config.yaml",
                )

            assert "symlink" in str(exc_info.value).lower()

    def test_regular_file_works_when_symlinks_disabled(self) -> None:
        """Regular (non-symlink) files work when allow_symlinks=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve to real path to avoid macOS /var -> /private/var symlink
            # The test verifies regular files work, not system symlink handling
            config_dir = Path(tmpdir).resolve()

            # Create regular config file (no symlink)
            config_file = config_dir / "config.yaml"
            config_file.write_text("handler_type: api\ntimeout_ms: 1000\n")

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                allow_symlinks=False,
            )
            resolver, _container = create_resolver(config)

            # Should work with regular file
            result = resolver.resolve(
                handler_type="api",
                config_ref="file:config.yaml",
            )

            assert result.handler_type == "api"
            assert result.timeout_ms == 1000

    def test_symlink_in_parent_path_rejected_when_disabled(self) -> None:
        """Symlinks in parent directories are rejected when allow_symlinks=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve to real path to avoid macOS /var -> /private/var symlink
            # This ensures we test explicit symlink parent rejection
            base_dir = Path(tmpdir).resolve()

            # Create real directory with config
            real_configs = base_dir / "real_configs"
            real_configs.mkdir()
            config_file = real_configs / "db.yaml"
            config_file.write_text("handler_type: db\ntimeout_ms: 5000\n")

            # Create symlink directory pointing to real_configs
            symlink_dir = base_dir / "configs_symlink"
            symlink_dir.symlink_to(real_configs)

            config = ModelBindingConfigResolverConfig(
                config_dir=symlink_dir,  # symlink as config_dir
                allow_symlinks=False,
            )
            resolver, _container = create_resolver(config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:db.yaml",
                )

            assert "symlink" in str(exc_info.value).lower()

    def test_symlink_outside_config_dir_blocked_even_when_allowed(self) -> None:
        """Symlinks pointing outside config_dir are blocked (path traversal protection)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            config_dir = base_dir / "configs"
            config_dir.mkdir()

            # Create config outside config_dir
            outside_config = base_dir / "outside_config.yaml"
            outside_config.write_text("handler_type: evil\n")

            # Create symlink inside config_dir pointing outside
            evil_symlink = config_dir / "evil.yaml"
            evil_symlink.symlink_to(outside_config)

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
                allow_symlinks=True,  # Even with symlinks allowed
            )
            resolver, _container = create_resolver(config)

            # Should be blocked by path traversal protection
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="evil",
                    config_ref="file:evil.yaml",
                )

            assert "traversal" in str(exc_info.value).lower()


__all__: list[str] = [
    "TestBindingConfigResolverBasic",
    "TestBindingConfigResolverFileSource",
    "TestBindingConfigResolverEnvSource",
    "TestBindingConfigResolverEnvOverrides",
    "TestBindingConfigResolverVaultSource",
    "TestBindingConfigResolverCaching",
    "TestBindingConfigResolverAsync",
    "TestBindingConfigResolverThreadSafety",
    "TestBindingConfigResolverValidation",
    "TestBindingConfigResolverSecurity",
    "TestModelConfigRef",
    "TestModelBindingConfig",
    "TestModelRetryPolicy",
    "TestBindingConfigResolverInlinePrecedence",
    "TestBindingConfigResolverRecursionDepth",
    "TestAsyncKeyLockCleanup",
    "TestVaultReferencesInLists",
    "TestDeferredConfigDirValidation",
    "TestSymlinkValidation",
]
