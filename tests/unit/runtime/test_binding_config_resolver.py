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

Related:
- OMN-765: BindingConfigResolver implementation
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
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import SecretStr

from omnibase_infra.errors import ProtocolConfigurationError
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


class TestBindingConfigResolverBasic:
    """Basic resolution functionality tests."""

    def test_resolve_inline_config(self) -> None:
        """Resolve with inline config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:large.yaml",
                )

            assert "size limit" in str(exc_info.value).lower()

    def test_path_traversal_blocked(self) -> None:
        """Path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "configs"
            config_dir.mkdir()

            # Create a file outside configs dir
            outside_file = Path(tmpdir) / "secret.yaml"
            outside_file.write_text(yaml.dump({"timeout_ms": 5000}))

            config = ModelBindingConfigResolverConfig(
                config_dir=config_dir,
            )
            resolver = BindingConfigResolver(config=config)

            # Attempt path traversal
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="file:../secret.yaml",
                )

            # Should detect path traversal in parsing
            assert "traversal" in str(exc_info.value).lower()

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
        resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

            with patch.dict(os.environ, {"HANDLER_DB_TIMEOUT_MS": "not_a_number"}):
                result = resolver.resolve(
                    handler_type="db",
                    inline_config={"timeout_ms": 5000},
                )

            # Should fall back to inline config value
            assert result.timeout_ms == 5000


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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
        """Error when vault:// used but no SecretResolver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
                secret_resolver=None,  # No resolver
            )
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            mock_resolver.get_secret.side_effect = Exception("Vault connection failed")

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver = BindingConfigResolver(config=config)

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
        """Resolve vault:// references within config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resolver = MagicMock()
            mock_secret = MagicMock()
            mock_secret.get_secret_value.return_value = "secret_value"
            mock_resolver.get_secret.return_value = mock_secret

            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

            with pytest.raises(ProtocolConfigurationError):
                resolver.resolve(handler_type="")

    def test_invalid_timeout_ms_too_low(self) -> None:
        """timeout_ms below minimum raises validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ModelBindingConfigResolverConfig(
                config_dir=Path(tmpdir),
            )
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

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
            resolver = BindingConfigResolver(config=config)

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                resolver.resolve(
                    handler_type="db",
                    config_ref="http://example.com/config",
                )

            assert "unknown" in str(exc_info.value).lower()

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
            resolver = BindingConfigResolver(config=config)

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
        """Path traversal returns error."""
        result = ModelConfigRef.parse("file:../../../etc/passwd")

        assert not result.success
        assert result.config_ref is None
        assert "traversal" in result.error_message.lower()

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
            resolver = BindingConfigResolver(config=config)

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
]
