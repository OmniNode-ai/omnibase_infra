# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ModelGatewayConfig.

Tests verify that:
    - Config validation (realm and runtime_id required)
    - Default values are correct
    - Optional fields work as expected
    - Config is frozen (immutable)

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.gateway import ModelGatewayConfig

pytestmark = pytest.mark.integration


class TestGatewayConfigRequired:
    """Tests for required configuration fields."""

    def test_realm_is_required(self) -> None:
        """Realm field is required."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                runtime_id="test-runtime",
                # Missing realm
            )  # type: ignore[call-arg]

        assert "realm" in str(exc_info.value)

    def test_runtime_id_is_required(self) -> None:
        """Runtime ID field is required."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                realm="test",
                # Missing runtime_id
            )  # type: ignore[call-arg]

        assert "runtime_id" in str(exc_info.value)

    def test_realm_cannot_be_empty(self) -> None:
        """Realm field cannot be empty string."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                realm="",  # Empty
                runtime_id="test-runtime",
            )

        assert "realm" in str(exc_info.value).lower()

    def test_runtime_id_cannot_be_empty(self) -> None:
        """Runtime ID field cannot be empty string."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                realm="test",
                runtime_id="",  # Empty
            )

        assert "runtime_id" in str(exc_info.value).lower()

    def test_realm_max_length(self) -> None:
        """Realm field has maximum length of 64 characters."""
        # Arrange
        long_realm = "a" * 65  # 65 chars, exceeds limit

        # Act & Assert
        with pytest.raises(ValidationError):
            ModelGatewayConfig(
                realm=long_realm,
                runtime_id="test-runtime",
            )

    def test_runtime_id_max_length(self) -> None:
        """Runtime ID field has maximum length of 128 characters."""
        # Arrange
        long_runtime_id = "a" * 129  # 129 chars, exceeds limit

        # Act & Assert
        with pytest.raises(ValidationError):
            ModelGatewayConfig(
                realm="test",
                runtime_id=long_runtime_id,
            )


class TestGatewayConfigDefaults:
    """Tests for default configuration values."""

    def test_enabled_defaults_to_true(self) -> None:
        """Enabled field defaults to True."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        # Assert
        assert config.enabled is True

    def test_private_key_path_defaults_to_none(self) -> None:
        """Private key path defaults to None."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        # Assert
        assert config.private_key_path is None

    def test_public_key_path_defaults_to_none(self) -> None:
        """Public key path defaults to None."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        # Assert
        assert config.public_key_path is None

    def test_allowed_topics_defaults_to_empty_tuple(self) -> None:
        """Allowed topics defaults to empty tuple."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        # Assert
        assert config.allowed_topics == ()
        assert isinstance(config.allowed_topics, tuple)

    def test_reject_unsigned_defaults_to_true(self) -> None:
        """Reject unsigned defaults to True (secure by default)."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        # Assert
        assert config.reject_unsigned is True


class TestGatewayConfigOptionalFields:
    """Tests for optional configuration fields."""

    def test_private_key_path_accepts_path(self) -> None:
        """Private key path accepts Path object."""
        # Arrange
        key_path = Path("/etc/onex/keys/private.pem")

        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            private_key_path=key_path,
        )

        # Assert
        assert config.private_key_path == key_path
        assert isinstance(config.private_key_path, Path)

    def test_public_key_path_accepts_path(self) -> None:
        """Public key path accepts Path object."""
        # Arrange
        key_path = Path("/etc/onex/keys/public.pem")

        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            public_key_path=key_path,
        )

        # Assert
        assert config.public_key_path == key_path

    def test_allowed_topics_accepts_tuple(self) -> None:
        """Allowed topics accepts tuple of strings."""
        # Arrange
        topics = ("events.*", "commands.*")

        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            allowed_topics=topics,
        )

        # Assert
        assert config.allowed_topics == topics

    def test_enabled_can_be_false(self) -> None:
        """Enabled can be set to False."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            enabled=False,
        )

        # Assert
        assert config.enabled is False

    def test_reject_unsigned_can_be_false(self) -> None:
        """Reject unsigned can be set to False."""
        # Act
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            reject_unsigned=False,
        )

        # Assert
        assert config.reject_unsigned is False


class TestGatewayConfigImmutability:
    """Tests for config immutability."""

    def test_config_is_frozen(self) -> None:
        """Config model is frozen (immutable)."""
        # Arrange
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        # Act & Assert
        with pytest.raises(ValidationError):
            config.realm = "modified"  # type: ignore[misc]

    def test_allowed_topics_is_immutable_tuple(self) -> None:
        """Allowed topics is stored as immutable tuple."""
        # Arrange
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            allowed_topics=("events.*",),
        )

        # Assert
        assert isinstance(config.allowed_topics, tuple)


class TestGatewayConfigValidation:
    """Tests for config validation rules."""

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are forbidden (strict validation)."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                realm="test",
                runtime_id="test-runtime",
                unknown_field="value",  # type: ignore[call-arg]
            )

        assert "extra" in str(exc_info.value).lower()

    def test_realm_with_valid_characters(self) -> None:
        """Realm accepts valid characters."""
        # Arrange - various valid realm names
        valid_realms = ["dev", "staging", "production", "tenant-123", "my_realm"]

        # Act & Assert
        for realm in valid_realms:
            config = ModelGatewayConfig(
                realm=realm,
                runtime_id="test-runtime",
            )
            assert config.realm == realm

    def test_runtime_id_with_valid_characters(self) -> None:
        """Runtime ID accepts valid characters."""
        # Arrange - various valid runtime IDs
        valid_ids = [
            "runtime-001",
            "runtime_dev_001",
            "my-runtime.local",
            "Runtime123",
        ]

        # Act & Assert
        for runtime_id in valid_ids:
            config = ModelGatewayConfig(
                realm="test",
                runtime_id=runtime_id,
            )
            assert config.runtime_id == runtime_id


class TestKeyPathAbsoluteValidation:
    """Tests for _validate_key_path_is_absolute path traversal prevention."""

    def test_relative_private_key_path_rejected(self) -> None:
        """Relative private_key_path is rejected to prevent path traversal."""
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                realm="test",
                runtime_id="test-runtime",
                private_key_path=Path("../../../etc/shadow"),
            )

        assert "absolute" in str(exc_info.value).lower()

    def test_relative_public_key_path_rejected(self) -> None:
        """Relative public_key_path is rejected to prevent path traversal."""
        with pytest.raises(ValidationError) as exc_info:
            ModelGatewayConfig(
                realm="test",
                runtime_id="test-runtime",
                public_key_path=Path("keys/public.pem"),
            )

        assert "absolute" in str(exc_info.value).lower()

    def test_absolute_private_key_path_accepted(self) -> None:
        """Absolute private_key_path is accepted."""
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            private_key_path=Path("/etc/onex/keys/private.pem"),
        )

        assert config.private_key_path == Path("/etc/onex/keys/private.pem")

    def test_absolute_public_key_path_accepted(self) -> None:
        """Absolute public_key_path is accepted."""
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            public_key_path=Path("/etc/onex/keys/public.pem"),
        )

        assert config.public_key_path == Path("/etc/onex/keys/public.pem")

    def test_none_key_path_accepted(self) -> None:
        """None key paths (default) pass validation."""
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
        )

        assert config.private_key_path is None
        assert config.public_key_path is None

    def test_dot_slash_relative_path_rejected(self) -> None:
        """Dot-slash relative paths are rejected."""
        with pytest.raises(ValidationError):
            ModelGatewayConfig(
                realm="test",
                runtime_id="test-runtime",
                private_key_path=Path("./keys/private.pem"),
            )


class TestGatewayConfigUseCases:
    """Tests for common configuration use cases."""

    def test_development_config_signing_disabled(self) -> None:
        """Development configuration with signing disabled."""
        # Arrange & Act
        config = ModelGatewayConfig(
            realm="dev",
            runtime_id="runtime-local",
            enabled=False,  # Signing disabled for dev
            reject_unsigned=False,  # Accept unsigned for dev
        )

        # Assert
        assert config.realm == "dev"
        assert config.enabled is False
        assert config.reject_unsigned is False
        assert config.private_key_path is None
        assert config.public_key_path is None

    def test_production_config_full_signing(self) -> None:
        """Production configuration with full signing enabled."""
        # Arrange & Act
        config = ModelGatewayConfig(
            realm="prod",
            runtime_id="runtime-prod-001",
            private_key_path=Path("/etc/onex/keys/private.pem"),
            public_key_path=Path("/etc/onex/keys/public.pem"),
            allowed_topics=("events.*", "commands.*"),
            reject_unsigned=True,
        )

        # Assert
        assert config.realm == "prod"
        assert config.enabled is True
        assert config.reject_unsigned is True
        assert config.private_key_path is not None
        assert config.public_key_path is not None
        assert len(config.allowed_topics) == 2

    def test_multi_tenant_config(self) -> None:
        """Multi-tenant configuration with tenant-specific realm."""
        # Arrange & Act
        config = ModelGatewayConfig(
            realm="tenant-abc-123",
            runtime_id="runtime-tenant-abc-123-001",
            allowed_topics=("tenant.abc-123.*",),
        )

        # Assert
        assert config.realm == "tenant-abc-123"
        assert "tenant-abc-123" in config.runtime_id


class TestGatewayConfigSerialization:
    """Tests for config serialization."""

    def test_model_dump(self) -> None:
        """Config can be serialized to dictionary."""
        # Arrange
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            allowed_topics=("events.*",),
        )

        # Act
        data = config.model_dump()

        # Assert
        assert data["realm"] == "test"
        assert data["runtime_id"] == "test-runtime"
        assert data["allowed_topics"] == ("events.*",)
        assert data["enabled"] is True

    def test_model_dump_json(self) -> None:
        """Config can be serialized to JSON-compatible dictionary."""
        # Arrange
        config = ModelGatewayConfig(
            realm="test",
            runtime_id="test-runtime",
            private_key_path=Path("/etc/keys/private.pem"),
        )

        # Act
        data = config.model_dump(mode="json")

        # Assert
        assert data["realm"] == "test"
        # Path should be serialized as string in JSON mode
        assert isinstance(data["private_key_path"], str)
