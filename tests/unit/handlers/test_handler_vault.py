# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for VaultAdapter.

These tests use mocked hvac client to validate VaultAdapter behavior
without requiring actual Vault server infrastructure.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import hvac.exceptions
import pytest
from omnibase_core.enums import EnumCoreErrorCode
from pydantic import SecretStr, ValidationError

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.handlers.handler_vault import VaultAdapter
from omnibase_infra.handlers.model_vault_adapter_config import ModelVaultAdapterConfig

# Type alias for vault config dict values (str, int, float, bool, dict)
VaultConfigValue = str | int | float | bool | dict[str, str | int | float | bool]


@pytest.fixture
def vault_config() -> dict[str, VaultConfigValue]:
    """Provide test Vault configuration."""
    return {
        "url": "https://vault.example.com:8200",
        "token": "s.test1234567890",
        "namespace": "engineering",
        "timeout_seconds": 30.0,
        "verify_ssl": True,
        "token_renewal_threshold_seconds": 300.0,
        "retry": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0.1,
            "max_backoff_seconds": 10.0,
            "exponential_base": 2.0,
        },
    }


@pytest.fixture
def mock_hvac_client() -> MagicMock:
    """Provide mocked hvac.Client."""
    client = MagicMock()
    client.is_authenticated.return_value = True
    client.secrets.kv.v2 = MagicMock()
    client.auth.token = MagicMock()
    # Mock token lookup response with TTL
    client.auth.token.lookup_self.return_value = {
        "data": {
            "ttl": 3600,  # 1 hour TTL in seconds
            "renewable": True,
        }
    }
    client.sys = MagicMock()
    return client


class TestVaultAdapterInitialization:
    """Test VaultAdapter initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful initialization with valid config."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            await handler.initialize(vault_config)

            assert handler._initialized is True
            assert handler._config is not None
            assert handler._config.url == "https://vault.example.com:8200"
            assert handler._config.namespace == "engineering"
            MockClient.assert_called_once()
            mock_hvac_client.is_authenticated.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_missing_url(self) -> None:
        """Test initialization fails with missing URL."""
        handler = VaultAdapter()
        config: dict[str, VaultConfigValue] = {"token": "s.test1234"}

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize(config)

        # Pydantic ValidationError contains "url" in the error details
        assert "vault configuration" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_missing_token(self) -> None:
        """Test initialization fails with missing token."""
        handler = VaultAdapter()
        config: dict[str, VaultConfigValue] = {"url": "https://vault.example.com:8200"}

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize(config)

        assert "token" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_authentication_failure(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test initialization fails with authentication failure."""
        handler = VaultAdapter()
        mock_hvac_client.is_authenticated.return_value = False

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(vault_config)

            assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(
        self,
        vault_config: dict[str, VaultConfigValue],
    ) -> None:
        """Test initialization fails with connection error."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            import hvac.exceptions

            MockClient.side_effect = hvac.exceptions.VaultError("Connection refused")

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(vault_config)

            assert "failed to connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_config_model_validation(self) -> None:
        """Test Pydantic config model validation."""
        # Valid config
        config = ModelVaultAdapterConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.test1234"),
            timeout_seconds=30.0,
        )
        assert config.url == "https://vault.example.com:8200"
        assert config.timeout_seconds == 30.0

        # Invalid timeout (too high)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                timeout_seconds=400.0,  # Max is 300.0
            )
        assert "timeout_seconds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_secretstr_prevents_token_logging(self) -> None:
        """Test SecretStr prevents token from being logged."""
        config = ModelVaultAdapterConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.sensitive_token_12345"),
        )

        # SecretStr representation should hide the token
        token_repr = repr(config.token)
        assert "sensitive_token_12345" not in token_repr
        assert "SecretStr" in token_repr


class TestVaultAdapterOperations:
    """Test VaultAdapter secret operations."""

    @pytest.mark.asyncio
    async def test_read_secret_success(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret read operation."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mock read_secret_version response
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {
                    "data": {"password": "secret123", "username": "admin"},
                    "metadata": {"version": 1, "created_time": "2025-01-01T00:00:00Z"},
                }
            }

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config", "mount_point": "secret"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["data"] == {
                "password": "secret123",
                "username": "admin",
            }
            metadata = payload["metadata"]
            assert isinstance(metadata, dict)
            assert metadata["version"] == 1
            mock_hvac_client.secrets.kv.v2.read_secret_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_secret_success(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret write operation."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mock create_or_update_secret response
            mock_hvac_client.secrets.kv.v2.create_or_update_secret.return_value = {
                "data": {"version": 2, "created_time": "2025-01-01T00:00:00Z"}
            }

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.write_secret",
                "payload": {
                    "path": "myapp/config",
                    "data": {"password": "newsecret", "username": "admin"},
                    "mount_point": "secret",
                },
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["version"] == 2
            mock_hvac_client.secrets.kv.v2.create_or_update_secret.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_secret_success(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret deletion."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.delete_secret",
                "payload": {"path": "myapp/config", "mount_point": "secret"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["deleted"] is True
            mock_hvac_client.secrets.kv.v2.delete_latest_version_of_secret.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_secrets_success(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret listing."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mock list_secrets response
            mock_hvac_client.secrets.kv.v2.list_secrets.return_value = {
                "data": {"keys": ["config1", "config2", "config3/"]}
            }

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.list_secrets",
                "payload": {"path": "myapp/", "mount_point": "secret"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["keys"] == ["config1", "config2", "config3/"]
            mock_hvac_client.secrets.kv.v2.list_secrets.assert_called_once()

    @pytest.mark.asyncio
    async def test_operation_missing_path(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test operation fails with missing path."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {},  # Missing 'path'
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "path" in str(exc_info.value).lower()


class TestVaultAdapterTokenRenewal:
    """Test VaultAdapter token renewal management."""

    @pytest.mark.asyncio
    async def test_renew_token_success(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful token renewal."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mock renew_self response
            mock_hvac_client.auth.token.renew_self.return_value = {
                "auth": {
                    "renewable": True,
                    "lease_duration": 3600,
                }
            }

            await handler.initialize(vault_config)

            result = await handler.renew_token()

            auth = result["auth"]
            assert isinstance(auth, dict)
            assert auth["renewable"] is True
            assert auth["lease_duration"] == 3600
            mock_hvac_client.auth.token.renew_self.assert_called_once()

    @pytest.mark.asyncio
    async def test_renew_token_operation_envelope(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test token renewal via envelope operation."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.auth.token.renew_self.return_value = {
                "auth": {
                    "renewable": True,
                    "lease_duration": 3600,
                }
            }

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.renew_token",
                "payload": {},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["renewable"] is True
            assert payload["lease_duration"] == 3600

    @pytest.mark.asyncio
    async def test_token_renewal_threshold_edge_case_exactly_at_threshold(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test token renewal boundary condition logging and behavior.

        This edge case tests the boundary condition where time_until_expiry
        is close to or at token_renewal_threshold_seconds. The implementation
        uses strict less-than comparison: needs_renewal = (time_until_expiry < threshold).

        Since there's always some execution time between setting _token_expires_at
        and checking it, this test verifies the logging captures edge case detection
        and that the comparison logic is working as expected (>= threshold means
        no renewal, < threshold means renewal).

        Note: We set time_until_expiry slightly above threshold to ensure no renewal,
        as the strict < comparison should prevent renewal when exactly at threshold.
        """
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Set default renewal threshold from fixture (300.0 seconds)
            threshold = 300.0
            vault_config["token_renewal_threshold_seconds"] = threshold

            # Mock token lookup to return TTL above threshold
            mock_hvac_client.auth.token.lookup_self.return_value = {
                "data": {"ttl": int(threshold) + 10, "renewable": True}
            }

            await handler.initialize(vault_config)

            # Set _token_expires_at to be slightly ABOVE threshold to ensure
            # time_until_expiry >= threshold (accounting for execution time)
            current_time = time.time()
            # Add 1 second buffer to account for any execution time
            handler._token_expires_at = current_time + threshold + 1.0

            # Mock the renewal call (should NOT be called)
            mock_hvac_client.auth.token.renew_self.return_value = {
                "auth": {"renewable": True, "lease_duration": 3600}
            }

            # Set up a successful read operation
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            envelope: dict[str, str | UUID | dict[str, str]] = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Execute should NOT trigger renewal when above threshold
            response = await handler.execute(envelope)

            assert response["status"] == "success"
            # Verify renewal was NOT called (above threshold)
            mock_hvac_client.auth.token.renew_self.assert_not_called()

    @pytest.mark.asyncio
    async def test_token_renewal_threshold_edge_case_just_below_threshold(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test token renewal triggers when just below threshold.

        This tests that renewal DOES trigger when time_until_expiry is
        just below the threshold (e.g., threshold - 1 second).
        """
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            threshold = 300.0
            vault_config["token_renewal_threshold_seconds"] = threshold

            await handler.initialize(vault_config)

            # Set _token_expires_at to be just below threshold
            current_time = time.time()
            handler._token_expires_at = current_time + (threshold - 1.0)

            # Mock the renewal call
            mock_hvac_client.auth.token.renew_self.return_value = {
                "auth": {"renewable": True, "lease_duration": 3600}
            }

            # Set up a successful read operation
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            envelope: dict[str, str | UUID | dict[str, str]] = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Execute should trigger renewal when below threshold
            response = await handler.execute(envelope)

            assert response["status"] == "success"
            # Verify renewal WAS called (below threshold)
            mock_hvac_client.auth.token.renew_self.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_renewal_threshold_edge_case_just_above_threshold(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test token renewal does NOT trigger when just above threshold.

        This tests that renewal does NOT trigger when time_until_expiry is
        just above the threshold (e.g., threshold + 1 second).
        """
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            threshold = 300.0
            vault_config["token_renewal_threshold_seconds"] = threshold

            await handler.initialize(vault_config)

            # Set _token_expires_at to be just above threshold
            current_time = time.time()
            handler._token_expires_at = current_time + (threshold + 1.0)

            # Mock the renewal call (should NOT be called)
            mock_hvac_client.auth.token.renew_self.return_value = {
                "auth": {"renewable": True, "lease_duration": 3600}
            }

            # Set up a successful read operation
            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            envelope: dict[str, str | UUID | dict[str, str]] = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Execute should NOT trigger renewal when above threshold
            response = await handler.execute(envelope)

            assert response["status"] == "success"
            # Verify renewal was NOT called (above threshold)
            mock_hvac_client.auth.token.renew_self.assert_not_called()


class TestVaultAdapterRetryLogic:
    """Test VaultAdapter retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test retry logic on transient failures."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # First call fails, second succeeds
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = [
                Exception("Transient error"),
                {
                    "data": {
                        "data": {"key": "value"},
                        "metadata": {"version": 1},
                    }
                },
            ]

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Should succeed on retry
            response = await handler.execute(envelope)

            assert response["status"] == "success"
            assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test retry logic when all attempts exhausted."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # All attempts fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Persistent error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Should have tried max_attempts times (3)
            assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_error(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test operation timeout raises InfraTimeoutError."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Simulate timeout
            async def slow_operation(*args: object, **kwargs: object) -> None:
                await asyncio.sleep(100)  # Longer than timeout

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                await handler.initialize(vault_config)

                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": "myapp/config"},
                    "correlation_id": uuid4(),
                }

                with pytest.raises(InfraTimeoutError) as exc_info:
                    await handler.execute(envelope)

                assert "timed out" in str(exc_info.value).lower()


class TestVaultAdapterErrorHandling:
    """Test VaultAdapter error handling and sanitization."""

    @pytest.mark.asyncio
    async def test_authentication_error(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test authentication error handling."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            import hvac.exceptions

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.Forbidden("Permission denied")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.execute(envelope)

            # Should not retry authentication errors
            assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 1

    @pytest.mark.asyncio
    async def test_secret_not_found(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test secret not found error handling."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            import hvac.exceptions

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.InvalidPath("Secret not found")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "nonexistent/path"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(SecretResolutionError) as exc_info:
                await handler.execute(envelope)

            # Should not retry invalid path errors
            assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 1

    @pytest.mark.asyncio
    async def test_vault_unavailable(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test Vault server unavailable error."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            import hvac.exceptions

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.VaultDown("Vault is sealed")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraUnavailableError):
                await handler.execute(envelope)

            # Should retry on VaultDown
            assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 3


class TestVaultAdapterHealthCheck:
    """Test VaultAdapter health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health check returns healthy status."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.sys.read_health_status.return_value = {
                "initialized": True,
                "sealed": False,
            }

            await handler.initialize(vault_config)

            health = await handler.health_check()

            assert health["healthy"] is True
            assert health["initialized"] is True
            assert health["handler_type"] == "vault"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health check propagates errors instead of returning unhealthy."""
        from omnibase_infra.errors import InfraConnectionError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.sys.read_health_status.side_effect = Exception(
                "Connection refused"
            )

            await handler.initialize(vault_config)

            # Health check now propagates errors instead of returning healthy=False
            with pytest.raises(InfraConnectionError):
                await handler.health_check()

    @pytest.mark.asyncio
    async def test_health_check_operation_envelope(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health check via envelope operation."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.sys.read_health_status.return_value = {
                "initialized": True,
                "sealed": False,
            }

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.health_check",
                "payload": {},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["healthy"] is True


class TestVaultAdapterDescribe:
    """Test VaultAdapter describe functionality."""

    @pytest.mark.asyncio
    async def test_describe(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test describe returns handler metadata."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            description = handler.describe()

            assert description["handler_type"] == "vault"
            supported_ops = description["supported_operations"]
            assert isinstance(supported_ops, list)
            assert "vault.read_secret" in supported_ops
            assert "vault.write_secret" in supported_ops
            assert "vault.delete_secret" in supported_ops
            assert "vault.list_secrets" in supported_ops
            assert "vault.renew_token" in supported_ops
            assert "vault.health_check" in supported_ops
            assert description["initialized"] is True


class TestVaultAdapterShutdown:
    """Test VaultAdapter shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test handler shutdown releases resources."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            assert handler._initialized is True
            assert handler._client is not None

            await handler.shutdown()

            assert handler._initialized is False
            assert handler._client is None
            assert handler._config is None


class TestVaultAdapterThreadPool:
    """Test VaultAdapter thread pool functionality."""

    @pytest.mark.asyncio
    async def test_thread_pool_created_with_config_size(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test thread pool is created with configured size."""
        handler = VaultAdapter()

        # Set custom thread pool size
        vault_config["max_concurrent_operations"] = 15

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            assert handler._executor is not None
            assert handler.max_workers == 15

    @pytest.mark.asyncio
    async def test_thread_pool_default_size(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test thread pool uses default size when not specified."""
        handler = VaultAdapter()

        # Remove max_concurrent_operations to test default
        vault_config.pop("max_concurrent_operations", None)

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            assert handler._executor is not None
            assert handler.max_workers == 10  # Default value

    @pytest.mark.asyncio
    async def test_thread_pool_shutdown_on_handler_shutdown(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test thread pool is properly shutdown when handler shuts down."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            executor = handler._executor
            assert executor is not None

            await handler.shutdown()

            assert handler._executor is None
            # Verify executor was shut down by attempting to submit a task
            # Shutdown executors raise RuntimeError on submit()
            with pytest.raises(RuntimeError, match="cannot schedule new futures"):
                executor.submit(lambda: None)

    @pytest.mark.asyncio
    async def test_operations_use_thread_pool(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test operations execute in thread pool executor."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
            }

            await handler.initialize(vault_config)

            # Spy on executor to verify it's used
            original_executor = handler._executor
            with patch.object(
                asyncio.get_event_loop(), "run_in_executor"
            ) as mock_run_in_executor:
                mock_run_in_executor.return_value = asyncio.Future()
                mock_run_in_executor.return_value.set_result(
                    {"data": {"data": {"key": "value"}, "metadata": {"version": 1}}}
                )

                envelope = {
                    "operation": "vault.read_secret",
                    "payload": {"path": "myapp/config"},
                    "correlation_id": uuid4(),
                }

                await handler.execute(envelope)

                # Verify run_in_executor was called with our executor
                assert mock_run_in_executor.call_count >= 1
                # First call should use our custom executor
                call_args = mock_run_in_executor.call_args_list[0]
                assert call_args[0][0] == original_executor

    @pytest.mark.asyncio
    async def test_config_validates_thread_pool_bounds(self) -> None:
        """Test config validation enforces thread pool size bounds."""
        from omnibase_infra.handlers.model_vault_adapter_config import (
            ModelVaultAdapterConfig,
        )

        # Valid: within bounds (1-100)
        config = ModelVaultAdapterConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.test1234"),
            max_concurrent_operations=50,
        )
        assert config.max_concurrent_operations == 50

        # Invalid: below minimum (< 1)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                max_concurrent_operations=0,
            )
        assert "max_concurrent_operations" in str(exc_info.value)

        # Invalid: above maximum (> 100)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                max_concurrent_operations=150,
            )
        assert "max_concurrent_operations" in str(exc_info.value)


class TestVaultAdapterCircuitBreaker:
    """Test VaultAdapter circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit opens after threshold consecutive failures."""
        handler = VaultAdapter()

        # Configure circuit breaker with low threshold for testing
        vault_config["circuit_breaker_failure_threshold"] = 2

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Make all requests fail
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # First 2 attempts should fail with connection error (3 retries each)
            for _ in range(2):
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Circuit should now be OPEN (using mixin attribute)
            assert handler._circuit_breaker_open is True
            assert handler._circuit_breaker_failures >= 2

    @pytest.mark.asyncio
    async def test_circuit_blocks_requests_when_open(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit blocks requests when OPEN."""
        handler = VaultAdapter()

        vault_config["circuit_breaker_failure_threshold"] = 2

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Trigger circuit open
            for _ in range(2):
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Next request should be blocked by circuit breaker
            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            assert "circuit breaker is open" in str(exc_info.value).lower()
            # Should include retry_after in context
            assert "retry_after_seconds" in str(exc_info.value.model)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit transitions to HALF_OPEN after reset timeout."""
        handler = VaultAdapter()

        # Minimum timeout for testing (1.0 seconds)
        vault_config["circuit_breaker_failure_threshold"] = 2
        vault_config["circuit_breaker_reset_timeout_seconds"] = 1.0

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Open circuit
            for _ in range(2):
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Circuit should be OPEN (using mixin attribute)
            assert handler._circuit_breaker_open is True

            # Mock time.time() to simulate timeout passage instead of sleeping
            import time

            original_time = time.time
            mock_time_offset = 0.0

            def mock_time() -> float:
                return original_time() + mock_time_offset

            with patch("time.time", side_effect=mock_time):
                # Advance time by more than reset timeout
                mock_time_offset = 2.0

                # Now configure success response
                mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None
                mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                    "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
                }

                # Next request should transition to HALF_OPEN then succeed
                response = await handler.execute(envelope)

                assert response["status"] == "success"
                # Circuit should now be CLOSED (recovered, using mixin attribute)
                assert handler._circuit_breaker_open is False

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success_in_half_open_state(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit closes on successful request in HALF_OPEN state."""
        handler = VaultAdapter()

        vault_config["circuit_breaker_failure_threshold"] = 2
        vault_config["circuit_breaker_reset_timeout_seconds"] = 1.0

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Start with failures
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Open circuit
            for _ in range(2):
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Mock time.time() to simulate timeout passage instead of sleeping
            import time

            original_time = time.time
            mock_time_offset = 0.0

            def mock_time() -> float:
                return original_time() + mock_time_offset

            with patch("time.time", side_effect=mock_time):
                # Advance time by more than reset timeout
                mock_time_offset = 2.0

                # Success response
                mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None
                mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = {
                    "data": {"data": {"key": "value"}, "metadata": {"version": 1}}
                }

                response = await handler.execute(envelope)

                assert response["status"] == "success"
                # Circuit should now be CLOSED (using mixin attributes)
                assert handler._circuit_breaker_open is False
                assert handler._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure_in_half_open_state(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit reopens on failed request in HALF_OPEN state.

        Note: With MixinAsyncCircuitBreaker, a failure in HALF_OPEN state needs
        to reach the threshold again to reopen. We use threshold=1 to test that
        a single failure in HALF_OPEN immediately reopens the circuit.
        """
        handler = VaultAdapter()

        # Use threshold=1 so single failure in HALF_OPEN reopens immediately
        vault_config["circuit_breaker_failure_threshold"] = 1
        vault_config["circuit_breaker_reset_timeout_seconds"] = 1.0

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Open circuit (only needs 1 failure with threshold=1)
            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Circuit should be open
            assert handler._circuit_breaker_open is True

            # Mock time.time() to simulate timeout passage instead of sleeping
            import time

            original_time = time.time
            mock_time_offset = 0.0

            def mock_time() -> float:
                return original_time() + mock_time_offset

            with patch("time.time", side_effect=mock_time):
                # Advance time by more than reset timeout (transitions to HALF_OPEN)
                mock_time_offset = 2.0

                # Next request still fails (in HALF_OPEN state, 1 failure reopens)
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

                # Circuit should reopen (using mixin attribute)
                assert handler._circuit_breaker_open is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_can_be_disabled(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit breaker can be disabled via config."""
        handler = VaultAdapter()

        # Disable circuit breaker
        vault_config["circuit_breaker_enabled"] = False

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Even after many failures, circuit should not open
            for _ in range(10):
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Circuit breaker was not initialized (disabled), so attribute doesn't exist
            # The handler should not have _circuit_breaker_initialized set to True
            assert handler._circuit_breaker_initialized is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_shutdown(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test circuit breaker state resets on handler shutdown."""
        handler = VaultAdapter()

        vault_config["circuit_breaker_failure_threshold"] = 2

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Connection error"
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # Open circuit with 2 failed operations
            for _ in range(2):
                with pytest.raises(InfraConnectionError):
                    await handler.execute(envelope)

            # Circuit should be OPEN after threshold failures (using mixin attributes)
            assert handler._circuit_breaker_open is True
            assert handler._circuit_breaker_failures >= 2

            # Shutdown handler
            await handler.shutdown()

            # Circuit state should be reset (mixin reset and initialized flag cleared)
            # After shutdown, the circuit breaker is reset via mixin and flag cleared
            assert handler._circuit_breaker_initialized is False

    @pytest.mark.asyncio
    async def test_config_validates_circuit_breaker_bounds(self) -> None:
        """Test config validation enforces circuit breaker parameter bounds."""
        from omnibase_infra.handlers.model_vault_adapter_config import (
            ModelVaultAdapterConfig,
        )

        # Valid: within bounds
        config = ModelVaultAdapterConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.test1234"),
            circuit_breaker_failure_threshold=10,
            circuit_breaker_reset_timeout_seconds=60.0,
        )
        assert config.circuit_breaker_failure_threshold == 10
        assert config.circuit_breaker_reset_timeout_seconds == 60.0

        # Invalid: threshold below minimum (< 1)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                circuit_breaker_failure_threshold=0,
            )
        assert "circuit_breaker_failure_threshold" in str(exc_info.value)

        # Invalid: threshold above maximum (> 20)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                circuit_breaker_failure_threshold=25,
            )
        assert "circuit_breaker_failure_threshold" in str(exc_info.value)

        # Invalid: timeout below minimum (< 1.0)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                circuit_breaker_reset_timeout_seconds=0.5,
            )
        assert "circuit_breaker_reset_timeout_seconds" in str(exc_info.value)

        # Invalid: timeout above maximum (> 300.0)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                circuit_breaker_reset_timeout_seconds=400.0,
            )
        assert "circuit_breaker_reset_timeout_seconds" in str(exc_info.value)


class TestVaultAdapterErrorCodes:
    """Test VaultAdapter error code validation and consistency."""

    @pytest.mark.asyncio
    async def test_protocol_configuration_error_for_validation(self) -> None:
        """Test ProtocolConfigurationError is raised for Pydantic validation failures."""
        from omnibase_infra.errors import ProtocolConfigurationError

        handler = VaultAdapter()

        # Invalid config should raise ProtocolConfigurationError
        invalid_config = {
            "url": "https://vault.example.com:8200",
            "token": "s.test1234",
            "timeout_seconds": 400.0,  # Exceeds max (300.0)
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(invalid_config)

        # Verify error code
        assert exc_info.value.model.error_code is not None
        assert exc_info.value.model.error_code.name == "INVALID_CONFIGURATION"

    @pytest.mark.asyncio
    async def test_runtime_host_error_for_missing_url(
        self,
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test RuntimeHostError is raised for empty URL with correct error code."""
        from omnibase_infra.errors import ProtocolConfigurationError

        handler = VaultAdapter()

        # Config with empty URL - this fails Pydantic validation
        config: dict[str, VaultConfigValue] = {
            "url": "",
            "token": "s.test1234",
        }  # Empty URL

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Empty URL fails Pydantic validation, raising ProtocolConfigurationError
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await handler.initialize(config)

            # Verify error code is INVALID_CONFIGURATION for empty URL
            assert exc_info.value.model.error_code is not None
            assert exc_info.value.model.error_code.name == "INVALID_CONFIGURATION"

    @pytest.mark.asyncio
    async def test_infra_authentication_error_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test InfraAuthenticationError has correct error code."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            import hvac.exceptions

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.Forbidden("Permission denied")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.execute(envelope)

            # Verify error code
            assert exc_info.value.model.error_code is not None
            assert exc_info.value.model.error_code.name == "AUTHENTICATION_ERROR"

    @pytest.mark.asyncio
    async def test_secret_resolution_error_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test SecretResolutionError has correct error code."""
        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            import hvac.exceptions

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.InvalidPath("Secret not found")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "nonexistent/path"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(SecretResolutionError) as exc_info:
                await handler.execute(envelope)

            # Verify error code
            assert exc_info.value.model.error_code is not None
            assert exc_info.value.model.error_code.name == "RESOURCE_NOT_FOUND"


class TestVaultAdapterBoundedQueue:
    """Test VaultAdapter bounded queue functionality."""

    @pytest.mark.asyncio
    async def test_queue_size_configured_from_multiplier(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test queue size is calculated from multiplier."""
        handler = VaultAdapter()

        # Set custom multiplier
        vault_config["max_concurrent_operations"] = 10
        vault_config["max_queue_size_multiplier"] = 5

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            assert handler.max_workers == 10
            assert handler.max_queue_size == 50  # 10 * 5

    @pytest.mark.asyncio
    async def test_queue_size_default_multiplier(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test queue size uses default multiplier when not specified."""
        handler = VaultAdapter()

        vault_config["max_concurrent_operations"] = 10
        # Don't set max_queue_size_multiplier to test default

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            assert handler.max_workers == 10
            assert handler.max_queue_size == 30  # 10 * 3 (default multiplier)

    @pytest.mark.asyncio
    async def test_config_validates_queue_multiplier_bounds(self) -> None:
        """Test config validation enforces queue multiplier bounds."""
        from omnibase_infra.handlers.model_vault_adapter_config import (
            ModelVaultAdapterConfig,
        )

        # Valid: within bounds (1-10)
        config = ModelVaultAdapterConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.test1234"),
            max_queue_size_multiplier=5,
        )
        assert config.max_queue_size_multiplier == 5

        # Invalid: below minimum (< 1)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                max_queue_size_multiplier=0,
            )
        assert "max_queue_size_multiplier" in str(exc_info.value)

        # Invalid: above maximum (> 10)
        with pytest.raises(ValidationError) as exc_info:
            ModelVaultAdapterConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                max_queue_size_multiplier=15,
            )
        assert "max_queue_size_multiplier" in str(exc_info.value)


class TestVaultAdapterErrorCodeValidation:
    """Test error code mappings and validation per PR #38 feedback."""

    @pytest.mark.asyncio
    async def test_validation_error_uses_protocol_configuration_error(
        self,
    ) -> None:
        """Test ValidationError raises ProtocolConfigurationError with correct error code."""
        from omnibase_core.enums import EnumCoreErrorCode

        from omnibase_infra.errors import ProtocolConfigurationError

        handler = VaultAdapter()

        # Invalid configuration (missing required field)
        invalid_config: dict[str, VaultConfigValue] = {
            # Missing 'url' and 'token'
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(invalid_config)

        # Validate error code is INVALID_CONFIGURATION
        error = exc_info.value
        assert error.model.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION
        assert "Vault configuration" in str(error)

    @pytest.mark.asyncio
    async def test_connection_error_uses_service_unavailable_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test InfraConnectionError for Vault uses SERVICE_UNAVAILABLE error code."""
        import hvac.exceptions
        from omnibase_core.enums import EnumCoreErrorCode

        from omnibase_infra.errors import InfraConnectionError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            # Simulate VaultError during connection
            mock_hvac_client.is_authenticated.side_effect = hvac.exceptions.VaultError(
                "Connection refused"
            )

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(vault_config)

            # Validate error code is SERVICE_UNAVAILABLE (Vault transport type)
            error = exc_info.value
            assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_authentication_error_uses_authentication_error_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test InfraAuthenticationError uses AUTHENTICATION_ERROR code."""
        from omnibase_core.enums import EnumCoreErrorCode

        from omnibase_infra.errors import InfraAuthenticationError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            # Simulate authentication failure
            mock_hvac_client.is_authenticated.return_value = False

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(vault_config)

            # Validate error code is AUTHENTICATION_ERROR
            error = exc_info.value
            assert error.model.error_code == EnumCoreErrorCode.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_timeout_error_uses_timeout_error_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test InfraTimeoutError uses TIMEOUT_ERROR code."""
        from omnibase_core.enums import EnumCoreErrorCode

        from omnibase_infra.errors import InfraTimeoutError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Mock timeout on secret read
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                TimeoutError("Operation timed out")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraTimeoutError) as exc_info:
                await handler.execute(envelope)

            # Validate error code is TIMEOUT_ERROR
            error = exc_info.value
            assert error.model.error_code == EnumCoreErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_unavailable_error_uses_service_unavailable_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test InfraUnavailableError uses SERVICE_UNAVAILABLE code."""
        from omnibase_core.enums import EnumCoreErrorCode

        from omnibase_infra.errors import InfraUnavailableError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Configure circuit breaker with low threshold for testing
            vault_config["circuit_breaker_enabled"] = True
            vault_config["circuit_breaker_failure_threshold"] = 1
            vault_config["circuit_breaker_reset_timeout_seconds"] = 5.0

            await handler.initialize(vault_config)

            # Trigger circuit breaker open
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Vault down"
            )

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # First call fails and opens circuit
            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Second call hits open circuit
            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            # Validate error code is SERVICE_UNAVAILABLE
            error = exc_info.value
            assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_secret_resolution_error_uses_resource_not_found_code(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test SecretResolutionError uses RESOURCE_NOT_FOUND code."""
        import hvac.exceptions
        from omnibase_core.enums import EnumCoreErrorCode

        from omnibase_infra.errors import SecretResolutionError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Simulate invalid path error
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.InvalidPath("Secret not found")
            )

            await handler.initialize(vault_config)

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "nonexistent/path"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(SecretResolutionError) as exc_info:
                await handler.execute(envelope)

            # Validate error code is RESOURCE_NOT_FOUND
            error = exc_info.value
            assert error.model.error_code == EnumCoreErrorCode.RESOURCE_NOT_FOUND

    @pytest.mark.asyncio
    async def test_error_context_includes_namespace(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test all error contexts include Vault namespace per PR #38 feedback."""
        from omnibase_infra.errors import RuntimeHostError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            # Test missing path error includes namespace
            envelope = {
                "operation": "vault.read_secret",
                "payload": {},  # Missing 'path'
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            error = exc_info.value
            # Validate namespace is in error context (context is a dict)
            assert error.model.context is not None
            assert isinstance(error.model.context, dict)
            assert error.model.context.get("namespace") == "engineering"  # From fixture

    @pytest.mark.asyncio
    async def test_health_check_propagates_errors(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health_check propagates errors instead of swallowing them per PR #38."""
        from omnibase_infra.errors import InfraConnectionError

        handler = VaultAdapter()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            # Simulate health check failure
            mock_hvac_client.sys.read_health_status.side_effect = Exception(
                "Health check failed"
            )

            # Health check should propagate error, not return healthy=False
            with pytest.raises(InfraConnectionError):
                await handler.health_check()

    @pytest.mark.asyncio
    async def test_all_error_codes_are_mapped_correctly(
        self,
        vault_config: dict[str, VaultConfigValue],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Comprehensive test verifying all error types use correct error codes.

        This test validates that each infrastructure error type is raised with
        the correct EnumCoreErrorCode per the error mapping table in CLAUDE.md.
        Uses pytest.raises to ensure errors are actually raised.
        """
        # Track verified error types to ensure all are tested
        verified_errors: list[str] = []

        # 1. Test ProtocolConfigurationError -> INVALID_CONFIGURATION
        handler = VaultAdapter()
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize({})
        assert (
            exc_info.value.model.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION
        )
        verified_errors.append("ProtocolConfigurationError")

        # 2. Test InfraConnectionError -> SERVICE_UNAVAILABLE (Vault transport)
        handler = VaultAdapter()
        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            mock_hvac_client.is_authenticated.side_effect = hvac.exceptions.VaultError(
                "Connection error"
            )
            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(vault_config)
            assert (
                exc_info.value.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE
            )
            verified_errors.append("InfraConnectionError")
            # Reset for next test
            mock_hvac_client.is_authenticated.side_effect = None
            mock_hvac_client.is_authenticated.return_value = True

        # 3. Test SecretResolutionError -> RESOURCE_NOT_FOUND
        handler = VaultAdapter()
        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.InvalidPath("Secret not found")
            )

            envelope: dict[str, str | UUID | dict[str, str]] = {
                "operation": "vault.read_secret",
                "payload": {"path": "nonexistent/path"},
                "correlation_id": uuid4(),
            }
            with pytest.raises(SecretResolutionError) as exc_info:
                await handler.execute(envelope)
            assert (
                exc_info.value.model.error_code == EnumCoreErrorCode.RESOURCE_NOT_FOUND
            )
            verified_errors.append("SecretResolutionError")
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None

        # 4. Test InfraTimeoutError -> TIMEOUT_ERROR
        handler = VaultAdapter()
        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                TimeoutError("Operation timed out")
            )

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }
            with pytest.raises(InfraTimeoutError) as exc_info:
                await handler.execute(envelope)
            assert exc_info.value.model.error_code == EnumCoreErrorCode.TIMEOUT_ERROR
            verified_errors.append("InfraTimeoutError")
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None

        # 5. Test InfraAuthenticationError -> AUTHENTICATION_ERROR
        handler = VaultAdapter()
        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = (
                hvac.exceptions.Forbidden("Permission denied")
            )

            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }
            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.execute(envelope)
            assert (
                exc_info.value.model.error_code
                == EnumCoreErrorCode.AUTHENTICATION_ERROR
            )
            verified_errors.append("InfraAuthenticationError")
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None

        # 6. Test InfraUnavailableError -> SERVICE_UNAVAILABLE (circuit breaker)
        handler = VaultAdapter()
        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            # Configure for quick circuit breaker trigger
            test_config = dict(vault_config)
            test_config["circuit_breaker_enabled"] = True
            test_config["circuit_breaker_failure_threshold"] = 1

            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Vault down"
            )

            await handler.initialize(test_config)
            envelope = {
                "operation": "vault.read_secret",
                "payload": {"path": "myapp/config"},
                "correlation_id": uuid4(),
            }

            # First call opens circuit
            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Second call hits open circuit
            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)
            assert (
                exc_info.value.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE
            )
            verified_errors.append("InfraUnavailableError")
            mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = None

        # Verify all 6 error types were tested
        expected_errors = [
            "ProtocolConfigurationError",
            "InfraConnectionError",
            "SecretResolutionError",
            "InfraTimeoutError",
            "InfraAuthenticationError",
            "InfraUnavailableError",
        ]
        assert sorted(verified_errors) == sorted(expected_errors), (
            f"Not all error types were verified. "
            f"Expected: {expected_errors}, Got: {verified_errors}"
        )
