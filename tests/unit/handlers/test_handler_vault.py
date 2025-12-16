# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for VaultHandler.

These tests use mocked hvac client to validate VaultHandler behavior
without requiring actual Vault server infrastructure.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.handlers.handler_vault import VaultHandler
from omnibase_infra.handlers.model_vault_handler_config import (
    ModelVaultHandlerConfig,
    ModelVaultRetryConfig,
)


@pytest.fixture
def vault_config() -> dict[str, object]:
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
    client.sys = MagicMock()
    return client


class TestVaultHandlerInitialization:
    """Test VaultHandler initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful initialization with valid config."""
        handler = VaultHandler()

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
        handler = VaultHandler()
        config = {"token": "s.test1234"}

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize(config)

        # Pydantic ValidationError contains "url" in the error details
        assert "vault configuration" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_missing_token(self) -> None:
        """Test initialization fails with missing token."""
        handler = VaultHandler()
        config = {"url": "https://vault.example.com:8200"}

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.initialize(config)

        assert "token" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_authentication_failure(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test initialization fails with authentication failure."""
        handler = VaultHandler()
        mock_hvac_client.is_authenticated.return_value = False

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.initialize(vault_config)

            assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(
        self,
        vault_config: dict[str, object],
    ) -> None:
        """Test initialization fails with connection error."""
        handler = VaultHandler()

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
        config = ModelVaultHandlerConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.test1234"),
            timeout_seconds=30.0,
        )
        assert config.url == "https://vault.example.com:8200"
        assert config.timeout_seconds == 30.0

        # Invalid timeout (too high)
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelVaultHandlerConfig(
                url="https://vault.example.com:8200",
                token=SecretStr("s.test1234"),
                timeout_seconds=400.0,  # Max is 300.0
            )

    @pytest.mark.asyncio
    async def test_secretstr_prevents_token_logging(self) -> None:
        """Test SecretStr prevents token from being logged."""
        config = ModelVaultHandlerConfig(
            url="https://vault.example.com:8200",
            token=SecretStr("s.sensitive_token_12345"),
        )

        # SecretStr representation should hide the token
        token_repr = repr(config.token)
        assert "sensitive_token_12345" not in token_repr
        assert "SecretStr" in token_repr


class TestVaultHandlerOperations:
    """Test VaultHandler secret operations."""

    @pytest.mark.asyncio
    async def test_read_secret_success(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret read operation."""
        handler = VaultHandler()

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
            assert response["payload"]["data"] == {
                "password": "secret123",
                "username": "admin",
            }
            assert response["payload"]["metadata"]["version"] == 1
            mock_hvac_client.secrets.kv.v2.read_secret_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_secret_success(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret write operation."""
        handler = VaultHandler()

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
            assert response["payload"]["version"] == 2
            mock_hvac_client.secrets.kv.v2.create_or_update_secret.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_secret_success(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret deletion."""
        handler = VaultHandler()

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
            assert response["payload"]["deleted"] is True
            mock_hvac_client.secrets.kv.v2.delete_latest_version_of_secret.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_secrets_success(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful secret listing."""
        handler = VaultHandler()

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
            assert response["payload"]["keys"] == ["config1", "config2", "config3/"]
            mock_hvac_client.secrets.kv.v2.list_secrets.assert_called_once()

    @pytest.mark.asyncio
    async def test_operation_missing_path(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test operation fails with missing path."""
        handler = VaultHandler()

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


class TestVaultHandlerTokenRenewal:
    """Test VaultHandler token renewal management."""

    @pytest.mark.asyncio
    async def test_renew_token_success(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test successful token renewal."""
        handler = VaultHandler()

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

            assert result["auth"]["renewable"] is True
            assert result["auth"]["lease_duration"] == 3600
            mock_hvac_client.auth.token.renew_self.assert_called_once()

    @pytest.mark.asyncio
    async def test_renew_token_operation_envelope(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test token renewal via envelope operation."""
        handler = VaultHandler()

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
            assert response["payload"]["renewable"] is True
            assert response["payload"]["lease_duration"] == 3600


class TestVaultHandlerRetryLogic:
    """Test VaultHandler retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test retry logic on transient failures."""
        handler = VaultHandler()

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
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test retry logic when all attempts exhausted."""
        handler = VaultHandler()

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
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test operation timeout raises InfraTimeoutError."""
        handler = VaultHandler()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            # Simulate timeout
            async def slow_operation(*args: Any, **kwargs: Any) -> None:
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


class TestVaultHandlerErrorHandling:
    """Test VaultHandler error handling and sanitization."""

    @pytest.mark.asyncio
    async def test_authentication_error(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test authentication error handling."""
        handler = VaultHandler()

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
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test secret not found error handling."""
        handler = VaultHandler()

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
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test Vault server unavailable error."""
        handler = VaultHandler()

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


class TestVaultHandlerHealthCheck:
    """Test VaultHandler health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health check returns healthy status."""
        handler = VaultHandler()

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
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health check returns unhealthy on error."""
        handler = VaultHandler()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client

            mock_hvac_client.sys.read_health_status.side_effect = Exception(
                "Connection refused"
            )

            await handler.initialize(vault_config)

            health = await handler.health_check()

            assert health["healthy"] is False
            assert health["initialized"] is True

    @pytest.mark.asyncio
    async def test_health_check_operation_envelope(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test health check via envelope operation."""
        handler = VaultHandler()

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
            assert response["payload"]["healthy"] is True


class TestVaultHandlerDescribe:
    """Test VaultHandler describe functionality."""

    @pytest.mark.asyncio
    async def test_describe(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test describe returns handler metadata."""
        handler = VaultHandler()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            description = handler.describe()

            assert description["handler_type"] == "vault"
            assert "vault.read_secret" in description["supported_operations"]
            assert "vault.write_secret" in description["supported_operations"]
            assert "vault.delete_secret" in description["supported_operations"]
            assert "vault.list_secrets" in description["supported_operations"]
            assert "vault.renew_token" in description["supported_operations"]
            assert "vault.health_check" in description["supported_operations"]
            assert description["initialized"] is True


class TestVaultHandlerShutdown:
    """Test VaultHandler shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        vault_config: dict[str, object],
        mock_hvac_client: MagicMock,
    ) -> None:
        """Test handler shutdown releases resources."""
        handler = VaultHandler()

        with patch("omnibase_infra.handlers.handler_vault.hvac.Client") as MockClient:
            MockClient.return_value = mock_hvac_client
            await handler.initialize(vault_config)

            assert handler._initialized is True
            assert handler._client is not None

            await handler.shutdown()

            assert handler._initialized is False
            assert handler._client is None
            assert handler._config is None
