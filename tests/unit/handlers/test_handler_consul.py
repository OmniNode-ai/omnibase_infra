# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ConsulHandler.

These tests use mocked consul client to validate ConsulHandler behavior
without requiring actual Consul server infrastructure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import consul
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
)
from omnibase_infra.handlers.handler_consul import ConsulHandler
from omnibase_infra.handlers.model_consul_handler_config import ModelConsulHandlerConfig


@pytest.fixture
def consul_config() -> dict[str, object]:
    """Provide test Consul configuration."""
    return {
        "host": "consul.example.com",
        "port": 8500,
        "scheme": "http",
        "token": "acl-token-abc123",
        "timeout_seconds": 30.0,
        "connect_timeout_seconds": 10.0,
        "datacenter": "dc1",
        "health_check_interval_seconds": 30.0,
        "retry": {
            "max_attempts": 3,
            "initial_delay_seconds": 1.0,
            "max_delay_seconds": 30.0,
            "exponential_base": 2.0,
        },
    }


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Provide mocked consul.Consul client."""
    client = MagicMock()

    # Mock KV store operations
    client.kv = MagicMock()
    client.kv.get = MagicMock(
        return_value=(
            0,
            {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
        )
    )
    client.kv.put = MagicMock(return_value=True)
    client.kv.delete = MagicMock(return_value=True)

    # Mock Agent operations (service registration)
    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = MagicMock(
        return_value=None
    )  # Returns None on success
    client.agent.service.deregister = MagicMock(
        return_value=None
    )  # Returns None on success
    client.agent.services = MagicMock(
        return_value={
            "service-1": {
                "ID": "service-1",
                "Service": "my-service",
                "Address": "192.168.1.100",
                "Port": 8080,
            }
        }
    )

    # Mock Catalog operations
    client.catalog = MagicMock()
    client.catalog.services = MagicMock(
        return_value=(0, {"my-service": [], "other-service": []})
    )
    client.catalog.service = MagicMock(
        return_value=(
            0,
            [
                {
                    "ServiceID": "service-1",
                    "ServiceName": "my-service",
                    "ServiceAddress": "192.168.1.100",
                    "ServicePort": 8080,
                }
            ],
        )
    )

    # Mock Health operations
    client.health = MagicMock()
    client.health.service = MagicMock(
        return_value=(
            0,
            [
                {
                    "Service": {
                        "ID": "service-1",
                        "Service": "my-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                    },
                    "Checks": [
                        {"Status": "passing", "Name": "Service check"},
                    ],
                }
            ],
        )
    )
    client.health.checks = MagicMock(
        return_value=(
            0,
            [
                {
                    "Status": "passing",
                    "Name": "Service check",
                    "ServiceID": "service-1",
                },
            ],
        )
    )

    # Mock Status operations (for health check)
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    return client


class TestConsulHandlerInitialization:
    """Test ConsulHandler initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test successful initialization with valid config."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            assert handler._initialized is True
            assert handler._config is not None
            assert handler._config.host == "consul.example.com"
            assert handler._config.port == 8500
            assert handler._config.datacenter == "dc1"
            MockClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_default_config(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test initialization with default config values."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize({})

            assert handler._initialized is True
            assert handler._config is not None
            assert handler._config.host == "localhost"
            assert handler._config.port == 8500

    @pytest.mark.asyncio
    async def test_initialize_with_token(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test initialization with ACL token."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            assert handler._config is not None
            assert handler._config.token is not None
            assert isinstance(handler._config.token, SecretStr)

    @pytest.mark.asyncio
    async def test_initialize_invalid_port(self) -> None:
        """Test initialization fails with invalid port."""
        handler = ConsulHandler()
        config: dict[str, object] = {"port": 0}

        with pytest.raises((ProtocolConfigurationError, RuntimeHostError)):
            await handler.initialize(config)

    @pytest.mark.asyncio
    async def test_initialize_invalid_scheme(self) -> None:
        """Test initialization fails with invalid scheme."""
        handler = ConsulHandler()
        config: dict[str, object] = {"scheme": "ftp"}

        with pytest.raises((ProtocolConfigurationError, RuntimeHostError)):
            await handler.initialize(config)

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(
        self,
        consul_config: dict[str, object],
    ) -> None:
        """Test initialization fails with connection error."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.side_effect = consul.ConsulException("Connection refused")

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(consul_config)

            assert "consul connection failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_no_leader(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test initialization fails when cluster has no leader."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            mock_consul_client.status.leader.return_value = ""

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.initialize(consul_config)

            assert "no leader" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_config_model_validation(self) -> None:
        """Test Pydantic config model validation."""
        # Valid config
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8500,
            timeout_seconds=30.0,
        )
        assert config.host == "consul.example.com"
        assert config.timeout_seconds == 30.0

        # Invalid timeout (too high)
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(
                timeout_seconds=400.0,  # Max is 300.0
            )
        assert "timeout_seconds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_secretstr_prevents_token_logging(self) -> None:
        """Test SecretStr prevents token from being logged."""
        config = ModelConsulHandlerConfig(
            token=SecretStr("sensitive_acl_token_12345"),
        )

        # SecretStr representation should hide the token
        token_repr = repr(config.token)
        assert "sensitive_acl_token_12345" not in token_repr
        assert "SecretStr" in token_repr


class TestConsulHandlerType:
    """Test ConsulHandler type property."""

    def test_handler_type_property(self) -> None:
        """Test handler_type property returns correct type."""
        handler = ConsulHandler()
        handler_type = handler.handler_type
        assert handler_type == "consul"


class TestConsulHandlerKVOperations:
    """Test ConsulHandler KV store operations."""

    @pytest.mark.asyncio
    async def test_kv_get_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test successful KV get operation."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            mock_consul_client.kv.get.return_value = (
                0,
                {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100},
            )

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["value"] == "test-value"
            assert payload["key"] == "test/key"
            assert payload["found"] is True
            mock_consul_client.kv.get.assert_called()

    @pytest.mark.asyncio
    async def test_kv_get_not_found(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test KV get when key not found."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Return None for non-existent key
            mock_consul_client.kv.get.return_value = (0, None)

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "nonexistent/key"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            # Should return success with found=False
            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload.get("found") is False
            assert payload.get("value") is None

    @pytest.mark.asyncio
    async def test_kv_get_recurse(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test KV get with recurse option."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # Return list for recurse mode
            mock_consul_client.kv.get.return_value = (
                0,
                [
                    {
                        "Key": "test/key1",
                        "Value": b"value1",
                        "Flags": 0,
                        "ModifyIndex": 1,
                    },
                    {
                        "Key": "test/key2",
                        "Value": b"value2",
                        "Flags": 0,
                        "ModifyIndex": 2,
                    },
                ],
            )

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/", "recurse": True},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["found"] is True
            assert payload["count"] == 2
            assert "items" in payload

    @pytest.mark.asyncio
    async def test_kv_put_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test successful KV put operation."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            mock_consul_client.kv.put.return_value = True

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_put",
                "payload": {"key": "test/key", "value": "new-value"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["success"] is True
            assert payload["key"] == "test/key"
            mock_consul_client.kv.put.assert_called()

    @pytest.mark.asyncio
    async def test_kv_put_with_flags(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test KV put with flags parameter."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            mock_consul_client.kv.put.return_value = True

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_put",
                "payload": {"key": "test/key", "value": "new-value", "flags": 42},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"

    @pytest.mark.asyncio
    async def test_kv_operation_missing_key(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test KV operation fails with missing key."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {},  # Missing 'key'
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "key" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_kv_put_missing_value(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test KV put fails with missing value."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_put",
                "payload": {"key": "test/key"},  # Missing 'value'
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "value" in str(exc_info.value).lower()


class TestConsulHandlerServiceOperations:
    """Test ConsulHandler service discovery operations."""

    @pytest.mark.asyncio
    async def test_register_service_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test successful service registration."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "my-service",
                    "service_id": "my-service-1",
                    "address": "192.168.1.100",
                    "port": 8080,
                    "tags": ["web", "api"],
                },
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["registered"] is True
            assert payload["name"] == "my-service"
            assert payload["service_id"] == "my-service-1"

    @pytest.mark.asyncio
    async def test_register_service_minimal(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test service registration with minimal parameters."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.register",
                "payload": {"name": "my-service"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["registered"] is True
            # service_id defaults to name when not provided
            assert payload["service_id"] == "my-service"

    @pytest.mark.asyncio
    async def test_register_service_missing_name(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test service registration fails with missing name."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.register",
                "payload": {"service_id": "my-service-1"},  # Missing 'name'
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "name" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_deregister_service_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test successful service deregistration."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.deregister",
                "payload": {"service_id": "my-service-1"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["deregistered"] is True
            assert payload["service_id"] == "my-service-1"

    @pytest.mark.asyncio
    async def test_deregister_service_missing_id(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test service deregistration fails with missing service_id."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.deregister",
                "payload": {},  # Missing 'service_id'
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "service_id" in str(exc_info.value).lower()


class TestConsulHandlerHealthOperations:
    """Test ConsulHandler health check operations."""

    @pytest.mark.asyncio
    async def test_health_check_operation_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test health check via execute operation."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.health_check",
                "payload": {},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)

            assert response["status"] == "success"
            payload = response["payload"]
            assert isinstance(payload, dict)
            assert payload["healthy"] is True

    @pytest.mark.asyncio
    async def test_handler_health_check_success(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test handler health check returns healthy status."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            health = await handler.health_check()

            assert health["healthy"] is True
            assert health["initialized"] is True
            assert health["handler_type"] == "consul"

    @pytest.mark.asyncio
    async def test_handler_health_check_unhealthy(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test health check propagates errors."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            # Now set error for subsequent health check calls
            mock_consul_client.status.leader.side_effect = consul.ConsulException(
                "Connection refused"
            )

            # Health check should raise error per PR #38 pattern
            with pytest.raises(InfraConnectionError):
                await handler.health_check()


class TestConsulHandlerExecuteRouting:
    """Test ConsulHandler execute operation routing."""

    @pytest.mark.asyncio
    async def test_execute_routes_to_kv_get(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test execute routes to KV get operation."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            response = await handler.execute(envelope)
            assert response["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_unsupported_operation(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test execute with unsupported operation raises error."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.unsupported_operation",
                "payload": {},
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "not supported" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_not_initialized(self) -> None:
        """Test execute fails when handler not initialized."""
        handler = ConsulHandler()

        envelope = {
            "operation": "consul.kv_get",
            "payload": {"key": "test/key"},
            "correlation_id": uuid4(),
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_missing_operation(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test execute fails with missing operation."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "operation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_missing_payload(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test execute fails with missing payload."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "correlation_id": uuid4(),
            }

            with pytest.raises(RuntimeHostError) as exc_info:
                await handler.execute(envelope)

            assert "payload" in str(exc_info.value).lower()


class TestConsulHandlerCorrelationId:
    """Test ConsulHandler correlation ID handling."""

    @pytest.mark.asyncio
    async def test_correlation_id_extraction_uuid(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test correlation ID extraction from UUID."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            test_uuid = uuid4()
            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": test_uuid,
            }

            response = await handler.execute(envelope)
            assert response["correlation_id"] == test_uuid

    @pytest.mark.asyncio
    async def test_correlation_id_extraction_string(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test correlation ID extraction from string."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            test_uuid = uuid4()
            test_uuid_str = str(test_uuid)
            envelope: dict[str, object] = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": test_uuid_str,
            }

            response = await handler.execute(envelope)
            assert response["correlation_id"] == test_uuid

    @pytest.mark.asyncio
    async def test_correlation_id_generation(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test correlation ID is generated when not provided."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope: dict[str, object] = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
            }

            response = await handler.execute(envelope)
            # Should have a correlation_id generated
            assert "correlation_id" in response
            assert isinstance(response["correlation_id"], UUID)


class TestConsulHandlerDescribe:
    """Test ConsulHandler describe functionality."""

    @pytest.mark.asyncio
    async def test_describe(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test describe returns handler metadata."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            description = handler.describe()

            assert "handler_type" in description
            assert description["handler_type"] == "consul"
            assert "supported_operations" in description
            supported_ops = description["supported_operations"]
            assert isinstance(supported_ops, list)
            # Should support KV and service operations
            assert "consul.kv_get" in supported_ops
            assert "consul.kv_put" in supported_ops
            assert "consul.register" in supported_ops
            assert "consul.deregister" in supported_ops
            assert "consul.health_check" in supported_ops
            assert description["initialized"] is True
            assert description["version"] == "0.1.0-mvp"


class TestConsulHandlerShutdown:
    """Test ConsulHandler shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test handler shutdown releases resources."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            assert handler._initialized is True
            assert handler._client is not None

            await handler.shutdown()

            assert handler._initialized is False
            assert handler._client is None
            assert handler._config is None


class TestConsulHandlerErrorHandling:
    """Test ConsulHandler error handling and sanitization."""

    @pytest.mark.asyncio
    async def test_connection_error_handling(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test connection error is properly wrapped."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Connection refused"
            )

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test timeout error is properly wrapped."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            mock_consul_client.kv.get.side_effect = consul.Timeout(
                "Operation timed out"
            )

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraTimeoutError):
                await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_authentication_error_handling(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test authentication error is properly wrapped."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            # ACL denied error
            mock_consul_client.kv.get.side_effect = consul.ACLPermissionDenied(
                "Permission denied"
            )

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraAuthenticationError):
                await handler.execute(envelope)


class TestConsulHandlerRetryLogic:
    """Test ConsulHandler retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test retry logic on transient failures."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # First call fails, second succeeds
            mock_consul_client.kv.get.side_effect = [
                Exception("Transient error"),
                (0, {"Value": b"test-value", "Key": "test/key", "ModifyIndex": 100}),
            ]

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            # Should succeed on retry
            response = await handler.execute(envelope)

            assert response["status"] == "success"
            assert mock_consul_client.kv.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test retry logic when all attempts exhausted."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            # All attempts fail
            mock_consul_client.kv.get.side_effect = Exception("Persistent error")

            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraConnectionError):
                await handler.execute(envelope)

            # Should have tried max_attempts times (3)
            assert mock_consul_client.kv.get.call_count == 3


class TestConsulHandlerErrorCodes:
    """Test ConsulHandler error code validation and consistency."""

    @pytest.mark.asyncio
    async def test_protocol_configuration_error_code(self) -> None:
        """Test ProtocolConfigurationError has correct error code."""
        handler = ConsulHandler()

        # Invalid configuration
        invalid_config: dict[str, object] = {
            "timeout_seconds": 400.0,  # Exceeds max
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(invalid_config)

        # Verify error code
        assert exc_info.value.model.error_code is not None
        assert exc_info.value.model.error_code.name == "INVALID_CONFIGURATION"

    @pytest.mark.asyncio
    async def test_connection_error_code(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test InfraConnectionError has correct error code."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Connection error"
            )

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            # Verify error code is SERVICE_UNAVAILABLE for Consul transport
            assert exc_info.value.model.error_code is not None
            assert exc_info.value.model.error_code.name == "SERVICE_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_authentication_error_code(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test InfraAuthenticationError has correct error code."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(consul_config)

            mock_consul_client.kv.get.side_effect = consul.ACLPermissionDenied(
                "Permission denied"
            )

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraAuthenticationError) as exc_info:
                await handler.execute(envelope)

            # Verify error code
            assert exc_info.value.model.error_code is not None
            assert exc_info.value.model.error_code.name == "AUTHENTICATION_ERROR"


class TestConsulHandlerSecuritySanitization:
    """Test ConsulHandler security sanitization of error messages."""

    @pytest.mark.asyncio
    async def test_validation_error_does_not_expose_token(self) -> None:
        """Test that validation errors do not expose token values.

        Security: Pydantic ValidationError can contain actual field values,
        which could expose sensitive tokens. This test verifies the handler
        sanitizes validation errors to only show field names, not values.
        """
        handler = ConsulHandler()

        # Configuration with an invalid token type that will trigger validation error
        # The sensitive value should NOT appear in the error message
        sensitive_token = "super_secret_acl_token_that_should_never_appear"
        invalid_config: dict[str, object] = {
            "token": sensitive_token,
            "timeout_seconds": 500.0,  # Invalid - max is 300.0
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(invalid_config)

        error_message = str(exc_info.value)

        # The sensitive token should NEVER appear in the error message
        assert sensitive_token not in error_message

        # The error should indicate which field failed validation
        assert "validation failed for fields" in error_message
        assert "timeout_seconds" in error_message

    @pytest.mark.asyncio
    async def test_error_messages_exclude_credentials(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that error messages never include credential information."""
        handler = ConsulHandler()

        sensitive_token = "my_acl_token_credential_xyz123"
        config: dict[str, object] = {
            "host": "consul.example.com",
            "port": 8500,
            "token": sensitive_token,
        }

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client

            await handler.initialize(config)

            # Simulate connection error
            mock_consul_client.kv.get.side_effect = consul.ConsulException(
                "Connection failed"
            )

            envelope = {
                "operation": "consul.kv_get",
                "payload": {"key": "test/key"},
                "correlation_id": uuid4(),
            }

            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(envelope)

            error_message = str(exc_info.value)

            # Sensitive token should never appear in error message
            assert sensitive_token not in error_message


class TestConsulHandlerThreadPool:
    """Test ConsulHandler thread pool functionality."""

    @pytest.mark.asyncio
    async def test_thread_pool_default_size(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test thread pool uses default size when not specified."""
        handler = ConsulHandler()

        # Remove max_concurrent_operations to test default
        consul_config.pop("max_concurrent_operations", None)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            assert handler._executor is not None
            assert handler.max_workers == 10  # Default value

    @pytest.mark.asyncio
    async def test_thread_pool_shutdown_on_handler_shutdown(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
    ) -> None:
        """Test thread pool is properly shutdown when handler shuts down."""
        handler = ConsulHandler()

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            executor = handler._executor
            assert executor is not None

            await handler.shutdown()

            assert handler._executor is None


__all__: list[str] = [
    "TestConsulHandlerInitialization",
    "TestConsulHandlerType",
    "TestConsulHandlerKVOperations",
    "TestConsulHandlerServiceOperations",
    "TestConsulHandlerHealthOperations",
    "TestConsulHandlerExecuteRouting",
    "TestConsulHandlerCorrelationId",
    "TestConsulHandlerDescribe",
    "TestConsulHandlerShutdown",
    "TestConsulHandlerErrorHandling",
    "TestConsulHandlerRetryLogic",
    "TestConsulHandlerErrorCodes",
    "TestConsulHandlerSecuritySanitization",
    "TestConsulHandlerThreadPool",
]
