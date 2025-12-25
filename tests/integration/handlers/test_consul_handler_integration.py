# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ConsulHandler against remote Consul infrastructure.

These tests validate ConsulHandler behavior against actual Consul infrastructure
running on the remote server (192.168.86.200). They require Consul to be available
and will be skipped gracefully if Consul is not reachable.

Test categories:
- Health Check Tests: Validate connectivity and health reporting
- KV Store Tests: Verify key-value store operations (put, get)
- Service Registration Tests: Test service register/deregister

Environment Variables:
    CONSUL_HOST: Consul server hostname (default: 192.168.86.200)
    CONSUL_PORT: Consul server port (default: 28500)
    CONSUL_SCHEME: HTTP scheme (default: http)
    CONSUL_TOKEN: Optional ACL token for authentication

Related Ticket: OMN-816
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from tests.integration.handlers.conftest import CONSUL_AVAILABLE

if TYPE_CHECKING:
    from omnibase_core.types import JsonValue

    from omnibase_infra.handlers import ConsulHandler

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Module-level markers - skip all tests if Consul is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not CONSUL_AVAILABLE,
        reason="Consul not available (cannot connect to remote infrastructure)",
    ),
]


# =============================================================================
# Health Check Tests - Validate basic connectivity
# =============================================================================


class TestConsulHandlerHealthCheck:
    """Tests for ConsulHandler health check and connectivity."""

    @pytest.mark.asyncio
    async def test_consul_health_check(
        self, consul_config: dict[str, JsonValue]
    ) -> None:
        """Test Consul handler connectivity via health check.

        Verifies that:
        - Handler can connect to remote Consul
        - Health check reports healthy status
        - Handler reports correct operational state
        """
        from omnibase_infra.handlers import ConsulHandler

        handler = ConsulHandler()
        await handler.initialize(consul_config)

        try:
            result = await handler.health_check()

            assert result["healthy"] is True
            assert result["initialized"] is True
            assert result["handler_type"] == "consul"
            assert result["timeout_seconds"] == 30.0
            # Circuit breaker should be closed (healthy)
            assert result["circuit_breaker_state"] == "closed"
            assert result["circuit_breaker_failure_count"] == 0
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_consul_health_check_via_execute(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test health check operation via execute method.

        Verifies that:
        - Health check works through envelope-based dispatch
        - Response includes correlation tracking
        - All health metrics are present
        """
        envelope = {
            "operation": "consul.health_check",
            "payload": {},
            "correlation_id": str(uuid4()),
        }

        result = await initialized_consul_handler.execute(envelope)

        assert result.result.status == "success"
        payload = result.result.payload.data
        assert payload.healthy is True
        assert payload.initialized is True
        assert payload.handler_type == "consul"

    @pytest.mark.asyncio
    async def test_handler_describe(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test handler describe returns correct metadata.

        Verifies that:
        - Describe returns supported operations
        - Handler reports correct type and version
        """
        description = initialized_consul_handler.describe()

        assert description["handler_type"] == "consul"
        assert description["initialized"] is True
        assert "consul.kv_get" in description["supported_operations"]
        assert "consul.kv_put" in description["supported_operations"]
        assert "consul.register" in description["supported_operations"]
        assert "consul.deregister" in description["supported_operations"]
        assert "consul.health_check" in description["supported_operations"]


# =============================================================================
# KV Store Tests - Key-Value operations
# =============================================================================


class TestConsulHandlerKVStore:
    """Tests for ConsulHandler KV store operations."""

    @pytest.mark.asyncio
    async def test_consul_kv_put_and_get(
        self,
        initialized_consul_handler: ConsulHandler,
        unique_kv_key: str,
    ) -> None:
        """Test storing and retrieving a value from KV store.

        Verifies that:
        - Values can be stored in KV store
        - Values can be retrieved accurately
        - Response includes metadata (index, modify_index)
        """
        test_value = f"test-value-{uuid4().hex[:8]}"

        # Put the value
        put_envelope = {
            "operation": "consul.kv_put",
            "payload": {
                "key": unique_kv_key,
                "value": test_value,
            },
            "correlation_id": str(uuid4()),
        }

        put_result = await initialized_consul_handler.execute(put_envelope)
        assert put_result.result.status == "success"
        assert put_result.result.payload.data.success is True
        assert put_result.result.payload.data.key == unique_kv_key

        # Get the value
        get_envelope = {
            "operation": "consul.kv_get",
            "payload": {
                "key": unique_kv_key,
            },
            "correlation_id": str(uuid4()),
        }

        get_result = await initialized_consul_handler.execute(get_envelope)
        assert get_result.result.status == "success"

        payload = get_result.result.payload.data
        assert payload.key == unique_kv_key
        assert payload.value == test_value
        # Should have metadata
        assert payload.index is not None

    @pytest.mark.asyncio
    async def test_consul_kv_get_not_found(
        self,
        initialized_consul_handler: ConsulHandler,
    ) -> None:
        """Test retrieving a non-existent key from KV store.

        Verifies that:
        - Non-existent keys return not-found response
        - Response includes proper metadata
        """
        non_existent_key = f"integration-test/nonexistent/{uuid4().hex}"

        get_envelope = {
            "operation": "consul.kv_get",
            "payload": {
                "key": non_existent_key,
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_consul_handler.execute(get_envelope)
        assert result.result.status == "success"

        # The payload should indicate key not found
        # Based on the handler code, this returns ModelConsulKVGetNotFoundPayload
        payload = result.result.payload.data
        assert payload.key == non_existent_key
        # Not-found payload has value=None
        assert not hasattr(payload, "value") or payload.value is None

    @pytest.mark.asyncio
    async def test_consul_kv_put_with_flags(
        self,
        initialized_consul_handler: ConsulHandler,
        unique_kv_key: str,
    ) -> None:
        """Test storing a value with custom flags.

        Verifies that:
        - Custom flags can be set on KV entries
        - Flags are preserved and retrievable
        """
        test_value = "flagged-value"
        test_flags = 42

        put_envelope = {
            "operation": "consul.kv_put",
            "payload": {
                "key": unique_kv_key,
                "value": test_value,
                "flags": test_flags,
            },
            "correlation_id": str(uuid4()),
        }

        put_result = await initialized_consul_handler.execute(put_envelope)
        assert put_result.result.status == "success"
        assert put_result.result.payload.data.success is True

        # Get and verify flags
        get_envelope = {
            "operation": "consul.kv_get",
            "payload": {"key": unique_kv_key},
            "correlation_id": str(uuid4()),
        }

        get_result = await initialized_consul_handler.execute(get_envelope)
        payload = get_result.result.payload.data
        assert payload.value == test_value
        assert payload.flags == test_flags

    @pytest.mark.asyncio
    async def test_consul_kv_overwrite(
        self,
        initialized_consul_handler: ConsulHandler,
        unique_kv_key: str,
    ) -> None:
        """Test overwriting an existing KV entry.

        Verifies that:
        - Existing values can be overwritten
        - New value is correctly stored
        """
        initial_value = "initial-value"
        updated_value = "updated-value"

        # Put initial value
        put_envelope_1 = {
            "operation": "consul.kv_put",
            "payload": {"key": unique_kv_key, "value": initial_value},
            "correlation_id": str(uuid4()),
        }
        await initialized_consul_handler.execute(put_envelope_1)

        # Overwrite with new value
        put_envelope_2 = {
            "operation": "consul.kv_put",
            "payload": {"key": unique_kv_key, "value": updated_value},
            "correlation_id": str(uuid4()),
        }
        result = await initialized_consul_handler.execute(put_envelope_2)
        assert result.result.payload.data.success is True

        # Verify updated value
        get_envelope = {
            "operation": "consul.kv_get",
            "payload": {"key": unique_kv_key},
            "correlation_id": str(uuid4()),
        }
        get_result = await initialized_consul_handler.execute(get_envelope)
        assert get_result.result.payload.data.value == updated_value


# =============================================================================
# Service Registration Tests
# =============================================================================


class TestConsulHandlerServiceRegistration:
    """Tests for ConsulHandler service registration operations."""

    @pytest.mark.asyncio
    async def test_consul_service_register_and_deregister(
        self,
        initialized_consul_handler: ConsulHandler,
        unique_service_name: str,
    ) -> None:
        """Test registering and deregistering a service.

        Verifies that:
        - Services can be registered with Consul
        - Services can be deregistered successfully
        """
        service_id = f"{unique_service_name}-001"

        # Register service
        register_envelope = {
            "operation": "consul.register",
            "payload": {
                "name": unique_service_name,
                "service_id": service_id,
                "address": "127.0.0.1",
                "port": 8080,
                "tags": ["integration-test", "auto-cleanup"],
            },
            "correlation_id": str(uuid4()),
        }

        register_result = await initialized_consul_handler.execute(register_envelope)
        assert register_result.result.status == "success"
        payload = register_result.result.payload.data
        assert payload.registered is True
        assert payload.name == unique_service_name
        assert payload.consul_service_id == service_id

        # Deregister service
        deregister_envelope = {
            "operation": "consul.deregister",
            "payload": {
                "service_id": service_id,
            },
            "correlation_id": str(uuid4()),
        }

        deregister_result = await initialized_consul_handler.execute(
            deregister_envelope
        )
        assert deregister_result.result.status == "success"
        assert deregister_result.result.payload.data.deregistered is True
        assert deregister_result.result.payload.data.consul_service_id == service_id

    @pytest.mark.asyncio
    async def test_consul_service_register_minimal(
        self,
        initialized_consul_handler: ConsulHandler,
        unique_service_name: str,
    ) -> None:
        """Test registering a service with minimal configuration.

        Verifies that:
        - Services can be registered with just name
        - Default service_id is used when not provided
        """
        try:
            # Register with minimal config
            register_envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": unique_service_name,
                },
                "correlation_id": str(uuid4()),
            }

            register_result = await initialized_consul_handler.execute(
                register_envelope
            )
            assert register_result.result.status == "success"
            payload = register_result.result.payload.data
            assert payload.registered is True
            assert payload.name == unique_service_name
            # service_id defaults to name when not provided
            assert payload.consul_service_id == unique_service_name

        finally:
            # Cleanup - deregister the service
            deregister_envelope = {
                "operation": "consul.deregister",
                "payload": {"service_id": unique_service_name},
                "correlation_id": str(uuid4()),
            }
            try:
                await initialized_consul_handler.execute(deregister_envelope)
            except Exception:
                pass  # Ignore cleanup errors

    @pytest.mark.asyncio
    async def test_consul_deregister_nonexistent_service(
        self,
        initialized_consul_handler: ConsulHandler,
    ) -> None:
        """Test deregistering a non-existent service.

        Verifies that:
        - Deregistering non-existent service succeeds (Consul's behavior)
        - No error is raised for missing service
        """
        nonexistent_service_id = f"nonexistent-svc-{uuid4().hex[:8]}"

        deregister_envelope = {
            "operation": "consul.deregister",
            "payload": {
                "service_id": nonexistent_service_id,
            },
            "correlation_id": str(uuid4()),
        }

        # Consul's deregister is idempotent - succeeds even for non-existent services
        result = await initialized_consul_handler.execute(deregister_envelope)
        assert result.result.status == "success"
        assert result.result.payload.data.deregistered is True

    @pytest.mark.asyncio
    async def test_consul_service_register_with_tags(
        self,
        initialized_consul_handler: ConsulHandler,
        unique_service_name: str,
    ) -> None:
        """Test registering a service with tags.

        Verifies that:
        - Services can be registered with tags
        - Multiple tags are supported
        """
        service_id = f"{unique_service_name}-tagged"
        tags = ["environment:test", "version:1.0.0", "integration-test"]

        try:
            register_envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": unique_service_name,
                    "service_id": service_id,
                    "address": "10.0.0.1",
                    "port": 9090,
                    "tags": tags,
                },
                "correlation_id": str(uuid4()),
            }

            result = await initialized_consul_handler.execute(register_envelope)
            assert result.result.status == "success"
            assert result.result.payload.data.registered is True

        finally:
            # Cleanup
            deregister_envelope = {
                "operation": "consul.deregister",
                "payload": {"service_id": service_id},
                "correlation_id": str(uuid4()),
            }
            try:
                await initialized_consul_handler.execute(deregister_envelope)
            except Exception:
                pass


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestConsulHandlerErrorHandling:
    """Tests for ConsulHandler error handling."""

    @pytest.mark.asyncio
    async def test_execute_without_initialize_raises_error(
        self, consul_config: dict[str, JsonValue]
    ) -> None:
        """Test that executing without initialization raises appropriate error.

        Verifies that:
        - RuntimeHostError is raised for uninitialized handler
        - Error message indicates initialization is required
        """
        from omnibase_infra.errors import RuntimeHostError
        from omnibase_infra.handlers import ConsulHandler

        handler = ConsulHandler()
        # Don't initialize

        envelope = {
            "operation": "consul.health_check",
            "payload": {},
        }

        with pytest.raises(RuntimeHostError, match="not initialized"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_invalid_operation_raises_error(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test that invalid operation raises appropriate error.

        Verifies that:
        - RuntimeHostError is raised for unsupported operations
        - Error message lists available operations
        """
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "consul.invalid_operation",
            "payload": {},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="not supported"):
            await initialized_consul_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_key_in_kv_get_raises_error(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test that missing key parameter raises appropriate error.

        Verifies that:
        - RuntimeHostError is raised for missing required parameter
        - Error message indicates missing 'key' parameter
        """
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "consul.kv_get",
            "payload": {},  # Missing 'key'
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="key"):
            await initialized_consul_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_value_in_kv_put_raises_error(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test that missing value parameter raises appropriate error.

        Verifies that:
        - RuntimeHostError is raised for missing required parameter
        - Error message indicates missing 'value' parameter
        """
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "consul.kv_put",
            "payload": {"key": "test-key"},  # Missing 'value'
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="value"):
            await initialized_consul_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_name_in_register_raises_error(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test that missing name parameter in register raises error.

        Verifies that:
        - RuntimeHostError is raised for missing service name
        - Error message indicates missing 'name' parameter
        """
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "consul.register",
            "payload": {"address": "127.0.0.1"},  # Missing 'name'
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="name"):
            await initialized_consul_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_service_id_in_deregister_raises_error(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test that missing service_id parameter in deregister raises error.

        Verifies that:
        - RuntimeHostError is raised for missing service_id
        - Error message indicates missing parameter
        """
        from omnibase_infra.errors import RuntimeHostError

        envelope = {
            "operation": "consul.deregister",
            "payload": {},  # Missing 'service_id'
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError, match="service_id"):
            await initialized_consul_handler.execute(envelope)


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestConsulHandlerLifecycle:
    """Tests for ConsulHandler lifecycle management."""

    @pytest.mark.asyncio
    async def test_shutdown_and_reinitialize(
        self, consul_config: dict[str, JsonValue]
    ) -> None:
        """Test that handler can be shutdown and reinitialized.

        Verifies that:
        - Handler can be properly shutdown
        - Handler can be reinitialized after shutdown
        - Health check works after reinitialization
        """
        from omnibase_infra.handlers import ConsulHandler

        handler = ConsulHandler()

        # First initialization
        await handler.initialize(consul_config)
        health1 = await handler.health_check()
        assert health1["healthy"] is True

        # Shutdown
        await handler.shutdown()

        # Reinitialize
        await handler.initialize(consul_config)
        health2 = await handler.health_check()
        assert health2["healthy"] is True

        # Final cleanup
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls_safe(
        self, initialized_consul_handler: ConsulHandler
    ) -> None:
        """Test that multiple shutdown calls are safe.

        Verifies that:
        - Calling shutdown multiple times doesn't raise errors
        - Handler is properly cleaned up
        """
        # Note: initialized_consul_handler fixture will also call shutdown
        # This tests that double-shutdown is safe
        await initialized_consul_handler.shutdown()

        # Should be safe to call again
        await initialized_consul_handler.shutdown()
