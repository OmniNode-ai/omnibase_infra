"""
Unit tests for registry node Kafka connection behavior.

Tests verify that KafkaClient.connect() is called during initialization
to prevent the "Kafka client not connected" error during consumption.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry


@pytest.mark.asyncio
async def test_registry_connects_kafka_client_during_initialization():
    """
    Verify registry calls connect() on KafkaClient during initialization.

    This test ensures the fix for Issue #6 is working:
    - KafkaClient.connect() MUST be called after client creation
    - This sets _connected = True, allowing consume_messages() to work
    - Without this, tests timeout waiting for consumer to receive messages
    """
    # Create mock container with minimal required config
    container = MagicMock()
    container.get_service.return_value = None  # Force client creation
    container.value = {
        "kafka_broker_url": "localhost:29092",
        "postgres_password": "test_password",
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_database": "test_db",
        "postgres_user": "test_user",
        "consul_host": "localhost",
        "consul_port": 8500,
    }

    # Mock KafkaClient class - patch at import location
    with (
        patch("omninode_bridge.services.kafka_client.KafkaClient") as MockKafkaClient,
        patch(
            "omninode_bridge.services.postgres_client.PostgresClient"
        ) as MockPostgresClient,
    ):
        # Create mock Kafka instance with async connect method
        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance._connected = False

        # After connect() is called, set _connected to True
        async def mock_connect():
            mock_client_instance._connected = True

        mock_client_instance.connect.side_effect = mock_connect
        MockKafkaClient.return_value = mock_client_instance

        # Create mock Postgres instance
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.connect = AsyncMock()
        mock_postgres_instance.close = AsyncMock()
        MockPostgresClient.return_value = mock_postgres_instance

        # Create registry node
        registry = NodeBridgeRegistry(container, environment="dev")

        # Initialize services (this should call connect())
        await registry._initialize_services_async(container)

        # ✅ ASSERT: connect() was called exactly once
        mock_client_instance.connect.assert_called_once()

        # ✅ ASSERT: _connected flag is now True
        assert mock_client_instance._connected is True

        # ✅ ASSERT: Client was registered with container
        container.register_service.assert_called()


@pytest.mark.asyncio
async def test_registry_handles_kafka_connection_failure_in_test_mode():
    """
    Verify registry handles Kafka connection failures gracefully in test mode.

    In test/dev environments, connection failures should:
    - Log a warning
    - Set kafka_client to None
    - NOT raise an exception

    This allows tests to run without real Kafka when needed.
    """
    container = MagicMock()
    container.get_service.return_value = None
    container.value = {
        "kafka_broker_url": "localhost:29092",
        "postgres_password": "test_password",
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_database": "test_db",
        "postgres_user": "test_user",
        "consul_host": "localhost",
        "consul_port": 8500,
    }

    with (
        patch("omninode_bridge.services.kafka_client.KafkaClient") as MockKafkaClient,
        patch(
            "omninode_bridge.services.postgres_client.PostgresClient"
        ) as MockPostgresClient,
    ):
        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_client_instance._connected = False
        MockKafkaClient.return_value = mock_client_instance

        # Create mock Postgres instance
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.connect = AsyncMock()
        mock_postgres_instance.close = AsyncMock()
        MockPostgresClient.return_value = mock_postgres_instance

        registry = NodeBridgeRegistry(container, environment="dev")

        # Should NOT raise exception in test mode
        await registry._initialize_services_async(container)

        # ✅ ASSERT: Client is set to None after connection failure
        assert registry.kafka_client is None

        # ✅ ASSERT: No exception was raised (execution continues)


@pytest.mark.asyncio
async def test_start_consuming_fails_if_client_not_connected():
    """
    Verify start_consuming() raises clear error if client not connected.

    This test ensures the safety check in start_consuming() works:
    - If kafka_client._connected is False, raise OnexError
    - Error message should be clear and actionable
    - Context should include resolution steps
    """
    from omninode_bridge.nodes.registry.v1_0_0.node import EnumCoreErrorCode, OnexError

    container = MagicMock()
    container.value = {
        "kafka_broker_url": "localhost:29092",
        "postgres_password": "test_password",
    }

    # Create registry
    registry = NodeBridgeRegistry(container, environment="dev")

    # Mock kafka_client as existing but NOT connected
    mock_client = MagicMock()
    mock_client._connected = False
    registry.kafka_client = mock_client

    # ✅ ASSERT: start_consuming() raises OnexError
    with pytest.raises(OnexError) as exc_info:
        await registry.start_consuming()

    # ✅ ASSERT: Error code is CONFIGURATION_ERROR
    assert exc_info.value.error_code == EnumCoreErrorCode.CONFIGURATION_ERROR

    # ✅ ASSERT: Error message mentions "not connected"
    assert "not connected" in str(exc_info.value.message).lower()

    # ✅ ASSERT: Context includes resolution steps
    # Note: Context may be nested in additional_context wrapper
    context_data = exc_info.value.context
    if "additional_context" in context_data:
        context_data = context_data["additional_context"].get("context", context_data)
    assert "resolution" in context_data or "error_type" in context_data


@pytest.mark.asyncio
async def test_registry_connects_existing_kafka_client_from_container():
    """
    Verify registry connects existing KafkaClient from container.

    When kafka_client is provided by container (e.g., from test fixture):
    - Registry should check if it's connected
    - If not connected, call connect()
    - If connection fails in production, raise exception
    """
    container = MagicMock()
    container.value = {
        "kafka_broker_url": "localhost:29092",
        "postgres_password": "test_password",
    }

    # Mock existing kafka_client from container (like test fixtures provide)
    mock_existing_client = MagicMock()
    mock_existing_client.connect = AsyncMock()
    mock_existing_client._connected = False

    # After connect() is called, set _connected to True
    async def mock_connect():
        mock_existing_client._connected = True

    mock_existing_client.connect.side_effect = mock_connect

    # Container provides existing client
    def mock_get_service(service_name):
        if service_name == "kafka_client":
            return mock_existing_client
        return None

    container.get_service.side_effect = mock_get_service

    # Mock PostgresClient to prevent connection attempts
    with patch(
        "omninode_bridge.services.postgres_client.PostgresClient"
    ) as MockPostgresClient:
        mock_postgres_instance = MagicMock()
        mock_postgres_instance.connect = AsyncMock()
        mock_postgres_instance.close = AsyncMock()
        MockPostgresClient.return_value = mock_postgres_instance

        registry = NodeBridgeRegistry(container, environment="dev")

        # Initialize services (should connect existing client)
        await registry._initialize_services_async(container)

        # ✅ ASSERT: connect() was called on existing client
        mock_existing_client.connect.assert_called_once()

        # ✅ ASSERT: _connected flag is now True
        assert mock_existing_client._connected is True

        # ✅ ASSERT: Registry uses the existing client
        assert registry.kafka_client is mock_existing_client
