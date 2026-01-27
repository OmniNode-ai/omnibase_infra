# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeHostProcess.get_subscribers_for_topic() method.

Tests for OMN-1613: Add event bus topic storage to registry for dynamic topic discovery.

This module tests the topic subscriber query functionality that enables dynamic
routing based on Consul-stored topic subscriptions.

Test Categories:
    - Basic functionality: Returns node UUIDs for known topics
    - Edge cases: Empty lists, missing handlers, malformed data
    - Error handling: Consul errors, JSON parsing, UUID validation
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess


class TestGetSubscribersForTopic:
    """Tests for RuntimeHostProcess.get_subscribers_for_topic() method."""

    @pytest.fixture
    def runtime(self) -> RuntimeHostProcess:
        """Create a RuntimeHostProcess instance for testing."""
        return RuntimeHostProcess()

    @pytest.fixture
    def mock_consul_handler(self) -> MagicMock:
        """Create a mock Consul handler."""
        handler = MagicMock()
        handler.execute = AsyncMock()
        return handler

    @pytest.fixture
    def sample_node_ids(self) -> list[UUID]:
        """Return sample node UUIDs for testing."""
        return [
            UUID("12345678-1234-1234-1234-123456789abc"),
            UUID("87654321-4321-4321-4321-cba987654321"),
            UUID("abcdef00-1111-2222-3333-444455556666"),
        ]

    def _create_kv_found_response(
        self, value: str, key: str = "onex/topics/test.topic/subscribers"
    ) -> MagicMock:
        """Create a mock response for a found KV key.

        Args:
            value: The value to return (usually JSON string of node IDs)
            key: The KV key path

        Returns:
            Mock response matching ModelHandlerOutput structure.
        """
        # Create the payload data (ModelConsulKVGetFoundPayload)
        payload_data = MagicMock()
        payload_data.found = True
        payload_data.key = key
        payload_data.value = value
        payload_data.operation_type = "kv_get_found"

        # Create the payload wrapper (ModelConsulHandlerPayload)
        payload = MagicMock()
        payload.data = payload_data

        # Create the response (ModelConsulHandlerResponse)
        response = MagicMock()
        response.payload = payload

        # Create the handler output (ModelHandlerOutput)
        result = MagicMock()
        result.result = response

        return result

    def _create_kv_not_found_response(
        self, key: str = "onex/topics/test.topic/subscribers"
    ) -> MagicMock:
        """Create a mock response for a not-found KV key.

        Args:
            key: The KV key path that was not found

        Returns:
            Mock response matching ModelHandlerOutput structure.
        """
        # Create the payload data (ModelConsulKVGetNotFoundPayload)
        payload_data = MagicMock()
        payload_data.found = False
        payload_data.key = key
        payload_data.operation_type = "kv_get_not_found"

        # Create the payload wrapper
        payload = MagicMock()
        payload.data = payload_data

        # Create the response
        response = MagicMock()
        response.payload = payload

        # Create the handler output
        result = MagicMock()
        result.result = response

        return result

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_returns_node_ids(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
        sample_node_ids: list[UUID],
    ) -> None:
        """Test that get_subscribers_for_topic returns list of UUIDs."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Configure mock to return node IDs
        node_id_strings = [str(uid) for uid in sample_node_ids]
        mock_response = self._create_kv_found_response(json.dumps(node_id_strings))
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        topic = "dev.onex.evt.intent-classified.v1"
        result = await runtime.get_subscribers_for_topic(topic)

        # Verify
        assert result == sample_node_ids
        assert len(result) == 3
        assert all(isinstance(uid, UUID) for uid in result)

        # Verify envelope sent to handler
        mock_consul_handler.execute.assert_called_once()
        call_args = mock_consul_handler.execute.call_args[0][0]
        assert call_args["operation"] == "consul.kv_get"
        assert call_args["payload"]["key"] == f"onex/topics/{topic}/subscribers"

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_empty_list_unknown_topic(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test that unknown topic returns empty list (not error)."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Configure mock to return "not found" response
        mock_response = self._create_kv_not_found_response()
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("unknown.topic.v1")

        # Verify: Empty list, not an error
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_no_consul_handler(
        self,
        runtime: RuntimeHostProcess,
    ) -> None:
        """Test that missing consul handler returns empty list."""
        # No handler registered - get_handler("consul") returns None

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list, not an error
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_multiple_subscribers(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling of multiple subscribers for a topic."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Create 5 unique node IDs
        node_ids = [uuid4() for _ in range(5)]
        node_id_strings = [str(uid) for uid in node_ids]
        mock_response = self._create_kv_found_response(json.dumps(node_id_strings))
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.multi.v1")

        # Verify
        assert len(result) == 5
        assert result == node_ids

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_consul_error(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test that Consul errors return empty list (graceful degradation)."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Configure mock to raise an exception
        mock_consul_handler.execute.side_effect = Exception("Consul connection failed")

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list (graceful degradation), not an exception
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_invalid_json(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling of invalid JSON in topic subscribers value."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Return invalid JSON
        mock_response = self._create_kv_found_response("not valid json {{{")
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list, not an error
        assert result == []

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_invalid_uuid(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling of invalid UUID strings in subscriber list."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Return list with invalid UUID
        invalid_list = ["not-a-valid-uuid", "also-invalid"]
        mock_response = self._create_kv_found_response(json.dumps(invalid_list))
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list due to invalid UUIDs
        assert result == []

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_empty_value(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling of empty value in KV store."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Return empty string value
        mock_response = self._create_kv_found_response("")
        # Set value to None to simulate empty
        mock_response.result.payload.data.value = None
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_empty_json_array(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling of empty JSON array in subscriber list."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Return empty JSON array
        mock_response = self._create_kv_found_response("[]")
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_non_array_json(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling when JSON value is not an array."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Return JSON object instead of array
        mock_response = self._create_kv_found_response('{"key": "value"}')
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list (not an array)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_null_response(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test handling when handler returns None."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Return None
        mock_consul_handler.execute.return_value = None

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_single_subscriber(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test topic with exactly one subscriber."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Single node ID
        single_id = uuid4()
        mock_response = self._create_kv_found_response(json.dumps([str(single_id)]))
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.single.v1")

        # Verify
        assert len(result) == 1
        assert result[0] == single_id

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_correlation_id_generated(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test that a correlation ID is generated for the Consul request."""
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Configure mock response
        mock_response = self._create_kv_found_response("[]")
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: correlation_id is present and valid
        call_args = mock_consul_handler.execute.call_args[0][0]
        assert "correlation_id" in call_args
        # Should be a valid UUID string
        UUID(call_args["correlation_id"])  # Raises if invalid

    @pytest.mark.asyncio
    async def test_get_subscribers_for_topic_mixed_valid_invalid_uuids(
        self,
        runtime: RuntimeHostProcess,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Test that invalid UUIDs are skipped, valid ones returned.

        The implementation uses graceful degradation - invalid UUIDs are
        logged and skipped rather than causing complete failure. This is
        preferred for distributed systems where one corrupt entry shouldn't
        break the entire routing mechanism.
        """
        # Setup: Register mock consul handler
        runtime.register_handler("consul", mock_consul_handler)

        # Setup: Mix of valid and invalid UUIDs
        valid_id_1 = uuid4()
        valid_id_2 = uuid4()
        mixed_list = [str(valid_id_1), "not-a-uuid", str(valid_id_2)]
        mock_response = self._create_kv_found_response(json.dumps(mixed_list))
        mock_consul_handler.execute.return_value = mock_response

        # Execute
        result = await runtime.get_subscribers_for_topic("dev.onex.evt.test.v1")

        # Verify: Only valid UUIDs returned (invalid skipped with warning)
        assert len(result) == 2
        assert valid_id_1 in result
        assert valid_id_2 in result


class TestGetSubscribersForTopicIntegration:
    """Integration-style tests for topic subscriber query.

    These tests verify the method works correctly with more realistic
    response structures.
    """

    @pytest.fixture
    def runtime(self) -> RuntimeHostProcess:
        """Create a RuntimeHostProcess instance for testing."""
        return RuntimeHostProcess()

    @pytest.mark.asyncio
    async def test_topic_key_format(
        self,
        runtime: RuntimeHostProcess,
    ) -> None:
        """Test that the correct Consul key format is used."""
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(return_value=None)
        runtime.register_handler("consul", mock_handler)

        # Test various topic formats
        topics = [
            "dev.onex.evt.intent-classified.v1",
            "prod.onex.cmd.register-node.v1",
            "test.custom.domain.event.v2",
        ]

        for topic in topics:
            await runtime.get_subscribers_for_topic(topic)

        # Verify all calls used correct key format
        assert mock_handler.execute.call_count == len(topics)
        for i, call in enumerate(mock_handler.execute.call_args_list):
            envelope = call[0][0]
            expected_key = f"onex/topics/{topics[i]}/subscribers"
            assert envelope["payload"]["key"] == expected_key


__all__: list[str] = [
    "TestGetSubscribersForTopic",
    "TestGetSubscribersForTopicIntegration",
]
