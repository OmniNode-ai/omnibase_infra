# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeHostProcess idempotency guard integration (OMN-945).

This module tests the idempotency guard integration in RuntimeHostProcess,
verifying duplicate message detection and replay-safe behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.idempotency import (
    InMemoryIdempotencyStore,
    ModelIdempotencyGuardConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """Create a mock event bus for testing."""
    bus = MagicMock()
    bus.start = AsyncMock()
    bus.close = AsyncMock()
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    bus.publish = AsyncMock()
    bus.publish_envelope = AsyncMock()
    bus.health_check = AsyncMock(return_value={"healthy": True})
    return bus


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create a mock handler for testing."""
    handler = MagicMock()
    handler.execute = AsyncMock(return_value={"success": True, "result": "processed"})
    return handler


@pytest.fixture
def idempotency_config() -> dict[str, ModelIdempotencyGuardConfig]:
    """Create idempotency configuration for testing."""
    return {
        "idempotency": ModelIdempotencyGuardConfig(
            enabled=True,
            store_type="memory",
            domain_from_operation=True,
            skip_operations=["health.check", "metrics.get"],
        )
    }


class TestIdempotencyGuardDisabled:
    """Tests for idempotency guard when disabled (default)."""

    @pytest.mark.asyncio
    async def test_idempotency_not_configured_by_default(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Idempotency guard is not configured when no config provided."""
        process = RuntimeHostProcess(event_bus=mock_event_bus)

        assert process._idempotency_store is None
        assert process._idempotency_config is None

    @pytest.mark.asyncio
    async def test_idempotency_disabled_skips_initialization(
        self, mock_event_bus: MagicMock, mock_handler: MagicMock
    ) -> None:
        """Idempotency guard skips initialization when explicitly disabled."""
        config = {
            "idempotency": {
                "enabled": False,
                "store_type": "memory",
            }
        }

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(event_bus=mock_event_bus, config=config)
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        assert process._idempotency_config is not None
        assert not process._idempotency_config.enabled
        assert process._idempotency_store is None

        await process.stop()


class TestIdempotencyGuardEnabled:
    """Tests for idempotency guard when enabled."""

    @pytest.mark.asyncio
    async def test_idempotency_guard_initializes_memory_store(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Idempotency guard initializes InMemory store when configured."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        assert process._idempotency_config is not None
        assert process._idempotency_config.enabled
        assert process._idempotency_store is not None
        assert isinstance(process._idempotency_store, InMemoryIdempotencyStore)

        await process.stop()

    @pytest.mark.asyncio
    async def test_idempotency_guard_cleanup_on_stop(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Idempotency store is cleaned up during stop()."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        assert process._idempotency_store is not None

        await process.stop()

        assert process._idempotency_store is None


class TestDuplicateMessageDetection:
    """Tests for duplicate message detection behavior."""

    @pytest.mark.asyncio
    async def test_first_message_is_processed(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """First message with unique message_id is processed."""
        # Create mock handler
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(
            return_value={"success": True, "result": "processed"}
        )

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        message_id = uuid4()
        envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
            "message_id": message_id,
            "correlation_id": uuid4(),
        }

        await process._handle_envelope(envelope)

        # Handler should be called
        mock_handler.execute.assert_called_once()

        await process.stop()

    @pytest.mark.asyncio
    async def test_duplicate_message_is_rejected(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Duplicate message with same message_id is rejected."""
        # Create mock handler
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(
            return_value={"success": True, "result": "processed"}
        )

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        message_id = uuid4()
        envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
            "message_id": message_id,
            "correlation_id": uuid4(),
        }

        # First call - should process
        await process._handle_envelope(envelope)
        assert mock_handler.execute.call_count == 1

        # Second call with same message_id - should reject
        await process._handle_envelope(envelope)
        assert mock_handler.execute.call_count == 1  # Still 1, not called again

        await process.stop()

    @pytest.mark.asyncio
    async def test_duplicate_message_publishes_duplicate_response(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Duplicate message publishes success response with status=duplicate."""
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(
            return_value={"success": True, "result": "processed"}
        )

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        message_id = uuid4()
        correlation_id = uuid4()
        envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
            "message_id": message_id,
            "correlation_id": correlation_id,
        }

        # First call
        await process._handle_envelope(envelope)
        first_publish_count = mock_event_bus.publish_envelope.call_count

        # Second call (duplicate)
        await process._handle_envelope(envelope)

        # Should have published a duplicate response
        assert mock_event_bus.publish_envelope.call_count == first_publish_count + 1

        # Get the last published message
        last_call = mock_event_bus.publish_envelope.call_args_list[-1]
        # publish_envelope receives (envelope_dict, topic)
        published_envelope = (
            last_call.args[0] if last_call.args else last_call.kwargs.get("envelope")
        )

        assert published_envelope["success"] is True
        assert published_envelope["status"] == "duplicate"
        assert published_envelope["message"] == "Message already processed"

        await process.stop()


class TestDomainIsolation:
    """Tests for domain-based idempotency isolation."""

    @pytest.mark.asyncio
    async def test_same_message_id_different_domains_both_process(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Same message_id in different domains are processed independently."""
        mock_db_handler = MagicMock()
        mock_db_handler.execute = AsyncMock(return_value={"success": True})

        mock_http_handler = MagicMock()
        mock_http_handler.execute = AsyncMock(return_value={"success": True})

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_db_handler, "http": mock_http_handler}
            await process.start()

        message_id = uuid4()

        # First domain (db) - with required payload
        db_envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
            "message_id": message_id,
            "correlation_id": uuid4(),
        }
        await process._handle_envelope(db_envelope)
        assert mock_db_handler.execute.call_count == 1

        # Different domain (http) with same message_id - with required payload
        http_envelope = {
            "operation": "http.get",
            "payload": {"url": "http://example.com"},
            "message_id": message_id,
            "correlation_id": uuid4(),
        }
        await process._handle_envelope(http_envelope)
        assert mock_http_handler.execute.call_count == 1  # Both processed

        await process.stop()

    @pytest.mark.asyncio
    async def test_domain_extracted_from_operation_prefix(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Domain is correctly extracted from operation prefix."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        envelope = {"operation": "db.query", "payload": {}}
        domain = process._extract_idempotency_domain(envelope)

        assert domain == "db"

        await process.stop()


class TestSkipOperations:
    """Tests for skip_operations configuration."""

    @pytest.mark.asyncio
    async def test_skip_operations_bypass_idempotency_check(
        self, mock_event_bus: MagicMock
    ) -> None:
        """Operations in skip_operations list bypass idempotency check."""
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(return_value={"success": True})

        # Configure with db.health as a skip operation (uses db handler which is registered)
        config = {
            "idempotency": {
                "enabled": True,
                "store_type": "memory",
                "domain_from_operation": True,
                "skip_operations": ["db.health"],  # Skip health check operations
            }
        }

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(event_bus=mock_event_bus, config=config)
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        message_id = uuid4()

        # db.health is in skip_operations
        envelope = {
            "operation": "db.health",
            "payload": {},  # Health checks may not require payload
            "message_id": message_id,
            "correlation_id": uuid4(),
        }

        # Both calls should process (not deduplicated)
        await process._handle_envelope(envelope)
        await process._handle_envelope(envelope)

        assert mock_handler.execute.call_count == 2

        await process.stop()


class TestMessageIdExtraction:
    """Tests for message_id extraction from envelope."""

    @pytest.mark.asyncio
    async def test_extracts_message_id_from_headers(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Message ID is extracted from envelope headers."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        message_id = uuid4()
        correlation_id = uuid4()
        envelope = {
            "headers": {"message_id": message_id},
            "operation": "db.query",
        }

        extracted = process._extract_message_id(envelope, correlation_id)

        assert extracted == message_id

        await process.stop()

    @pytest.mark.asyncio
    async def test_extracts_message_id_from_top_level(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Message ID is extracted from top-level envelope field."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        message_id = uuid4()
        correlation_id = uuid4()
        envelope = {
            "message_id": message_id,
            "operation": "db.query",
        }

        extracted = process._extract_message_id(envelope, correlation_id)

        assert extracted == message_id

        await process.stop()

    @pytest.mark.asyncio
    async def test_falls_back_to_correlation_id(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Falls back to correlation_id when message_id not present."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        correlation_id = uuid4()
        envelope = {"operation": "db.query"}

        extracted = process._extract_message_id(envelope, correlation_id)

        assert extracted == correlation_id

        await process.stop()

    @pytest.mark.asyncio
    async def test_extracts_string_message_id(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """String message_id is parsed to UUID."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        message_id = uuid4()
        correlation_id = uuid4()
        envelope = {
            "message_id": str(message_id),
            "operation": "db.query",
        }

        extracted = process._extract_message_id(envelope, correlation_id)

        assert extracted == message_id

        await process.stop()


class TestFailOpenBehavior:
    """Tests for fail-open behavior on store errors."""

    @pytest.mark.asyncio
    async def test_store_error_allows_message_through(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Store errors result in fail-open (message processed)."""
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(return_value={"success": True})

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        # Make store raise an error
        process._idempotency_store.check_and_record = AsyncMock(
            side_effect=Exception("Store unavailable")
        )

        envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
            "message_id": uuid4(),
            "correlation_id": uuid4(),
        }

        # Should still process despite store error
        await process._handle_envelope(envelope)

        assert mock_handler.execute.call_count == 1

        await process.stop()


class TestReplaySafeBehavior:
    """Tests verifying replay-safe behavior under at-least-once delivery."""

    @pytest.mark.asyncio
    async def test_replay_of_processed_message_is_ignored(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Replayed messages (simulating Kafka redelivery) are properly ignored."""
        mock_handler = MagicMock()
        mock_handler.execute = AsyncMock(return_value={"success": True})

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        message_id = uuid4()
        correlation_id = uuid4()
        envelope = {
            "operation": "db.query",
            "payload": {"sql": "INSERT INTO orders VALUES (1)"},
            "message_id": message_id,
            "correlation_id": correlation_id,
        }

        # Simulate multiple deliveries (at-least-once)
        for _ in range(5):
            await process._handle_envelope(envelope)

        # Handler should only be called once
        assert mock_handler.execute.call_count == 1

        await process.stop()

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_messages_only_one_processes(
        self,
        mock_event_bus: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Concurrent duplicate messages result in only one processing."""
        import asyncio

        mock_handler = MagicMock()
        call_count = 0

        async def slow_execute(envelope: dict) -> dict:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"success": True}

        mock_handler.execute = slow_execute

        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"db": mock_handler}
            await process.start()

        message_id = uuid4()
        envelope = {
            "operation": "db.query",
            "payload": {"sql": "SELECT 1"},
            "message_id": message_id,
            "correlation_id": uuid4(),
        }

        # Submit multiple concurrent requests
        tasks = [process._handle_envelope(envelope) for _ in range(10)]
        await asyncio.gather(*tasks)

        # Only one should have been processed
        assert call_count == 1

        await process.stop()


class TestDuplicateResponse:
    """Tests for duplicate response creation."""

    @pytest.mark.asyncio
    async def test_duplicate_response_format(
        self,
        mock_event_bus: MagicMock,
        mock_handler: MagicMock,
        idempotency_config: dict[str, ModelIdempotencyGuardConfig],
    ) -> None:
        """Duplicate response has correct format."""
        with patch.object(
            RuntimeHostProcess,
            "_populate_handlers_from_registry",
            new_callable=AsyncMock,
        ):
            process = RuntimeHostProcess(
                event_bus=mock_event_bus, config=idempotency_config
            )
            # Set handlers to avoid fail-fast validation
            process._handlers = {"mock": mock_handler}
            await process.start()

        message_id = uuid4()
        correlation_id = uuid4()

        response = process._create_duplicate_response(message_id, correlation_id)

        # Response is now a dict for envelope publishing compatibility
        assert response["success"] is True
        assert response["status"] == "duplicate"
        assert response["message"] == "Message already processed"
        assert response["message_id"] == message_id
        assert response["correlation_id"] == correlation_id

        await process.stop()
