# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the ONEX topic format enforcement gate in EventBusKafka.publish()."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka


@pytest.fixture
def make_bus():
    """Factory to create an EventBusKafka with a specific enforcement mode."""

    def _factory(mode: str = "warn") -> EventBusKafka:
        with patch.dict("os.environ", {"ONEX_TOPIC_ENFORCEMENT_MODE": mode}):
            bus = EventBusKafka()
        # Pretend the bus is started so publish() doesn't bail out early
        bus._started = True
        # Mock the producer so we don't need a real Kafka connection
        bus._producer = AsyncMock()
        bus._producer.send_and_wait = AsyncMock()
        return bus

    return _factory


class TestOnexTopicEnforcementGate:
    """Tests for _enforce_onex_topic_format in EventBusKafka."""

    @pytest.mark.asyncio
    async def test_valid_topic_passes_in_reject_mode(self, make_bus) -> None:
        """Valid ONEX topics should pass without raising, even in reject mode."""
        bus = make_bus("reject")
        correlation_id = uuid4()
        # Should not raise
        await bus._enforce_onex_topic_format(
            "onex.evt.omniclaude.session-started.v1", correlation_id
        )

    @pytest.mark.asyncio
    async def test_invalid_topic_raises_in_reject_mode(self, make_bus) -> None:
        """Invalid topics should raise ProtocolConfigurationError in reject mode."""
        bus = make_bus("reject")
        correlation_id = uuid4()
        with pytest.raises(ProtocolConfigurationError):
            await bus._enforce_onex_topic_format("agent-actions", correlation_id)

    @pytest.mark.asyncio
    async def test_invalid_topic_warns_in_warn_mode(self, make_bus) -> None:
        """Invalid topics should log a warning but not raise in warn mode."""
        bus = make_bus("warn")
        correlation_id = uuid4()
        # Mock the alerter to verify it's called
        bus._topic_violation_alerter = AsyncMock()
        bus._topic_violation_alerter.maybe_alert = AsyncMock(return_value=True)

        # Should NOT raise
        await bus._enforce_onex_topic_format("agent-actions", correlation_id)

        # Alerter should have been called
        bus._topic_violation_alerter.maybe_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_off_mode_skips_validation(self, make_bus) -> None:
        """Off mode should skip validation entirely."""
        bus = make_bus("off")
        correlation_id = uuid4()
        # Should not raise, even for invalid topics
        await bus._enforce_onex_topic_format("agent-actions", correlation_id)

    @pytest.mark.asyncio
    async def test_kafka_internal_topics_pass(self, make_bus) -> None:
        """Kafka internal topics (__ prefix) should be skipped."""
        bus = make_bus("reject")
        correlation_id = uuid4()
        await bus._enforce_onex_topic_format("__consumer_offsets", correlation_id)

    @pytest.mark.asyncio
    async def test_legacy_dlq_passes(self, make_bus) -> None:
        """Legacy DLQ topics should pass (valid_legacy_dlq)."""
        bus = make_bus("reject")
        correlation_id = uuid4()
        await bus._enforce_onex_topic_format("local.dlq.intents.v1", correlation_id)

    @pytest.mark.asyncio
    async def test_default_mode_is_warn(self) -> None:
        """Default enforcement mode should be 'warn'."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove the env var if it exists
            import os

            os.environ.pop("ONEX_TOPIC_ENFORCEMENT_MODE", None)
            bus = EventBusKafka()
        assert bus._topic_enforcement_mode == "warn"

    @pytest.mark.asyncio
    async def test_alerter_initialized_in_warn_mode(self, make_bus) -> None:
        """TopicViolationAlerter should be initialized when mode is not 'off'."""
        bus = make_bus("warn")
        assert bus._topic_violation_alerter is not None

    @pytest.mark.asyncio
    async def test_alerter_none_in_off_mode(self, make_bus) -> None:
        """TopicViolationAlerter should be None when mode is 'off'."""
        bus = make_bus("off")
        assert bus._topic_violation_alerter is None
