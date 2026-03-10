# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeHostProcess.get_subscribers_for_topic() method.

HandlerConsul was removed in OMN-3540.  get_subscribers_for_topic() is now a
stub that always returns [] regardless of any registered handlers.

All tests verify the stub contract: the method is callable, always returns an
empty list, and never raises.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config


class TestGetSubscribersForTopic:
    """Tests for RuntimeHostProcess.get_subscribers_for_topic() stub.

    HandlerConsul was removed in OMN-3540; the method now unconditionally
    returns [] to preserve the calling interface without silent errors.
    """

    @pytest.fixture
    def runtime(self) -> RuntimeHostProcess:
        """Create a RuntimeHostProcess instance for testing."""
        return RuntimeHostProcess(config=make_runtime_config())

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, runtime: RuntimeHostProcess) -> None:
        """Stub always returns an empty list."""
        result = await runtime.get_subscribers_for_topic(
            "onex.evt.intent-classified.v1"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_list_type(self, runtime: RuntimeHostProcess) -> None:
        """Return value is always a list, never None."""
        result = await runtime.get_subscribers_for_topic("onex.evt.test.v1")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_no_handler_registered(self, runtime: RuntimeHostProcess) -> None:
        """Returns empty list when no handler is registered."""
        result = await runtime.get_subscribers_for_topic("onex.evt.test.v1")
        assert result == []

    @pytest.mark.asyncio
    async def test_various_topic_formats(self, runtime: RuntimeHostProcess) -> None:
        """Returns empty list for every topic format — stub is topic-agnostic."""
        topics = [
            "onex.evt.intent-classified.v1",
            "onex.cmd.register-node.v1",
            "test.custom.domain.event.v2",
            "unknown.topic.v1",
            "",
        ]
        for topic in topics:
            result = await runtime.get_subscribers_for_topic(topic)
            assert result == [], f"Expected [] for topic {topic!r}, got {result!r}"

    @pytest.mark.asyncio
    async def test_does_not_raise(self, runtime: RuntimeHostProcess) -> None:
        """Stub never raises, regardless of topic string."""
        try:
            await runtime.get_subscribers_for_topic("onex.evt.test.v1")
        except Exception as exc:
            pytest.fail(f"get_subscribers_for_topic raised unexpectedly: {exc}")


class TestGetSubscribersForTopicIntegration:
    """Integration-style tests for the topic subscriber stub.

    Verifies the stub is accessible on a fully constructed RuntimeHostProcess
    and behaves consistently across repeated calls.
    """

    @pytest.fixture
    def runtime(self) -> RuntimeHostProcess:
        """Create a RuntimeHostProcess instance for testing."""
        return RuntimeHostProcess(config=make_runtime_config())

    @pytest.mark.asyncio
    async def test_repeated_calls_all_return_empty(
        self, runtime: RuntimeHostProcess
    ) -> None:
        """Repeated calls for the same topic all return []."""
        topic = "onex.evt.intent-classified.v1"
        for _ in range(3):
            result = await runtime.get_subscribers_for_topic(topic)
            assert result == []


__all__: list[str] = [
    "TestGetSubscribersForTopic",
    "TestGetSubscribersForTopicIntegration",
]
