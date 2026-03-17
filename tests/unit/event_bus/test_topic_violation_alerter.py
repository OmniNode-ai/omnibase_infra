# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the debounced topic violation Slack alerter."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.event_bus.topic_violation_alerter import TopicViolationAlerter


@pytest.fixture
def alerter() -> TopicViolationAlerter:
    """Create an alerter with a short debounce window and fake credentials."""
    return TopicViolationAlerter(
        debounce_seconds=1,
        channel="#test-alerts",
        slack_token="xoxb-fake-token",
    )


class TestTopicViolationAlerter:
    """Test suite for TopicViolationAlerter."""

    @pytest.mark.asyncio
    async def test_first_violation_fires_alert(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """First violation for a topic should dispatch an alert."""
        with patch.object(alerter, "_send_alert", new_callable=AsyncMock) as mock_send:
            result = await alerter.maybe_alert("bad-topic", "does not match format")

        assert result is True
        # Give the fire-and-forget task a chance to start
        await asyncio.sleep(0.05)
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_topic_within_window_suppressed(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """Same topic within debounce window should be suppressed."""
        with patch.object(alerter, "_send_alert", new_callable=AsyncMock):
            first = await alerter.maybe_alert("bad-topic", "reason")
            second = await alerter.maybe_alert("bad-topic", "reason")

        assert first is True
        assert second is False

    @pytest.mark.asyncio
    async def test_same_topic_after_window_fires_again(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """Same topic after debounce window expires should fire again."""
        with patch.object(alerter, "_send_alert", new_callable=AsyncMock):
            # Manually set last_alert_time to the past
            await alerter.maybe_alert("bad-topic", "reason")
            # Simulate time passing beyond the debounce window
            entry = alerter._entries["bad-topic"]
            entry.last_alert_time = time.monotonic() - 2.0  # 2s ago, window is 1s

            result = await alerter.maybe_alert("bad-topic", "reason")

        assert result is True

    @pytest.mark.asyncio
    async def test_different_topics_independent(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """Different topics should have independent debounce state."""
        with patch.object(alerter, "_send_alert", new_callable=AsyncMock):
            first = await alerter.maybe_alert("topic-a", "reason")
            second = await alerter.maybe_alert("topic-b", "reason")

        assert first is True
        assert second is True

    @pytest.mark.asyncio
    async def test_suppressed_count_tracked(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """Suppression counter should increment for each suppressed alert."""
        with patch.object(alerter, "_send_alert", new_callable=AsyncMock):
            await alerter.maybe_alert("bad-topic", "reason")
            await alerter.maybe_alert("bad-topic", "reason")
            await alerter.maybe_alert("bad-topic", "reason")

        assert alerter.get_suppressed_count("bad-topic") == 2

    @pytest.mark.asyncio
    async def test_missing_token_logs_warning(self) -> None:
        """Missing SLACK_BOT_TOKEN should result in a logged warning, not a raise."""
        alerter = TopicViolationAlerter(
            debounce_seconds=1,
            channel="#test",
            slack_token="",
        )
        # Should not raise
        result = await alerter.maybe_alert("bad-topic", "reason")
        assert result is True
        # The fire-and-forget _send_alert will log a warning but not raise
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_slack_failure_does_not_raise(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """Slack send failure should be caught and logged, not raised."""
        with patch.object(
            alerter,
            "_send_alert",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            # maybe_alert dispatches fire-and-forget, so it won't propagate
            result = await alerter.maybe_alert("bad-topic", "reason")
            assert result is True
            await asyncio.sleep(0.05)

    def test_get_suppressed_count_unknown_topic(
        self, alerter: TopicViolationAlerter
    ) -> None:
        """Querying suppression count for an unknown topic should return 0."""
        assert alerter.get_suppressed_count("never-seen") == 0
