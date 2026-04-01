# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for WaitlistSignupNotifier.

Tests message handling and Slack notification dispatch.
All tests mock aiokafka and Slack — no real Kafka or Slack required.

Related Tickets:
    - OMN-7199: Slack notification on new waitlist signup
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.handlers.models.model_slack_alert import ModelSlackAlertResult
from omnibase_infra.services.waitlist_signup_notifier.config import (
    ConfigWaitlistSignupNotifier,
)
from omnibase_infra.services.waitlist_signup_notifier.consumer import (
    WaitlistSignupNotifier,
)


@pytest.fixture
def config() -> ConfigWaitlistSignupNotifier:
    return ConfigWaitlistSignupNotifier(
        kafka_bootstrap_servers="localhost:19092",
        slack_bot_token="xoxb-test-token",
        slack_channel_id="C0123456789",
    )


@pytest.fixture
def notifier(config: ConfigWaitlistSignupNotifier) -> WaitlistSignupNotifier:
    return WaitlistSignupNotifier(config)


@pytest.mark.unit
class TestHandleMessage:
    """Test _handle_message processes events and dispatches Slack alerts."""

    @pytest.mark.asyncio
    async def test_sends_slack_notification_on_valid_event(
        self, notifier: WaitlistSignupNotifier
    ) -> None:
        """Valid waitlist signup event triggers a Slack notification."""
        mock_result = ModelSlackAlertResult(
            success=True,
            duration_ms=50.0,
            correlation_id=uuid4(),
            retry_count=0,
        )

        with patch.object(
            notifier._slack_handler, "handle", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_result

            event = {
                "envelope_id": str(uuid4()),
                "topic": "onex.evt.omniweb.waitlist-signup.v1",
                "tenant_id": "",
                "tenant_slug": "",
                "occurred_at": "2026-03-24T15:37:00.000Z",
                "payload": {"email_domain": "example.com"},
            }

            await notifier._handle_message(event)

            mock_handle.assert_called_once()
            alert = mock_handle.call_args[0][0]
            assert "example.com" in alert.message
            assert "2026-03-24 15:37 UTC" in alert.message
            assert alert.title == "Waitlist Signup"

    @pytest.mark.asyncio
    async def test_handles_missing_payload_gracefully(
        self, notifier: WaitlistSignupNotifier
    ) -> None:
        """Event with missing payload fields still sends notification."""
        mock_result = ModelSlackAlertResult(
            success=True,
            duration_ms=50.0,
            correlation_id=uuid4(),
            retry_count=0,
        )

        with patch.object(
            notifier._slack_handler, "handle", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_result

            event = {"payload": {}}
            await notifier._handle_message(event)

            mock_handle.assert_called_once()
            alert = mock_handle.call_args[0][0]
            assert "unknown" in alert.message

    @pytest.mark.asyncio
    async def test_fail_open_on_slack_error(
        self, notifier: WaitlistSignupNotifier
    ) -> None:
        """Slack handler failure does not raise — fail-open semantics."""
        with patch.object(
            notifier._slack_handler,
            "handle",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Slack exploded"),
        ):
            event = {
                "occurred_at": "2026-03-24T15:37:00.000Z",
                "payload": {"email_domain": "test.com"},
            }
            # Should not raise
            await notifier._handle_message(event)

    @pytest.mark.asyncio
    async def test_fail_open_on_slack_delivery_failure(
        self, notifier: WaitlistSignupNotifier
    ) -> None:
        """Slack returns failure result — logged but does not raise."""
        mock_result = ModelSlackAlertResult(
            success=False,
            duration_ms=50.0,
            correlation_id=uuid4(),
            retry_count=3,
            error="Slack rate limited",
            error_code="SLACK_RATE_LIMITED",
        )

        with patch.object(
            notifier._slack_handler, "handle", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_result

            event = {
                "occurred_at": "2026-03-24T15:37:00.000Z",
                "payload": {"email_domain": "test.com"},
            }
            # Should not raise
            await notifier._handle_message(event)


@pytest.mark.unit
class TestConfig:
    """Test configuration defaults and overrides."""

    def test_default_topic(self) -> None:
        config = ConfigWaitlistSignupNotifier()
        assert config.kafka_topic == "onex.evt.omniweb.waitlist-signup.v1"

    def test_default_group_id(self) -> None:
        config = ConfigWaitlistSignupNotifier()
        assert config.kafka_group_id == "waitlist-signup-slack-notifier"
