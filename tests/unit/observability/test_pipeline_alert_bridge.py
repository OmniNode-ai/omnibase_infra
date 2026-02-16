# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for AdapterPipelineAlertBridge - Scenario 6: Failure notification wiring.

These tests validate the automatic alerting pathways that bridge pipeline
failure detection to Slack notifications:

1. DLQ -> Slack: DLQ events trigger Slack alerts automatically
2. Wiring health degradation -> Slack: Health checks trigger alerts
3. Cold-start blocked -> Slack: Delayed bootstrap triggers alerts
4. Recovery -> Slack: Recovery sends resolution notifications
5. Rate limiting: Alert storms are prevented
6. Correlation IDs: All alerts include correlation_id for traceability

Related Tickets:
    - OMN-2291: Intelligence pipeline resilience testing
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.models import ModelDlqEvent
from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
    ModelSlackAlertResult,
)
from omnibase_infra.observability.adapter_pipeline_alert_bridge import (
    AdapterPipelineAlertBridge,
)
from omnibase_infra.observability.wiring_health.model_wiring_health_alert import (
    ModelWiringHealthAlert,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_slack_handler() -> HandlerSlackWebhook:
    """Create a mock Slack handler with successful delivery."""
    handler = HandlerSlackWebhook(webhook_url="https://test.slack.com/webhook")
    handler.handle = AsyncMock(  # type: ignore[method-assign]
        return_value=ModelSlackAlertResult(
            success=True,
            duration_ms=50.0,
            correlation_id=uuid4(),
            retry_count=0,
        )
    )
    return handler


@pytest.fixture
def bridge(mock_slack_handler: HandlerSlackWebhook) -> AdapterPipelineAlertBridge:
    """Create a AdapterPipelineAlertBridge with mock handler."""
    return AdapterPipelineAlertBridge(
        slack_handler=mock_slack_handler,
        environment="test",
        rate_limit_window_seconds=60.0,
        max_alerts_per_window=5,
        cold_start_timeout_seconds=300.0,
    )


@pytest.fixture
def sample_dlq_event() -> ModelDlqEvent:
    """Create a sample DLQ event for testing."""
    return ModelDlqEvent(
        original_topic="dev.intelligence.code-analysis.v1",
        dlq_topic="dev.dlq.intents.v1",
        correlation_id=uuid4(),
        error_type="ValidationError",
        error_message="Invalid payload format",
        retry_count=3,
        message_offset="42",
        message_partition=0,
        success=True,
        timestamp=datetime.now(UTC),
        environment="test",
        consumer_group="test-consumer-group",
    )


@pytest.fixture
def critical_dlq_event() -> ModelDlqEvent:
    """Create a critical DLQ event (DLQ publish itself failed)."""
    return ModelDlqEvent(
        original_topic="dev.intelligence.code-analysis.v1",
        dlq_topic="dev.dlq.intents.v1",
        correlation_id=uuid4(),
        error_type="ValueError",
        error_message="Handler failed",
        retry_count=5,
        success=False,  # DLQ publish failed - message may be lost
        dlq_error_type="ProducerUnavailable",
        dlq_error_message="Producer not initialized or closed",
        timestamp=datetime.now(UTC),
        environment="test",
        consumer_group="test-consumer-group",
    )


@pytest.fixture
def sample_wiring_alert() -> ModelWiringHealthAlert:
    """Create a sample wiring health alert."""
    return ModelWiringHealthAlert(
        environment="test",
        unhealthy_topics=("session-outcome", "code-analysis"),
        threshold=0.05,
        summary="2 topics exceed 5.0% mismatch threshold",
        details=(
            {
                "topic": "session-outcome",
                "emit_count": 100,
                "consume_count": 85,
                "mismatch_ratio": "15.00%",
            },
            {
                "topic": "code-analysis",
                "emit_count": 50,
                "consume_count": 40,
                "mismatch_ratio": "20.00%",
            },
        ),
    )


# =============================================================================
# Scenario 6.1: DLQ -> Slack
# =============================================================================


class TestDlqToSlack:
    """Tests for DLQ event -> Slack alert pathway."""

    @pytest.mark.asyncio
    async def test_dlq_event_triggers_slack_alert(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        sample_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify DLQ events trigger Slack alerts automatically."""
        await bridge.on_dlq_event(sample_dlq_event)

        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]
        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.severity == EnumAlertSeverity.WARNING
        assert "dev.intelligence.code-analysis.v1" in alert.message
        assert alert.correlation_id == sample_dlq_event.correlation_id

    @pytest.mark.asyncio
    async def test_critical_dlq_event_triggers_critical_alert(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        critical_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify critical DLQ events (publish failure) send CRITICAL severity."""
        await bridge.on_dlq_event(critical_dlq_event)

        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]
        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.severity == EnumAlertSeverity.CRITICAL
        assert alert.title is not None
        assert "DLQ Publish Failed" in alert.title
        assert "lost" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_dlq_alert_includes_correlation_id(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        sample_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify all DLQ alerts include correlation_id for traceability."""
        await bridge.on_dlq_event(sample_dlq_event)

        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.correlation_id == sample_dlq_event.correlation_id

    @pytest.mark.asyncio
    async def test_dlq_alert_includes_context_details(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        sample_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify DLQ alerts include essential context in details."""
        await bridge.on_dlq_event(sample_dlq_event)

        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert "Environment" in alert.details
        assert "Original Topic" in alert.details
        assert "Error Type" in alert.details
        assert "Consumer Group" in alert.details


# =============================================================================
# Scenario 6.2: Wiring Health -> Slack
# =============================================================================


class TestWiringHealthToSlack:
    """Tests for wiring health degradation -> Slack alert pathway."""

    @pytest.mark.asyncio
    async def test_unhealthy_wiring_triggers_alert(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        sample_wiring_alert: ModelWiringHealthAlert,
    ) -> None:
        """Verify wiring health degradation triggers Slack alerts."""
        correlation_id = uuid4()
        await bridge.on_wiring_health_check(
            is_healthy=False,
            alert=sample_wiring_alert,
            correlation_id=correlation_id,
        )

        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]
        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.severity == EnumAlertSeverity.WARNING
        assert alert.title is not None
        assert "Wiring Health Degraded" in alert.title
        assert alert.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_healthy_wiring_no_alert(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify healthy wiring does not trigger alerts."""
        await bridge.on_wiring_health_check(
            is_healthy=True,
            alert=None,
        )

        mock_slack_handler.handle.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_recovery_triggers_info_alert(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        sample_wiring_alert: ModelWiringHealthAlert,
    ) -> None:
        """Verify recovery from unhealthy to healthy sends INFO alert."""
        # First: mark as unhealthy
        await bridge.on_wiring_health_check(
            is_healthy=False,
            alert=sample_wiring_alert,
        )
        mock_slack_handler.handle.reset_mock()  # type: ignore[attr-defined]

        # Then: mark as recovered
        await bridge.on_wiring_health_check(
            is_healthy=True,
            alert=None,
        )

        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]
        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.severity == EnumAlertSeverity.INFO
        title_str = alert.title or ""
        assert "Recovered" in title_str or "recovered" in alert.message.lower()


# =============================================================================
# Scenario 6.3: Cold-Start Blocked -> Slack
# =============================================================================


class TestColdStartToSlack:
    """Tests for cold-start blocked -> Slack alert pathway."""

    @pytest.mark.asyncio
    async def test_cold_start_blocked_triggers_alert_after_timeout(
        self,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify cold-start blocked >5 minutes triggers alert."""
        bridge = AdapterPipelineAlertBridge(
            slack_handler=mock_slack_handler,
            environment="test",
            cold_start_timeout_seconds=300.0,
        )

        # Before timeout: no alert
        await bridge.on_cold_start_blocked("PostgreSQL", elapsed_seconds=60.0)
        mock_slack_handler.handle.assert_not_called()  # type: ignore[attr-defined]

        # After timeout: alert sent
        await bridge.on_cold_start_blocked("PostgreSQL", elapsed_seconds=301.0)
        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]
        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.severity == EnumAlertSeverity.WARNING
        assert alert.title is not None
        assert "Cold-Start Blocked" in alert.title
        assert "PostgreSQL" in alert.message

    @pytest.mark.asyncio
    async def test_cold_start_alert_sent_only_once(
        self,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify cold-start blocked alert is not repeated."""
        bridge = AdapterPipelineAlertBridge(
            slack_handler=mock_slack_handler,
            environment="test",
            cold_start_timeout_seconds=10.0,
        )

        await bridge.on_cold_start_blocked("PostgreSQL", elapsed_seconds=15.0)
        await bridge.on_cold_start_blocked("PostgreSQL", elapsed_seconds=30.0)
        await bridge.on_cold_start_blocked("PostgreSQL", elapsed_seconds=60.0)

        # Only one alert should be sent
        assert mock_slack_handler.handle.call_count == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_cold_start_resolved_sends_recovery(
        self,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify cold-start resolution sends INFO recovery alert."""
        bridge = AdapterPipelineAlertBridge(
            slack_handler=mock_slack_handler,
            environment="test",
            cold_start_timeout_seconds=10.0,
        )

        # Trigger cold-start blocked alert
        await bridge.on_cold_start_blocked("PostgreSQL", elapsed_seconds=15.0)
        mock_slack_handler.handle.reset_mock()  # type: ignore[attr-defined]

        # Resolve
        await bridge.on_cold_start_resolved("PostgreSQL")

        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]
        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.severity == EnumAlertSeverity.INFO
        assert alert.title is not None
        assert "Resolved" in alert.title
        assert "PostgreSQL" in alert.message

    @pytest.mark.asyncio
    async def test_cold_start_resolved_no_alert_if_not_blocked(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify no recovery alert if cold-start was never blocked."""
        await bridge.on_cold_start_resolved("PostgreSQL")
        mock_slack_handler.handle.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_cold_start_includes_correlation_id(
        self,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify cold-start alerts include correlation_id."""
        bridge = AdapterPipelineAlertBridge(
            slack_handler=mock_slack_handler,
            environment="test",
            cold_start_timeout_seconds=10.0,
        )

        correlation_id = uuid4()
        await bridge.on_cold_start_blocked(
            "PostgreSQL",
            elapsed_seconds=15.0,
            correlation_id=correlation_id,
        )

        alert: ModelSlackAlert = mock_slack_handler.handle.call_args[0][0]  # type: ignore[attr-defined]
        assert alert.correlation_id == correlation_id


# =============================================================================
# Scenario 6.4: Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting to prevent alert storms."""

    @pytest.mark.asyncio
    async def test_rate_limiting_suppresses_excess_alerts(
        self,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify rate limiting prevents alert storms."""
        bridge = AdapterPipelineAlertBridge(
            slack_handler=mock_slack_handler,
            environment="test",
            rate_limit_window_seconds=60.0,
            max_alerts_per_window=3,
        )

        # Send 5 DLQ events - only first 3 should trigger alerts
        for i in range(5):
            event = ModelDlqEvent(
                original_topic=f"test.topic.{i}",
                dlq_topic="test.dlq.v1",
                correlation_id=uuid4(),
                error_type="ValueError",
                error_message=f"Error {i}",
                success=True,
                environment="test",
                consumer_group="test-group",
            )
            await bridge.on_dlq_event(event)

        assert mock_slack_handler.handle.call_count == 3  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_rate_limiting_per_category(
        self,
        mock_slack_handler: HandlerSlackWebhook,
    ) -> None:
        """Verify rate limiting is per-category (DLQ and wiring_health independent)."""
        bridge = AdapterPipelineAlertBridge(
            slack_handler=mock_slack_handler,
            environment="test",
            rate_limit_window_seconds=60.0,
            max_alerts_per_window=2,
        )

        # Send 3 DLQ events (2 allowed)
        for i in range(3):
            event = ModelDlqEvent(
                original_topic=f"test.topic.{i}",
                dlq_topic="test.dlq.v1",
                correlation_id=uuid4(),
                error_type="ValueError",
                error_message=f"Error {i}",
                success=True,
                environment="test",
                consumer_group="test-group",
            )
            await bridge.on_dlq_event(event)

        dlq_count = mock_slack_handler.handle.call_count  # type: ignore[attr-defined]
        assert dlq_count == 2

        # Send wiring health alert (should be allowed - different category)
        wiring_alert = ModelWiringHealthAlert(
            environment="test",
            unhealthy_topics=("topic-a",),
            threshold=0.05,
            summary="1 topic exceeds threshold",
        )
        await bridge.on_wiring_health_check(
            is_healthy=False,
            alert=wiring_alert,
        )

        # Total should be 2 (DLQ) + 1 (wiring health) = 3
        assert mock_slack_handler.handle.call_count == 3  # type: ignore[attr-defined]


# =============================================================================
# Scenario 6.5: Delivery Failure Handling
# =============================================================================


class TestDeliveryFailureHandling:
    """Tests for graceful handling of Slack delivery failures."""

    @pytest.mark.asyncio
    async def test_slack_delivery_failure_does_not_raise(
        self,
        sample_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify Slack delivery failure does not crash the bridge."""
        handler = HandlerSlackWebhook(webhook_url="https://test.slack.com/webhook")
        handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=ModelSlackAlertResult(
                success=False,
                duration_ms=100.0,
                correlation_id=uuid4(),
                error="SLACK_CONNECTION_ERROR",
                error_code="SLACK_CONNECTION_ERROR",
                retry_count=3,
            )
        )

        bridge = AdapterPipelineAlertBridge(
            slack_handler=handler,
            environment="test",
        )

        # Should not raise
        await bridge.on_dlq_event(sample_dlq_event)

    @pytest.mark.asyncio
    async def test_slack_handler_exception_suppressed_to_protect_caller(
        self,
        sample_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify handler exceptions are caught and suppressed.

        The alert bridge wraps all handler.handle() calls in try/except to
        ensure that a Slack outage (or any unexpected handler exception) never
        crashes the caller (DLQ processing, health checks, etc.). The exception
        is logged but not re-raised.
        """
        handler = HandlerSlackWebhook(webhook_url="https://test.slack.com/webhook")
        handler.handle = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("Connection refused")
        )

        bridge = AdapterPipelineAlertBridge(
            slack_handler=handler,
            environment="test",
        )

        # Should NOT raise -- the bridge catches and logs the exception
        await bridge.on_dlq_event(sample_dlq_event)


# =============================================================================
# Scenario 6.6: Integration Pattern - DLQ Callback Registration
# =============================================================================


class TestDlqCallbackRegistration:
    """Tests verifying the bridge works as a DLQ callback."""

    @pytest.mark.asyncio
    async def test_bridge_callback_signature_compatible(
        self,
        bridge: AdapterPipelineAlertBridge,
    ) -> None:
        """Verify on_dlq_event has the correct signature for register_dlq_callback."""
        import inspect

        # Verify it's an async callable accepting ModelDlqEvent
        assert asyncio.iscoroutinefunction(bridge.on_dlq_event)

        sig = inspect.signature(bridge.on_dlq_event)
        params = list(sig.parameters.keys())
        assert "event" in params

    @pytest.mark.asyncio
    async def test_bridge_works_as_dlq_callback(
        self,
        bridge: AdapterPipelineAlertBridge,
        mock_slack_handler: HandlerSlackWebhook,
        sample_dlq_event: ModelDlqEvent,
    ) -> None:
        """Verify the bridge callback produces expected alert when invoked."""
        # Simulate what register_dlq_callback would do: call the callback
        callback = bridge.on_dlq_event
        await callback(sample_dlq_event)

        mock_slack_handler.handle.assert_called_once()  # type: ignore[attr-defined]


__all__: list[str] = []
