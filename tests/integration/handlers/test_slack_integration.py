# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerSlackWebhook.

Tests the handler with a mock HTTP server simulating Slack webhook responses.
Uses pytest-httpserver for realistic HTTP interaction testing.
"""

from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Response

from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
)


@pytest.fixture(scope="session")
def httpserver_ssl_context():
    """Disable SSL for local test server."""
    return


class TestSlackIntegration:
    """Integration tests using mock HTTP server."""

    @pytest.fixture
    def alert(self) -> ModelSlackAlert:
        """Create test alert."""
        return ModelSlackAlert(
            severity=EnumAlertSeverity.ERROR,
            message="Integration test alert",
            title="Test Alert",
            details={"test": "true", "environment": "test"},
        )

    @pytest.mark.asyncio
    async def test_successful_webhook_delivery(
        self, httpserver: HTTPServer, alert: ModelSlackAlert
    ) -> None:
        """Test successful webhook delivery against mock server."""
        # Configure mock server to return 200
        httpserver.expect_request("/webhook", method="POST").respond_with_response(
            Response("ok", status=200)
        )

        handler = HandlerSlackWebhook(
            webhook_url=httpserver.url_for("/webhook"),
            timeout=5.0,
        )

        result = await handler.handle(alert)

        assert result.success is True
        assert result.duration_ms > 0
        assert result.retry_count == 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_rate_limit_then_success(
        self, httpserver: HTTPServer, alert: ModelSlackAlert
    ) -> None:
        """Test retry on 429 followed by success."""
        # First request returns 429, second returns 200
        call_count = [0]

        def rate_limit_handler(request):
            call_count[0] += 1
            if call_count[0] == 1:
                return Response("rate_limited", status=429)
            return Response("ok", status=200)

        httpserver.expect_request("/webhook", method="POST").respond_with_handler(
            rate_limit_handler
        )

        handler = HandlerSlackWebhook(
            webhook_url=httpserver.url_for("/webhook"),
            max_retries=2,
            retry_backoff=(0.1, 0.2),
            timeout=5.0,
        )

        result = await handler.handle(alert)

        assert result.success is True
        assert result.retry_count == 1
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_server_error_exhausts_retries(
        self, httpserver: HTTPServer, alert: ModelSlackAlert
    ) -> None:
        """Test that server errors exhaust retries."""
        httpserver.expect_request("/webhook", method="POST").respond_with_response(
            Response("Internal Server Error", status=500)
        )

        handler = HandlerSlackWebhook(
            webhook_url=httpserver.url_for("/webhook"),
            max_retries=2,
            retry_backoff=(0.05, 0.1),
            timeout=5.0,
        )

        result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_HTTP_500"
        assert result.retry_count == 2

    @pytest.mark.asyncio
    async def test_block_kit_payload_format(
        self, httpserver: HTTPServer, alert: ModelSlackAlert
    ) -> None:
        """Test that the payload sent to Slack is valid Block Kit format."""
        received_payload = []

        def capture_payload(request):
            import json

            received_payload.append(json.loads(request.data))
            return Response("ok", status=200)

        httpserver.expect_request("/webhook", method="POST").respond_with_handler(
            capture_payload
        )

        handler = HandlerSlackWebhook(
            webhook_url=httpserver.url_for("/webhook"),
        )

        await handler.handle(alert)

        # Verify Block Kit structure
        assert len(received_payload) == 1
        payload = received_payload[0]
        assert "blocks" in payload
        blocks = payload["blocks"]

        # Should have: header, divider, message section, fields section, context
        assert len(blocks) >= 4

        # Verify header block
        header = blocks[0]
        assert header["type"] == "header"
        assert (
            "Error Alert" in header["text"]["text"]
            or "Test Alert" in header["text"]["text"]
        )

        # Verify divider
        assert blocks[1]["type"] == "divider"

        # Verify message section
        message_section = blocks[2]
        assert message_section["type"] == "section"
        assert "Integration test alert" in message_section["text"]["text"]

    @pytest.mark.asyncio
    async def test_multiple_alerts_concurrent(self, httpserver: HTTPServer) -> None:
        """Test sending multiple alerts concurrently."""
        import asyncio

        httpserver.expect_request("/webhook", method="POST").respond_with_response(
            Response("ok", status=200)
        )

        handler = HandlerSlackWebhook(
            webhook_url=httpserver.url_for("/webhook"),
        )

        alerts = [
            ModelSlackAlert(
                severity=EnumAlertSeverity.INFO,
                message=f"Concurrent alert {i}",
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*[handler.handle(alert) for alert in alerts])

        assert all(r.success for r in results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_connection_to_unreachable_server(
        self, alert: ModelSlackAlert
    ) -> None:
        """Test handling of unreachable server."""
        handler = HandlerSlackWebhook(
            webhook_url="http://127.0.0.1:1",  # Unreachable port
            max_retries=1,
            retry_backoff=(0.1,),
            timeout=1.0,
        )

        result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_CONNECTION_ERROR"
