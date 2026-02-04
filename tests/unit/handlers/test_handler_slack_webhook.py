# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerSlackWebhook.

Tests the Slack webhook handler's core functionality including:
- Block Kit message formatting
- Retry logic with exponential backoff
- Rate limit handling (HTTP 429)
- Error handling and sanitization
- Configuration validation

All tests use mocked HTTP responses to avoid external dependencies.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import aiohttp
import pytest

from omnibase_infra.handlers.handler_slack_webhook import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_RETRY_BACKOFF_SECONDS,
    _SEVERITY_EMOJI,
    _SEVERITY_TITLES,
    HandlerSlackWebhook,
)
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
    ModelSlackAlertResult,
)


class TestModelSlackAlert:
    """Tests for ModelSlackAlert input model."""

    def test_minimal_alert(self) -> None:
        """Test creating an alert with only required fields."""
        alert = ModelSlackAlert(message="Test message")

        assert alert.message == "Test message"
        assert alert.severity == EnumAlertSeverity.INFO
        assert alert.title is None
        assert alert.details == {}
        assert alert.channel is None
        assert alert.correlation_id is not None

    def test_full_alert(self) -> None:
        """Test creating an alert with all fields."""
        correlation_id = uuid4()
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.CRITICAL,
            message="Critical error occurred",
            title="System Alert",
            details={"service": "consul", "error_code": "CONN_FAILED"},
            channel="#alerts",
            correlation_id=correlation_id,
        )

        assert alert.severity == EnumAlertSeverity.CRITICAL
        assert alert.message == "Critical error occurred"
        assert alert.title == "System Alert"
        assert alert.details == {"service": "consul", "error_code": "CONN_FAILED"}
        assert alert.channel == "#alerts"
        assert alert.correlation_id == correlation_id

    def test_alert_is_frozen(self) -> None:
        """Test that ModelSlackAlert is immutable."""
        alert = ModelSlackAlert(message="Test")
        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen
            alert.message = "Changed"  # type: ignore[misc]


class TestModelSlackAlertResult:
    """Tests for ModelSlackAlertResult output model."""

    def test_success_result(self) -> None:
        """Test creating a successful result."""
        correlation_id = uuid4()
        result = ModelSlackAlertResult(
            success=True,
            duration_ms=123.45,
            correlation_id=correlation_id,
            retry_count=0,
        )

        assert result.success is True
        assert result.duration_ms == 123.45
        assert result.correlation_id == correlation_id
        assert result.error is None
        assert result.error_code is None
        assert result.retry_count == 0

    def test_failure_result(self) -> None:
        """Test creating a failure result."""
        correlation_id = uuid4()
        result = ModelSlackAlertResult(
            success=False,
            duration_ms=500.0,
            correlation_id=correlation_id,
            error="Connection failed",
            error_code="SLACK_CONNECTION_ERROR",
            retry_count=3,
        )

        assert result.success is False
        assert result.error == "Connection failed"
        assert result.error_code == "SLACK_CONNECTION_ERROR"
        assert result.retry_count == 3


class TestHandlerSlackWebhook:
    """Tests for HandlerSlackWebhook."""

    @pytest.fixture
    def webhook_url(self) -> str:
        """Return test webhook URL."""
        return "https://hooks.slack.com/services/T00/B00/XXX"

    @pytest.fixture
    def handler(self, webhook_url: str) -> HandlerSlackWebhook:
        """Create handler with test webhook URL."""
        return HandlerSlackWebhook(
            webhook_url=webhook_url,
            max_retries=2,
            retry_backoff=(0.01, 0.02),  # Fast retries for testing
            timeout=1.0,
        )

    @pytest.fixture
    def alert(self) -> ModelSlackAlert:
        """Create test alert."""
        return ModelSlackAlert(
            severity=EnumAlertSeverity.ERROR,
            message="Test error message",
            title="Test Alert",
            details={"key": "value"},
        )

    def test_handler_initialization_with_url(self, webhook_url: str) -> None:
        """Test handler initializes with provided URL."""
        handler = HandlerSlackWebhook(webhook_url=webhook_url)
        assert handler._webhook_url == webhook_url

    def test_handler_initialization_from_env(self) -> None:
        """Test handler reads URL from environment."""
        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://test.slack.com"}):
            handler = HandlerSlackWebhook()
            assert handler._webhook_url == "https://test.slack.com"

    def test_handler_initialization_no_url(self) -> None:
        """Test handler with no URL configured."""
        with patch.dict("os.environ", {}, clear=True):
            handler = HandlerSlackWebhook(webhook_url=None)
            assert handler._webhook_url == ""

    @pytest.mark.asyncio
    async def test_handle_success(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test successful alert delivery."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.duration_ms > 0
        assert result.correlation_id == alert.correlation_id
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_handle_not_configured(self, alert: ModelSlackAlert) -> None:
        """Test handling when webhook URL is not configured."""
        handler = HandlerSlackWebhook(webhook_url="")
        result = await handler.handle(alert)

        assert result.success is False
        assert result.error == "SLACK_WEBHOOK_URL not configured"
        assert result.error_code == "SLACK_NOT_CONFIGURED"

    @pytest.mark.asyncio
    async def test_handle_rate_limited_with_retry(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test retry on rate limit (429)."""
        # First call returns 429, second returns 200
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        mock_response_429.__aexit__ = AsyncMock(return_value=None)

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.__aenter__ = AsyncMock(return_value=mock_response_200)
        mock_response_200.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(
            side_effect=[mock_response_429, mock_response_200]
        )
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_handle_rate_limited_exhausted(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test failure when retries exhausted on rate limit."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_RATE_LIMITED"
        assert result.retry_count == 2  # max_retries=2

    @pytest.mark.asyncio
    async def test_handle_timeout(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test timeout handling."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(side_effect=TimeoutError())
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_TIMEOUT"
        assert result.error == "Request timeout"

    @pytest.mark.asyncio
    async def test_handle_connection_error(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test connection error handling."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("Connection refused")
            )
        )
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_handle_http_error(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test HTTP error handling."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_HTTP_500"


class TestBlockKitFormatting:
    """Tests for Block Kit message formatting."""

    @pytest.fixture
    def handler(self) -> HandlerSlackWebhook:
        """Create handler for formatting tests."""
        return HandlerSlackWebhook(webhook_url="https://test.slack.com")

    def test_format_critical_alert(self, handler: HandlerSlackWebhook) -> None:
        """Test formatting critical severity alert."""
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.CRITICAL,
            message="System is down",
        )
        payload = handler._format_block_kit_message(alert)

        assert "blocks" in payload
        blocks = payload["blocks"]
        header = blocks[0]
        assert header["type"] == "header"
        assert ":red_circle:" in header["text"]["text"]
        assert "Critical Alert" in header["text"]["text"]

    def test_format_with_custom_title(self, handler: HandlerSlackWebhook) -> None:
        """Test formatting with custom title."""
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.WARNING,
            message="High memory usage",
            title="Resource Warning",
        )
        payload = handler._format_block_kit_message(alert)

        header = payload["blocks"][0]
        assert "Resource Warning" in header["text"]["text"]

    def test_format_with_details(self, handler: HandlerSlackWebhook) -> None:
        """Test formatting with detail fields."""
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.ERROR,
            message="Connection failed",
            details={"service": "postgres", "retry_count": "3"},
        )
        payload = handler._format_block_kit_message(alert)

        blocks = payload["blocks"]
        # Find fields section
        fields_block = None
        for block in blocks:
            if block.get("type") == "section" and "fields" in block:
                fields_block = block
                break

        assert fields_block is not None
        fields = fields_block["fields"]
        assert len(fields) == 2

    def test_format_message_at_max_length(self, handler: HandlerSlackWebhook) -> None:
        """Test that messages at max length are handled correctly."""
        # Model enforces max_length=3000, so use max allowed length
        max_message = "x" * 3000
        alert = ModelSlackAlert(message=max_message)
        payload = handler._format_block_kit_message(alert)

        message_block = payload["blocks"][2]  # After header and divider
        # Message should be exactly 3000 chars (at the Slack limit)
        assert len(message_block["text"]["text"]) == 3000

    def test_format_correlation_id_context(self, handler: HandlerSlackWebhook) -> None:
        """Test that correlation ID is included in context."""
        correlation_id = uuid4()
        alert = ModelSlackAlert(message="Test", correlation_id=correlation_id)
        payload = handler._format_block_kit_message(alert)

        # Find context block (last block)
        context_block = payload["blocks"][-1]
        assert context_block["type"] == "context"
        assert str(correlation_id)[:16] in context_block["elements"][0]["text"]


class TestSeverityMappings:
    """Tests for severity emoji and title mappings."""

    def test_all_severities_have_emoji(self) -> None:
        """Test that all severity levels have emoji mappings."""
        for severity in EnumAlertSeverity:
            assert severity in _SEVERITY_EMOJI

    def test_all_severities_have_titles(self) -> None:
        """Test that all severity levels have title mappings."""
        for severity in EnumAlertSeverity:
            assert severity in _SEVERITY_TITLES

    def test_emoji_format(self) -> None:
        """Test that emojis use Slack colon format."""
        for emoji in _SEVERITY_EMOJI.values():
            assert emoji.startswith(":")
            assert emoji.endswith(":")
