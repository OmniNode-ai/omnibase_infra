# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerSlackWebhook.

Tests the Slack webhook handler's core functionality including:
- Block Kit message formatting
- Retry logic with exponential backoff
- Rate limit handling (HTTP 429)
- Error handling and sanitization
- Configuration validation
- Web API mode (chat.postMessage) with threading
- Webhook fallback mode

All tests use mocked HTTP responses to avoid external dependencies.
"""

from __future__ import annotations

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
    _SLACK_WEB_API_URL,
    HandlerSlackWebhook,
)
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
    ModelSlackAlertResult,
)


@pytest.fixture(autouse=True)
def _clear_slack_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient Slack credentials from the environment for test isolation.

    Prevents SLACK_BOT_TOKEN, SLACK_WEBHOOK_URL, and SLACK_CHANNEL_ID present in
    the developer's shell from leaking into handler constructor env-var lookups and
    affecting test assertions about mode selection.
    """
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)


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
        assert alert.thread_ts is None
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
            thread_ts="1234567890.123456",
            correlation_id=correlation_id,
        )

        assert alert.severity == EnumAlertSeverity.CRITICAL
        assert alert.message == "Critical error occurred"
        assert alert.title == "System Alert"
        assert alert.details == {"service": "consul", "error_code": "CONN_FAILED"}
        assert alert.channel == "#alerts"
        assert alert.thread_ts == "1234567890.123456"
        assert alert.correlation_id == correlation_id

    def test_alert_is_frozen(self) -> None:
        """Test that ModelSlackAlert is immutable."""
        alert = ModelSlackAlert(message="Test")
        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen
            alert.message = "Changed"  # type: ignore[misc]

    def test_alert_thread_ts_default_none(self) -> None:
        """Test that thread_ts defaults to None."""
        alert = ModelSlackAlert(message="Test")
        assert alert.thread_ts is None


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
        assert result.thread_ts is None

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
        assert result.thread_ts is None

    def test_result_with_thread_ts(self) -> None:
        """Test result with thread_ts from Web API."""
        correlation_id = uuid4()
        result = ModelSlackAlertResult(
            success=True,
            duration_ms=123.45,
            correlation_id=correlation_id,
            thread_ts="1234567890.123456",
        )

        assert result.thread_ts == "1234567890.123456"


class TestHandlerSlackWebhook:
    """Tests for HandlerSlackWebhook webhook mode."""

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
        assert handler._bot_token == ""
        assert handler._default_channel == ""
        assert handler.uses_web_api is False

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

    def test_handler_initialization_with_bot_token(self) -> None:
        """Test handler initializes with bot token for Web API mode."""
        handler = HandlerSlackWebhook(
            bot_token="xoxb-test-token",
            default_channel="C01234567",
        )
        assert handler._bot_token == "xoxb-test-token"
        assert handler._default_channel == "C01234567"
        assert handler.uses_web_api is True

    def test_handler_initialization_bot_token_from_env(self) -> None:
        """Test handler reads bot token from environment."""
        with patch.dict(
            "os.environ",
            {"SLACK_BOT_TOKEN": "xoxb-env-token", "SLACK_CHANNEL_ID": "C99999"},
        ):
            handler = HandlerSlackWebhook()
            assert handler._bot_token == "xoxb-env-token"
            assert handler._default_channel == "C99999"
            assert handler.uses_web_api is True

    @pytest.mark.asyncio
    async def test_handle_success(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test successful alert delivery."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="ok")
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
        assert result.thread_ts is None  # Webhook mode doesn't return thread_ts

    @pytest.mark.asyncio
    async def test_handle_webhook_error_body(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test webhook returns 200 with error text body (e.g. invalid_payload)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="invalid_payload")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_WEBHOOK_ERROR"
        assert "invalid_payload" in (result.error or "")

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
        mock_response_429.headers = {}
        mock_response_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        mock_response_429.__aexit__ = AsyncMock(return_value=None)

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.text = AsyncMock(return_value="ok")
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
        mock_response.headers = {}
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


class TestHandlerWebApiMode:
    """Tests for HandlerSlackWebhook Web API mode."""

    @pytest.fixture
    def handler(self) -> HandlerSlackWebhook:
        """Create handler in Web API mode."""
        return HandlerSlackWebhook(
            bot_token="xoxb-test-token",
            default_channel="C01234567",
            max_retries=2,
            retry_backoff=(0.01, 0.02),
            timeout=1.0,
        )

    @pytest.fixture
    def alert(self) -> ModelSlackAlert:
        """Create test alert."""
        return ModelSlackAlert(
            severity=EnumAlertSeverity.ERROR,
            message="Test error message",
            title="Test Alert",
        )

    @pytest.mark.asyncio
    async def test_web_api_success_returns_thread_ts(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test Web API success returns thread_ts."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "ts": "1234567890.123456", "channel": "C01234567"}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.thread_ts == "1234567890.123456"
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_web_api_passes_thread_ts_to_api(
        self, handler: HandlerSlackWebhook
    ) -> None:
        """Test that thread_ts from alert is passed to the API call."""
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.INFO,
            message="Thread reply",
            thread_ts="1111111111.111111",
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "ts": "2222222222.222222", "channel": "C01234567"}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        # Verify the call included thread_ts
        call_kwargs = mock_session.post.call_args
        posted_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert posted_payload["thread_ts"] == "1111111111.111111"
        assert result.success is True
        assert result.thread_ts == "2222222222.222222"

    @pytest.mark.asyncio
    async def test_web_api_no_thread_ts_when_not_provided(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test that thread_ts is not sent when not provided in alert."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": True, "ts": "1234567890.123456"}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await handler.handle(alert)

        call_kwargs = mock_session.post.call_args
        posted_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "thread_ts" not in posted_payload

    @pytest.mark.asyncio
    async def test_web_api_sends_auth_header(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test that Web API sends Bearer auth header."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await handler.handle(alert)

        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer xoxb-test-token"

    @pytest.mark.asyncio
    async def test_web_api_uses_correct_url(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test that Web API uses the chat.postMessage URL."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await handler.handle(alert)

        call_args = mock_session.post.call_args
        assert call_args[0][0] == _SLACK_WEB_API_URL

    @pytest.mark.asyncio
    async def test_web_api_channel_from_alert(self) -> None:
        """Test that channel from alert overrides default."""
        handler = HandlerSlackWebhook(
            bot_token="xoxb-test",
            default_channel="C_DEFAULT",
            max_retries=0,
            timeout=1.0,
        )
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.INFO,
            message="Test",
            channel="C_OVERRIDE",
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await handler.handle(alert)

        call_kwargs = mock_session.post.call_args
        posted_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert posted_payload["channel"] == "C_OVERRIDE"

    @pytest.mark.asyncio
    async def test_web_api_no_channel_configured(self) -> None:
        """Test Web API mode fails when no channel is configured."""
        handler = HandlerSlackWebhook(
            bot_token="xoxb-test",
            default_channel="",
            max_retries=0,
            timeout=1.0,
        )
        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.INFO,
            message="Test",
        )

        result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_NOT_CONFIGURED"
        assert "SLACK_CHANNEL_ID" in (result.error or "")

    @pytest.mark.asyncio
    async def test_web_api_ok_false_non_retryable(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test non-retryable Slack API error (invalid_auth)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"ok": False, "error": "invalid_auth"}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_API_ERROR"
        assert "invalid_auth" in (result.error or "")
        # Should NOT retry for non-retryable errors
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_web_api_ok_false_retryable(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test retryable Slack API error (ratelimited via ok=false)."""
        mock_response_ratelimited = AsyncMock()
        mock_response_ratelimited.status = 200
        mock_response_ratelimited.json = AsyncMock(
            return_value={"ok": False, "error": "ratelimited"}
        )
        mock_response_ratelimited.__aenter__ = AsyncMock(
            return_value=mock_response_ratelimited
        )
        mock_response_ratelimited.__aexit__ = AsyncMock(return_value=None)

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        mock_response_ok.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(
            side_effect=[mock_response_ratelimited, mock_response_ok]
        )
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_web_api_http_429_retries(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test HTTP 429 rate limiting in Web API mode."""
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {}
        mock_response_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        mock_response_429.__aexit__ = AsyncMock(return_value=None)

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        mock_response_ok.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(side_effect=[mock_response_429, mock_response_ok])
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_web_api_retry_after_header_respected(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test that Retry-After header from 429 is respected."""
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {"Retry-After": "0.01"}
        mock_response_429.__aenter__ = AsyncMock(return_value=mock_response_429)
        mock_response_429.__aexit__ = AsyncMock(return_value=None)

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        mock_response_ok.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(side_effect=[mock_response_429, mock_response_ok])
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_web_api_non_json_response_handled(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test that non-JSON HTTP 200 response is handled gracefully."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(
                MagicMock(), MagicMock(), message="Attempt to decode JSON"
            )
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_API_ERROR"
        assert "non-JSON" in (result.error or "")

    @pytest.mark.asyncio
    async def test_web_api_includes_fallback_text(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test that Web API payload includes fallback text."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await handler.handle(alert)

        call_kwargs = mock_session.post.call_args
        posted_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "text" in posted_payload
        assert "blocks" in posted_payload

    @pytest.mark.asyncio
    async def test_web_api_timeout(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test Web API timeout handling."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(side_effect=TimeoutError())
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_TIMEOUT"

    @pytest.mark.asyncio
    async def test_web_api_4xx_fail_fast(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test 4xx client errors fail fast without retry in Web API mode."""
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value="missing_scope")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_HTTP_403"
        assert result.retry_count == 0
        # Should only have been called once (no retries for 4xx)
        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_web_api_5xx_retries(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test 5xx server errors are retried in Web API mode."""
        mock_response_500 = AsyncMock()
        mock_response_500.status = 500
        mock_response_500.text = AsyncMock(return_value="internal error")
        mock_response_500.__aenter__ = AsyncMock(return_value=mock_response_500)
        mock_response_500.__aexit__ = AsyncMock(return_value=None)

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        mock_response_ok.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(side_effect=[mock_response_500, mock_response_ok])
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_web_api_connection_error(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test connection error handling in Web API mode."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                MagicMock(), OSError("Connection refused")
            )
        )
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_web_api_unexpected_status(
        self, handler: HandlerSlackWebhook, alert: ModelSlackAlert
    ) -> None:
        """Test unexpected HTTP status (e.g., 301) in Web API mode."""
        mock_response = AsyncMock()
        mock_response.status = 301
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is False
        assert result.error_code == "SLACK_HTTP_301"
        assert result.retry_count == 0


class TestModeResolution:
    """Tests for handler mode resolution (Web API vs webhook)."""

    @pytest.mark.asyncio
    async def test_bot_token_prefers_web_api(self) -> None:
        """When bot_token is set, Web API is used even if webhook_url is also set."""
        handler = HandlerSlackWebhook(
            webhook_url="https://hooks.slack.com/test",
            bot_token="xoxb-test",
            default_channel="C01234567",
            max_retries=0,
            timeout=1.0,
        )

        alert = ModelSlackAlert(message="Test")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True, "ts": "123.456"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        # Should use Web API (returns thread_ts)
        assert result.thread_ts == "123.456"
        # Should have called chat.postMessage URL, not webhook
        call_args = mock_session.post.call_args
        assert call_args[0][0] == _SLACK_WEB_API_URL

    @pytest.mark.asyncio
    async def test_no_bot_token_uses_webhook(self) -> None:
        """Without bot_token, falls back to webhook mode."""
        handler = HandlerSlackWebhook(
            webhook_url="https://hooks.slack.com/test",
            max_retries=0,
            timeout=1.0,
        )

        alert = ModelSlackAlert(message="Test")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="ok")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle(alert)

        assert result.success is True
        assert result.thread_ts is None
        # Should have called webhook URL
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://hooks.slack.com/test"


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
