"""
Mock Webhook Tests for Hook Node.

Tests webhook delivery validation including:
- Slack webhook format validation and delivery
- Discord webhook format validation and delivery
- Generic webhook delivery with custom payloads
- Authentication header generation for all three methods
- Retry behavior on failures with backoff strategies
- Circuit breaker activation and protection
"""

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_auth_type import EnumAuthType
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_spi.protocols.core import ProtocolHttpResponse

from omnibase_infra.models.notification.model_notification_auth import (
    ModelNotificationAuth,
)

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import (
    ModelNotificationRequest,
)
from omnibase_infra.models.notification.model_notification_retry_policy import (
    ModelNotificationRetryPolicy,
)
from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import (
    ModelHookNodeInput,
)

# Hook Node implementation
from omnibase_infra.nodes.hook_node.v1_0_0.node import (
    CircuitBreakerState,
    NodeHookEffect,
)

# Test-specific strongly typed models
from tests.models.test_webhook_models import (
    MockWebhookFailureConfigModel,
    MockWebhookRequestModel,
    MockWebhookResponseConfigModel,
)


class MockWebhookServer:
    """Mock webhook server for testing various webhook endpoints."""

    def __init__(self):
        self.received_requests: list[MockWebhookRequestModel] = []
        self.response_config: MockWebhookResponseConfigModel = MockWebhookResponseConfigModel()
        self.failure_config: MockWebhookFailureConfigModel | None = None
        self.failure_count: int = 0
        self.request_count: int = 0

    def configure_responses(self, success_config: MockWebhookResponseConfigModel | None = None, failure_config: MockWebhookFailureConfigModel | None = None):
        """Configure mock server responses."""
        if success_config:
            self.response_config = success_config
        self.failure_config = failure_config

    def reset(self):
        """Reset mock server state."""
        self.received_requests.clear()
        self.failure_count = 0
        self.request_count = 0

    async def handle_request(self, method: str, url: str, headers: Dict[str, str], body: str) -> ProtocolHttpResponse:
        """Handle incoming webhook request."""
        self.request_count += 1

        # Record the request
        request_data = MockWebhookRequestModel(
            url=url,
            method=method,
            headers=headers,
            body=body,
            timestamp=time.time(),
        )
        self.received_requests.append(request_data)

        # Simulate processing delay
        if self.response_config.delay_ms > 0:
            await asyncio.sleep(self.response_config.delay_ms / 1000)

        # Check if we should fail this request
        if self.failure_config and self.failure_count < self.failure_config.fail_count:
            self.failure_count += 1
            return ProtocolHttpResponse(
                status_code=self.failure_config.status_code,
                headers=self.failure_config.headers,
                body=self.failure_config.body,
                execution_time_ms=self.response_config.delay_ms,
                is_success=False,
            )

        # Return success response
        return ProtocolHttpResponse(
            status_code=self.response_config.status_code,
            headers=self.response_config.headers,
            body=self.response_config.body,
            execution_time_ms=self.response_config.delay_ms,
            is_success=True,
        )

    def get_last_request(self) -> MockWebhookRequestModel | None:
        """Get the most recent request received."""
        return self.received_requests[-1] if self.received_requests else None

    def get_requests_for_url(self, url_pattern: str) -> list[MockWebhookRequestModel]:
        """Get all requests matching a URL pattern."""
        return [req for req in self.received_requests if url_pattern in req.url]


class TestSlackWebhookDelivery:
    """Test suite for Slack webhook delivery validation."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def webhook_server(self):
        """Create a mock webhook server."""
        return MockWebhookServer()

    @pytest.fixture
    def hook_node(self, container, webhook_server):
        """Create Hook Node with mocked HTTP client."""
        # Mock HTTP client
        mock_http_client = AsyncMock()
        mock_http_client.post = webhook_server.handle_request

        # Mock event bus
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_slack_webhook_basic_message(self, hook_node, webhook_server):
        """Test basic Slack webhook message delivery."""
        slack_payload = {
            "text": "🚨 System Alert: Database connection lost",
            "channel": "#infrastructure-alerts",
            "username": "ONEX Monitor",
        }

        request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/T1234567890/B1234567890/XXXXXXXXXXXXXXXXXXXXXXXX",
            method=EnumNotificationMethod.POST,
            payload=slack_payload,
        )

        # Configure Slack-like response
        webhook_server.configure_responses(
            success_config=MockWebhookResponseConfigModel(
                status_code=200,
                body="ok",
                headers={"Content-Type": "text/plain"},
                delay_ms=150,
            ),
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True
        assert result.notification_result.final_status_code == 200

        # Verify Slack payload format
        last_request = webhook_server.get_last_request()
        assert last_request is not None

        received_payload = json.loads(last_request.body)
        assert received_payload["text"] == "🚨 System Alert: Database connection lost"
        assert received_payload["channel"] == "#infrastructure-alerts"
        assert received_payload["username"] == "ONEX Monitor"

    @pytest.mark.asyncio
    async def test_slack_webhook_rich_attachments(self, hook_node, webhook_server):
        """Test Slack webhook with rich attachments."""
        slack_payload = {
            "text": "Infrastructure Alert",
            "channel": "#dev-alerts",
            "attachments": [
                {
                    "color": "danger",
                    "title": "Database Connection Failure",
                    "fields": [
                        {"title": "Service", "value": "PostgreSQL", "short": True},
                        {"title": "Severity", "value": "Critical", "short": True},
                        {"title": "Duration", "value": "5 minutes", "short": True},
                        {"title": "Affected Users", "value": "All", "short": True},
                    ],
                    "footer": "ONEX Infrastructure",
                    "ts": int(time.time()),
                },
            ],
        }

        request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/TEAM/CHANNEL/TOKEN",
            method=EnumNotificationMethod.POST,
            payload=slack_payload,
        )

        webhook_server.configure_responses(
            success_config=MockWebhookResponseConfigModel(delay_ms=200),
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify rich attachment structure is preserved
        last_request = webhook_server.get_last_request()
        received_payload = json.loads(last_request.body)

        assert "attachments" in received_payload
        attachment = received_payload["attachments"][0]
        assert attachment["color"] == "danger"
        assert attachment["title"] == "Database Connection Failure"
        assert len(attachment["fields"]) == 4
        assert attachment["fields"][0]["title"] == "Service"
        assert attachment["fields"][0]["value"] == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_slack_webhook_authentication(self, hook_node, webhook_server):
        """Test Slack webhook with authentication (Bearer token)."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BEARER,
            credentials={"token": "xoxb-slack-bot-token"},
        )

        request = ModelNotificationRequest(
            url="https://slack.com/api/chat.postMessage",
            method=EnumNotificationMethod.POST,
            payload={
                "channel": "C1234567890",
                "text": "Authenticated Slack message",
                "as_user": True,
            },
            auth=auth,
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify authentication header
        last_request = webhook_server.get_last_request()
        assert "Authorization" in last_request["headers"]
        assert last_request["headers"]["Authorization"] == "Bearer xoxb-slack-bot-token"

    @pytest.mark.asyncio
    async def test_slack_webhook_failure_and_retry(self, hook_node, webhook_server):
        """Test Slack webhook failure handling and retry behavior."""
        retry_policy = ModelNotificationRetryPolicy(
            max_attempts=3,
            strategy=EnumBackoffStrategy.EXPONENTIAL,
            base_delay_ms=100,
            max_delay_ms=1000,
            backoff_multiplier=2.0,
        )

        request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/unreliable/webhook/url",
            method=EnumNotificationMethod.POST,
            payload={"text": "Test retry behavior"},
            retry_policy=retry_policy,
        )

        # Configure server to fail first 2 attempts, then succeed
        webhook_server.configure_responses(
            success_config=MockWebhookResponseConfigModel(delay_ms=50),
            failure_config=MockWebhookFailureConfigModel(
                fail_count=2,
                status_code=500,
                body="Internal Server Error",
            ),
        )

        with patch("asyncio.sleep"):  # Speed up test
            input_data = ModelHookNodeInput(notification_request=request)
            result = await hook_node.process(input_data)

        assert result.success is True
        assert result.notification_result.total_attempts == 3

        # Verify all attempts were made
        requests = webhook_server.get_requests_for_url("unreliable")
        assert len(requests) == 3


class TestDiscordWebhookDelivery:
    """Test suite for Discord webhook delivery validation."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def webhook_server(self):
        """Create a mock webhook server."""
        return MockWebhookServer()

    @pytest.fixture
    def hook_node(self, container, webhook_server):
        """Create Hook Node with mocked HTTP client."""
        mock_http_client = AsyncMock()
        mock_http_client.post = webhook_server.handle_request
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_discord_webhook_basic_message(self, hook_node, webhook_server):
        """Test basic Discord webhook message delivery."""
        discord_payload = {
            "content": "🔥 **CRITICAL ALERT**\nDatabase connection has been lost!",
            "username": "ONEX Infrastructure Bot",
            "avatar_url": "https://example.com/bot-avatar.png",
        }

        request = ModelNotificationRequest(
            url="https://discord.com/api/webhooks/1234567890123456789/very-long-webhook-token-here",
            method=EnumNotificationMethod.POST,
            payload=discord_payload,
        )

        # Configure Discord-like response
        webhook_server.configure_responses(
            success_config=MockWebhookResponseConfigModel(
                status_code=204,  # Discord returns 204 for successful webhooks
                body="",
                headers={},
                delay_ms=120,
            ),
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True
        assert result.notification_result.final_status_code == 204

        # Verify Discord payload format
        last_request = webhook_server.get_last_request()
        received_payload = json.loads(last_request.body)

        assert received_payload["content"] == "🔥 **CRITICAL ALERT**\nDatabase connection has been lost!"
        assert received_payload["username"] == "ONEX Infrastructure Bot"
        assert received_payload["avatar_url"] == "https://example.com/bot-avatar.png"

    @pytest.mark.asyncio
    async def test_discord_webhook_embeds(self, hook_node, webhook_server):
        """Test Discord webhook with rich embeds."""
        discord_payload = {
            "content": "Infrastructure Status Update",
            "embeds": [
                {
                    "title": "System Health Report",
                    "description": "Current status of all infrastructure services",
                    "color": 16711680,  # Red color
                    "fields": [
                        {"name": "Database", "value": "❌ Offline", "inline": True},
                        {"name": "Cache", "value": "✅ Healthy", "inline": True},
                        {"name": "Load Balancer", "value": "⚠️ Degraded", "inline": True},
                    ],
                    "footer": {"text": "ONEX Infrastructure Monitor"},
                    "timestamp": datetime.utcnow().isoformat(),
                },
            ],
        }

        request = ModelNotificationRequest(
            url="https://discord.com/api/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN",
            method=EnumNotificationMethod.POST,
            payload=discord_payload,
        )

        webhook_server.configure_responses(
            success_config=MockWebhookResponseConfigModel(status_code=204, body=""),
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify embed structure
        last_request = webhook_server.get_last_request()
        received_payload = json.loads(last_request.body)

        assert "embeds" in received_payload
        embed = received_payload["embeds"][0]
        assert embed["title"] == "System Health Report"
        assert embed["color"] == 16711680
        assert len(embed["fields"]) == 3
        assert embed["fields"][0]["name"] == "Database"
        assert embed["fields"][0]["value"] == "❌ Offline"

    @pytest.mark.asyncio
    async def test_discord_webhook_rate_limit_handling(self, hook_node, webhook_server):
        """Test Discord webhook rate limit handling."""
        request = ModelNotificationRequest(
            url="https://discord.com/api/webhooks/rate-limited/webhook",
            method=EnumNotificationMethod.POST,
            payload={"content": "Rate limit test message"},
        )

        # Configure rate limit response
        webhook_server.configure_responses(
            failure_config=MockWebhookFailureConfigModel(
                fail_count=1,
                status_code=429,  # Rate limited
                body='{"retry_after": 1000}',
                headers={"Retry-After": "1"},
            ),
        )

        with patch("asyncio.sleep"):  # Speed up test
            input_data = ModelHookNodeInput(notification_request=request)
            result = await hook_node.process(input_data)

        assert result.success is True  # Should eventually succeed after rate limit


class TestGenericWebhookDelivery:
    """Test suite for generic webhook delivery validation."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def webhook_server(self):
        """Create a mock webhook server."""
        return MockWebhookServer()

    @pytest.fixture
    def hook_node(self, container, webhook_server):
        """Create Hook Node with mocked HTTP client."""
        mock_http_client = AsyncMock()
        mock_http_client.post = webhook_server.handle_request
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_generic_webhook_custom_payload(self, hook_node, webhook_server):
        """Test generic webhook with custom payload structure."""
        custom_payload = {
            "event_type": "infrastructure.alert",
            "severity": "critical",
            "source": {
                "service": "hook_node",
                "environment": "production",
                "region": "us-west-2",
            },
            "alert": {
                "title": "Database Connection Pool Exhausted",
                "description": "All database connections are currently in use",
                "timestamp": datetime.utcnow().isoformat(),
                "tags": ["database", "connection-pool", "critical"],
            },
            "metadata": {
                "correlation_id": str(uuid4()),
                "version": "1.0",
                "schema": "onex-infrastructure-alert-v1",
            },
        }

        request = ModelNotificationRequest(
            url="https://api.example.com/v1/webhooks/infrastructure-alerts",
            method=EnumNotificationMethod.POST,
            payload=custom_payload,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Source": "ONEX-Infrastructure",
                "X-Event-Type": "infrastructure.alert",
            },
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify custom payload structure is preserved
        last_request = webhook_server.get_last_request()
        received_payload = json.loads(last_request.body)

        assert received_payload["event_type"] == "infrastructure.alert"
        assert received_payload["severity"] == "critical"
        assert received_payload["source"]["service"] == "hook_node"
        assert received_payload["alert"]["title"] == "Database Connection Pool Exhausted"
        assert len(received_payload["alert"]["tags"]) == 3

        # Verify custom headers
        assert last_request["headers"]["X-Webhook-Source"] == "ONEX-Infrastructure"
        assert last_request["headers"]["X-Event-Type"] == "infrastructure.alert"

    @pytest.mark.asyncio
    async def test_generic_webhook_api_key_authentication(self, hook_node, webhook_server):
        """Test generic webhook with API key authentication."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.API_KEY_HEADER,
            credentials={
                "header_name": "X-API-Key",
                "api_key": "ak_1234567890abcdef",
            },
        )

        request = ModelNotificationRequest(
            url="https://api.monitoring-service.com/webhooks/alerts",
            method=EnumNotificationMethod.POST,
            payload={
                "alert_type": "infrastructure_failure",
                "message": "Service degradation detected",
                "priority": "high",
            },
            auth=auth,
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify API key header
        last_request = webhook_server.get_last_request()
        assert "X-API-Key" in last_request["headers"]
        assert last_request["headers"]["X-API-Key"] == "ak_1234567890abcdef"

    @pytest.mark.asyncio
    async def test_generic_webhook_basic_authentication(self, hook_node, webhook_server):
        """Test generic webhook with Basic authentication."""
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BASIC,
            credentials={
                "username": "webhook-user",
                "password": "secure-password-123",
            },
        )

        request = ModelNotificationRequest(
            url="https://internal.company.com/api/webhooks/infrastructure",
            method=EnumNotificationMethod.POST,
            payload={
                "system": "ONEX Infrastructure",
                "event": "circuit_breaker_opened",
                "details": "Circuit breaker opened for failing service",
            },
            auth=auth,
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify Basic auth header
        last_request = webhook_server.get_last_request()
        assert "Authorization" in last_request["headers"]
        auth_header = last_request["headers"]["Authorization"]
        assert auth_header.startswith("Basic ")

        # Decode and verify credentials
        import base64
        encoded_creds = auth_header.split(" ")[1]
        decoded_creds = base64.b64decode(encoded_creds).decode()
        assert decoded_creds == "webhook-user:secure-password-123"

    @pytest.mark.asyncio
    async def test_webhook_put_method_support(self, hook_node, webhook_server):
        """Test webhook delivery using PUT method."""
        request = ModelNotificationRequest(
            url="https://api.custom-service.com/v2/notifications/update",
            method=EnumNotificationMethod.PUT,  # Using PUT instead of POST
            payload={
                "notification_id": "notif_12345",
                "status": "acknowledged",
                "updated_by": "onex_infrastructure",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Mock PUT method on HTTP client
        webhook_server.handle_request = AsyncMock(return_value=ProtocolHttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"updated": true}',
            execution_time_ms=80.0,
            is_success=True,
        ))

        # Patch the HTTP client to support PUT
        with patch.object(hook_node._http_client, "put", webhook_server.handle_request):
            input_data = ModelHookNodeInput(notification_request=request)
            result = await hook_node.process(input_data)

        assert result.success is True
        assert result.notification_result.final_status_code == 200


class TestWebhookCircuitBreakerBehavior:
    """Test suite for circuit breaker behavior with webhook failures."""

    @pytest.fixture
    def container(self):
        """Create a test container."""
        return ModelONEXContainer()

    @pytest.fixture
    def webhook_server(self):
        """Create a mock webhook server."""
        return MockWebhookServer()

    @pytest.fixture
    def hook_node(self, container, webhook_server):
        """Create Hook Node with mocked HTTP client."""
        mock_http_client = AsyncMock()
        mock_http_client.post = webhook_server.handle_request
        mock_event_bus = AsyncMock()

        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return NodeHookEffect(container)

    @pytest.mark.asyncio
    async def test_circuit_breaker_per_destination(self, hook_node, webhook_server):
        """Test that circuit breakers are maintained per destination URL."""
        # Create requests to different URLs
        slack_request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/failing/webhook",
            method=EnumNotificationMethod.POST,
            payload={"text": "Slack test"},
        )

        discord_request = ModelNotificationRequest(
            url="https://discord.com/api/webhooks/working/webhook",
            method=EnumNotificationMethod.POST,
            payload={"content": "Discord test"},
        )

        # Configure server to fail only Slack requests
        def selective_handler(method: str, url: str, headers: Dict[str, str], body: str) -> ProtocolHttpResponse:
            if "slack.com" in url:
                return ProtocolHttpResponse(
                    status_code=500,
                    headers={},
                    body="Internal Server Error",
                    execution_time_ms=100.0,
                    is_success=False,
                )
            return ProtocolHttpResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body='{"status": "ok"}',
                execution_time_ms=50.0,
                is_success=True,
            )

        hook_node._http_client.post = selective_handler

        # Send multiple failing requests to Slack to trigger circuit breaker
        for i in range(6):  # Exceed failure threshold
            try:
                input_data = ModelHookNodeInput(notification_request=slack_request)
                await hook_node.process(input_data)
            except OnexError:
                pass  # Expected failures

        # Verify Slack circuit breaker is OPEN
        slack_cb = hook_node._circuit_breakers.get(str(slack_request.url))
        assert slack_cb is not None
        assert slack_cb.state == CircuitBreakerState.OPEN

        # Discord should still work (separate circuit breaker)
        input_data = ModelHookNodeInput(notification_request=discord_request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Verify Discord circuit breaker is not affected
        discord_cb = hook_node._circuit_breakers.get(str(discord_request.url))
        assert discord_cb is None or discord_cb.state != CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_attempt(self, hook_node, webhook_server):
        """Test circuit breaker recovery attempt after timeout."""
        request = ModelNotificationRequest(
            url="https://recovery-test.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"test": "recovery"},
        )

        # Configure server to fail initially
        webhook_server.configure_responses(
            failure_config=MockWebhookFailureConfigModel(
                fail_count=10,  # Many failures
                status_code=503,
                body="Service Unavailable",
            ),
        )

        # Trigger circuit breaker opening
        for i in range(6):
            try:
                input_data = ModelHookNodeInput(notification_request=request)
                await hook_node.process(input_data)
            except OnexError:
                pass

        # Verify circuit breaker is OPEN
        url_key = str(request.url)
        circuit_breaker = hook_node._circuit_breakers[url_key]
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Manually advance time to trigger recovery attempt
        circuit_breaker.last_failure_time = time.time() - 70  # 70 seconds ago (past recovery timeout)

        # Configure server to now succeed
        webhook_server.configure_responses(
            success_config=MockWebhookResponseConfigModel(status_code=200),
        )

        # Next request should attempt recovery
        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node.process(input_data)

        assert result.success is True

        # Circuit breaker should transition to HALF_OPEN then CLOSED
        # Note: Implementation details may vary, but it should eventually close
        assert circuit_breaker.state in [CircuitBreakerState.HALF_OPEN, CircuitBreakerState.CLOSED]
