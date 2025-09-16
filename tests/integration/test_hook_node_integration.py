"""
Integration Tests for Hook Node.

Tests the Hook Node integration with:
- Event bus message processing (ModelOnexEvent)
- Circuit breaker notifications integration
- HTTP client protocol dependency injection
- Container injection and registry setup
- End-to-end notification delivery flows
"""

import asyncio
import json
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4
from typing import List

from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_core.enums.enum_auth_type import EnumAuthType
from omnibase_spi.protocols.core import ProtocolHttpClient, ProtocolHttpResponse
from omnibase_spi.protocols.event_bus import ProtocolEventBus

# Hook Node implementation
from omnibase_infra.nodes.hook_node.v1_0_0.node import NodeHookEffect, CircuitBreakerState
from omnibase_infra.nodes.hook_node.v1_0_0.models.model_hook_node_input import ModelHookNodeInput
from omnibase_infra.nodes.hook_node.v1_0_0.registry.registry_hook_node import HookNodeRegistry

# Shared notification models
from omnibase_infra.models.notification.model_notification_request import ModelNotificationRequest
from omnibase_infra.models.notification.model_notification_auth import ModelNotificationAuth

# Test-specific strongly typed models
from tests.models.test_webhook_models import IntegrationTestRequestModel


class MockHttpClient:
    """Mock HTTP client that implements ProtocolHttpClient interface."""

    def __init__(self):
        self.requests_made: List[IntegrationTestRequestModel] = []
        self.response_sequence: List[ProtocolHttpResponse] = []
        self.current_response_index = 0

    def set_response_sequence(self, responses: List[ProtocolHttpResponse]):
        """Set a sequence of responses to return."""
        self.response_sequence = responses
        self.current_response_index = 0

    async def post(self, url: str, headers: Dict[str, str] = None, body: str = None, timeout: float = 30.0) -> ProtocolHttpResponse:
        """Mock POST request implementation."""
        # Record the request
        self.requests_made.append(IntegrationTestRequestModel(
            url=url,
            method="POST",
            headers=headers or {},
            payload={"body": body or "", "timeout": timeout},
            timestamp=time.time(),
            correlation_id=str(uuid4())
        ))

        # Return next response in sequence
        if self.current_response_index < len(self.response_sequence):
            response = self.response_sequence[self.current_response_index]
            self.current_response_index += 1
            return response
        else:
            # Default success response
            return ProtocolHttpResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body='{"status": "ok"}',
                execution_time_ms=100.0,
                is_success=True
            )

    async def get(self, url: str, headers: Dict[str, str] = None, timeout: float = 30.0) -> ProtocolHttpResponse:
        """Mock GET request implementation."""
        self.requests_made.append(IntegrationTestRequestModel(
            url=url,
            method="GET",
            headers=headers or {},
            payload={"timeout": timeout},
            timestamp=time.time(),
            correlation_id=str(uuid4())
        ))
        return ProtocolHttpResponse(
            status_code=200,
            headers={},
            body='{"status": "ok"}',
            execution_time_ms=50.0,
            is_success=True
        )


class MockEventBus:
    """Mock event bus that implements ProtocolEventBus interface."""

    def __init__(self):
        self.published_events: List[ModelOnexEvent] = []
        self.event_handlers: Dict[str, callable] = {}

    async def publish(self, event: ModelOnexEvent) -> bool:
        """Mock event publishing."""
        self.published_events.append(event)
        return True

    async def subscribe(self, event_type: str, handler: callable) -> bool:
        """Mock event subscription."""
        self.event_handlers[event_type] = handler
        return True

    def get_published_events_by_type(self, event_type: str) -> List[ModelOnexEvent]:
        """Get published events filtered by type."""
        return [event for event in self.published_events if event.event_type == event_type]


class TestHookNodeIntegration:
    """Integration test suite for Hook Node with real dependencies."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        return MockHttpClient()

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return MockEventBus()

    @pytest.fixture
    def container_with_mocks(self, mock_http_client, mock_event_bus):
        """Create a container with mock dependencies."""
        container = ModelONEXContainer()

        # Provide mocked protocol implementations
        container.provide("protocol_http_client", mock_http_client)
        container.provide("protocol_event_bus", mock_event_bus)

        return container

    @pytest.fixture
    def hook_node_integration(self, container_with_mocks):
        """Create a Hook Node with integration-ready dependencies."""
        return NodeHookEffect(container_with_mocks)

    @pytest.mark.asyncio
    async def test_container_dependency_injection(self, container_with_mocks):
        """Test that Hook Node properly receives injected dependencies."""
        hook_node = NodeHookEffect(container_with_mocks)

        # Verify dependencies are properly injected
        assert hook_node._http_client is not None
        assert hook_node._event_bus is not None
        assert isinstance(hook_node._http_client, MockHttpClient)
        assert isinstance(hook_node._event_bus, MockEventBus)

    @pytest.mark.asyncio
    async def test_event_bus_integration_success_notification(self, hook_node_integration, mock_event_bus):
        """Test successful notification triggers circuit breaker success event."""
        # Setup successful response
        success_response = ProtocolHttpResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"message": "received"}',
            execution_time_ms=120.0,
            is_success=True
        )
        hook_node_integration._http_client.set_response_sequence([success_response])

        # Create notification request
        request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/integration-test",
            method=EnumNotificationMethod.POST,
            payload={"text": "Integration test notification"}
        )

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node_integration.process(input_data)

        assert result.success is True

        # Verify circuit breaker success event was published
        success_events = mock_event_bus.get_published_events_by_type("circuit_breaker.success")
        assert len(success_events) >= 1

        latest_event = success_events[-1]
        assert "https://hooks.slack.com/services/integration-test" in str(latest_event.payload)

    @pytest.mark.asyncio
    async def test_event_bus_integration_failure_notification(self, hook_node_integration, mock_event_bus):
        """Test failed notification triggers circuit breaker failure event."""
        # Setup failure response
        failure_response = ProtocolHttpResponse(
            status_code=500,
            headers={},
            body='{"error": "Internal server error"}',
            execution_time_ms=30000.0,  # Timeout
            is_success=False
        )
        hook_node_integration._http_client.set_response_sequence([failure_response])

        # Create notification request
        request = ModelNotificationRequest(
            url="https://failing.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"alert": "System failure"}
        )

        input_data = ModelHookNodeInput(notification_request=request)

        # Process should not raise exception but should report failure
        result = await hook_node_integration.process(input_data)
        assert result.success is False

        # Verify circuit breaker failure event was published
        failure_events = mock_event_bus.get_published_events_by_type("circuit_breaker.failure")
        assert len(failure_events) >= 1

        latest_event = failure_events[-1]
        assert "https://failing.webhook.com/api/notify" in str(latest_event.payload)

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_change_events(self, hook_node_integration, mock_event_bus):
        """Test circuit breaker state change events are published."""
        # Setup consistent failures to trigger state change
        failure_response = ProtocolHttpResponse(
            status_code=503,
            headers={},
            body='{"error": "Service unavailable"}',
            execution_time_ms=5000.0,
            is_success=False
        )

        # Set up enough failures to trigger state change
        hook_node_integration._http_client.set_response_sequence([failure_response] * 10)

        request = ModelNotificationRequest(
            url="https://circuit-breaker-test.com/webhook",
            method=EnumNotificationMethod.POST,
            payload={"test": "circuit breaker integration"}
        )

        input_data = ModelHookNodeInput(notification_request=request)

        # Send multiple requests to trigger circuit breaker
        for i in range(6):  # Default threshold is 5
            try:
                await hook_node_integration.process(input_data)
            except OnexError:
                pass  # Expected failures

        # Verify circuit breaker opened event was published
        state_change_events = mock_event_bus.get_published_events_by_type("circuit_breaker.state_change")

        # Should have at least one state change event (to OPEN)
        assert len(state_change_events) >= 1

        # Find the OPEN state change event
        open_events = [
            event for event in state_change_events
            if "OPEN" in str(event.payload)
        ]
        assert len(open_events) >= 1

    @pytest.mark.asyncio
    async def test_http_client_authentication_integration(self, hook_node_integration):
        """Test HTTP client receives proper authentication headers."""
        # Setup authentication
        auth = ModelNotificationAuth(
            auth_type=EnumAuthType.BEARER,
            credentials={"token": "integration-test-token"}
        )

        request = ModelNotificationRequest(
            url="https://authenticated.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"message": "Authenticated integration test"},
            auth=auth
        )

        success_response = ProtocolHttpResponse(
            status_code=200,
            headers={},
            body='{"authenticated": true}',
            execution_time_ms=80.0,
            is_success=True
        )
        hook_node_integration._http_client.set_response_sequence([success_response])

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node_integration.process(input_data)

        assert result.success is True

        # Verify authentication header was included
        requests_made = hook_node_integration._http_client.requests_made
        assert len(requests_made) == 1

        auth_header = requests_made[0].headers.get("Authorization")
        assert auth_header == "Bearer integration-test-token"

    @pytest.mark.asyncio
    async def test_http_client_timeout_integration(self, hook_node_integration):
        """Test HTTP client timeout configuration is respected."""
        request = ModelNotificationRequest(
            url="https://timeout-test.webhook.com/slow",
            method=EnumNotificationMethod.POST,
            payload={"test": "timeout integration"}
        )

        input_data = ModelHookNodeInput(notification_request=request)

        # Mock timeout exception
        async def mock_post_timeout(*args, **kwargs):
            raise asyncio.TimeoutError("Request timeout after 30 seconds")

        hook_node_integration._http_client.post = mock_post_timeout

        with pytest.raises(OnexError) as exc_info:
            await hook_node_integration.process(input_data)

        assert exc_info.value.code == CoreErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_end_to_end_slack_notification_flow(self, hook_node_integration, mock_event_bus):
        """Test complete end-to-end Slack notification flow."""
        # Setup Slack webhook format
        slack_payload = {
            "text": "🚨 Integration Test Alert",
            "channel": "#dev-alerts",
            "username": "ONEX Infrastructure",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {
                            "title": "Service",
                            "value": "hook_node_integration_test",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.utcnow().isoformat(),
                            "short": True
                        }
                    ]
                }
            ]
        }

        request = ModelNotificationRequest(
            url="https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
            method=EnumNotificationMethod.POST,
            payload=slack_payload
        )

        # Mock Slack's expected response
        slack_response = ProtocolHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/plain"},
            body="ok",
            execution_time_ms=150.0,
            is_success=True
        )
        hook_node_integration._http_client.set_response_sequence([slack_response])

        input_data = ModelHookNodeInput(notification_request=request)
        result = await hook_node_integration.process(input_data)

        # Verify successful delivery
        assert result.success is True
        assert result.notification_result.final_status_code == 200

        # Verify request was properly formatted for Slack
        requests_made = hook_node_integration._http_client.requests_made
        assert len(requests_made) == 1

        request_body = json.loads(requests_made[0].payload["body"])
        assert request_body["text"] == "🚨 Integration Test Alert"
        assert request_body["channel"] == "#dev-alerts"
        assert "attachments" in request_body

        # Verify success event was published
        success_events = mock_event_bus.get_published_events_by_type("circuit_breaker.success")
        assert len(success_events) >= 1

    @pytest.mark.asyncio
    async def test_end_to_end_retry_with_circuit_breaker(self, hook_node_integration, mock_event_bus):
        """Test end-to-end retry flow with circuit breaker interaction."""
        request = ModelNotificationRequest(
            url="https://unreliable.webhook.com/api/notify",
            method=EnumNotificationMethod.POST,
            payload={"message": "Retry integration test"}
        )

        # Setup response sequence: fail, fail, succeed
        responses = [
            ProtocolHttpResponse(
                status_code=500,
                headers={},
                body='{"error": "Temporary failure"}',
                execution_time_ms=200.0,
                is_success=False
            ),
            ProtocolHttpResponse(
                status_code=502,
                headers={},
                body='{"error": "Bad gateway"}',
                execution_time_ms=300.0,
                is_success=False
            ),
            ProtocolHttpResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body='{"status": "delivered"}',
                execution_time_ms=100.0,
                is_success=True
            )
        ]
        hook_node_integration._http_client.set_response_sequence(responses)

        with patch('asyncio.sleep'):  # Speed up test by mocking sleep
            input_data = ModelHookNodeInput(notification_request=request)
            result = await hook_node_integration.process(input_data)

        # Verify eventual success
        assert result.success is True
        assert result.notification_result.total_attempts == 3
        assert result.notification_result.final_status_code == 200

        # Verify all requests were made
        requests_made = hook_node_integration._http_client.requests_made
        assert len(requests_made) == 3

        # Verify circuit breaker events for failures and final success
        failure_events = mock_event_bus.get_published_events_by_type("circuit_breaker.failure")
        success_events = mock_event_bus.get_published_events_by_type("circuit_breaker.success")

        assert len(failure_events) == 2  # Two failures
        assert len(success_events) == 1  # One final success

    @pytest.mark.asyncio
    async def test_registry_integration(self, container_with_mocks):
        """Test Hook Node registry integration and dependency resolution."""
        # Test registry can validate Hook Node configuration
        is_valid = HookNodeRegistry.validate_configuration(container_with_mocks)
        assert is_valid, "Registry should validate container configuration as valid"

        # Registry should be able to register dependencies in container
        HookNodeRegistry.register_dependencies(container_with_mocks)

        # Create Hook Node directly with the configured container
        hook_node = NodeHookEffect(container_with_mocks)
        assert hook_node is not None
        assert hook_node._http_client is not None
        assert hook_node._event_bus is not None
        assert hook_node.node_type == "effect"
        assert hook_node.domain == "infrastructure"

    @pytest.mark.asyncio
    async def test_concurrent_notifications_integration(self, hook_node_integration):
        """Test concurrent notification processing doesn't interfere."""
        # Create multiple notification requests
        requests = [
            ModelNotificationRequest(
                url=f"https://concurrent-test-{i}.webhook.com/notify",
                method=EnumNotificationMethod.POST,
                payload={"message": f"Concurrent test {i}", "test_id": i}
            )
            for i in range(5)
        ]

        # Setup responses for all requests
        responses = [
            ProtocolHttpResponse(
                status_code=200,
                headers={},
                body=f'{{"received": {i}}}',
                execution_time_ms=100.0 + (i * 10),  # Slightly different timing
                is_success=True
            )
            for i in range(5)
        ]
        hook_node_integration._http_client.set_response_sequence(responses)

        # Process all notifications concurrently
        input_data_list = [ModelHookNodeInput(notification_request=req) for req in requests]

        tasks = [hook_node_integration.process(input_data) for input_data in input_data_list]
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success is True
            assert result.notification_result.final_status_code == 200

        # Verify all requests were made
        requests_made = hook_node_integration._http_client.requests_made
        assert len(requests_made) == 5

        # Verify each URL was called exactly once
        urls_called = [req["url"] for req in requests_made]
        expected_urls = [str(req.url) for req in requests]

        for expected_url in expected_urls:
            assert urls_called.count(expected_url) == 1

    @pytest.mark.asyncio
    async def test_event_bus_integration_error_scenarios(self, hook_node_integration, mock_event_bus):
        """Test event bus integration handles various error scenarios."""
        # Test with network exception
        async def mock_post_network_error(*args, **kwargs):
            raise ConnectionError("Network unreachable")

        hook_node_integration._http_client.post = mock_post_network_error

        request = ModelNotificationRequest(
            url="https://network-error-test.webhook.com/notify",
            method=EnumNotificationMethod.POST,
            payload={"test": "network error"}
        )

        input_data = ModelHookNodeInput(notification_request=request)

        with pytest.raises(OnexError):
            await hook_node_integration.process(input_data)

        # Verify error event was published
        error_events = mock_event_bus.get_published_events_by_type("circuit_breaker.error")
        assert len(error_events) >= 1

        latest_error_event = error_events[-1]
        event_payload = str(latest_error_event.payload)
        assert "network-error-test.webhook.com" in event_payload
        assert "ConnectionError" in event_payload or "Network" in event_payload