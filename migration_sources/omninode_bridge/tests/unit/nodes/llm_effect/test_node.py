#!/usr/bin/env python3
"""
Unit tests for NodeLLMEffect.

Tests node initialization, circuit breaker, retry logic, and API integration (mocked).
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from src.omninode_bridge.nodes.llm_effect.v1_0_0.models import EnumLLMTier
from src.omninode_bridge.nodes.llm_effect.v1_0_0.node import NodeLLMEffect


class TestNodeLLMEffectInitialization:
    """Test cases for NodeLLMEffect initialization."""

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key_12345",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_init_with_config(self):
        """Test initialization with configuration."""
        container = ModelContainer(
            value={
                "zai_api_key": "test_key_12345",
                "zai_base_url": "https://api.z.ai/api/anthropic",
            },
            container_type="config",
        )

        node = NodeLLMEffect(container)

        assert node.config.zai_api_key == "test_key_12345"
        assert node.config.zai_base_url == "https://api.z.ai/api/anthropic"
        assert node.tier_models[EnumLLMTier.CLOUD_FAST] == "glm-4.5"
        assert node.circuit_breaker.failure_threshold == 5
        # Circuit breaker should be in closed state (can execute, no failures)
        assert node.circuit_breaker.can_execute() is True
        assert node.circuit_breaker.failure_count == 0

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        container = ModelContainer(
            value={},
            container_type="config",
        )

        # Should raise either ModelOnexError or Pydantic ValidationError
        with pytest.raises((ModelOnexError, Exception)) as exc_info:
            NodeLLMEffect(container)

        # Either ONEX error or Pydantic validation error is acceptable
        if isinstance(exc_info.value, ModelOnexError):
            assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_INPUT
            assert "ZAI_API_KEY" in str(exc_info.value.message)

    @patch.dict("os.environ", {"ZAI_API_KEY": "env_key_67890"})
    def test_init_from_environment(self):
        """Test initialization from environment variables."""
        container = ModelContainer(
            value={},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        assert node.config.zai_api_key == "env_key_67890"

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_tier_models_configuration(self):
        """Test tier models are correctly configured."""
        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        assert node.tier_models[EnumLLMTier.LOCAL] is None  # Not implemented
        assert node.tier_models[EnumLLMTier.CLOUD_FAST] == "glm-4.5"
        assert node.tier_models[EnumLLMTier.CLOUD_PREMIUM] == "glm-4.6"

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_cost_configuration(self):
        """Test cost per 1M tokens is correctly configured."""
        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        assert node.cost_per_1m_input[EnumLLMTier.CLOUD_FAST] == 0.20
        assert node.cost_per_1m_output[EnumLLMTier.CLOUD_FAST] == 0.20


class TestNodeLLMEffectCircuitBreaker:
    """Test cases for circuit breaker logic."""

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_circuit_breaker_closed_initially(self):
        """Test circuit breaker is closed initially."""
        from omnibase_core.nodes.enum_effect_types import EnumCircuitBreakerState

        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        assert node.circuit_breaker.can_execute() is True
        assert node.circuit_breaker.failure_count == 0
        assert node.circuit_breaker.state == EnumCircuitBreakerState.CLOSED

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens after threshold failures."""
        from omnibase_core.nodes.enum_effect_types import EnumCircuitBreakerState

        container = ModelContainer(
            value={"zai_api_key": "test_key", "circuit_breaker_threshold": 3},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Record failures up to threshold
        for i in range(3):
            node.circuit_breaker.record_failure()

        assert node.circuit_breaker.failure_count == 3
        assert node.circuit_breaker.state == EnumCircuitBreakerState.OPEN
        assert node.circuit_breaker.can_execute() is False

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets after successful operation."""
        from omnibase_core.nodes.enum_effect_types import EnumCircuitBreakerState

        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Record some failures
        node.circuit_breaker.record_failure()
        node.circuit_breaker.record_failure()
        assert node.circuit_breaker.failure_count == 2

        # Reset on success
        node.circuit_breaker.record_success()
        # After success, failure_count should decrease (gradual recovery in CLOSED state)
        assert node.circuit_breaker.failure_count == 1
        assert node.circuit_breaker.state == EnumCircuitBreakerState.CLOSED

    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    def test_circuit_breaker_timeout(self):
        """Test circuit breaker closes after timeout."""
        from omnibase_core.nodes.enum_effect_types import EnumCircuitBreakerState

        container = ModelContainer(
            value={
                "zai_api_key": "test_key",
                "circuit_breaker_threshold": 2,
                "circuit_breaker_timeout_seconds": 1,  # 1 second timeout
            },
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Open circuit breaker
        node.circuit_breaker.record_failure()
        node.circuit_breaker.record_failure()
        assert node.circuit_breaker.can_execute() is False

        # Wait for timeout
        time.sleep(1.1)

        # Circuit should transition to HALF_OPEN and allow execution
        assert node.circuit_breaker.can_execute() is True
        assert node.circuit_breaker.state == EnumCircuitBreakerState.HALF_OPEN


class TestNodeLLMEffectExecution:
    """Test cases for execute_effect method."""

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_execute_effect_not_implemented_tiers(self):
        """Test execution fails for not-implemented tiers."""
        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Test LOCAL tier
        contract = ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Test",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state={
                "prompt": "Test prompt",
                "tier": "LOCAL",
            },
            io_operations=[
                {
                    "operation_type": "WRITE",
                    "resource_identifier": "llm_api",
                }
            ],
        )

        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract)

        assert exc_info.value.error_code == EnumCoreErrorCode.UNSUPPORTED_OPERATION
        assert "LOCAL" in str(exc_info.value.message)

        # Test CLOUD_PREMIUM tier
        contract.input_state["tier"] = "CLOUD_PREMIUM"

        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract)

        assert exc_info.value.error_code == EnumCoreErrorCode.UNSUPPORTED_OPERATION
        assert "CLOUD_PREMIUM" in str(exc_info.value.message)

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_execute_effect_circuit_breaker_open(self):
        """Test execution fails when circuit breaker is open."""
        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Force circuit breaker open with recent failure
        from omnibase_core.nodes.enum_effect_types import EnumCircuitBreakerState

        node.circuit_breaker.state = EnumCircuitBreakerState.OPEN
        node.circuit_breaker.failure_count = 5
        node.circuit_breaker.last_failure_time = datetime.now()

        contract = ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Test",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state={
                "prompt": "Test prompt",
                "tier": "CLOUD_FAST",
            },
            io_operations=[
                {
                    "operation_type": "WRITE",
                    "resource_identifier": "llm_api",
                }
            ],
        )

        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract)

        assert exc_info.value.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE
        assert "Circuit breaker is OPEN" in str(exc_info.value.message)

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_execute_effect_success_cloud_fast(self):
        """Test successful execution for CLOUD_FAST tier."""
        from omnibase_core.nodes.enum_effect_types import EnumCircuitBreakerState

        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "def fibonacci(n): return n"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 15,
            },
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        node.http_client = mock_client

        contract = ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Test",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state={
                "prompt": "Generate Fibonacci function",
                "tier": "CLOUD_FAST",
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            io_operations=[
                {
                    "operation_type": "WRITE",
                    "resource_identifier": "llm_api",
                }
            ],
        )

        response = await node.execute_effect(contract)

        # Verify response
        assert response.generated_text == "def fibonacci(n): return n"
        assert response.model_used == "glm-4.5"
        assert response.tier_used == EnumLLMTier.CLOUD_FAST
        assert response.tokens_input == 25
        assert response.tokens_output == 15
        assert response.tokens_total == 40
        assert response.finish_reason == "stop"
        assert response.cost_usd > 0  # Cost should be calculated

        # Verify circuit breaker was reset
        assert node.circuit_breaker.failure_count == 0
        assert node.circuit_breaker.state == EnumCircuitBreakerState.CLOSED


class TestNodeLLMEffectRetryLogic:
    """Test cases for retry logic."""

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_retry_on_rate_limit(self):
        """Test retry logic on 429 rate limit errors."""
        container = ModelContainer(
            value={
                "zai_api_key": "test_key",
                "retry_initial_backoff_seconds": 0.1,  # Fast retry for testing
            },
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Mock HTTP client - fail twice, then succeed
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # Simulate rate limit
                response = MagicMock()
                response.status_code = 429
                response.text = "Rate limit exceeded"
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "429 Too Many Requests",
                    request=MagicMock(),
                    response=response,
                )
                raise response.raise_for_status.side_effect

            # Success on third attempt
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "choices": [
                    {
                        "message": {"content": "Success after retry"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            return response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        node.http_client = mock_client

        contract = ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Test",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state={
                "prompt": "Test prompt",
                "tier": "CLOUD_FAST",
                "max_retries": 3,
            },
            io_operations=[
                {
                    "operation_type": "WRITE",
                    "resource_identifier": "llm_api",
                }
            ],
        )

        response = await node.execute_effect(contract)

        # Verify success after retries
        assert response.generated_text == "Success after retry"
        assert response.retry_count == 2  # 2 retries before success
        assert call_count == 3  # 3 total attempts

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_retry_exhaustion(self):
        """Test failure after retry exhaustion."""
        container = ModelContainer(
            value={
                "zai_api_key": "test_key",
                "retry_initial_backoff_seconds": 0.1,  # Fast retry for testing
            },
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Mock HTTP client - always fail
        async def mock_post(*args, **kwargs):
            response = MagicMock()
            response.status_code = 500
            response.text = "Internal server error"
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=response,
            )
            raise response.raise_for_status.side_effect

        mock_client = AsyncMock()
        mock_client.post = mock_post
        node.http_client = mock_client

        contract = ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Test",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state={
                "prompt": "Test prompt",
                "tier": "CLOUD_FAST",
                "max_retries": 2,
            },
            io_operations=[
                {
                    "operation_type": "WRITE",
                    "resource_identifier": "llm_api",
                }
            ],
        )

        with pytest.raises(ModelOnexError) as exc_info:
            await node.execute_effect(contract)

        assert exc_info.value.error_code == EnumCoreErrorCode.NETWORK_ERROR
        assert "500" in str(exc_info.value.message)

        # Verify circuit breaker recorded failure
        assert node.circuit_breaker.failure_count > 0


class TestNodeLLMEffectCostCalculation:
    """Test cases for cost calculation."""

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_cost_calculation_accuracy(self):
        """Test cost calculation accuracy."""
        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,  # 1K input tokens
                "completion_tokens": 2000,  # 2K output tokens
            },
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        node.http_client = mock_client

        contract = ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            description="Test",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state={
                "prompt": "Test prompt",
                "tier": "CLOUD_FAST",
            },
            io_operations=[
                {
                    "operation_type": "WRITE",
                    "resource_identifier": "llm_api",
                }
            ],
        )

        response = await node.execute_effect(contract)

        # Calculate expected cost
        # CLOUD_FAST: $0.20 per 1M tokens (both input and output)
        expected_cost_input = (1000 / 1_000_000) * 0.20  # $0.0002
        expected_cost_output = (2000 / 1_000_000) * 0.20  # $0.0004
        expected_cost_total = expected_cost_input + expected_cost_output  # $0.0006

        # Verify cost calculation
        assert (
            abs(response.cost_usd - expected_cost_total) < 0.0000001
        )  # Float precision
        assert response.tokens_input == 1000
        assert response.tokens_output == 2000
        assert response.tokens_total == 3000


class TestNodeLLMEffectCleanup:
    """Test cases for resource cleanup."""

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "ZAI_API_KEY": "test_key",
            "ZAI_ENDPOINT": "https://api.z.ai/api/anthropic",
        },
        clear=True,
    )
    async def test_cleanup_closes_http_client(self):
        """Test cleanup properly closes HTTP client."""
        container = ModelContainer(
            value={"zai_api_key": "test_key"},
            container_type="config",
        )

        node = NodeLLMEffect(container)

        # Initialize HTTP client
        await node.initialize()
        assert node.http_client is not None

        # Cleanup
        await node.cleanup()
        assert node.http_client is None
