"""
Unit tests for LLM client implementations.

Tests abstract client and concrete implementations (Gemini, GLM, Codestral).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omninode_bridge.agents.workflows.llm_client import (
    CodestralClient,
    GeminiClient,
    GLMClient,
    LLMClient,
    MockLLMClient,
)
from omninode_bridge.agents.workflows.quorum_models import ValidationContext


@pytest.fixture
def validation_context():
    """Sample validation context."""
    return ValidationContext(
        node_type="effect",
        contract_summary="Test effect node contract",
        validation_criteria=[
            "ONEX v2.0 compliance",
            "Code quality",
            "Security",
        ],
    )


@pytest.fixture
def sample_code():
    """Sample code for validation."""
    return """
async def execute_effect(input_data: Dict[str, Any]) -> Dict[str, Any]:
    '''Test effect node.'''
    emit_log_event(EnumLogLevel.INFO, "Processing effect")
    return {"status": "success"}
"""


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_mock_client_initialization(self):
        """Test mock client initialization."""
        client = MockLLMClient(
            model_id="test",
            model_name="test-model",
            default_vote=True,
            default_confidence=0.9,
            latency_ms=50.0,
        )

        await client.initialize()

        assert client.model_id == "test"
        assert client.model_name == "test-model"
        assert client.default_vote is True
        assert client.default_confidence == 0.9
        assert client.latency_ms == 50.0

    @pytest.mark.asyncio
    async def test_mock_client_validate_code(self, validation_context, sample_code):
        """Test mock client validate_code."""
        client = MockLLMClient(
            model_id="test",
            model_name="test-model",
            default_vote=True,
            default_confidence=0.85,
            default_reasoning="Mock validation passed",
            latency_ms=100.0,
        )

        await client.initialize()

        import time

        start = time.perf_counter()
        vote, confidence, reasoning = await client.validate_code(
            sample_code, validation_context
        )
        duration = (time.perf_counter() - start) * 1000

        assert vote is True
        assert confidence == 0.85
        assert reasoning == "Mock validation passed"
        assert duration >= 100.0  # Should respect latency

    @pytest.mark.asyncio
    async def test_mock_client_close(self):
        """Test mock client close."""
        client = MockLLMClient(model_id="test", model_name="test-model")
        await client.initialize()
        await client.close()  # Should not raise


class TestLLMClientBase:
    """Tests for LLMClient abstract base class."""

    def test_build_validation_prompt_default(self, sample_code):
        """Test building validation prompt with default criteria."""
        # Create context without custom criteria to use defaults
        context = ValidationContext(
            node_type="effect",
            contract_summary="Test contract",
            validation_criteria=[],  # Empty to trigger default criteria
        )

        client = MockLLMClient(model_id="test", model_name="test-model")

        prompt = client._build_validation_prompt(sample_code, context)

        # Check for default criteria (case-insensitive)
        prompt_lower = prompt.lower()
        assert "onex v2.0 compliance" in prompt_lower
        assert "code quality" in prompt_lower
        assert "functional correctness" in prompt_lower
        assert "security" in prompt_lower
        assert "performance" in prompt_lower
        assert sample_code in prompt
        assert context.node_type in prompt
        assert context.contract_summary in prompt

    def test_build_validation_prompt_custom_criteria(self, sample_code):
        """Test building validation prompt with custom criteria."""
        context = ValidationContext(
            node_type="compute",
            contract_summary="Custom contract",
            validation_criteria=[
                "Custom criterion 1",
                "Custom criterion 2",
            ],
        )

        client = MockLLMClient(model_id="test", model_name="test-model")

        prompt = client._build_validation_prompt(sample_code, context)

        assert "Custom criterion 1" in prompt
        assert "Custom criterion 2" in prompt

    @pytest.mark.asyncio
    async def test_client_initialization_missing_api_key(self):
        """Test client initialization with missing API key."""

        class TestClient(LLMClient):
            async def validate_code(self, code, context):
                return (True, 0.9, "Test")

            def _get_headers(self):
                return {}

            def _parse_response(self, response):
                return (True, 0.9, "Test")

        client = TestClient(
            model_id="test",
            model_name="test-model",
            endpoint="http://test",
            api_key_env="NONEXISTENT_KEY",
        )

        with pytest.raises(ValueError, match="API key not found"):
            await client.initialize()


class TestGeminiClient:
    """Tests for GeminiClient."""

    @pytest.mark.asyncio
    async def test_gemini_client_initialization(self):
        """Test Gemini client initialization."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            client = GeminiClient(
                model_id="gemini",
                model_name="gemini-1.5-pro",
                endpoint="https://api.gemini.com",
                api_key_env="GEMINI_API_KEY",
            )

            await client.initialize()

            assert client._api_key == "test_key"
            assert client._session is not None

            await client.close()

    @pytest.mark.asyncio
    async def test_gemini_client_validate_code_success(
        self, validation_context, sample_code
    ):
        """Test Gemini client successful validation."""
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"vote": true, "confidence": 0.92, "reasoning": "Code meets all criteria"}'
                            }
                        ]
                    }
                }
            ]
        }

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            client = GeminiClient(
                model_id="gemini",
                model_name="gemini-1.5-pro",
                endpoint="https://api.gemini.com",
                api_key_env="GEMINI_API_KEY",
            )

            await client.initialize()

            # Mock HTTP request
            client._make_request_with_retry = AsyncMock(return_value=mock_response)

            vote, confidence, reasoning = await client.validate_code(
                sample_code, validation_context
            )

            assert vote is True
            assert confidence == 0.92
            assert reasoning == "Code meets all criteria"

            await client.close()

    @pytest.mark.asyncio
    async def test_gemini_client_parse_response_with_markdown(self):
        """Test Gemini client parsing response with markdown code blocks."""
        client = GeminiClient(
            model_id="gemini",
            model_name="gemini-1.5-pro",
            endpoint="https://api.gemini.com",
            api_key_env="GEMINI_API_KEY",
        )

        # Response with markdown code blocks
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '```json\n{"vote": true, "confidence": 0.88, "reasoning": "Good code"}\n```'
                            }
                        ]
                    }
                }
            ]
        }

        vote, confidence, reasoning = client._parse_response(response)

        assert vote is True
        assert confidence == 0.88
        assert reasoning == "Good code"

    @pytest.mark.asyncio
    async def test_gemini_client_parse_response_invalid(self):
        """Test Gemini client parsing invalid response."""
        client = GeminiClient(
            model_id="gemini",
            model_name="gemini-1.5-pro",
            endpoint="https://api.gemini.com",
            api_key_env="GEMINI_API_KEY",
        )

        # Empty candidates
        response = {"candidates": []}

        with pytest.raises(ValueError, match="No candidates"):
            client._parse_response(response)


class TestGLMClient:
    """Tests for GLMClient."""

    @pytest.mark.asyncio
    async def test_glm_client_initialization(self):
        """Test GLM client initialization."""
        with patch.dict("os.environ", {"GLM_API_KEY": "test_key"}):
            client = GLMClient(
                model_id="glm-4.5",
                model_name="glm-4-plus",
                endpoint="https://api.glm.com",
                api_key_env="GLM_API_KEY",
            )

            await client.initialize()

            assert client._api_key == "test_key"
            assert client._session is not None

            await client.close()

    @pytest.mark.asyncio
    async def test_glm_client_validate_code_success(
        self, validation_context, sample_code
    ):
        """Test GLM client successful validation."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"vote": false, "confidence": 0.75, "reasoning": "Code has issues"}'
                    }
                }
            ]
        }

        with patch.dict("os.environ", {"GLM_API_KEY": "test_key"}):
            client = GLMClient(
                model_id="glm-4.5",
                model_name="glm-4-plus",
                endpoint="https://api.glm.com",
                api_key_env="GLM_API_KEY",
            )

            await client.initialize()

            # Mock HTTP request
            client._make_request_with_retry = AsyncMock(return_value=mock_response)

            vote, confidence, reasoning = await client.validate_code(
                sample_code, validation_context
            )

            assert vote is False
            assert confidence == 0.75
            assert reasoning == "Code has issues"

            await client.close()

    @pytest.mark.asyncio
    async def test_glm_client_get_headers(self):
        """Test GLM client headers."""
        with patch.dict("os.environ", {"GLM_API_KEY": "test_key"}):
            client = GLMClient(
                model_id="glm-4.5",
                model_name="glm-4-plus",
                endpoint="https://api.glm.com",
                api_key_env="GLM_API_KEY",
            )

            await client.initialize()

            headers = client._get_headers()

            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test_key"
            assert headers["Content-Type"] == "application/json"

            await client.close()

    @pytest.mark.asyncio
    async def test_glm_client_parse_response_with_markdown(self):
        """Test GLM client parsing response with markdown."""
        client = GLMClient(
            model_id="glm",
            model_name="glm-model",
            endpoint="https://api.glm.com",
            api_key_env="GLM_KEY",
        )

        response = {
            "choices": [
                {
                    "message": {
                        "content": "```\n{\"vote\": true, \"confidence\": 0.9, \"reasoning\": \"OK\"}\n```"
                    }
                }
            ]
        }

        vote, confidence, reasoning = client._parse_response(response)

        assert vote is True
        assert confidence == 0.9
        assert reasoning == "OK"


class TestCodestralClient:
    """Tests for CodestralClient."""

    @pytest.mark.asyncio
    async def test_codestral_client_initialization(self):
        """Test Codestral client initialization."""
        with patch.dict("os.environ", {"CODESTRAL_API_KEY": "test_key"}):
            client = CodestralClient(
                model_id="codestral",
                model_name="codestral-latest",
                endpoint="https://api.mistral.ai",
                api_key_env="CODESTRAL_API_KEY",
            )

            await client.initialize()

            assert client._api_key == "test_key"
            assert client._session is not None

            await client.close()

    @pytest.mark.asyncio
    async def test_codestral_client_validate_code_success(
        self, validation_context, sample_code
    ):
        """Test Codestral client successful validation."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"vote": true, "confidence": 0.87, "reasoning": "Code is well-structured"}'
                    }
                }
            ]
        }

        with patch.dict("os.environ", {"CODESTRAL_API_KEY": "test_key"}):
            client = CodestralClient(
                model_id="codestral",
                model_name="codestral-latest",
                endpoint="https://api.mistral.ai",
                api_key_env="CODESTRAL_API_KEY",
            )

            await client.initialize()

            # Mock HTTP request
            client._make_request_with_retry = AsyncMock(return_value=mock_response)

            vote, confidence, reasoning = await client.validate_code(
                sample_code, validation_context
            )

            assert vote is True
            assert confidence == 0.87
            assert reasoning == "Code is well-structured"

            await client.close()

    @pytest.mark.asyncio
    async def test_codestral_client_get_headers(self):
        """Test Codestral client headers."""
        with patch.dict("os.environ", {"CODESTRAL_API_KEY": "test_key"}):
            client = CodestralClient(
                model_id="codestral",
                model_name="codestral-latest",
                endpoint="https://api.mistral.ai",
                api_key_env="CODESTRAL_API_KEY",
            )

            await client.initialize()

            headers = client._get_headers()

            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test_key"

            await client.close()


class TestRetryLogic:
    """Tests for retry logic in LLM clients."""

    @pytest.mark.asyncio
    async def test_retry_logic_basic(self):
        """
        Test retry logic is configured correctly.

        Note: Detailed retry testing is complex with async mocking.
        The retry logic is tested indirectly through integration tests.
        """
        client = MockLLMClient(
            model_id="test", model_name="test-model", latency_ms=10
        )
        await client.initialize()

        # Verify retry configuration
        assert client.max_retries == 2
        assert client.timeout == 30

        await client.close()
