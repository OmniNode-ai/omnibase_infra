"""
LLM client implementations for AI Quorum.

Abstract client and concrete implementations for Gemini, GLM, and Codestral.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import aiohttp

from omninode_bridge.agents.workflows.quorum_models import ValidationContext

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract LLM client for model-specific implementations.

    All concrete clients must implement validate_code method.
    """

    def __init__(
        self,
        model_id: str,
        model_name: str,
        endpoint: str,
        api_key_env: str,
        timeout: int = 30,
        max_retries: int = 2,
    ):
        """
        Initialize LLM client.

        Args:
            model_id: Unique model identifier
            model_name: Full model name
            endpoint: API endpoint URL
            api_key_env: Environment variable name for API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.model_id = model_id
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize client (load API key, create session)."""
        self._api_key = os.getenv(self.api_key_env)
        if not self._api_key:
            raise ValueError(f"API key not found in environment: {self.api_key_env}")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        logger.info(f"Initialized LLM client: {self.model_id}")

    async def close(self) -> None:
        """Close client session."""
        if self._session:
            await self._session.close()
            self._session = None

    @abstractmethod
    async def validate_code(
        self, code: str, context: ValidationContext
    ) -> Tuple[bool, float, str]:
        """
        Validate code and return vote.

        Args:
            code: Code to validate
            context: Validation context with node type, contract, etc.

        Returns:
            Tuple of (vote: bool, confidence: float, reasoning: str)

        Raises:
            Exception: If validation fails
        """
        pass

    def _build_validation_prompt(self, code: str, context: ValidationContext) -> str:
        """
        Build validation prompt for LLM.

        Args:
            code: Code to validate
            context: Validation context

        Returns:
            Formatted prompt string
        """
        criteria_text = "\n".join(
            f"   {i+1}. {criterion}"
            for i, criterion in enumerate(context.validation_criteria or [])
        )

        if not criteria_text:
            criteria_text = """   1. ONEX v2.0 Compliance (ModelOnexError, emit_log_event, async def, EnumLogLevel)
   2. Code Quality (PEP 8, type hints, docstrings, low complexity)
   3. Functional Correctness (implements contract requirements)
   4. Security (no hardcoded secrets, SQL injection, XSS vulnerabilities)
   5. Performance (efficient algorithms, no obvious bottlenecks)"""

        prompt = f"""You are a code quality validator for ONEX v2.0 compliant code generation.

Evaluate the following generated code:

```python
{code}
```

Context:
- Node Type: {context.node_type}
- Contract: {context.contract_summary}

Validation Criteria:
{criteria_text}

Respond with JSON in this exact format (no additional text):
{{
  "vote": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation"
}}"""

        return prompt

    async def _make_request_with_retry(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            payload: Request payload

        Returns:
            Response JSON

        Raises:
            Exception: If all retries fail
        """
        if not self._session:
            raise RuntimeError(f"Client not initialized: {self.model_id}")

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.post(
                    self.endpoint, json=payload, headers=self._get_headers()
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Request failed for {self.model_id}, attempt {attempt+1}/{self.max_retries+1}. "
                        f"Retrying in {wait_time}s. Error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"All retries failed for {self.model_id}. Last error: {e}"
                    )

        raise last_error or Exception(f"Request failed for {self.model_id}")

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """
        Get HTTP headers for API request.

        Returns:
            Headers dictionary
        """
        pass

    @abstractmethod
    def _parse_response(self, response: dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Parse API response and extract vote, confidence, reasoning.

        Args:
            response: API response JSON

        Returns:
            Tuple of (vote: bool, confidence: float, reasoning: str)

        Raises:
            Exception: If response parsing fails
        """
        pass


class GeminiClient(LLMClient):
    """
    Google Gemini client implementation.

    Uses Gemini 1.5 Pro for code validation.
    Weight: 2.0 (highest weight for code quality)
    """

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Gemini API."""
        return {"Content-Type": "application/json"}

    async def validate_code(
        self, code: str, context: ValidationContext
    ) -> Tuple[bool, float, str]:
        """Validate code using Gemini API."""
        prompt = self._build_validation_prompt(code, context)

        # Build Gemini API payload
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,  # Low temperature for consistent validation
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
        }

        # Add API key to URL (Gemini uses URL parameter)
        endpoint_with_key = f"{self.endpoint}?key={self._api_key}"

        # Temporarily override endpoint for this request
        original_endpoint = self.endpoint
        self.endpoint = endpoint_with_key

        try:
            response = await self._make_request_with_retry(payload)
            return self._parse_response(response)
        finally:
            self.endpoint = original_endpoint

    def _parse_response(self, response: dict[str, Any]) -> Tuple[bool, float, str]:
        """Parse Gemini API response."""
        try:
            # Extract text from Gemini response
            candidates = response.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in Gemini response")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise ValueError("No parts in Gemini response")

            text = parts[0].get("text", "")
            if not text:
                raise ValueError("Empty text in Gemini response")

            # Parse JSON from text
            # Handle potential markdown code blocks
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]  # Remove ```json
            if text.startswith("```"):
                text = text[3:]  # Remove ```
            if text.endswith("```"):
                text = text[:-3]  # Remove trailing ```
            text = text.strip()

            result = json.loads(text)

            vote = result.get("vote", False)
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")

            return (vote, confidence, reasoning)

        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            raise


class GLMClient(LLMClient):
    """
    GLM (Zhipu AI) client implementation.

    Supports both GLM-4.5 (weight 2.0) and GLM-Air (weight 1.5).
    """

    def _get_headers(self) -> dict[str, str]:
        """Get headers for GLM API."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def validate_code(
        self, code: str, context: ValidationContext
    ) -> Tuple[bool, float, str]:
        """Validate code using GLM API."""
        prompt = self._build_validation_prompt(code, context)

        # Build GLM API payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code quality validator. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        response = await self._make_request_with_retry(payload)
        return self._parse_response(response)

    def _parse_response(self, response: dict[str, Any]) -> Tuple[bool, float, str]:
        """Parse GLM API response."""
        try:
            # Extract text from GLM response
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in GLM response")

            message = choices[0].get("message", {})
            text = message.get("content", "")
            if not text:
                raise ValueError("Empty content in GLM response")

            # Parse JSON from text
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            vote = result.get("vote", False)
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")

            return (vote, confidence, reasoning)

        except Exception as e:
            logger.error(f"Failed to parse GLM response: {e}")
            raise


class CodestralClient(LLMClient):
    """
    Mistral Codestral client implementation.

    Specialized for code validation.
    Weight: 1.0 (specialized for code)
    """

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Codestral API."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def validate_code(
        self, code: str, context: ValidationContext
    ) -> Tuple[bool, float, str]:
        """Validate code using Codestral API."""
        prompt = self._build_validation_prompt(code, context)

        # Build Codestral API payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code quality validator. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        response = await self._make_request_with_retry(payload)
        return self._parse_response(response)

    def _parse_response(self, response: dict[str, Any]) -> Tuple[bool, float, str]:
        """Parse Codestral API response."""
        try:
            # Extract text from Codestral response (similar to OpenAI format)
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices in Codestral response")

            message = choices[0].get("message", {})
            text = message.get("content", "")
            if not text:
                raise ValueError("Empty content in Codestral response")

            # Parse JSON from text
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            vote = result.get("vote", False)
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")

            return (vote, confidence, reasoning)

        except Exception as e:
            logger.error(f"Failed to parse Codestral response: {e}")
            raise


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.

    Returns configurable responses without making real API calls.
    """

    def __init__(
        self,
        model_id: str,
        model_name: str,
        default_vote: bool = True,
        default_confidence: float = 0.9,
        default_reasoning: str = "Mock validation",
        latency_ms: float = 100.0,
    ):
        """
        Initialize mock client.

        Args:
            model_id: Model identifier
            model_name: Model name
            default_vote: Default vote to return
            default_confidence: Default confidence to return
            default_reasoning: Default reasoning to return
            latency_ms: Simulated latency in milliseconds
        """
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            endpoint="http://mock",
            api_key_env="MOCK_API_KEY",
        )
        self.default_vote = default_vote
        self.default_confidence = default_confidence
        self.default_reasoning = default_reasoning
        self.latency_ms = latency_ms

    async def initialize(self) -> None:
        """Initialize mock client (no-op)."""
        self._api_key = "mock_key"
        logger.info(f"Initialized mock LLM client: {self.model_id}")

    async def close(self) -> None:
        """Close mock client (no-op)."""
        pass

    async def validate_code(
        self, code: str, context: ValidationContext
    ) -> Tuple[bool, float, str]:
        """Return mock validation result."""
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        return (self.default_vote, self.default_confidence, self.default_reasoning)

    def _get_headers(self) -> dict[str, str]:
        """Get mock headers."""
        return {}

    def _parse_response(self, response: dict[str, Any]) -> Tuple[bool, float, str]:
        """Parse mock response."""
        return (self.default_vote, self.default_confidence, self.default_reasoning)
