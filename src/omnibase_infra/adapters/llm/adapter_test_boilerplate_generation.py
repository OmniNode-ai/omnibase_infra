# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Test boilerplate generation adapter using Qwen3-14B-AWQ (Coder-14B).

Generates pytest test modules, class tests, and function tests using
Qwen3-14B-AWQ (endpoint :8001 on .201). Returns ``ContractDelegatedResponse``
with visible attribution and a tracked prompt version.

Architecture:
    - Receives a ``task_type`` (test_module, test_class, test_function) and
      ``source`` code to generate tests for
    - Delegates LLM inference to HandlerLlmOpenaiCompatible via
      TransportHolderLlmHttp pointing at the Coder-14B endpoint (:8001)
    - Returns ContractDelegatedResponse with attribution metadata

System Prompt Derivation:
    The system prompt is derived from OmniNode platform standards:
    - Python 3.12+ with PEP 604 type unions (``X | Y`` not ``Optional[X]``)
    - pytest with @pytest.mark.unit / @pytest.mark.integration decorators
    - Tests live in tests/unit/ or tests/integration/ matching source hierarchy
    - No backwards-compatibility shims, no over-engineering
    - Fixtures in conftest.py, tests import from omnibase_infra

Prompt Versioning:
    _PROMPT_VERSION is bumped whenever the system or user prompt templates
    change in a semantically meaningful way.  Callers can inspect
    ``result.attribution.prompt_version`` for reproducibility tracking.

Related Tickets:
    - OMN-2277: Test boilerplate generation handler (this file)
    - OMN-2254: ContractDelegatedResponse output contract
    - OMN-2271: Local model dispatch path
    - OMN-2276: Documentation generation handler (reference implementation)
    - OMN-2107: HandlerLlmOpenaiCompatible
"""

from __future__ import annotations

import logging
import os
import time
from uuid import UUID

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    TransportHolderLlmHttp,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumLlmOperationType,
)
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_spi.contracts.delegation.contract_delegated_response import (
    ContractDelegatedResponse,
)
from omnibase_spi.contracts.delegation.contract_delegation_attribution import (
    ContractDelegationAttribution,
)

logger = logging.getLogger(__name__)

# Prompt version -- bump when system/user prompt template changes.
_PROMPT_VERSION: str = "v1.0"

# Default model identifier sent to the Coder-14B endpoint.
# Must match the model ID returned by /v1/models (full HuggingFace path).
_DEFAULT_MODEL: str = "Qwen/Qwen3-14B-AWQ"

# Default maximum tokens for the test boilerplate response.
_DEFAULT_MAX_TOKENS: int = 2_048

# Default temperature -- low for deterministic, structured code generation.
_DEFAULT_TEMPERATURE: float = 0.1

# Per-request timeout in seconds.  Qwen3-14B is a fast coding model;
# 90 s gives headroom for a 2 048-token completion without exceeding the
# transport cap of 180 s.
# IMPORTANT: Must not exceed TransportHolderLlmHttp(max_timeout_seconds=180.0).
# If you raise this value, update the transport cap in __init__ accordingly.
_DEFAULT_TIMEOUT_SECONDS: float = 90.0

# Delegation confidence: test generation is a well-defined coding task.
_DELEGATION_CONFIDENCE: float = 0.92

# Maximum characters of source text sent to the LLM to avoid context overflow.
# Qwen3-14B has a 40K token context; cap conservatively at 36 000 chars.
_MAX_SOURCE_CHARS: int = 36_000

# Sentinel appended when source is truncated (included in _MAX_SOURCE_CHARS budget).
_TRUNCATION_SENTINEL: str = "\n... [source truncated]"

# Valid task type values.
TASK_TYPE_TEST_MODULE: str = "test_module"
TASK_TYPE_TEST_CLASS: str = "test_class"
TASK_TYPE_TEST_FUNCTION: str = "test_function"
_VALID_TASK_TYPES: frozenset[str] = frozenset(
    {TASK_TYPE_TEST_MODULE, TASK_TYPE_TEST_CLASS, TASK_TYPE_TEST_FUNCTION}
)

# System prompt derived from OmniNode CLAUDE.md standards and pytest conventions.
_SYSTEM_PROMPT: str = (
    "You are an expert Python test engineer for the OmniNode platform. "
    "Generate pytest test boilerplate that adheres to these standards:\n"
    "- Python 3.12+ with PEP 604 type unions (``X | Y``, never ``Optional[X]``)\n"
    "- Use @pytest.mark.unit for tests in tests/unit/, "
    "@pytest.mark.integration for tests/integration/\n"
    "- Group tests in classes prefixed with ``Test``; methods prefixed with ``test_``\n"
    "- Use ``from __future__ import annotations`` at the top of every test file\n"
    "- Add meaningful assertions -- do NOT just assert True or pass without logic\n"
    "- Include pytest fixtures in conftest.py where appropriate; reference existing "
    "fixtures (mock_container, event_bus) rather than duplicating them\n"
    "- No backwards-compatibility shims, no over-engineering\n"
    "- ONEX architecture: Effect nodes for I/O, Compute for transforms, "
    "Reducer for FSM state, Orchestrator for workflow coordination\n"
    "Produce test code that is syntactically correct, clearly structured, "
    "and ready to run with ``uv run pytest``."
)

# User prompt templates per task type.
_TEST_MODULE_TEMPLATE: str = """\
Generate a complete pytest test module for the following Python source file. Include:
1. Module-level docstring explaining what is being tested
2. ``from __future__ import annotations``
3. Imports (the module under test, pytest, unittest.mock as needed)
4. At least one ``@pytest.mark.unit`` test class per public class or function
5. At least 3 meaningful test methods per class covering: happy path, edge \
cases, and error cases
6. Use ``AsyncMock`` for coroutines and ``MagicMock`` for synchronous dependencies
7. Add a ``# ---------------------------------------------------------------------------``
   separator between test classes

Return only the test file content (no markdown fences, no explanation).

## Source

{source}
"""

_TEST_CLASS_TEMPLATE: str = """\
Generate a pytest test class for the following Python class. Include:
1. ``@pytest.mark.unit`` decorator on the test class
2. A docstring on the test class explaining what is being tested
3. At least 3 test methods: one happy path, one edge case, one error/exception case
4. Use ``AsyncMock`` for async methods and ``MagicMock`` for sync dependencies
5. Assertions must be specific -- use ``==``, ``is``, ``pytest.approx``,
   or ``pytest.raises`` appropriately

Return only the test class code block (no surrounding file boilerplate, no markdown fences).

## Source

{source}
"""

_TEST_FUNCTION_TEMPLATE: str = """\
Generate pytest test methods for the following Python function. Include:
1. At least 3 test methods covering: successful result, boundary/edge inputs, \
and raised exceptions
2. Each test method decorated with ``@pytest.mark.unit``
3. Use ``@pytest.mark.parametrize`` where the same logic applies to multiple inputs
4. Use ``AsyncMock`` if the function is a coroutine
5. Assertions must be specific and meaningful

Return only the test method definitions (no class wrapper, no file boilerplate, no \
markdown fences).

## Source

{source}
"""

_TASK_TYPE_TEMPLATES: dict[str, str] = {
    TASK_TYPE_TEST_MODULE: _TEST_MODULE_TEMPLATE,
    TASK_TYPE_TEST_CLASS: _TEST_CLASS_TEMPLATE,
    TASK_TYPE_TEST_FUNCTION: _TEST_FUNCTION_TEMPLATE,
}


class AdapterTestBoilerplateGeneration:
    """Test boilerplate generation adapter using Qwen3-14B-AWQ (Coder-14B).

    Generates pytest skeletons -- full test modules, test classes, and test
    function stubs -- from Python source code.  Returns
    ``ContractDelegatedResponse`` with visible attribution so callers can
    display model provenance in their output.

    Attributes:
        handler_type: ``INFRA_HANDLER`` -- infrastructure-level handler.
        handler_category: ``EFFECT`` -- performs external I/O (HTTP call).

    Example:
        >>> adapter = AdapterTestBoilerplateGeneration()
        >>> result = await adapter.generate(
        ...     task_type="test_module",
        ...     source="class MyService:\\n    def run(self) -> bool:\\n        ...",
        ... )
        >>> print(result.rendered_text)
        >>> print(result.attribution.model_name)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            base_url: Base URL of the Coder-14B endpoint.  Defaults to the
                ``LLM_CODER_FAST_URL`` environment variable, falling back to
                ``http://localhost:8001``.
            model: Model identifier string sent in inference requests.
            max_tokens: Maximum tokens for the test boilerplate completion.
            temperature: Sampling temperature (lower = more deterministic).
            api_key: Optional Bearer token for authenticated endpoints.

        Raises:
            ProtocolConfigurationError: If ``max_tokens`` is not in [1, 32768],
                ``temperature`` is not in [0.0, 2.0], or ``base_url``
                resolves to an empty string.
        """
        if max_tokens <= 0 or max_tokens > 32_768:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_config",
            )
            raise ProtocolConfigurationError(
                f"max_tokens must be in [1, 32768], got {max_tokens}",
                context=context,
            )
        if not (0.0 <= temperature <= 2.0):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_config",
            )
            raise ProtocolConfigurationError(
                f"temperature must be in [0.0, 2.0], got {temperature}",
                context=context,
            )

        if base_url is not None and not base_url.strip():
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_config",
            )
            raise ProtocolConfigurationError(
                "base_url must be a non-empty string; got an empty string. "
                "Provide a valid URL or set the LLM_CODER_FAST_URL environment variable.",
                context=context,
            )
        self._base_url: str = base_url or os.environ.get(
            "LLM_CODER_FAST_URL", "http://localhost:8001"
        )
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._api_key: str | None = api_key

        self._transport = TransportHolderLlmHttp(
            target_name="qwen3-14b-test-boilerplate",
            max_timeout_seconds=180.0,
        )
        self._handler = HandlerLlmOpenaiCompatible(self._transport)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: EFFECT (HTTP call to Coder-14B)."""
        return EnumHandlerTypeCategory.EFFECT

    async def generate(
        self,
        task_type: str,
        source: str,
        correlation_id: UUID | None = None,
    ) -> ContractDelegatedResponse:
        """Generate test boilerplate for the provided source code.

        Selects a task-specific prompt template, calls Qwen3-14B-AWQ, and
        returns a ``ContractDelegatedResponse`` with the generated test
        skeleton and full attribution metadata.

        Args:
            task_type: Test generation task.  One of:
                - ``"test_module"`` -- Full pytest test module for a source file
                - ``"test_class"`` -- Test class for a specific Python class
                - ``"test_function"`` -- Test methods for a specific function
            source: Source code to generate tests for.  Truncated to
                ``_MAX_SOURCE_CHARS`` characters before being sent to the model.
            correlation_id: Optional correlation ID for distributed tracing.
                Forwarded to error context only; not auto-generated.

        Returns:
            ``ContractDelegatedResponse`` with:

            - ``rendered_text``: Generated pytest boilerplate (Python source)
            - ``attribution``: Provenance metadata including model name,
              endpoint URL, latency, prompt version, and delegation confidence
            - ``structured_json``: ``{"task_type": task_type}`` for downstream
              routing and filtering

        Raises:
            ProtocolConfigurationError: If ``task_type`` is not one of the
                valid task types (``"test_module"``, ``"test_class"``,
                ``"test_function"``), or if ``source`` is empty or whitespace.
            RuntimeHostError: Propagated from ``HandlerLlmOpenaiCompatible``
                on connection failures, timeouts, or authentication errors.
        """
        if task_type not in _VALID_TASK_TYPES:
            valid = ", ".join(sorted(_VALID_TASK_TYPES))
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="generate_test_boilerplate",
            )
            raise ProtocolConfigurationError(
                f"Invalid task_type {task_type!r}. Must be one of: {valid}",
                context=context,
            )

        source_stripped = source.strip()
        if not source_stripped:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="generate_test_boilerplate",
            )
            raise ProtocolConfigurationError(
                "source must not be empty or whitespace-only; "
                "provide the source code to generate tests for.",
                context=context,
            )

        start = time.perf_counter()
        if len(source_stripped) > _MAX_SOURCE_CHARS:
            logger.debug(
                "Truncating source from %d to %d chars to fit context window.",
                len(source_stripped),
                _MAX_SOURCE_CHARS,
            )
            source_stripped = (
                source_stripped[: _MAX_SOURCE_CHARS - len(_TRUNCATION_SENTINEL)]
                + _TRUNCATION_SENTINEL
            )

        template = _TASK_TYPE_TEMPLATES[task_type]

        # Build user message by splitting on placeholder -- avoids str.format()
        # issues when source_stripped itself contains curly braces (e.g. JSON,
        # Python dicts, f-strings).
        parts = template.split("{source}", 1)
        if len(parts) == 2:
            user_message = parts[0] + source_stripped + parts[1]
        else:
            # Fallback: placeholder absent (should never happen, but degrade gracefully).
            user_message = template + "\n\n" + source_stripped

        request = ModelLlmInferenceRequest(
            base_url=self._base_url,
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            model=self._model,
            messages=({"role": "user", "content": user_message},),
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            api_key=self._api_key,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

        try:
            response = await self._handler.handle(request)
        except RuntimeHostError:
            logger.warning(
                "LLM handler raised error during test boilerplate generation",
                extra={"task_type": task_type, "model": self._model},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        rendered = (response.generated_text or "").strip()
        if not rendered:
            rendered = (
                f"# Test Boilerplate Unavailable\n\n"
                f"# Qwen3-14B did not return a response for task_type={task_type!r}.\n"
                f"# Re-run or check endpoint: {self._base_url}\n"
            )
            logger.warning(
                "Qwen3-14B returned empty generated_text for test boilerplate. "
                "task_type=%s model=%s latency_ms=%.1f",
                task_type,
                self._model,
                latency_ms,
            )

        attribution = ContractDelegationAttribution(
            model_name=self._model,
            endpoint_url=self._base_url,
            latency_ms=latency_ms,
            prompt_version=_PROMPT_VERSION,
            delegation_confidence=_DELEGATION_CONFIDENCE,
        )

        logger.debug(
            "Test boilerplate generation complete. task_type=%s model=%s "
            "latency_ms=%.1f rendered_chars=%d",
            task_type,
            self._model,
            latency_ms,
            len(rendered),
        )

        return ContractDelegatedResponse(
            rendered_text=rendered,
            attribution=attribution,
            structured_json={"task_type": task_type},
        )

    async def close(self) -> None:
        """Close the HTTP transport client."""
        await self._transport.close()


__all__: list[str] = ["AdapterTestBoilerplateGeneration"]
