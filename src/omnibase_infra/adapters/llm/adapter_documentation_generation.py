# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Documentation generation adapter using Qwen-72B.

Generates docstrings, README sections, and API documentation using Qwen-72B
(endpoint :8100). Returns ``ContractDelegatedResponse`` with visible
attribution and a tracked prompt version.

Architecture:
    - Receives a ``task_type`` (docstring, readme, api_doc) and ``source``
      text to document
    - Delegates LLM inference to HandlerLlmOpenaiCompatible via
      TransportHolderLlmHttp pointing at the Qwen-72B endpoint (:8100)
    - Returns ContractDelegatedResponse with attribution metadata

System Prompt Derivation:
    The system prompt is derived from OmniNode platform standards:
    - Python 3.12+ with PEP 604 type unions (``X | Y`` not ``Optional[X]``)
    - Strict, accurate docstrings (Args, Returns, Raises, Example sections)
    - Markdown-formatted README sections with clear structure
    - No backwards-compatibility shims, no over-engineering

Prompt Versioning:
    _PROMPT_VERSION is bumped whenever the system or user prompt templates
    change in a semantically meaningful way.  Callers can inspect
    ``result.attribution.prompt_version`` for reproducibility tracking.

Related Tickets:
    - OMN-2276: Documentation generation handler (this file)
    - OMN-2254: ContractDelegatedResponse output contract
    - OMN-2271: Local model dispatch path
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

# Default model identifier sent to the Qwen-72B endpoint.
_DEFAULT_MODEL: str = "qwen2.5-72b"

# Default maximum tokens for the documentation response.
_DEFAULT_MAX_TOKENS: int = 2_048

# Default temperature -- low for deterministic, high-quality documentation.
_DEFAULT_TEMPERATURE: float = 0.2

# Delegation confidence: documentation is a well-defined task for Qwen-72B.
_DELEGATION_CONFIDENCE: float = 0.95

# Maximum characters of source text sent to the LLM to avoid context overflow.
# Qwen-72B supports large context; cap conservatively at 32 000 chars.
_MAX_SOURCE_CHARS: int = 32_000

# Sentinel appended when source is truncated (included in _MAX_SOURCE_CHARS budget).
_TRUNCATION_SENTINEL: str = "\n... [source truncated]"

# Valid task type values.
TASK_TYPE_DOCSTRING: str = "docstring"
TASK_TYPE_README: str = "readme"
TASK_TYPE_API_DOC: str = "api_doc"
_VALID_TASK_TYPES: frozenset[str] = frozenset(
    {TASK_TYPE_DOCSTRING, TASK_TYPE_README, TASK_TYPE_API_DOC}
)

# System prompt derived from OmniNode CLAUDE.md standards.
_SYSTEM_PROMPT: str = (
    "You are an expert technical writer for the OmniNode platform, a Python-based "
    "infrastructure system. Apply these standards:\n"
    "- Python 3.12+ with PEP 604 type unions (``X | Y``, never ``Optional[X]``)\n"
    "- Accurate, complete docstrings with Args, Returns, Raises, and Example sections\n"
    "- Markdown-formatted output with clear structure and hierarchy\n"
    "- No backwards-compatibility shims, no over-engineering\n"
    "- ONEX architecture: Effect nodes for I/O, Compute for transforms, "
    "Reducer for FSM state, Orchestrator for workflow coordination\n"
    "Produce output that is precise, technically correct, and ready for production use."
)

# User prompt templates per task type.
_DOCSTRING_TEMPLATE: str = """\
Generate a complete Python docstring for the following code. Follow Google-style docstring \
conventions with Args, Returns, Raises, and Example sections where applicable. \
Use PEP 604 type syntax (``X | Y`` not ``Optional[X]``). Return only the docstring \
content (no surrounding code).

## Code

```python
{source}
```
"""

_README_TEMPLATE: str = """\
Generate a README section for the following code or module. Include:
1. **Overview** - What this does and why it exists
2. **Usage** - Concrete example with code
3. **Configuration** - Key parameters and environment variables
4. **Notes** - Important caveats or architectural decisions

## Source

```python
{source}
```
"""

_API_DOC_TEMPLATE: str = """\
Generate API documentation for the following code. Include:
1. **Description** - Purpose and behavior
2. **Parameters** - Each parameter with type, description, and default
3. **Returns** - Return type and description
4. **Raises** - Exceptions that may be raised
5. **Example** - Minimal usage example

Use Markdown formatting. Apply Python 3.12+ type syntax.

## Code

```python
{source}
```
"""

_TASK_TYPE_TEMPLATES: dict[str, str] = {
    TASK_TYPE_DOCSTRING: _DOCSTRING_TEMPLATE,
    TASK_TYPE_README: _README_TEMPLATE,
    TASK_TYPE_API_DOC: _API_DOC_TEMPLATE,
}


class AdapterDocumentationGeneration:
    """Documentation generation adapter using Qwen-72B.

    Generates docstrings, README sections, and API documentation from source
    code or module text.  Returns ``ContractDelegatedResponse`` with visible
    attribution so callers can display model provenance in their output.

    Attributes:
        handler_type: ``INFRA_HANDLER`` -- infrastructure-level handler.
        handler_category: ``EFFECT`` -- performs external I/O (HTTP call).

    Example:
        >>> adapter = AdapterDocumentationGeneration()
        >>> result = await adapter.generate(
        ...     task_type="docstring",
        ...     source="def add(a: int, b: int) -> int:\\n    return a + b",
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
            base_url: Base URL of the Qwen-72B endpoint.  Defaults to the
                ``LLM_QWEN_72B_URL`` environment variable, falling back to
                ``http://localhost:8100``.
            model: Model identifier string sent in inference requests.
            max_tokens: Maximum tokens for the documentation completion.
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

        if base_url is not None and not base_url:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_config",
            )
            raise ProtocolConfigurationError(
                "base_url must be a non-empty string; got an empty string. "
                "Provide a valid URL or set the LLM_QWEN_72B_URL environment variable.",
                context=context,
            )
        self._base_url: str = base_url or os.environ.get(
            "LLM_QWEN_72B_URL", "http://localhost:8100"
        )
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._api_key: str | None = api_key

        self._transport = TransportHolderLlmHttp(
            target_name="qwen-72b-documentation",
            max_timeout_seconds=180.0,
        )
        self._handler = HandlerLlmOpenaiCompatible(self._transport)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: EFFECT (HTTP call to Qwen-72B)."""
        return EnumHandlerTypeCategory.EFFECT

    async def generate(
        self,
        task_type: str,
        source: str,
        correlation_id: UUID | None = None,
    ) -> ContractDelegatedResponse:
        """Generate documentation for the provided source text.

        Selects a task-specific prompt template, calls Qwen-72B, and returns
        a ``ContractDelegatedResponse`` with the rendered documentation and
        full attribution metadata.

        Args:
            task_type: Documentation task.  One of:
                - ``"docstring"`` -- Python docstring for a function/class
                - ``"readme"`` -- README section for a module or component
                - ``"api_doc"`` -- API reference documentation
            source: Source code or descriptive text to document.  Truncated
                to ``_MAX_SOURCE_CHARS`` characters before being sent to the
                model.
            correlation_id: Optional correlation ID for distributed tracing.
                Auto-generated if not provided.

        Returns:
            ``ContractDelegatedResponse`` with:

            - ``rendered_text``: Markdown-formatted documentation output
            - ``attribution``: Provenance metadata including model name,
              endpoint URL, latency, prompt version, and delegation confidence
            - ``structured_json``: ``{"task_type": task_type}`` for downstream
              routing and filtering

        Raises:
            ProtocolConfigurationError: If ``task_type`` is not one of the
                valid task types (``"docstring"``, ``"readme"``, ``"api_doc"``),
                or if ``source`` is empty or contains only whitespace.
            RuntimeHostError: Propagated from ``HandlerLlmOpenaiCompatible``
                on connection failures, timeouts, or authentication errors.
        """
        if task_type not in _VALID_TASK_TYPES:
            valid = ", ".join(sorted(_VALID_TASK_TYPES))
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="generate_documentation",
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
                operation="generate_documentation",
            )
            raise ProtocolConfigurationError(
                "source must not be empty or whitespace-only; "
                "provide the code or text to document.",
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
        )

        try:
            response = await self._handler.handle(request)
        except RuntimeHostError:
            logger.warning(
                "LLM handler raised error during documentation generation",
                extra={"task_type": task_type, "model": self._model},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        rendered = (response.generated_text or "").strip()
        if not rendered:
            rendered = f"## Documentation Unavailable\n\nQwen-72B did not return a response for task_type={task_type!r}."
            logger.warning(
                "Qwen-72B returned empty generated_text for documentation. "
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
            "Documentation generation complete. task_type=%s model=%s "
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


__all__: list[str] = ["AdapterDocumentationGeneration"]
