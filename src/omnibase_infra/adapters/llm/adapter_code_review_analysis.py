# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Code review analysis adapter using Qwen3-14B-AWQ (Coder-14B).

Reviews Python code diffs for naming conventions, docstring quality, and type
annotation coverage.  Returns ``ContractDelegatedResponse`` with visible
attribution and a tracked prompt version.

Architecture:
    - Receives a ``review_type`` (naming, docstrings, types) and
      ``code_diff`` text to review
    - Delegates LLM inference to HandlerLlmOpenaiCompatible via
      TransportHolderLlmHttp pointing at the Coder-14B endpoint (:8001)
    - Returns ContractDelegatedResponse with attribution metadata

System Prompt Derivation:
    The system prompt is derived from OmniNode platform standards:
    - Python 3.12+ with PEP 604 type unions (``X | Y`` not ``Optional[X]``)
    - Strict naming conventions (snake_case, PascalCase, UPPER_CASE)
    - All public functions/classes/methods must have complete docstrings
    - No backwards-compatibility shims, no over-engineering
    - ONEX architecture: Effect/Compute/Reducer/Orchestrator node types

Prompt Versioning:
    _PROMPT_VERSION is bumped whenever the system or user prompt templates
    change in a semantically meaningful way.  Callers can inspect
    ``result.attribution.prompt_version`` for reproducibility tracking.

Related Tickets:
    - OMN-2278: Code review analysis handler (this file)
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

# Default model identifier sent to the Coder-14B endpoint.
# Must match the model ID returned by /v1/models.
_DEFAULT_MODEL: str = "qwen2.5-coder-14b-instruct"

# Default maximum tokens for the review response.
_DEFAULT_MAX_TOKENS: int = 1_024

# Default temperature -- low for deterministic, high-quality review output.
_DEFAULT_TEMPERATURE: float = 0.1

# Per-request timeout in seconds.  Coder-14B is fast; 90 s gives headroom.
_DEFAULT_TIMEOUT_SECONDS: float = 90.0

# Delegation confidence: code review is a well-defined task for Coder-14B.
_DELEGATION_CONFIDENCE: float = 0.92

# Maximum characters of code diff sent to the LLM to avoid context overflow.
# Coder-14B supports 40K context; cap conservatively at 32 000 chars.
_MAX_DIFF_CHARS: int = 32_000

# Sentinel appended when diff is truncated (included in _MAX_DIFF_CHARS budget).
_TRUNCATION_SENTINEL: str = "\n... [diff truncated]"

# Valid review type values.
REVIEW_TYPE_NAMING: str = "naming"
REVIEW_TYPE_DOCSTRINGS: str = "docstrings"
REVIEW_TYPE_TYPES: str = "types"
_VALID_REVIEW_TYPES: frozenset[str] = frozenset(
    {REVIEW_TYPE_NAMING, REVIEW_TYPE_DOCSTRINGS, REVIEW_TYPE_TYPES}
)

# System prompt derived from OmniNode CLAUDE.md standards.
_SYSTEM_PROMPT: str = (
    "You are an expert Python code reviewer specializing in the OmniNode platform.\n"
    "Review code strictly according to these standards:\n"
    "- Python 3.12+ with PEP 604 type unions (X | Y, never Optional[X])\n"
    "- All public functions/classes/methods must have complete docstrings\n"
    "- Naming: snake_case for variables/functions, PascalCase for classes, "
    "UPPER_CASE for constants\n"
    "- No backwards-compatibility shims, no over-engineering\n"
    "- ONEX architecture: Effect nodes for I/O, Compute for transforms, "
    "Reducer for FSM state\n"
    "Provide actionable, specific feedback with file/line references where possible.\n"
    "Format output as Markdown with clear sections."
)

# User prompt templates per review type.
_NAMING_TEMPLATE: str = """\
Review the following code diff for naming convention issues. Check:
1. Variables and functions: snake_case
2. Classes: PascalCase
3. Constants: UPPER_CASE
4. Module names: lowercase_with_underscores
5. Prefix conventions: Model*, Enum*, Protocol*, Handler*, Adapter*, Service*

For each issue found, provide: location, current name, suggested name, reason.
If no issues, state "No naming issues found."

## Code Diff

```diff
{code_diff}
```
"""

_DOCSTRINGS_TEMPLATE: str = """\
Review the following code diff for docstring quality. Check:
1. All public functions, methods, and classes have docstrings
2. Docstrings include Args, Returns, Raises, and Example sections where applicable
3. Type annotations in docstrings match the function signature (PEP 604: X | Y syntax)
4. Docstrings are accurate and not placeholder/boilerplate

For each issue found, provide: location, issue type, suggested improvement.
If no issues, state "No docstring issues found."

## Code Diff

```diff
{code_diff}
```
"""

_TYPES_TEMPLATE: str = """\
Review the following code diff for type annotation quality. Check:
1. All function parameters and return types are annotated
2. Uses PEP 604 union syntax: X | Y (never Optional[X] or Union[X, Y])
3. No use of Any without justification
4. Pydantic models use ConfigDict(frozen=True, extra="forbid")
5. Collections use list[T] / dict[K, V] / tuple[T, ...] (not List[T] etc.)

For each issue found, provide: location, current annotation, suggested annotation, reason.
If no issues, state "No type annotation issues found."

## Code Diff

```diff
{code_diff}
```
"""

_REVIEW_TYPE_TEMPLATES: dict[str, str] = {
    REVIEW_TYPE_NAMING: _NAMING_TEMPLATE,
    REVIEW_TYPE_DOCSTRINGS: _DOCSTRINGS_TEMPLATE,
    REVIEW_TYPE_TYPES: _TYPES_TEMPLATE,
}


class AdapterCodeReviewAnalysis:
    """Code review analysis adapter using Qwen3-14B-AWQ (Coder-14B).

    Reviews Python code diffs for naming conventions, docstring quality,
    and type annotation coverage.  Returns ``ContractDelegatedResponse``
    with visible attribution so callers can display model provenance.

    Attributes:
        handler_type: ``INFRA_HANDLER`` -- infrastructure-level handler.
        handler_category: ``EFFECT`` -- performs external I/O (HTTP call).

    Example:
        >>> adapter = AdapterCodeReviewAnalysis()
        >>> result = await adapter.review(
        ...     review_type="naming",
        ...     code_diff="- def Foo():\\n+ def foo():\\n",
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
            max_tokens: Maximum tokens for the review completion.
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
            target_name="coder-14b-code-review",
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

    async def review(
        self,
        review_type: str,
        code_diff: str,
        correlation_id: UUID | None = None,
    ) -> ContractDelegatedResponse:
        """Review a code diff for the specified review category.

        Selects a review-type-specific prompt template, calls Coder-14B, and
        returns a ``ContractDelegatedResponse`` with the rendered review and
        full attribution metadata.

        Args:
            review_type: Review category.  One of:
                - ``"naming"`` -- naming convention issues (snake_case,
                  PascalCase, UPPER_CASE, prefix conventions)
                - ``"docstrings"`` -- docstring presence and quality
                - ``"types"`` -- type annotation coverage and PEP 604 style
            code_diff: The code diff or code snippet to review.  Truncated
                to ``_MAX_DIFF_CHARS`` characters before being sent to the
                model.
            correlation_id: Optional correlation ID for distributed tracing.
                Forwarded to error context only; not auto-generated.

        Returns:
            ``ContractDelegatedResponse`` with:

            - ``rendered_text``: Markdown-formatted review output
            - ``attribution``: Provenance metadata including model name,
              endpoint URL, latency, prompt version, and delegation confidence
            - ``structured_json``: ``{"review_type": review_type}`` for
              downstream routing and filtering

        Raises:
            ProtocolConfigurationError: If ``review_type`` is not one of the
                valid review types (``"naming"``, ``"docstrings"``,
                ``"types"``), or if ``code_diff`` is empty or contains only
                whitespace.
            RuntimeHostError: Propagated from ``HandlerLlmOpenaiCompatible``
                on connection failures, timeouts, or authentication errors.
        """
        if review_type not in _VALID_REVIEW_TYPES:
            valid = ", ".join(sorted(_VALID_REVIEW_TYPES))
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="review_code",
            )
            raise ProtocolConfigurationError(
                f"Invalid review_type {review_type!r}. Must be one of: {valid}",
                context=context,
            )

        diff_stripped = code_diff.strip()
        if not diff_stripped:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="review_code",
            )
            raise ProtocolConfigurationError(
                "code_diff must not be empty or whitespace-only; "
                "provide the code diff or snippet to review.",
                context=context,
            )

        start = time.perf_counter()
        if len(diff_stripped) > _MAX_DIFF_CHARS:
            logger.debug(
                "Truncating code_diff from %d to %d chars to fit context window.",
                len(diff_stripped),
                _MAX_DIFF_CHARS,
            )
            diff_stripped = (
                diff_stripped[: _MAX_DIFF_CHARS - len(_TRUNCATION_SENTINEL)]
                + _TRUNCATION_SENTINEL
            )

        template = _REVIEW_TYPE_TEMPLATES[review_type]

        # Build user message by splitting on placeholder -- avoids str.format()
        # issues when diff_stripped itself contains curly braces (e.g. JSON,
        # Python dicts, f-strings).
        parts = template.split("{code_diff}", 1)
        if len(parts) == 2:
            user_message = parts[0] + diff_stripped + parts[1]
        else:
            # Fallback: placeholder absent (should never happen, but degrade gracefully).
            user_message = template + "\n\n" + diff_stripped

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
                "LLM handler raised error during code review analysis",
                extra={"review_type": review_type, "model": self._model},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        rendered = (response.generated_text or "").strip()
        if not rendered:
            rendered = f"## Review Unavailable\n\nCoder-14B did not return a response for review_type={review_type!r}."
            logger.warning(
                "Coder-14B returned empty generated_text for code review. "
                "review_type=%s model=%s latency_ms=%.1f",
                review_type,
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
            "Code review analysis complete. review_type=%s model=%s "
            "latency_ms=%.1f rendered_chars=%d",
            review_type,
            self._model,
            latency_ms,
            len(rendered),
        )

        return ContractDelegatedResponse(
            rendered_text=rendered,
            attribution=attribution,
            structured_json={"review_type": review_type},
        )

    async def close(self) -> None:
        """Close the HTTP transport client."""
        await self._transport.close()


__all__: list[str] = ["AdapterCodeReviewAnalysis"]
