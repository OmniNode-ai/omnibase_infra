# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM inference response model with text XOR tool_calls invariant.

This module provides ModelLlmInferenceResponse, the output model for the
LLM inference effect node. It enforces a critical invariant: responses
contain either generated text or tool calls, never both.

Empty String Policy:
    - None = no text output (content filter, tool calls only)
    - "" = valid empty completion with STOP
    - Both generated_text=None with tool_calls=() is valid (no output)

Related:
    - ModelLlmToolCall: Individual tool call in the response
    - ModelLlmUsage: Token usage tracking
    - ModelBackendResult: Backend operation result
    - EnumLlmFinishReason: Why generation stopped
    - EnumLlmOperationType: Type of LLM operation
    - OMN-2106: Phase 6 LLM inference response model
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelLlmInferenceResponse(BaseModel):
    """Output model for the LLM inference effect node.

    Captures the complete result of an LLM inference call including generated
    content, token usage, timing, and tracing metadata.

    Invariants:
        - **Text XOR Tool Calls**: A response contains either ``generated_text``
          or ``tool_calls``, never both. Empty string ``""`` counts as text.
        - **Backend Success**: ``backend_result.success`` must be ``True``.
          Failures are raised as exceptions, not encoded in the response.

    Empty String Policy:
        - ``generated_text=None`` means no text output (e.g., content filter,
          tool-call-only response).
        - ``generated_text=""`` means a valid empty completion (e.g., STOP with
          no content).
        - Both ``generated_text=None, tool_calls=()`` is valid (no output at all).
          This represents a legitimate empty response, such as content-filter
          suppression or a model returning no output. Callers should check for
          this state explicitly rather than assuming at least one field is populated.

    Attributes:
        status: Always ``"success"``. Non-success states are exceptions.
        generated_text: Generated text, or ``None`` if no text output.
        tool_calls: Tool calls from the model. Empty tuple if none.
        model_used: Model identifier used for inference.
        provider_id: Provider-specific ID, if available.
        operation_type: Type of LLM operation performed.
        finish_reason: Why generation stopped. Caller sets UNKNOWN if provider omits.
        truncated: Whether output was truncated by max_tokens.
        usage: Token usage summary. Always present; zeros allowed.
        latency_ms: End-to-end latency in milliseconds.
        retry_count: Number of retries before success.
        backend_result: Backend operation result. Must have success=True.
        correlation_id: Distributed trace correlation ID.
        execution_id: Unique execution identifier.
        timestamp: Timezone-aware completion timestamp.

    Example:
        >>> from datetime import datetime, timezone
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
        >>> from omnibase_infra.models.model_backend_result import ModelBackendResult
        >>> from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
        >>> resp = ModelLlmInferenceResponse(
        ...     generated_text="Hello, world!",
        ...     model_used="qwen2.5-coder-14b",
        ...     operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        ...     finish_reason=EnumLlmFinishReason.STOP,
        ...     usage=ModelLlmUsage(tokens_input=10, tokens_output=5),
        ...     latency_ms=150.0,
        ...     backend_result=ModelBackendResult(success=True, duration_ms=145.0),
        ...     correlation_id=uuid4(),
        ...     execution_id=uuid4(),
        ...     timestamp=datetime.now(timezone.utc),
        ... )
        >>> resp.status
        'success'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Status — Literal, not str. Non-success is an exception, not a field value.
    status: Literal["success"] = Field(
        default="success",
        description="Always 'success'. Errors are raised as exceptions.",
    )

    # Content (mutually exclusive — model_validator)
    generated_text: str | None = Field(
        default=None,
        description="Generated text. None = no text output; '' = valid empty completion.",
    )
    tool_calls: tuple[ModelLlmToolCall, ...] = Field(
        default_factory=tuple,
        description="Tool calls from the model. Empty tuple if none.",
    )

    # Model info
    model_used: str = Field(
        ...,
        min_length=1,
        description="Model identifier used for inference.",
    )
    provider_id: str | None = Field(
        default=None,
        description="Provider-specific ID, if available.",
    )

    # Completion metadata
    operation_type: EnumLlmOperationType = Field(
        ...,
        description="Type of LLM operation performed.",
    )
    finish_reason: EnumLlmFinishReason = Field(
        ...,
        description="Why generation stopped. Caller must set UNKNOWN if provider omits.",
    )
    truncated: bool = Field(
        default=False,
        description="Whether output was truncated by max_tokens.",
    )

    # Tokens
    usage: ModelLlmUsage = Field(
        ...,
        description="Token usage summary. Always present; zeros allowed.",
    )

    # Timing
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end latency in milliseconds.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries before success.",
    )

    # Backend
    backend_result: ModelBackendResult = Field(
        ...,
        description="Backend operation result. Must have success=True.",
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description="Distributed trace correlation ID.",
    )
    execution_id: UUID = Field(
        ...,
        description="Unique execution identifier.",
    )
    timestamp: datetime = Field(
        ...,
        description="Timezone-aware completion timestamp.",
    )

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime) -> datetime:
        """Reject naive datetimes and preserve tzinfo."""
        return validate_timezone_aware_datetime(v)

    @model_validator(mode="after")
    def _enforce_text_xor_tool_calls(self) -> Self:
        """Enforce mutual exclusivity of text and tool call content.

        A response contains either generated text or tool calls, never both.
        Empty string ``""`` counts as having text (valid empty completion).
        Both ``None`` text and empty tool_calls is allowed (no output).
        """
        has_text = self.generated_text is not None  # "" is valid text
        has_tools = len(self.tool_calls) > 0
        if has_text and has_tools:
            msg = "Response cannot have both generated_text and tool_calls"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _enforce_backend_success(self) -> Self:
        """Enforce that backend_result reports success.

        Non-success backend results should be raised as exceptions by the
        handler, not encoded in the response model.
        """
        if not self.backend_result.success:
            msg = "backend_result.success must be True; errors are raised as exceptions"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _enforce_finish_reason_error_consistency(self) -> Self:
        """Reject finish_reason=ERROR when backend reports success.

        If the model's finish reason indicates an error, the backend result
        cannot simultaneously report success -- these are semantically
        contradictory.
        """
        if (
            self.finish_reason == EnumLlmFinishReason.ERROR
            and self.backend_result.success
        ):
            msg = (
                "finish_reason=ERROR is contradictory with backend_result.success=True; "
                "error responses must not be encoded as successful"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _enforce_truncated_finish_reason_consistency(self) -> Self:
        """Reject contradictory truncated/finish_reason combinations.

        - truncated=True with finish_reason=STOP is invalid: output cannot
          be truncated if generation stopped naturally.
        - truncated=False with finish_reason=LENGTH is invalid: a length-
          limited response is by definition truncated.
        """
        if self.truncated and self.finish_reason == EnumLlmFinishReason.STOP:
            msg = (
                "truncated=True is contradictory with finish_reason=STOP; "
                "output cannot be truncated if generation completed naturally"
            )
            raise ValueError(msg)
        if not self.truncated and self.finish_reason == EnumLlmFinishReason.LENGTH:
            msg = (
                "truncated=False is contradictory with finish_reason=LENGTH; "
                "length-limited output is by definition truncated"
            )
            raise ValueError(msg)
        return self


__all__: list[str] = ["ModelLlmInferenceResponse"]
