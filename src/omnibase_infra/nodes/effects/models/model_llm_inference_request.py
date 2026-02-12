# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM inference request model for the inference effect node.

This module provides ModelLlmInferenceRequest, the input model for the LLM
inference effect node. Supports chat completions and legacy completions with
tool calling, generation parameters, resilience settings, and distributed
tracing fields.

Related:
    - ModelLlmMessage: Chat messages contained in the request
    - ModelLlmToolDefinition: Tool definitions for function calling
    - ModelLlmToolChoice: Constraint on tool selection
    - EnumLlmOperationType: Operation type discriminator
    - OMN-2105: Phase 5 LLM inference request model
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.nodes.effects.models.model_llm_message import ModelLlmMessage
from omnibase_infra.nodes.effects.models.model_llm_tool_choice import (
    ModelLlmToolChoice,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_definition import (
    ModelLlmToolDefinition,
)


class ModelLlmInferenceRequest(BaseModel):
    """Input model for the LLM inference effect node.

    Groups fields into logical sections: routing, operation, tool calling,
    generation parameters, streaming, resilience, and tracing. A model
    validator enforces cross-field invariants at construction time.

    Attributes:
        base_url: LLM provider endpoint URL.
        model: Model identifier (e.g. ``'gpt-4'``, ``'qwen2.5-coder-14b'``).
        provider_label: Provider label for observability; not used for routing.
        operation_type: LLM operation category.
        messages: Chat messages; required and non-empty when operation_type
            is CHAT_COMPLETION.
        prompt: Text prompt; required when operation_type is COMPLETION.
        system_prompt: System-level instruction prepended to the conversation.
        tools: Tool definitions available to the model.
        tool_choice: Constraint on tool selection; requires non-empty tools.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        stop: Stop sequences that halt generation.
        stream: Streaming is not supported in v1; type-level guard.
        timeout_seconds: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts on transient failures.
        correlation_id: Caller-provided or auto-generated correlation ID
            for distributed tracing.
        execution_id: Unique identifier for this specific inference call.
        metadata: Arbitrary key-value pairs for observability.

    Example:
        >>> req = ModelLlmInferenceRequest(
        ...     base_url="http://192.168.86.201:8000",
        ...     model="qwen2.5-coder-14b",
        ...     messages=(
        ...         ModelLlmMessage(role="user", content="Write a hello world"),
        ...     ),
        ... )
        >>> req.operation_type
        <EnumLlmOperationType.CHAT_COMPLETION: 'chat_completion'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # -- Routing --

    base_url: str = Field(
        ...,
        min_length=1,
        description="LLM provider endpoint URL.",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier (e.g. 'gpt-4', 'qwen2.5-coder-14b').",
    )
    provider_label: str = Field(
        default="",
        description="Provider label for observability; not used for routing.",
    )

    # -- Operation --

    operation_type: EnumLlmOperationType = Field(
        default=EnumLlmOperationType.CHAT_COMPLETION,
        description="LLM operation category.",
    )
    messages: tuple[ModelLlmMessage, ...] = Field(
        default_factory=tuple,
        description=(
            "Chat messages; required and non-empty when operation_type "
            "is CHAT_COMPLETION."
        ),
    )
    prompt: str | None = Field(
        default=None,
        description="Text prompt; required when operation_type is COMPLETION.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System-level instruction prepended to the conversation.",
    )

    # -- Tool calling --

    tools: tuple[ModelLlmToolDefinition, ...] = Field(
        default_factory=tuple,
        description="Tool definitions available to the model.",
    )
    tool_choice: ModelLlmToolChoice | None = Field(
        default=None,
        description="Constraint on tool selection; requires non-empty tools.",
    )

    # -- Generation --

    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=128_000,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.",
    )
    stop: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Stop sequences that halt generation.",
    )

    # -- Streaming --

    stream: Literal[False] = Field(
        default=False,
        description="Streaming is not supported in v1; type-level guard.",
    )

    # -- Resilience --

    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient failures.",
    )

    # -- Tracing --

    correlation_id: UUID = Field(
        default_factory=uuid4,
        description=(
            "Caller-provided or auto-generated correlation ID for distributed tracing."
        ),
    )
    execution_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this specific inference call.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for observability.",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        """Ensure base_url uses an HTTP(S) scheme."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _validate_request_invariants(self) -> ModelLlmInferenceRequest:
        """Enforce cross-field invariants for the inference request."""
        # -- Operation type consistency --
        if self.operation_type == EnumLlmOperationType.CHAT_COMPLETION:
            if not self.messages:
                raise ValueError(
                    "messages must be non-empty when operation_type is CHAT_COMPLETION."
                )
            if self.prompt is not None:
                raise ValueError(
                    "prompt must be None when operation_type is CHAT_COMPLETION."
                )
        elif self.operation_type == EnumLlmOperationType.COMPLETION:
            if self.prompt is None or not self.prompt.strip():
                raise ValueError(
                    "prompt must be non-None and non-empty when "
                    "operation_type is COMPLETION."
                )
            if self.messages:
                raise ValueError(
                    "messages must be empty when operation_type is COMPLETION."
                )
            if self.system_prompt is not None:
                raise ValueError(
                    "system_prompt must be None when operation_type is COMPLETION."
                )
            if self.tools:
                raise ValueError(
                    "tools must be empty when operation_type is COMPLETION."
                )
            if self.tool_choice is not None:
                raise ValueError(
                    "tool_choice must be None when operation_type is COMPLETION."
                )
        elif self.operation_type == EnumLlmOperationType.EMBEDDING:
            if self.prompt is None or not self.prompt.strip():
                raise ValueError(
                    "prompt must be non-None and non-empty when "
                    "operation_type is EMBEDDING."
                )
            if self.messages:
                raise ValueError(
                    "messages must be empty when operation_type is EMBEDDING."
                )
            if self.tools:
                raise ValueError(
                    "tools must be empty when operation_type is EMBEDDING."
                )
            if self.tool_choice is not None:
                raise ValueError(
                    "tool_choice must be None when operation_type is EMBEDDING."
                )
            if self.system_prompt is not None:
                raise ValueError(
                    "system_prompt must be None when operation_type is EMBEDDING."
                )
        else:
            raise ValueError(f"Unrecognized operation_type: {self.operation_type!r}.")

        # -- Tool consistency --
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tools must be non-empty when tool_choice is set.")

        return self


__all__ = ["ModelLlmInferenceRequest"]
