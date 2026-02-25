# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM inference request model for the LLM Inference Effect node.

This module provides ModelLlmInferenceRequest, the input model for the
LLM inference effect node. It captures all parameters needed to make an
inference call to an OpenAI-compatible LLM endpoint.

Related:
    - ModelLlmInferenceResponse: Output model for the inference call
    - ModelLlmToolDefinition: Tool definitions sent in the request
    - ModelLlmToolChoice: Caller constraint on tool selection
    - EnumLlmOperationType: Type of LLM operation (CHAT_COMPLETION, COMPLETION)
    - OMN-2107: Phase 7 OpenAI-compatible inference handler
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.nodes.effects.models.model_llm_tool_choice import (
    ModelLlmToolChoice,
)
from omnibase_infra.nodes.effects.models.model_llm_tool_definition import (
    ModelLlmToolDefinition,
)


class ModelLlmInferenceRequest(BaseModel):
    """Input model for LLM inference operations.

    Captures all parameters needed to make an inference call to an
    OpenAI-compatible LLM endpoint. The handler translates these
    fields into the provider-specific wire format.

    Attributes:
        base_url: Base URL of the LLM endpoint (e.g. ``"http://192.168.86.201:8000"``).
            The handler appends the appropriate path (``/v1/chat/completions`` or
            ``/v1/completions``) based on ``operation_type``.
        operation_type: Type of LLM operation to perform.
        model: Model identifier to use for inference.
        messages: Chat messages for CHAT_COMPLETION operations.
        prompt: Text prompt for COMPLETION operations.
        system_prompt: Optional system prompt prepended as a system message.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random).
        top_p: Nucleus sampling parameter.
        stop: Stop sequences that halt generation.
        tools: Tool definitions to make available to the model.
        tool_choice: Constraint on how the model should use tools.
        api_key: Optional API key for Bearer auth. If None, no auth header is sent.
        extra_headers: Additional HTTP headers injected into the outbound request.
            Used for custom authentication schemes (e.g. HMAC ``X-ONEX-Signature``).
        timeout_seconds: HTTP request timeout in seconds (default 30.0). Applied
            to both authenticated and unauthenticated calls. Must be between 1.0
            and 600.0 inclusive.

    Example:
        >>> req = ModelLlmInferenceRequest(
        ...     base_url="http://192.168.86.201:8000",
        ...     operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        ...     model="qwen2.5-coder-14b",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ... )
        >>> req.model
        'qwen2.5-coder-14b'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    base_url: str = Field(
        ...,
        min_length=1,
        description="Base URL of the LLM endpoint.",
    )
    operation_type: EnumLlmOperationType = Field(
        ...,
        description="Type of LLM operation to perform.",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier to use for inference.",
    )
    # ONEX_EXCLUDE: any_type - Chat messages follow the OpenAI wire format with
    # heterogeneous content (text, tool_call results, images). Strict typing would
    # require a large union type that mirrors the full OpenAI message spec.
    messages: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        description="Chat messages for CHAT_COMPLETION operations.",
    )
    prompt: str | None = Field(
        default=None,
        description="Text prompt for COMPLETION operations.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt prepended as a system message.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens to generate.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter.",
    )
    stop: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Stop sequences that halt generation.",
    )
    tools: tuple[ModelLlmToolDefinition, ...] = Field(
        default_factory=tuple,
        description="Tool definitions to make available to the model.",
    )
    tool_choice: ModelLlmToolChoice | None = Field(
        default=None,
        description="Constraint on how the model should use tools.",
    )
    api_key: str | None = Field(
        default=None,
        repr=False,
        description="Optional API key for Bearer auth.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        repr=False,
        description=(
            "Additional HTTP headers injected into the outbound request. "
            "Used for custom authentication schemes such as HMAC signatures "
            "(e.g. ``X-ONEX-Signature``). Keys and values must be ASCII strings."
        ),
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="HTTP request timeout in seconds.",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url_scheme(cls, v: str) -> str:
        """Validate that base_url starts with http:// or https://.

        Args:
            v: The base_url value to validate.

        Returns:
            The validated base_url value (unchanged).

        Raises:
            ValueError: If the URL does not start with ``http://`` or ``https://``.
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"base_url must start with 'http://' or 'https://', got: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _validate_prompt_or_messages(self) -> ModelLlmInferenceRequest:
        """Enforce that the correct input field is populated for the operation type.

        - CHAT_COMPLETION requires at least one message in ``messages``, or a
          non-empty ``system_prompt`` (the handler prepends it as a system message).
        - COMPLETION requires a non-None ``prompt``.

        Returns:
            The validated instance (unchanged).

        Raises:
            ValueError: If the required field is missing for the operation type.
        """
        if (
            self.operation_type is EnumLlmOperationType.CHAT_COMPLETION
            and len(self.messages) == 0
            and not self.system_prompt
        ):
            raise ValueError(
                "CHAT_COMPLETION requires at least one message in messages"
                " or a non-empty system_prompt"
            )
        if (
            self.operation_type is EnumLlmOperationType.COMPLETION
            and self.prompt is None
        ):
            raise ValueError("COMPLETION requires a non-None prompt")
        return self


__all__: list[str] = ["ModelLlmInferenceRequest"]
