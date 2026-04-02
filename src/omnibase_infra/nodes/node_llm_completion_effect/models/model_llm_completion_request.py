# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for LLM completion requests."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_llm_completion_effect.models.model_llm_completion_message import (
    ModelLLMCompletionMessage,
)


class ModelLLMCompletionRequest(BaseModel):
    """Request payload for an LLM chat completion.

    Mirrors the OpenAI ``/v1/chat/completions`` contract so that any
    OpenAI-compatible backend can serve the request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        default_factory=uuid4, description="Workflow correlation ID."
    )
    messages: tuple[ModelLLMCompletionMessage, ...] = Field(
        ..., description="Ordered chat messages."
    )
    model: str = Field(
        default="",
        description="Model name or alias.  Empty string means 'use default routing'.",
    )
    max_tokens: int = Field(
        default=1024, ge=1, description="Maximum tokens to generate."
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature."
    )
    endpoint_url: str = Field(
        default="",
        description="Override endpoint URL.  Empty means use env-configured default.",
    )
