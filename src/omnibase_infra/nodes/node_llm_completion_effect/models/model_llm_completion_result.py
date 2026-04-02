# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output model for LLM completion results."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLLMCompletionResult(BaseModel):
    """Result of an LLM chat completion request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    success: bool = Field(..., description="Whether the completion succeeded.")
    content: str = Field(default="", description="Generated text from the model.")
    model: str = Field(
        default="", description="Model that actually served the request."
    )
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt token count.")
    completion_tokens: int = Field(
        default=0, ge=0, description="Completion token count."
    )
    error_message: str = Field(
        default="", description="Error detail if success is False."
    )
