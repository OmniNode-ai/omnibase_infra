# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Delegation request model for the delegation pipeline.

Represents a command to delegate a task (test, document, research)
to a local LLM via the ONEX runtime event bus.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDelegationRequest(BaseModel):
    """Delegation command: prompt, task type, and source context.

    Attributes:
        prompt: The user prompt to delegate to the local LLM.
        task_type: Classification of the delegation task.
        source_session_id: Session that originated the delegation request.
        source_file_path: File context for the delegation, if any.
        correlation_id: Unique identifier for tracking through the pipeline.
        max_tokens: Maximum tokens for the LLM response.
        emitted_at: Timestamp when the request was created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    prompt: str = Field(
        ...,
        description="The user prompt to delegate to the local LLM.",
    )
    task_type: Literal["test", "document", "research"] = Field(
        ...,
        description="Classification of the delegation task.",
    )
    source_session_id: str | None = Field(
        default=None,
        description="Session that originated the delegation request.",
    )
    source_file_path: str | None = Field(
        default=None,
        description="File context for the delegation, if any.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Unique identifier for tracking through the pipeline.",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for the LLM response.",
    )
    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the request was created.",
    )


__all__: list[str] = ["ModelDelegationRequest"]
