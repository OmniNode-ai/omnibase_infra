# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model for a single chat message in OpenAI format."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLLMCompletionMessage(BaseModel):
    """A single chat message in OpenAI format."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(..., description="Message role: system, user, or assistant.")
    content: str = Field(..., description="Message content.")
