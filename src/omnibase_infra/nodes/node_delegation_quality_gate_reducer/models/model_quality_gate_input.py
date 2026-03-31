# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Quality gate input model for the delegation pipeline.

Represents the input to the quality gate reducer: LLM response content
and expected quality markers for validation.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelQualityGateInput(BaseModel):
    """Gate input: LLM response content and expected quality markers.

    Attributes:
        correlation_id: Tracks this input back to the original request.
        task_type: The task classification for type-specific checks.
        llm_response_content: The raw LLM response to evaluate.
        expected_markers: Strings expected in the response for the task type.
        min_response_length: Minimum acceptable response length in characters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Tracks this input back to the original request.",
    )
    task_type: str = Field(
        ...,
        description="The task classification for type-specific checks.",
    )
    llm_response_content: str = Field(
        ...,
        description="The raw LLM response to evaluate.",
    )
    expected_markers: tuple[str, ...] = Field(
        default=(),
        description="Strings expected in the response for the task type.",
    )
    min_response_length: int = Field(
        default=60,
        description="Minimum acceptable response length in characters.",
    )


__all__: list[str] = ["ModelQualityGateInput"]
