# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Quality gate result model for the delegation pipeline.

Represents the output of the quality gate reducer: pass/fail,
quality score, failure reasons, and fallback recommendation.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelQualityGateResult(BaseModel):
    """Gate output: pass/fail verdict, score, failure reasons, and fallback flag.

    Attributes:
        correlation_id: Tracks this result back to the original request.
        passed: Whether the LLM response passed the quality gate.
        quality_score: Quality score from 0.0 to 1.0.
        failure_reasons: Tuple of human-readable failure reason strings.
        fallback_recommended: Whether fallback to Claude is recommended.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Tracks this result back to the original request.",
    )
    passed: bool = Field(
        ...,
        description="Whether the LLM response passed the quality gate.",
    )
    quality_score: float = Field(
        ...,
        description="Quality score from 0.0 to 1.0.",
    )
    failure_reasons: tuple[str, ...] = Field(
        default=(),
        description="Tuple of human-readable failure reason strings.",
    )
    fallback_recommended: bool = Field(
        default=False,
        description="Whether fallback to Claude is recommended.",
    )


__all__: list[str] = ["ModelQualityGateResult"]
