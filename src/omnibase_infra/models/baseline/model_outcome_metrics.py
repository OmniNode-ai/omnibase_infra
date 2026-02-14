# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Outcome metrics captured for a single A/B run variant.

Tracks pass/fail outcome, flake rate, and review iterations to enable
quality-based ROI comparison between baseline and candidate runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelOutcomeMetrics(BaseModel):
    """Outcome metrics for a single run (baseline or candidate).

    Note:
        ``total_checks``, ``passed_checks``, and ``failed_checks`` are
        independent counters with no consistency validation. Some test
        frameworks don't report a per-check breakdown, so enforcing
        ``total == passed + failed`` would reject valid partial data.

    Attributes:
        passed: Whether the run passed validation.
        total_checks: Total number of checks executed. Not enforced as
            passed + failed sum; partial reporting is supported.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        flake_rate: Flake rate as a fraction in [0.0, 1.0].
        review_iterations: Number of review iterations required.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    passed: bool = Field(
        ...,
        description="Whether the run passed validation.",
    )
    total_checks: int = Field(
        default=0,
        ge=0,
        description="Total number of checks executed. Not enforced as passed + failed sum; partial reporting is supported.",
    )
    passed_checks: int = Field(
        default=0,
        ge=0,
        description="Number of checks that passed.",
    )
    failed_checks: int = Field(
        default=0,
        ge=0,
        description="Number of checks that failed.",
    )
    flake_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Flake rate as a fraction in [0.0, 1.0].",
    )
    review_iterations: int = Field(
        default=0,
        ge=0,
        description="Number of review iterations required.",
    )


__all__: list[str] = ["ModelOutcomeMetrics"]
