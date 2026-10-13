# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline event -- checks completed.

Published by the validation executor when every check of a plan has run;
consumed by the validation orchestrator. It is the executor result plus the
plan's score threshold, so the adjudication downstream scores against the bar
the plan set rather than a default.

Ticket: OMN-21056
"""

from __future__ import annotations

from pydantic import Field

from omnibase_infra.models.validation.model_executor_result import ModelExecutorResult


class ModelValidationChecksCompleted(ModelExecutorResult):
    """Aggregated check results for a plan, with the plan's score threshold.

    Attributes:
        score_threshold: Minimum score for a PASS verdict, copied from the plan.
    """

    score_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum score for PASS verdict."
    )


__all__: list[str] = ["ModelValidationChecksCompleted"]
