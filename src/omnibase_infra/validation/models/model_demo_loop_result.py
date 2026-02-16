# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Aggregate result model for the Demo Loop Gate (OMN-2297)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.models.model_assertion_result import (
    ModelAssertionResult,
)


class ModelDemoLoopResult(BaseModel):
    """Aggregate result of all demo loop assertions.

    Attributes:
        assertions: Tuple of individual assertion results.
        passed: Number of assertions that passed.
        failed: Number of assertions that failed.
        skipped: Number of assertions that were skipped.
        is_ready: True only when all non-skipped assertions passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    assertions: tuple[ModelAssertionResult, ...] = Field(
        default_factory=tuple,
        description="Individual assertion results",
    )
    passed: int = Field(default=0, description="Count of passed assertions")
    failed: int = Field(default=0, description="Count of failed assertions")
    skipped: int = Field(default=0, description="Count of skipped assertions")
    is_ready: bool = Field(
        default=False,
        description="True when all non-skipped assertions passed",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``is_ready`` is True. Differs from typical Pydantic behavior.
        """
        return self.is_ready
