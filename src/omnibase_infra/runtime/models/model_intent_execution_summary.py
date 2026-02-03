# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent execution summary model for batch execution results.

This module provides the ModelIntentExecutionSummary model that aggregates
execution results from intent batch processing in the IntentExecutionRouter.

Related:
    - IntentExecutionRouter: Uses this model for batch execution results
    - OMN-1869: Implementation ticket
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.models.model_backend_result import (
        ModelBackendResult,
    )


class ModelIntentExecutionSummary(BaseModel):
    """Summary of intent batch execution results.

    Provides an aggregated view of the batch execution including success/failure
    counts, timing, and individual results for observability and retry decisions.

    Attributes:
        total_intents: Number of intents processed in the batch.
        successful_count: Number of intents that executed successfully.
        failed_count: Number of intents that failed execution.
        total_duration_ms: Total time for batch execution in milliseconds.
        results: Individual execution results for each intent.
        correlation_id: Correlation ID for distributed tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    total_intents: int = Field(
        default=0, ge=0, description="Number of intents processed."
    )
    successful_count: int = Field(
        default=0, ge=0, description="Number of successful executions."
    )
    failed_count: int = Field(
        default=0, ge=0, description="Number of failed executions."
    )
    total_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total batch duration in milliseconds."
    )
    results: tuple[ModelBackendResult, ...] = Field(
        default_factory=tuple, description="Individual execution results."
    )
    correlation_id: UUID | None = Field(
        default=None, description="Correlation ID for tracing."
    )

    @property
    def all_successful(self) -> bool:
        """Check if all intents executed successfully."""
        return self.failed_count == 0 and self.total_intents > 0

    @property
    def partial_success(self) -> bool:
        """Check if batch had partial success (some passed, some failed)."""
        return self.successful_count > 0 and self.failed_count > 0

    @property
    def all_failed(self) -> bool:
        """Check if all intents failed."""
        return self.successful_count == 0 and self.total_intents > 0


# Rebuild model to resolve forward reference to ModelBackendResult.
# This deferred import avoids circular import during omnibase_infra package init.
def _rebuild_model() -> None:
    """Rebuild model with forward references resolved."""
    from omnibase_infra.nodes.effects.models.model_backend_result import (
        ModelBackendResult,
    )

    ModelIntentExecutionSummary.model_rebuild(
        _types_namespace={"ModelBackendResult": ModelBackendResult}
    )


_rebuild_model()

__all__ = ["ModelIntentExecutionSummary"]
