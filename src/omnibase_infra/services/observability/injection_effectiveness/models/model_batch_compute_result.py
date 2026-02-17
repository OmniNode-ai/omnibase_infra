# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Batch compute result model for injection effectiveness metrics.

Captures the outcome of a batch effectiveness computation run, including
row counts written to each measurement table and any errors encountered.

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelBatchComputeResult(BaseModel):
    """Result of a batch effectiveness computation run.

    Attributes:
        effectiveness_rows: Rows written to injection_effectiveness.
        latency_rows: Rows written to latency_breakdowns.
        pattern_rows: Rows written to pattern_hit_rates.
        errors: Error messages for any failed phases.
        started_at: Computation start timestamp.
        completed_at: Computation end timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    effectiveness_rows: int = Field(
        default=0, description="Rows written to injection_effectiveness"
    )
    latency_rows: int = Field(
        default=0, description="Rows written to latency_breakdowns"
    )
    pattern_rows: int = Field(
        default=0, description="Rows written to pattern_hit_rates"
    )
    errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Error messages from failed phases"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Computation start timestamp",
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Computation end timestamp",
    )

    @property
    def total_rows(self) -> int:
        """Total rows written across all tables."""
        return self.effectiveness_rows + self.latency_rows + self.pattern_rows

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred during computation."""
        return len(self.errors) > 0
