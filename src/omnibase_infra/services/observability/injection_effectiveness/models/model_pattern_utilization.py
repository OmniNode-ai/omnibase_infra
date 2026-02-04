# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pattern utilization model.

Represents per-pattern utilization metrics within a context utilization event.

Related Tickets:
    - OMN-1889: Emit injection metrics + utilization signal (producer)
    - OMN-1890: Store injection metrics with corrected schema (consumer)
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPatternUtilization(BaseModel):
    """Per-pattern utilization metrics.

    Represents utilization data for a single injected pattern, enabling
    pattern-level effectiveness analysis.

    Attributes:
        pattern_id: UUID identifier for the injected pattern.
        utilization_score: How much of the pattern was used (0.0-1.0).
        utilization_method: Detection method used for this pattern.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pattern_id: UUID = Field(
        ...,
        description="Pattern UUID identifier from injection system",
    )
    utilization_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Pattern utilization score 0.0-1.0",
    )
    utilization_method: str = Field(
        ...,
        description="Method: identifier_match, semantic, or timeout",
    )
