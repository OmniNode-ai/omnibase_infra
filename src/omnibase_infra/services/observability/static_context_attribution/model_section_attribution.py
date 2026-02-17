# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pydantic model for section-level attribution results.

ModelSectionAttribution captures the utilization score for a single section
of static context against a model response.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)


class ModelSectionAttribution(BaseModel):
    """Attribution result for a single static context section.

    Combines the parsed section metadata with a utilization score
    computed via edit-distance anchoring against model responses.

    Attributes:
        section: The parsed context section with token count.
        utilization_score: Score in [0.0, 1.0] indicating how much of
            this section's content appeared in the model response.
            0.0 = not used, 1.0 = fully utilized.
        matched_fragments: Number of content fragments from this section
            found in the response via edit-distance matching.
        total_fragments: Total number of content fragments in this section.
        attributed_tokens: Estimated tokens attributable to this section
            in the response (utilization_score * token_count).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    section: ModelContextSection = Field(
        ...,
        description="The parsed context section with token count.",
    )
    utilization_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Utilization score in [0.0, 1.0]. "
        "0.0 = not used, 1.0 = fully utilized.",
    )
    matched_fragments: int = Field(
        default=0,
        ge=0,
        description="Number of section fragments found in response.",
    )
    total_fragments: int = Field(
        default=0,
        ge=0,
        description="Total number of content fragments in section.",
    )

    @property
    def attributed_tokens(self) -> int:
        """Estimated tokens attributable to this section in the response.

        Computed as ``utilization_score * section.token_count``, rounded
        to the nearest integer.
        """
        return round(self.utilization_score * self.section.token_count)


__all__ = ["ModelSectionAttribution"]
