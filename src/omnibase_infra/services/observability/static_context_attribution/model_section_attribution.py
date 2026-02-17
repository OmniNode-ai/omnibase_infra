# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pydantic models for section-level attribution results and reports.

ModelSectionAttribution captures the utilization score for a single section.
ModelStaticContextReport aggregates all attributions with provenance metadata.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

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


class ModelStaticContextReport(BaseModel):
    """Full attribution report with provenance metadata.

    Aggregates all section attributions and records provenance
    information for reproducibility and auditing.

    Attributes:
        attributions: Per-section attribution results.
        total_tokens: Total tokens across all sections.
        total_attributed_tokens: Total tokens attributed to response.
        input_hash: SHA-256 hash of the full input context for
            reproducibility verification.
        response_hash: SHA-256 hash of the model response.
        code_version: Version of the attribution service code.
        created_at: Timestamp when the report was generated.
        source_files: List of source file paths included in analysis.
        llm_augmented: Whether LLM augmentation pass was applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    attributions: tuple[ModelSectionAttribution, ...] = Field(
        default_factory=tuple,
        description="Per-section attribution results.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens across all sections.",
    )
    total_attributed_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens attributed to response.",
    )
    input_hash: str = Field(
        default="",
        description="SHA-256 hash of full input context.",
    )
    response_hash: str = Field(
        default="",
        description="SHA-256 hash of model response.",
    )
    code_version: str = Field(
        default="0.1.0",
        description="Version of the attribution service code.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Timestamp when report was generated.",
    )
    source_files: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Source file paths included in analysis.",
    )
    llm_augmented: bool = Field(
        default=False,
        description="Whether LLM augmentation pass was applied.",
    )

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content for provenance tracking.

        Args:
            content: String content to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


__all__ = ["ModelSectionAttribution", "ModelStaticContextReport"]
