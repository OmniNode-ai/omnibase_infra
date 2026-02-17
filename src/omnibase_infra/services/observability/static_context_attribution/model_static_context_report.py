# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pydantic model for the full static context attribution report.

ModelStaticContextReport aggregates all section attributions with provenance
metadata for reproducibility and auditing.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.services.observability.static_context_attribution.model_section_attribution import (
    ModelSectionAttribution,
)


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

    @model_validator(mode="after")
    def _validate_token_consistency(self) -> ModelStaticContextReport:
        """Validate that token counts are consistent with attributions.

        Checks:
            1. total_attributed_tokens <= total_tokens
            2. Sum of attribution tokens does not exceed total_attributed_tokens

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If token counts are inconsistent.
        """
        if self.total_attributed_tokens > self.total_tokens:
            raise ValueError(
                f"total_attributed_tokens ({self.total_attributed_tokens}) "
                f"must not exceed total_tokens ({self.total_tokens})"
            )

        if self.attributions:
            attribution_sum = sum(
                a.attributed_tokens for a in self.attributions
            )
            if attribution_sum > self.total_attributed_tokens:
                raise ValueError(
                    f"Sum of attribution tokens ({attribution_sum}) "
                    f"must not exceed total_attributed_tokens "
                    f"({self.total_attributed_tokens})"
                )

        return self

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content for provenance tracking.

        Args:
            content: String content to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


__all__ = ["ModelStaticContextReport"]
