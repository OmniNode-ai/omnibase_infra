# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Per-model pricing entry.

Defines a single model's token cost data as loaded from the pricing
manifest YAML file.

Design Decisions:
    - D1: Costs are stored as USD per 1,000 tokens (not per-token) to
      match industry convention and avoid floating-point precision issues
      with very small per-token values.
    - D2: ``effective_date`` is a plain ``str`` (ISO-8601 date) rather than
      ``datetime.date`` to keep the model serialization-friendly and
      avoid timezone ambiguity. Validation enforces the format.
    - D3: ``note`` is optional free-form text for documentation (e.g.
      "Local model - zero API cost").

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation

.. versionadded:: 0.10.0
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ModelPricingEntry(BaseModel):
    """Token cost data for a single LLM model.

    Attributes:
        input_cost_per_1k: Cost in USD per 1,000 input (prompt) tokens.
        output_cost_per_1k: Cost in USD per 1,000 output (completion) tokens.
        effective_date: ISO-8601 date string (``YYYY-MM-DD``) when this
            pricing became effective.
        note: Optional human-readable note (e.g. "Local model - zero API cost").

    Example:
        >>> entry = ModelPricingEntry(
        ...     input_cost_per_1k=0.015,
        ...     output_cost_per_1k=0.075,
        ...     effective_date="2026-02-01",
        ... )
        >>> entry.input_cost_per_1k
        0.015
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    input_cost_per_1k: float = Field(
        ...,
        ge=0.0,
        description="Cost in USD per 1,000 input (prompt) tokens.",
    )
    output_cost_per_1k: float = Field(
        ...,
        ge=0.0,
        description="Cost in USD per 1,000 output (completion) tokens.",
    )
    effective_date: str = Field(
        ...,
        min_length=10,
        max_length=10,
        description="ISO-8601 date (YYYY-MM-DD) when pricing became effective.",
    )
    note: str = Field(
        default="",
        max_length=512,
        description="Optional human-readable note about this model's pricing.",
    )

    @field_validator("effective_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Ensure ``effective_date`` is a valid ISO-8601 date string.

        Args:
            v: The date string to validate.

        Returns:
            The validated date string.

        Raises:
            ValueError: If the string does not match ``YYYY-MM-DD``.
        """
        if not _ISO_DATE_RE.match(v):
            raise ValueError(f"effective_date must be YYYY-MM-DD format, got: {v!r}")
        return v


__all__: list[str] = ["ModelPricingEntry"]
