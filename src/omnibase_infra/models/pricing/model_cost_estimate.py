# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Cost estimation result model.

Returned by ``ModelPricingTable.estimate_cost()`` to convey the estimated
USD cost for a single LLM call, or ``None`` when the model is unknown.

Design Decisions:
    - D1: ``estimated_cost_usd`` is ``float | None``. ``None`` means the
      model was not found in the pricing manifest (unknown model).
      ``0.0`` means the model is known and free (e.g. local model).
      This distinction is critical per OMN-2239 requirements.
    - D2: The model carries the input parameters (model_id, prompt_tokens,
      completion_tokens) for traceability and debugging.

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation

.. versionadded:: 0.10.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCostEstimate(BaseModel):
    """Result of an LLM cost estimation.

    Attributes:
        model_id: Identifier of the LLM model used for the call.
        prompt_tokens: Number of input (prompt) tokens.
        completion_tokens: Number of output (completion) tokens.
        estimated_cost_usd: Estimated cost in USD, or ``None`` if the
            model is not in the pricing manifest.

    Example:
        >>> estimate = ModelCostEstimate(
        ...     model_id="claude-opus-4-6",
        ...     prompt_tokens=1000,
        ...     completion_tokens=500,
        ...     estimated_cost_usd=0.0525,
        ... )
        >>> estimate.estimated_cost_usd
        0.0525

    Example (unknown model):
        >>> estimate = ModelCostEstimate(
        ...     model_id="unknown-model-xyz",
        ...     prompt_tokens=1000,
        ...     completion_tokens=500,
        ...     estimated_cost_usd=None,
        ... )
        >>> estimate.estimated_cost_usd is None
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ONEX_EXCLUDE: pattern_validator - model_id is an LLM model name (e.g. "claude-opus-4-6"), not a UUID entity reference
    model_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the LLM model.",
    )
    prompt_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input (prompt) tokens.",
    )
    completion_tokens: int = Field(
        ...,
        ge=0,
        description="Number of output (completion) tokens.",
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        description=(
            "Estimated cost in USD. None if the model is not in the "
            "pricing manifest (unknown). 0.0 for known free/local models."
        ),
    )


__all__: list[str] = ["ModelCostEstimate"]
