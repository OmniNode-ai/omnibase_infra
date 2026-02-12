# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM token-usage model for Effect layer cost and usage tracking.

This module provides ModelLlmUsage, a lightweight value object that captures
token counts and optional cost metadata returned by LLM provider APIs.

Architecture:
    ModelLlmUsage is designed to be embedded inside larger response models
    (e.g. an LLM effect output) rather than used standalone.  It carries
    only the information that every major provider returns in its
    ``usage`` block: input tokens, output tokens, and a pre-computed total.

    Cost tracking is deliberately left as ``None`` in v1 because unit
    pricing varies across providers and is subject to change.

Related:
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLlmUsage(BaseModel):
    """Token-usage summary returned by an LLM provider.

    All token counts default to zero so callers may construct a usage
    object even when the provider omits individual fields.

    Attributes:
        tokens_input: Number of tokens in the prompt / input messages.
        tokens_output: Number of tokens generated in the completion.
        tokens_total: Total tokens consumed (input + output).
        cost_usd: Estimated cost in US dollars.  Always ``None`` in v1;
            reserved for future provider-specific billing integration.

    Example:
        >>> usage = ModelLlmUsage(tokens_input=120, tokens_output=45, tokens_total=165)
        >>> usage.tokens_total
        165
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tokens_input: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in the prompt / input messages.",
    )
    tokens_output: int = Field(
        default=0,
        ge=0,
        description="Number of tokens generated in the completion.",
    )
    tokens_total: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (input + output).",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Cost in USD. Always None in v1.",
    )


__all__ = ["ModelLlmUsage"]
