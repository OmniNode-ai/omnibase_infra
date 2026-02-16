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

    Since OMN-2318, ModelLlmUsage also carries usage provenance tracking
    via ``usage_source`` (API, ESTIMATED, or MISSING) and preserves the
    raw provider usage payload in ``raw_provider_usage`` for auditing.

Related:
    - OMN-2103: Phase 3 shared LLM models
    - OMN-2318: Integrate SPI 0.9.0 LLM cost tracking contracts
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_spi.contracts.measurement import ContractEnumUsageSource


class ModelLlmUsage(BaseModel):
    """Token-usage summary returned by an LLM provider.

    All token counts default to zero so callers may construct a usage
    object even when the provider omits individual fields.

    When ``tokens_total`` is not explicitly provided (i.e. left at the
    default of ``None``), it is auto-computed as
    ``tokens_input + tokens_output``.  When explicitly supplied, a
    consistency check ensures
    ``tokens_total == tokens_input + tokens_output``.

    Attributes:
        tokens_input: Number of tokens in the prompt / input messages.
        tokens_output: Number of tokens generated in the completion.
        tokens_total: Total tokens consumed (input + output).  Defaults to
            ``None`` which triggers auto-computation from
            ``tokens_input + tokens_output``.
        cost_usd: Estimated cost in US dollars.  ``None`` when cost has
            not been computed.
        usage_source: Provenance of the usage data.  ``API`` when the
            provider reported token counts directly, ``ESTIMATED`` when
            counts were derived locally (e.g. via a tokenizer),
            ``MISSING`` when no usage data was available.
        raw_provider_usage: Verbatim provider response ``usage`` block
            preserved for auditing.  ``None`` when the provider did not
            return a usage block or when the handler chose not to capture it.

    Example:
        >>> usage = ModelLlmUsage(tokens_input=120, tokens_output=45)
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
    tokens_total: int | None = Field(
        default=None,
        ge=0,
        description="Total tokens consumed (input + output). "
        "None triggers auto-computation.",
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost in USD. None when not computed.",
    )
    usage_source: ContractEnumUsageSource = Field(
        default=ContractEnumUsageSource.MISSING,
        description=(
            "Provenance of the usage data: API (provider-reported), "
            "ESTIMATED (locally computed), or MISSING (no data)."
        ),
    )
    # ONEX_EXCLUDE: any_type - Raw provider usage payload is an untyped dict
    # because each LLM provider (OpenAI, Ollama, vLLM) returns a different
    # wire format. The verbatim data is preserved for auditing, not processed.
    raw_provider_usage: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Verbatim provider response usage block for auditing. "
            "None when not captured."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _compute_or_validate_tokens_total(cls, values: object) -> object:
        """Auto-compute tokens_total or validate consistency.

        When tokens_total is omitted or ``None``, it is set to
        ``tokens_input + tokens_output``.  When explicitly provided
        (including zero), it must equal the sum of tokens_input and
        tokens_output.
        """
        if not isinstance(values, dict):
            return values

        tokens_input = values.get("tokens_input", 0)
        tokens_output = values.get("tokens_output", 0)
        tokens_total = values.get("tokens_total")

        expected = tokens_input + tokens_output
        if tokens_total is None:
            values["tokens_total"] = expected
        elif tokens_total != expected:
            raise ValueError(
                f"tokens_total ({tokens_total}) does not equal "
                f"tokens_input + tokens_output ({expected})."
            )
        return values


__all__ = ["ModelLlmUsage"]
