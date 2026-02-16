# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Converter bridging ModelLlmUsage to SPI LLM cost tracking contracts.

This module provides pure functions that translate between the infra-layer
``ModelLlmUsage`` value object and the SPI measurement contracts:

- ``ContractLlmCallMetrics``
- ``ContractLlmUsageRaw``
- ``ContractLlmUsageNormalized``
- ``ContractEnumUsageSource``

All functions are stateless and produce frozen Pydantic models suitable for
downstream measurement pipeline ingestion.

Related:
    - OMN-2318: Integrate SPI 0.9.0 LLM cost tracking contracts
    - ModelLlmUsage: Source infra-layer usage model
    - ContractLlmCallMetrics: Target SPI per-call metrics contract
"""

from __future__ import annotations

from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_spi.contracts.measurement import (
    ContractEnumUsageSource,
    ContractLlmCallMetrics,
    ContractLlmUsageNormalized,
    ContractLlmUsageRaw,
)


def to_usage_raw(
    usage: ModelLlmUsage,
    provider: str = "",
) -> ContractLlmUsageRaw:
    """Convert raw provider usage data from ModelLlmUsage to ContractLlmUsageRaw.

    When ``usage.raw_provider_usage`` is ``None``, returns an empty raw
    container with the given provider identifier.

    Args:
        usage: The infra-layer usage model.
        provider: Provider identifier (e.g. ``"openai"``, ``"ollama"``,
            ``"vllm"``).

    Returns:
        A frozen ``ContractLlmUsageRaw`` preserving the verbatim provider data.
    """
    return ContractLlmUsageRaw(
        provider=provider,
        raw_data=usage.raw_provider_usage or {},
    )


def to_usage_normalized(usage: ModelLlmUsage) -> ContractLlmUsageNormalized:
    """Convert ModelLlmUsage token counts to ContractLlmUsageNormalized.

    Maps infra-layer field names (``tokens_input``, ``tokens_output``,
    ``tokens_total``) to the SPI canonical names (``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``).

    The ``source`` and ``usage_is_estimated`` fields are derived from
    ``usage.usage_source``:

    - ``API``       -> ``source=API, usage_is_estimated=False``
    - ``ESTIMATED`` -> ``source=ESTIMATED, usage_is_estimated=True``
    - ``MISSING``   -> ``source=MISSING, usage_is_estimated=False``

    Args:
        usage: The infra-layer usage model.

    Returns:
        A frozen ``ContractLlmUsageNormalized`` with canonical token counts
        and provenance.
    """
    source = usage.usage_source
    usage_is_estimated = source == ContractEnumUsageSource.ESTIMATED
    # ModelLlmUsage.model_validator guarantees tokens_total is always set
    total = usage.tokens_total

    return ContractLlmUsageNormalized(
        prompt_tokens=usage.tokens_input,
        completion_tokens=usage.tokens_output,
        total_tokens=total,
        source=source,
        usage_is_estimated=usage_is_estimated,
    )


def to_call_metrics(
    usage: ModelLlmUsage,
    model_id: str,
    *,
    provider: str = "",
    latency_ms: float | None = None,
    timestamp_iso: str = "",
    reporting_source: str = "",
) -> ContractLlmCallMetrics:
    """Convert ModelLlmUsage to a full ContractLlmCallMetrics record.

    Builds both the raw and normalized usage sub-contracts and assembles
    them into the top-level per-call metrics contract.

    Args:
        usage: The infra-layer usage model.
        model_id: Identifier of the LLM model (e.g. ``"gpt-4o"``,
            ``"qwen2.5-coder-14b"``). Required and must be non-empty.
        provider: Provider identifier for the raw usage envelope.
        latency_ms: End-to-end call latency in milliseconds.
        timestamp_iso: ISO-8601 timestamp string of the call.
        reporting_source: Provenance label for this metrics record
            (e.g. ``"llm-inference-effect"``).

    Returns:
        A frozen ``ContractLlmCallMetrics`` with token counts, cost,
        raw/normalized usage, and metadata.
    """
    # ModelLlmUsage.model_validator guarantees tokens_total is always set
    total = usage.tokens_total
    is_estimated = usage.usage_source == ContractEnumUsageSource.ESTIMATED

    return ContractLlmCallMetrics(
        model_id=model_id,
        prompt_tokens=usage.tokens_input,
        completion_tokens=usage.tokens_output,
        total_tokens=total,
        estimated_cost_usd=usage.cost_usd,
        latency_ms=latency_ms,
        usage_raw=to_usage_raw(usage, provider=provider),
        usage_normalized=to_usage_normalized(usage),
        usage_is_estimated=is_estimated,
        timestamp_iso=timestamp_iso,
        reporting_source=reporting_source,
    )


__all__ = [
    "to_call_metrics",
    "to_usage_normalized",
    "to_usage_raw",
]
