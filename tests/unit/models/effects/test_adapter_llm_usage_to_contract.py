# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the ModelLlmUsage -> SPI contract adapter functions.

Tests cover:
- to_usage_raw: ModelLlmUsage -> ContractLlmUsageRaw
- to_usage_normalized: ModelLlmUsage -> ContractLlmUsageNormalized
- to_call_metrics: ModelLlmUsage -> ContractLlmCallMetrics
- Provenance mapping (API/ESTIMATED/MISSING)
- Edge cases (empty usage, zero tokens, None raw data)

OMN-2318: Integrate SPI 0.9.0 LLM cost tracking contracts.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.effects.models.adapter_llm_usage_to_contract import (
    to_call_metrics,
    to_usage_normalized,
    to_usage_raw,
)
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_spi.contracts.measurement import (
    ContractEnumUsageSource,
    ContractLlmCallMetrics,
    ContractLlmUsageNormalized,
    ContractLlmUsageRaw,
)

# ============================================================================
# to_usage_raw Tests
# ============================================================================


class TestToUsageRaw:
    """Tests for converting ModelLlmUsage to ContractLlmUsageRaw."""

    def test_with_raw_data(self) -> None:
        """Raw provider data is preserved in the contract."""
        raw_data = {"prompt_tokens": 100, "completion_tokens": 50}
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            raw_provider_usage=raw_data,
        )

        result = to_usage_raw(usage, provider="openai")

        assert isinstance(result, ContractLlmUsageRaw)
        assert result.provider == "openai"
        assert result.raw_data == raw_data

    def test_without_raw_data(self) -> None:
        """None raw_provider_usage produces empty raw_data dict."""
        usage = ModelLlmUsage()

        result = to_usage_raw(usage, provider="vllm")

        assert result.provider == "vllm"
        assert result.raw_data == {}

    def test_default_provider(self) -> None:
        """Default provider is empty string."""
        usage = ModelLlmUsage()

        result = to_usage_raw(usage)

        assert result.provider == ""

    def test_frozen_output(self) -> None:
        """Returned ContractLlmUsageRaw is frozen."""
        usage = ModelLlmUsage(raw_provider_usage={"tokens": 10})
        result = to_usage_raw(usage)

        with pytest.raises(Exception):
            result.provider = "changed"  # type: ignore[misc]


# ============================================================================
# to_usage_normalized Tests
# ============================================================================


class TestToUsageNormalized:
    """Tests for converting ModelLlmUsage to ContractLlmUsageNormalized."""

    def test_api_source_mapping(self) -> None:
        """API source maps to source=API, usage_is_estimated=False."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.API,
        )

        result = to_usage_normalized(usage)

        assert isinstance(result, ContractLlmUsageNormalized)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.source == ContractEnumUsageSource.API
        assert result.usage_is_estimated is False

    def test_estimated_source_mapping(self) -> None:
        """ESTIMATED source maps to source=ESTIMATED, usage_is_estimated=True."""
        usage = ModelLlmUsage(
            tokens_input=200,
            tokens_output=80,
            usage_source=ContractEnumUsageSource.ESTIMATED,
        )

        result = to_usage_normalized(usage)

        assert result.prompt_tokens == 200
        assert result.completion_tokens == 80
        assert result.total_tokens == 280
        assert result.source == ContractEnumUsageSource.ESTIMATED
        assert result.usage_is_estimated is True

    def test_missing_source_mapping(self) -> None:
        """MISSING source maps to source=MISSING, usage_is_estimated=False."""
        usage = ModelLlmUsage(
            usage_source=ContractEnumUsageSource.MISSING,
        )

        result = to_usage_normalized(usage)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.source == ContractEnumUsageSource.MISSING
        assert result.usage_is_estimated is False

    def test_field_name_mapping(self) -> None:
        """Infra field names (tokens_input/output) map to SPI names (prompt/completion)."""
        usage = ModelLlmUsage(tokens_input=42, tokens_output=17)

        result = to_usage_normalized(usage)

        assert result.prompt_tokens == 42
        assert result.completion_tokens == 17
        assert result.total_tokens == 59

    def test_frozen_output(self) -> None:
        """Returned ContractLlmUsageNormalized is frozen."""
        usage = ModelLlmUsage(tokens_input=10, tokens_output=5)
        result = to_usage_normalized(usage)

        with pytest.raises(Exception):
            result.prompt_tokens = 999  # type: ignore[misc]


# ============================================================================
# to_call_metrics Tests
# ============================================================================


class TestToCallMetrics:
    """Tests for converting ModelLlmUsage to ContractLlmCallMetrics."""

    def test_to_call_metrics_empty_model_id_raises(self) -> None:
        """to_call_metrics rejects empty model_id with ValueError."""
        usage = ModelLlmUsage(tokens_input=10, tokens_output=5)

        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            to_call_metrics(usage, model_id="")

    def test_basic_conversion(self) -> None:
        """Basic conversion with required fields."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.API,
            raw_provider_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        result = to_call_metrics(usage, model_id="gpt-4o")

        assert isinstance(result, ContractLlmCallMetrics)
        assert result.model_id == "gpt-4o"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.usage_is_estimated is False

    def test_with_optional_fields(self) -> None:
        """Conversion with all optional fields populated."""
        usage = ModelLlmUsage(
            tokens_input=200,
            tokens_output=80,
            cost_usd=0.05,
            usage_source=ContractEnumUsageSource.API,
            raw_provider_usage={"prompt_tokens": 200},
        )

        result = to_call_metrics(
            usage,
            model_id="qwen2.5-coder-14b",
            provider="vllm",
            latency_ms=150.0,
            timestamp_iso="2026-02-15T10:00:00Z",
            reporting_source="llm-inference-effect",
        )

        assert result.model_id == "qwen2.5-coder-14b"
        assert result.estimated_cost_usd == pytest.approx(0.05)
        assert result.latency_ms == pytest.approx(150.0)
        assert result.timestamp_iso == "2026-02-15T10:00:00Z"
        assert result.reporting_source == "llm-inference-effect"
        assert result.usage_raw is not None
        assert result.usage_raw.provider == "vllm"
        assert result.usage_normalized is not None
        assert result.usage_normalized.prompt_tokens == 200

    def test_estimated_usage(self) -> None:
        """Estimated usage sets usage_is_estimated=True."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.ESTIMATED,
        )

        result = to_call_metrics(usage, model_id="test-model")

        assert result.usage_is_estimated is True
        assert result.usage_normalized is not None
        assert result.usage_normalized.usage_is_estimated is True
        assert result.usage_normalized.source == ContractEnumUsageSource.ESTIMATED

    def test_missing_usage(self) -> None:
        """Missing usage source produces zero tokens."""
        usage = ModelLlmUsage()

        result = to_call_metrics(usage, model_id="test-model")

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.usage_is_estimated is False

    def test_no_cost(self) -> None:
        """None cost_usd passes through as None."""
        usage = ModelLlmUsage(tokens_input=10)

        result = to_call_metrics(usage, model_id="test-model")

        assert result.estimated_cost_usd is None

    def test_usage_raw_included(self) -> None:
        """usage_raw sub-contract is populated from raw_provider_usage."""
        raw = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        usage = ModelLlmUsage(
            tokens_input=50,
            tokens_output=20,
            raw_provider_usage=raw,
        )

        result = to_call_metrics(usage, model_id="model", provider="openai")

        assert result.usage_raw is not None
        assert result.usage_raw.raw_data == raw
        assert result.usage_raw.provider == "openai"

    def test_usage_normalized_included(self) -> None:
        """usage_normalized sub-contract is populated and consistent."""
        usage = ModelLlmUsage(
            tokens_input=300,
            tokens_output=100,
            usage_source=ContractEnumUsageSource.API,
        )

        result = to_call_metrics(usage, model_id="model")

        assert result.usage_normalized is not None
        assert result.usage_normalized.prompt_tokens == 300
        assert result.usage_normalized.completion_tokens == 100
        assert result.usage_normalized.total_tokens == 400
        assert result.usage_normalized.source == ContractEnumUsageSource.API
        assert result.usage_normalized.usage_is_estimated is False

    def test_token_consistency_between_top_level_and_normalized(self) -> None:
        """Top-level and normalized token counts are always consistent.

        This verifies that the ContractLlmCallMetrics validator does not
        reject the output (it requires top-level == normalized).
        """
        usage = ModelLlmUsage(
            tokens_input=42,
            tokens_output=17,
            usage_source=ContractEnumUsageSource.API,
        )

        result = to_call_metrics(usage, model_id="model")

        assert result.prompt_tokens == result.usage_normalized.prompt_tokens  # type: ignore[union-attr]
        assert result.completion_tokens == result.usage_normalized.completion_tokens  # type: ignore[union-attr]
        assert result.total_tokens == result.usage_normalized.total_tokens  # type: ignore[union-attr]

    def test_frozen_output(self) -> None:
        """Returned ContractLlmCallMetrics is frozen."""
        usage = ModelLlmUsage(tokens_input=10, tokens_output=5)
        result = to_call_metrics(usage, model_id="model")

        with pytest.raises(Exception):
            result.model_id = "changed"  # type: ignore[misc]

    def test_default_string_fields(self) -> None:
        """Default optional string fields are empty strings."""
        usage = ModelLlmUsage(tokens_input=10)
        result = to_call_metrics(usage, model_id="model")

        assert result.timestamp_iso == ""
        assert result.reporting_source == ""


# ============================================================================
# Provenance Round-Trip
# ============================================================================


class TestProvenanceRoundTrip:
    """Verify provenance is correctly threaded through the full conversion
    pipeline: ModelLlmUsage -> ContractLlmCallMetrics."""

    @pytest.mark.parametrize(
        ("source", "expected_estimated"),
        [
            (ContractEnumUsageSource.API, False),
            (ContractEnumUsageSource.ESTIMATED, True),
            (ContractEnumUsageSource.MISSING, False),
        ],
    )
    def test_provenance_propagation(
        self,
        source: ContractEnumUsageSource,
        expected_estimated: bool,
    ) -> None:
        """Provenance source is consistently mapped through all layers."""
        usage = ModelLlmUsage(
            tokens_input=50,
            tokens_output=25,
            usage_source=source,
        )

        metrics = to_call_metrics(usage, model_id="test")

        # Top-level flag
        assert metrics.usage_is_estimated is expected_estimated
        # Normalized sub-contract
        assert metrics.usage_normalized is not None
        assert metrics.usage_normalized.source == source
        assert metrics.usage_normalized.usage_is_estimated is expected_estimated
