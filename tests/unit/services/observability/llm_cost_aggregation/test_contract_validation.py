# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Input validation and edge case tests for SPI LLM cost tracking contracts.

Tests confirm that:
1. Malformed input is rejected (not silently coerced)
2. Edge cases are handled (zero tokens, null cost, negative values)
3. Token sum consistency is enforced
4. Type coercion is rejected for financial data

The SPI contracts (ContractLlmCallMetrics, ContractLlmUsageNormalized,
ContractLlmUsageRaw) are the canonical wire-format contracts for LLM cost
data.  Cost data is financial data -- it must be strict.  Silent coercion
or zero-filling would corrupt cost analytics.

Related Tickets:
    - OMN-2295: LLM cost tracking: input validation and edge case tests
    - OMN-2235: SPI LLM cost tracking contracts
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_spi.contracts.measurement import (
    ContractEnumUsageSource,
    ContractLlmCallMetrics,
    ContractLlmUsageNormalized,
    ContractLlmUsageRaw,
)

# =============================================================================
# ContractLlmCallMetrics: Negative Token Rejection
# =============================================================================


class TestContractLlmCallMetricsNegativeTokens:
    """Negative token counts must be rejected at the contract boundary."""

    @pytest.mark.unit
    def test_negative_prompt_tokens_rejected(self) -> None:
        """prompt_tokens=-1 is rejected by ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=-1,
            )

        error_text = str(exc_info.value).lower()
        assert "prompt_tokens" in error_text

    @pytest.mark.unit
    def test_negative_completion_tokens_rejected(self) -> None:
        """completion_tokens=-1 is rejected by ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                completion_tokens=-1,
            )

        error_text = str(exc_info.value).lower()
        assert "completion_tokens" in error_text

    @pytest.mark.unit
    def test_negative_total_tokens_rejected(self) -> None:
        """total_tokens=-1 is rejected by ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                total_tokens=-1,
            )

        error_text = str(exc_info.value).lower()
        assert "total_tokens" in error_text

    @pytest.mark.unit
    def test_large_negative_tokens_rejected(self) -> None:
        """Large negative token values are rejected."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=-999999,
            )


# =============================================================================
# ContractLlmCallMetrics: String/Type Coercion Rejection
# =============================================================================


class TestContractLlmCallMetricsTypeCoercion:
    """Non-numeric types must not be silently coerced to numbers."""

    @pytest.mark.unit
    def test_string_cost_rejected(self) -> None:
        """String cost value 'abc' is rejected (not silently coerced)."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                estimated_cost_usd="abc",  # type: ignore[arg-type]
            )

    @pytest.mark.unit
    def test_string_tokens_rejected(self) -> None:
        """String token value is rejected."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens="ten",  # type: ignore[arg-type]
            )

    @pytest.mark.unit
    def test_dict_tokens_rejected(self) -> None:
        """Dict passed as token value is rejected."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens={"value": 10},  # type: ignore[arg-type]
            )

    @pytest.mark.unit
    def test_list_tokens_rejected(self) -> None:
        """List passed as token value is rejected."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=[10],  # type: ignore[arg-type]
            )

    @pytest.mark.unit
    def test_empty_model_id_rejected(self) -> None:
        """Empty string model_id is rejected by min_length=1."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(model_id="")

    @pytest.mark.unit
    def test_missing_model_id_rejected(self) -> None:
        """model_id is a required field and missing raises ValidationError."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics()  # type: ignore[call-arg]

    @pytest.mark.unit
    def test_extra_fields_rejected(self) -> None:
        """Extra fields are rejected by extra='forbid' config."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                unknown_field="surprise",  # type: ignore[call-arg]
            )


# =============================================================================
# ContractLlmCallMetrics: Null-Cost Handling (NULL, not 0)
# =============================================================================


class TestContractLlmCallMetricsCostHandling:
    """Cost field must distinguish NULL (unknown) from 0 (free tier)."""

    @pytest.mark.unit
    def test_null_cost_allowed(self) -> None:
        """None estimated_cost_usd is accepted (unknown model cost)."""
        metrics = ContractLlmCallMetrics(
            model_id="unknown-model",
            estimated_cost_usd=None,
        )
        assert metrics.estimated_cost_usd is None

    @pytest.mark.unit
    def test_null_cost_is_not_zero(self) -> None:
        """None cost and zero cost are semantically distinct."""
        null_cost = ContractLlmCallMetrics(
            model_id="model-a",
            estimated_cost_usd=None,
        )
        zero_cost = ContractLlmCallMetrics(
            model_id="model-a",
            estimated_cost_usd=0.0,
        )

        assert null_cost.estimated_cost_usd is None
        assert zero_cost.estimated_cost_usd == 0.0
        assert null_cost.estimated_cost_usd != zero_cost.estimated_cost_usd

    @pytest.mark.unit
    def test_zero_cost_accepted(self) -> None:
        """Zero cost is valid (free tier models)."""
        metrics = ContractLlmCallMetrics(
            model_id="free-tier-model",
            estimated_cost_usd=0.0,
        )
        assert metrics.estimated_cost_usd == 0.0

    @pytest.mark.unit
    def test_small_positive_cost_accepted(self) -> None:
        """Very small positive cost is accepted."""
        metrics = ContractLlmCallMetrics(
            model_id="cheap-model",
            estimated_cost_usd=0.000001,
        )
        assert metrics.estimated_cost_usd == pytest.approx(0.000001)

    @pytest.mark.unit
    def test_negative_cost_rejected(self) -> None:
        """Negative cost is rejected by ge=0.0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                estimated_cost_usd=-0.01,
            )

        error_text = str(exc_info.value).lower()
        assert "estimated_cost_usd" in error_text

    @pytest.mark.unit
    def test_negative_latency_rejected(self) -> None:
        """Negative latency is rejected by ge=0.0 constraint."""
        with pytest.raises(ValidationError):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                latency_ms=-1.0,
            )

    @pytest.mark.unit
    def test_null_latency_allowed(self) -> None:
        """None latency is accepted (failed calls without timing)."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o",
            latency_ms=None,
        )
        assert metrics.latency_ms is None


# =============================================================================
# ContractLlmCallMetrics: Token Sum Consistency
# =============================================================================


class TestContractLlmCallMetricsTokenConsistency:
    """total_tokens must equal prompt_tokens + completion_tokens."""

    @pytest.mark.unit
    def test_mismatched_total_rejected(self) -> None:
        """total_tokens != prompt + completion raises ValidationError."""
        with pytest.raises(ValidationError, match="total_tokens"):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=999,
            )

    @pytest.mark.unit
    def test_consistent_total_accepted(self) -> None:
        """total_tokens == prompt + completion is accepted."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert metrics.total_tokens == 30

    @pytest.mark.unit
    def test_zero_total_with_nonzero_parts_rejected(self) -> None:
        """total_tokens=0 with nonzero prompt+completion is rejected."""
        with pytest.raises(ValidationError, match="total_tokens"):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=0,
            )

    @pytest.mark.unit
    def test_all_zero_tokens_accepted(self) -> None:
        """All token counts at zero is a valid state."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0

    @pytest.mark.unit
    def test_default_tokens_are_zero_and_consistent(self) -> None:
        """Default token values (all zero) are consistent."""
        metrics = ContractLlmCallMetrics(model_id="gpt-4o")
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0

    @pytest.mark.unit
    def test_normalized_token_mismatch_rejected(self) -> None:
        """Tokens in usage_normalized disagreeing with top-level are rejected."""
        with pytest.raises(ValidationError, match="disagree"):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                usage_normalized=ContractLlmUsageNormalized(
                    prompt_tokens=200,
                    completion_tokens=50,
                    total_tokens=250,
                    source=ContractEnumUsageSource.API,
                ),
            )

    @pytest.mark.unit
    def test_normalized_estimated_flag_mismatch_rejected(self) -> None:
        """usage_is_estimated disagreeing with normalized is rejected."""
        with pytest.raises(ValidationError, match="usage_is_estimated"):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                usage_is_estimated=True,
                usage_normalized=ContractLlmUsageNormalized(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    source=ContractEnumUsageSource.API,
                    usage_is_estimated=False,
                ),
            )


# =============================================================================
# ContractLlmCallMetrics: Frozen Immutability
# =============================================================================


class TestContractLlmCallMetricsImmutability:
    """Contracts must be immutable (frozen=True)."""

    @pytest.mark.unit
    def test_frozen_model_id(self) -> None:
        """Cannot mutate model_id after construction."""
        metrics = ContractLlmCallMetrics(model_id="gpt-4o")

        with pytest.raises(ValidationError):
            metrics.model_id = "changed"  # type: ignore[misc]

    @pytest.mark.unit
    def test_frozen_token_counts(self) -> None:
        """Cannot mutate token counts after construction."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        with pytest.raises(ValidationError):
            metrics.prompt_tokens = 999  # type: ignore[misc]

    @pytest.mark.unit
    def test_frozen_cost(self) -> None:
        """Cannot mutate cost after construction."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o", estimated_cost_usd=0.05
        )

        with pytest.raises(ValidationError):
            metrics.estimated_cost_usd = 999.0  # type: ignore[misc]


# =============================================================================
# ContractLlmCallMetrics: Timestamp Validation
# =============================================================================


class TestContractLlmCallMetricsTimestamp:
    """timestamp_iso field validation."""

    @pytest.mark.unit
    def test_valid_iso_timestamp_accepted(self) -> None:
        """Standard ISO-8601 timestamp is accepted."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o",
            timestamp_iso="2026-02-15T10:00:00Z",
        )
        assert metrics.timestamp_iso == "2026-02-15T10:00:00Z"

    @pytest.mark.unit
    def test_empty_timestamp_accepted(self) -> None:
        """Empty string timestamp is accepted (default)."""
        metrics = ContractLlmCallMetrics(
            model_id="gpt-4o",
            timestamp_iso="",
        )
        assert metrics.timestamp_iso == ""

    @pytest.mark.unit
    def test_invalid_timestamp_rejected(self) -> None:
        """Non-ISO timestamp string is rejected."""
        with pytest.raises(ValidationError, match="timestamp_iso"):
            ContractLlmCallMetrics(
                model_id="gpt-4o",
                timestamp_iso="not-a-date",
            )


# =============================================================================
# ContractLlmUsageNormalized: Validation
# =============================================================================


class TestContractLlmUsageNormalizedValidation:
    """Tests for ContractLlmUsageNormalized input validation."""

    @pytest.mark.unit
    def test_negative_prompt_tokens_rejected(self) -> None:
        """Negative prompt_tokens are rejected."""
        with pytest.raises(ValidationError):
            ContractLlmUsageNormalized(prompt_tokens=-1)

    @pytest.mark.unit
    def test_negative_completion_tokens_rejected(self) -> None:
        """Negative completion_tokens are rejected."""
        with pytest.raises(ValidationError):
            ContractLlmUsageNormalized(completion_tokens=-1)

    @pytest.mark.unit
    def test_negative_total_tokens_rejected(self) -> None:
        """Negative total_tokens are rejected."""
        with pytest.raises(ValidationError):
            ContractLlmUsageNormalized(total_tokens=-1)

    @pytest.mark.unit
    def test_token_sum_mismatch_rejected(self) -> None:
        """total_tokens != prompt + completion is rejected."""
        with pytest.raises(ValidationError, match="total_tokens"):
            ContractLlmUsageNormalized(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=999,
            )

    @pytest.mark.unit
    def test_estimated_source_requires_estimated_flag(self) -> None:
        """source=ESTIMATED with usage_is_estimated=False raises."""
        with pytest.raises(ValidationError, match="usage_is_estimated"):
            ContractLlmUsageNormalized(
                source=ContractEnumUsageSource.ESTIMATED,
                usage_is_estimated=False,
            )

    @pytest.mark.unit
    def test_api_source_rejects_estimated_flag(self) -> None:
        """source=API with usage_is_estimated=True raises."""
        with pytest.raises(ValidationError, match="usage_is_estimated"):
            ContractLlmUsageNormalized(
                source=ContractEnumUsageSource.API,
                usage_is_estimated=True,
            )

    @pytest.mark.unit
    def test_zero_tokens_accepted(self) -> None:
        """All-zero tokens is a valid state (no usage data)."""
        normalized = ContractLlmUsageNormalized(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        assert normalized.prompt_tokens == 0
        assert normalized.total_tokens == 0

    @pytest.mark.unit
    def test_frozen_immutability(self) -> None:
        """ContractLlmUsageNormalized is frozen."""
        normalized = ContractLlmUsageNormalized()

        with pytest.raises(ValidationError):
            normalized.prompt_tokens = 100  # type: ignore[misc]

    @pytest.mark.unit
    def test_extra_fields_rejected(self) -> None:
        """Extra fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            ContractLlmUsageNormalized(
                unknown_field="bad",  # type: ignore[call-arg]
            )


# =============================================================================
# ContractLlmUsageRaw: Validation
# =============================================================================


class TestContractLlmUsageRawValidation:
    """Tests for ContractLlmUsageRaw input validation."""

    @pytest.mark.unit
    def test_default_empty_raw_data(self) -> None:
        """Default raw_data is an empty dict."""
        raw = ContractLlmUsageRaw()
        assert raw.raw_data == {}

    @pytest.mark.unit
    def test_provider_stored(self) -> None:
        """Provider identifier is stored."""
        raw = ContractLlmUsageRaw(provider="openai")
        assert raw.provider == "openai"

    @pytest.mark.unit
    def test_frozen_immutability(self) -> None:
        """ContractLlmUsageRaw is frozen."""
        raw = ContractLlmUsageRaw(provider="openai")

        with pytest.raises(ValidationError):
            raw.provider = "changed"  # type: ignore[misc]

    @pytest.mark.unit
    def test_extra_fields_rejected(self) -> None:
        """Extra fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            ContractLlmUsageRaw(
                mystery="field",  # type: ignore[call-arg]
            )


# =============================================================================
# Aggregate Consistency: NULL costs excluded, not zero-filled
# =============================================================================


class TestAggregateConsistencyContracts:
    """Verify that contract-level data preserves NULL/zero distinction.

    When ContractLlmCallMetrics records flow through to aggregation,
    NULL costs must remain NULL (excluded from sums), not silently
    become 0 (which would corrupt averages and totals).
    """

    @pytest.mark.unit
    def test_null_cost_record_preserves_null(self) -> None:
        """A record with None cost preserves that distinction."""
        record_null = ContractLlmCallMetrics(
            model_id="unknown-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=None,
        )
        assert record_null.estimated_cost_usd is None

    @pytest.mark.unit
    def test_zero_cost_record_preserves_zero(self) -> None:
        """A record with 0.0 cost preserves that value."""
        record_zero = ContractLlmCallMetrics(
            model_id="free-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.0,
        )
        assert record_zero.estimated_cost_usd == 0.0

    @pytest.mark.unit
    def test_mixed_cost_records_preserve_distinction(self) -> None:
        """Multiple records with mixed NULL/zero/positive costs are all distinct."""
        null_cost = ContractLlmCallMetrics(
            model_id="m1", estimated_cost_usd=None
        )
        zero_cost = ContractLlmCallMetrics(
            model_id="m2", estimated_cost_usd=0.0
        )
        positive_cost = ContractLlmCallMetrics(
            model_id="m3", estimated_cost_usd=0.005
        )

        costs = [null_cost.estimated_cost_usd, zero_cost.estimated_cost_usd, positive_cost.estimated_cost_usd]

        # NULL should be filtered out before aggregation, not treated as 0
        non_null_costs = [c for c in costs if c is not None]
        assert len(non_null_costs) == 2
        assert 0.0 in non_null_costs
        assert pytest.approx(0.005) in non_null_costs

    @pytest.mark.unit
    def test_serialization_preserves_null_cost(self) -> None:
        """model_dump preserves None cost (not coerced to 0)."""
        record = ContractLlmCallMetrics(
            model_id="model",
            estimated_cost_usd=None,
        )
        dumped = record.model_dump()
        assert dumped["estimated_cost_usd"] is None
