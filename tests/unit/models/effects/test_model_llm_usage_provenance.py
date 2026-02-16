# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelLlmUsage provenance tracking fields.

Tests the ``usage_source`` and ``raw_provider_usage`` fields added in
OMN-2318 to bridge ModelLlmUsage with SPI LLM cost tracking contracts.

Tests cover:
- Default values for new fields (backwards compatibility)
- Explicit usage_source construction for all enum members
- raw_provider_usage preservation and serialization
- Combination of new fields with existing token fields
- Frozen model enforcement on new fields
- Serialization round-trips with new fields
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_spi.contracts.measurement import ContractEnumUsageSource

# ============================================================================
# Backwards Compatibility
# ============================================================================


class TestBackwardsCompatibility:
    """Verify that existing callers constructing ModelLlmUsage without
    new fields continue to work identically."""

    def test_default_construction_unchanged(self) -> None:
        """Default ModelLlmUsage() has MISSING source and no raw data."""
        usage = ModelLlmUsage()

        assert usage.usage_source == ContractEnumUsageSource.MISSING
        assert usage.raw_provider_usage is None
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0
        assert usage.tokens_total == 0
        assert usage.cost_usd is None

    def test_token_only_construction_unchanged(self) -> None:
        """Construction with only token fields defaults new fields."""
        usage = ModelLlmUsage(tokens_input=100, tokens_output=50)

        assert usage.tokens_total == 150
        assert usage.usage_source == ContractEnumUsageSource.MISSING
        assert usage.raw_provider_usage is None


# ============================================================================
# Usage Source Field
# ============================================================================


class TestUsageSource:
    """Tests for the usage_source provenance field."""

    def test_source_api(self) -> None:
        """usage_source=API is accepted."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.API,
        )

        assert usage.usage_source == ContractEnumUsageSource.API

    def test_source_estimated(self) -> None:
        """usage_source=ESTIMATED is accepted."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.ESTIMATED,
        )

        assert usage.usage_source == ContractEnumUsageSource.ESTIMATED

    def test_source_missing(self) -> None:
        """usage_source=MISSING is accepted (and is the default)."""
        usage = ModelLlmUsage(
            usage_source=ContractEnumUsageSource.MISSING,
        )

        assert usage.usage_source == ContractEnumUsageSource.MISSING

    def test_source_from_string(self) -> None:
        """usage_source accepts string values matching enum."""
        usage = ModelLlmUsage(
            tokens_input=10,
            usage_source="api",  # type: ignore[arg-type]
        )

        assert usage.usage_source == ContractEnumUsageSource.API

    def test_source_invalid_rejected(self) -> None:
        """Invalid usage_source string is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmUsage(usage_source="invalid_source")  # type: ignore[arg-type]


# ============================================================================
# Raw Provider Usage Field
# ============================================================================


class TestRawProviderUsage:
    """Tests for the raw_provider_usage audit trail field."""

    def test_none_by_default(self) -> None:
        """raw_provider_usage defaults to None."""
        usage = ModelLlmUsage()

        assert usage.raw_provider_usage is None

    def test_preserves_openai_format(self) -> None:
        """OpenAI-style usage dict is preserved verbatim."""
        raw = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            raw_provider_usage=raw,
        )

        assert usage.raw_provider_usage == raw

    def test_preserves_ollama_format(self) -> None:
        """Ollama-style usage dict with different field names is preserved."""
        raw = {
            "prompt_eval_count": 100,
            "eval_count": 50,
            "total_duration": 123456789,
            "load_duration": 1000,
        }
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            raw_provider_usage=raw,
        )

        assert usage.raw_provider_usage == raw
        assert usage.raw_provider_usage["total_duration"] == 123456789

    def test_empty_dict_accepted(self) -> None:
        """Empty dict is valid for raw_provider_usage."""
        usage = ModelLlmUsage(raw_provider_usage={})

        assert usage.raw_provider_usage == {}

    def test_nested_dicts_preserved(self) -> None:
        """Nested structures in raw usage data are preserved."""
        raw = {
            "prompt_tokens": 100,
            "completion_tokens_details": {
                "reasoning_tokens": 30,
                "accepted_prediction_tokens": 20,
            },
        }
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            raw_provider_usage=raw,
        )

        assert usage.raw_provider_usage is not None
        assert (
            usage.raw_provider_usage["completion_tokens_details"]["reasoning_tokens"]
            == 30
        )  # type: ignore[index]


# ============================================================================
# Immutability on New Fields
# ============================================================================


class TestImmutabilityNewFields:
    """Verify frozen=True applies to new fields."""

    def test_usage_source_immutable(self) -> None:
        """Cannot reassign usage_source on frozen model."""
        usage = ModelLlmUsage(usage_source=ContractEnumUsageSource.API)

        with pytest.raises(ValidationError):
            usage.usage_source = ContractEnumUsageSource.MISSING  # type: ignore[misc]

    def test_raw_provider_usage_immutable(self) -> None:
        """Cannot reassign raw_provider_usage on frozen model."""
        usage = ModelLlmUsage(raw_provider_usage={"prompt_tokens": 100})

        with pytest.raises(ValidationError):
            usage.raw_provider_usage = {}  # type: ignore[misc]


# ============================================================================
# Serialization with New Fields
# ============================================================================


class TestSerializationNewFields:
    """Tests for serialization round-trips including new fields."""

    def test_model_dump_includes_new_fields(self) -> None:
        """model_dump includes usage_source and raw_provider_usage."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.API,
            raw_provider_usage={"prompt_tokens": 100},
        )

        data = usage.model_dump()

        assert data["usage_source"] == "api"
        assert data["raw_provider_usage"] == {"prompt_tokens": 100}

    def test_model_dump_json_roundtrip(self) -> None:
        """JSON round-trip preserves new fields."""
        usage = ModelLlmUsage(
            tokens_input=100,
            tokens_output=50,
            usage_source=ContractEnumUsageSource.API,
            raw_provider_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        json_str = usage.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["usage_source"] == "api"
        assert parsed["raw_provider_usage"]["prompt_tokens"] == 100
        assert parsed["raw_provider_usage"]["completion_tokens"] == 50

    def test_model_validate_roundtrip(self) -> None:
        """model_dump -> model_validate round-trip preserves all fields."""
        original = ModelLlmUsage(
            tokens_input=200,
            tokens_output=80,
            usage_source=ContractEnumUsageSource.ESTIMATED,
            raw_provider_usage={"estimated_via": "tiktoken"},
        )

        data = original.model_dump()
        restored = ModelLlmUsage.model_validate(data)

        assert restored == original
        assert restored.usage_source == ContractEnumUsageSource.ESTIMATED
        assert restored.raw_provider_usage == {"estimated_via": "tiktoken"}

    def test_model_dump_none_raw_usage(self) -> None:
        """Serialization handles None raw_provider_usage."""
        usage = ModelLlmUsage()

        data = usage.model_dump()

        assert data["raw_provider_usage"] is None
        assert data["usage_source"] == "missing"


# ============================================================================
# Combined Field Interactions
# ============================================================================


class TestCombinedFieldInteractions:
    """Tests for interactions between token fields and new provenance fields."""

    def test_api_source_with_full_tokens(self) -> None:
        """API source with complete token data and raw usage."""
        usage = ModelLlmUsage(
            tokens_input=500,
            tokens_output=200,
            cost_usd=0.05,
            usage_source=ContractEnumUsageSource.API,
            raw_provider_usage={
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
            },
        )

        assert usage.tokens_total == 700
        assert usage.cost_usd == pytest.approx(0.05)
        assert usage.usage_source == ContractEnumUsageSource.API

    def test_estimated_source_with_zero_cost(self) -> None:
        """Estimated source with zero tokens and no cost."""
        usage = ModelLlmUsage(
            tokens_input=0,
            tokens_output=0,
            usage_source=ContractEnumUsageSource.ESTIMATED,
        )

        assert usage.tokens_total == 0
        assert usage.usage_source == ContractEnumUsageSource.ESTIMATED

    def test_missing_source_with_zero_tokens(self) -> None:
        """Missing source results in all-zero tokens."""
        usage = ModelLlmUsage(
            usage_source=ContractEnumUsageSource.MISSING,
        )

        assert usage.tokens_input == 0
        assert usage.tokens_output == 0
        assert usage.tokens_total == 0
