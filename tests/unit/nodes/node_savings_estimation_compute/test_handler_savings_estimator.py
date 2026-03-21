# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for HandlerSavingsEstimator compute handler.

Covers all 5 savings categories, tier separation, missing signals,
and unknown model handling.

Tracking:
    - OMN-5547: Create HandlerSavingsEstimator compute handler
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable
from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_estimator import (
    HandlerSavingsEstimator,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    ModelInjectionSignal,
    ModelLlmCallRecord,
    ModelSavingsBaselineConfig,
    ModelSavingsInput,
    ModelValidatorCatchSignal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PRICING_DATA = {
    "schema_version": "1.0.0",
    "models": {
        "claude-opus-4-6": {
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
            "effective_date": "2026-02-01",
        },
        "qwen3-coder-30b-a3b": {
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "effective_date": "2026-03-19",
        },
        "qwen3-14b": {
            "input_cost_per_1k": 0.0,
            "output_cost_per_1k": 0.0,
            "effective_date": "2026-03-19",
        },
    },
}


@pytest.fixture
def pricing_table() -> ModelPricingTable:
    return ModelPricingTable.from_dict(_PRICING_DATA)


@pytest.fixture
def handler(pricing_table: ModelPricingTable) -> HandlerSavingsEstimator:
    return HandlerSavingsEstimator(pricing_table)


# ---------------------------------------------------------------------------
# Category 1: Local routing → exact reference pricing delta
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_routing_savings(handler: HandlerSavingsEstimator) -> None:
    """Local model calls produce direct savings equal to reference model cost."""
    savings_input = ModelSavingsInput(
        session_id="sess-1",
        correlation_id="corr-1",
        treatment_group="treatment",
        llm_calls=[
            ModelLlmCallRecord(
                model_id="qwen3-coder-30b-a3b",
                prompt_tokens=1000,
                completion_tokens=500,
            ),
        ],
    )

    result = await handler.handle(savings_input)

    # Reference cost = (1000/1000 * 0.015) + (500/1000 * 0.075) = 0.015 + 0.0375 = 0.0525
    assert result["direct_savings_usd"] == pytest.approx(0.0525, abs=1e-6)
    assert result["direct_tokens_saved"] == 1500

    # Find local_routing category
    local_cats = [c for c in result["categories"] if c["category"] == "local_routing"]
    assert len(local_cats) == 1
    assert local_cats[0]["tier"] == "direct"
    assert local_cats[0]["confidence"] == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_local_calls(handler: HandlerSavingsEstimator) -> None:
    """Multiple local model calls accumulate savings."""
    savings_input = ModelSavingsInput(
        session_id="sess-2",
        correlation_id="corr-2",
        treatment_group="treatment",
        llm_calls=[
            ModelLlmCallRecord(
                model_id="qwen3-coder-30b-a3b",
                prompt_tokens=1000,
                completion_tokens=500,
            ),
            ModelLlmCallRecord(
                model_id="qwen3-14b", prompt_tokens=2000, completion_tokens=1000
            ),
        ],
    )

    result = await handler.handle(savings_input)

    # Call 1: (1000/1000*0.015) + (500/1000*0.075) = 0.0525
    # Call 2: (2000/1000*0.015) + (1000/1000*0.075) = 0.03 + 0.075 = 0.105
    # Total: 0.1575
    assert result["direct_savings_usd"] == pytest.approx(0.1575, abs=1e-6)
    assert result["direct_tokens_saved"] == 4500


# ---------------------------------------------------------------------------
# Category 2: Injection signal → multiplier applied
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pattern_injection_savings(handler: HandlerSavingsEstimator) -> None:
    """Injection signals produce heuristic savings via regen multiplier."""
    savings_input = ModelSavingsInput(
        session_id="sess-3",
        correlation_id="corr-3",
        treatment_group="treatment",
        injection_signals=[
            ModelInjectionSignal(tokens_injected=500, patterns_count=2),
        ],
        baseline_config=ModelSavingsBaselineConfig(regen_multiplier=3.0),
    )

    result = await handler.handle(savings_input)

    # tokens_saved = 500 * 3.0 = 1500 (input tokens at reference price)
    # cost = 1500/1000 * 0.015 = 0.0225
    injection_cats = [
        c for c in result["categories"] if c["category"] == "pattern_injection"
    ]
    assert len(injection_cats) == 1
    assert injection_cats[0]["tokens_saved"] == 1500
    assert injection_cats[0]["cost_saved_usd"] == pytest.approx(0.0225, abs=1e-6)
    assert injection_cats[0]["confidence"] == 0.8
    assert injection_cats[0]["tier"] == "heuristic"


# ---------------------------------------------------------------------------
# Category 5: Mixed severity catches → severity-weighted
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validator_catches_severity_weighted(
    handler: HandlerSavingsEstimator,
) -> None:
    """Validator catches produce severity-weighted heuristic savings."""
    savings_input = ModelSavingsInput(
        session_id="sess-5",
        correlation_id="corr-5",
        treatment_group="treatment",
        validator_catches=[
            ModelValidatorCatchSignal(validator_type="pre_commit", severity="error"),
            ModelValidatorCatchSignal(validator_type="poly_enforcer", severity="error"),
            ModelValidatorCatchSignal(validator_type="code_review", severity="warning"),
        ],
    )

    result = await handler.handle(savings_input)

    # error_pre_commit: 15000 tokens, confidence 0.7
    # error_poly_enforcer: 10000 tokens, confidence 0.9
    # warning_code_review: 5000 tokens, confidence 0.5
    # total tokens: 30000
    catch_cats = [
        c for c in result["categories"] if c["category"] == "validator_catches"
    ]
    assert len(catch_cats) == 1
    assert catch_cats[0]["tokens_saved"] == 30000
    assert catch_cats[0]["tier"] == "heuristic"

    # Weighted confidence = (0.7*15000 + 0.9*10000 + 0.5*5000) / 30000
    # = (10500 + 9000 + 2500) / 30000 = 22000 / 30000 = 0.7333...
    assert catch_cats[0]["confidence"] == pytest.approx(0.7333, abs=1e-3)


# ---------------------------------------------------------------------------
# Tier separation: direct_savings only includes Category 1
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tier_separation(handler: HandlerSavingsEstimator) -> None:
    """direct_savings_usd only includes Tier A (local routing)."""
    savings_input = ModelSavingsInput(
        session_id="sess-6",
        correlation_id="corr-6",
        treatment_group="treatment",
        llm_calls=[
            ModelLlmCallRecord(
                model_id="qwen3-coder-30b-a3b",
                prompt_tokens=1000,
                completion_tokens=500,
            ),
        ],
        injection_signals=[
            ModelInjectionSignal(tokens_injected=500, patterns_count=1),
        ],
        validator_catches=[
            ModelValidatorCatchSignal(validator_type="pre_commit", severity="error"),
        ],
    )

    result = await handler.handle(savings_input)

    # Direct savings = only local routing
    assert result["direct_savings_usd"] == pytest.approx(0.0525, abs=1e-6)
    # Total includes heuristic categories too
    assert result["estimated_total_savings_usd"] > result["direct_savings_usd"]


# ---------------------------------------------------------------------------
# Missing signals: absent categories report zero with confidence 0.0
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_signals_zero_confidence(
    handler: HandlerSavingsEstimator,
) -> None:
    """Absent categories report zero savings and 0.0 confidence."""
    savings_input = ModelSavingsInput(
        session_id="sess-7",
        correlation_id="corr-7",
        treatment_group="treatment",
    )

    result = await handler.handle(savings_input)

    assert result["direct_savings_usd"] == 0.0
    assert result["estimated_total_savings_usd"] == 0.0
    assert result["direct_confidence"] == 0.0
    assert result["heuristic_confidence_avg"] == 0.0
    assert result["completeness_status"] == "phase_limited"

    # Delegation and RAG categories present with 0 confidence
    delegation_cats = [
        c for c in result["categories"] if c["category"] == "agent_delegation"
    ]
    assert len(delegation_cats) == 1
    assert delegation_cats[0]["confidence"] == 0.0
    assert delegation_cats[0]["tokens_saved"] == 0


# ---------------------------------------------------------------------------
# Unknown model: returns completeness_status="partial"
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_model_partial_completeness(
    handler: HandlerSavingsEstimator,
) -> None:
    """Unknown model in LLM calls sets completeness_status to 'partial'."""
    savings_input = ModelSavingsInput(
        session_id="sess-8",
        correlation_id="corr-8",
        treatment_group="treatment",
        llm_calls=[
            ModelLlmCallRecord(
                model_id="unknown-model-xyz", prompt_tokens=1000, completion_tokens=500
            ),
        ],
    )

    result = await handler.handle(savings_input)

    assert result["completeness_status"] == "partial"
