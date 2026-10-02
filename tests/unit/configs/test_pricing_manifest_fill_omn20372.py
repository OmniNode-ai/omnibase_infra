# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20372: the shipped manifest prices missing paid and local models."""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.models.pricing.model_pricing_table import (
    _DEFAULT_MANIFEST_PATH,
    ModelPricingTable,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "input_cost_per_1k", "output_cost_per_1k"),
    [
        ("claude-haiku-4-5", 0.001, 0.005),
        ("claude-sonnet-4-6", 0.003, 0.015),
        ("claude-opus-4-5", 0.005, 0.025),
        ("gemini-2.5-flash", 0.0003, 0.0025),
        ("gemini-2.5-flash-lite", 0.0001, 0.0004),
    ],
)
def test_paid_models_have_documented_rates(
    model_id: str, input_cost_per_1k: float, output_cost_per_1k: float
) -> None:
    table = ModelPricingTable.from_yaml()
    entry = table.get_entry(model_id)
    assert entry is not None, f"{model_id} is not priced in the shipped manifest"
    assert entry.input_cost_per_1k == pytest.approx(input_cost_per_1k)
    assert entry.output_cost_per_1k == pytest.approx(output_cost_per_1k)

    raw = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_entry = raw["models"][model_id]
    assert raw_entry["confidence"] == "LOW_CONFIDENCE"
    assert raw_entry["source"] == "FALLBACK_PROVIDER_DOCUMENTATION"
    assert raw_entry["evidence"]["authoritative"] is False
    assert raw_entry["evidence"]["source_url"].startswith("https://")
    assert raw_entry["evidence"]["retrieved_at"] == raw_entry["effective_date"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id", ["qwen3.8", "Qwen3.8-27B", "Qwen3-Coder-30B", "DeepSeek-R1-14B"]
)
def test_local_models_have_zero_api_cost(model_id: str) -> None:
    table = ModelPricingTable.from_yaml()
    entry = table.get_entry(model_id)
    assert entry is not None, f"{model_id} is not priced in the shipped manifest"
    assert entry.input_cost_per_1k == 0.0
    assert entry.output_cost_per_1k == 0.0

    raw = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert raw["models"][model_id]["source"] == "LOCAL_ZERO_API_COST_POLICY"


@pytest.mark.unit
def test_haiku_cost_estimate_uses_the_documented_rate() -> None:
    table = ModelPricingTable.from_yaml()
    estimate = table.estimate_cost("claude-haiku-4-5", 1000, 1000)
    assert estimate.estimated_cost_usd == pytest.approx(0.006)
