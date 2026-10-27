# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21431: the shipped manifest prices the Opus 5.5 family.

A Claude Code export is mostly ``claude-opus-5-5`` turns, which were shown
as not priced. These tests fail if any of the four family rows
(``claude-opus-5-5``, ``claude-opus-5``, ``claude-fable-5``,
``claude-fable-5-1``) is missing, if either rate differs from Anthropic's
published pricing table, or if the evidence block does not cite the
provider page with the correct retrieval date and confidence.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.models.pricing.model_pricing_table import (
    _DEFAULT_MANIFEST_PATH,
    ModelPricingTable,
)

# USD per 1 000 tokens, read from the provider page on 2026-10-27.
_PUBLISHED: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (0.004, 0.020),
    "claude-opus-5": (0.005, 0.025),
    "claude-fable-5": (0.010, 0.050),
    "claude-fable-5-1": (0.010, 0.050),
}

_PROVIDER_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_EFFECTIVE_DATE = "2026-10-27"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "input_per_1k", "output_per_1k"),
    [(k, *v) for k, v in _PUBLISHED.items()],
)
def test_opus_5_5_family_row_matches_the_published_rate(
    model_id: str,
    input_per_1k: float,
    output_per_1k: float,
) -> None:
    table = ModelPricingTable.from_yaml()
    entry = table.get_entry(model_id)
    assert entry is not None, f"{model_id} is not priced in the shipped manifest"
    assert entry.input_cost_per_1k == pytest.approx(input_per_1k)
    assert entry.output_cost_per_1k == pytest.approx(output_per_1k)
    assert entry.source == "PROVIDER_PUBLISHED_PRICING"
    assert entry.effective_date == _EFFECTIVE_DATE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "input_per_1k", "output_per_1k"),
    [(k, *v) for k, v in _PUBLISHED.items()],
)
def test_opus_5_5_family_row_cites_the_provider_page(
    model_id: str,
    input_per_1k: float,
    output_per_1k: float,
) -> None:
    raw = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = raw["models"][model_id]
    assert entry["evidence"]["source_url"] == _PROVIDER_URL
    assert entry["evidence"]["retrieved_at"] == _EFFECTIVE_DATE
    assert entry["evidence"]["authoritative"] is False
    assert entry["evidence"]["note"], f"{model_id} evidence note must be non-empty"
    assert entry["confidence"] == "LOW_CONFIDENCE"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "input_per_1k", "output_per_1k"),
    [(k, *v) for k, v in _PUBLISHED.items()],
)
def test_opus_5_5_family_prompt_is_priced(
    model_id: str,
    input_per_1k: float,
    output_per_1k: float,
) -> None:
    table = ModelPricingTable.from_yaml()
    # 1 000 input + 1 000 output tokens
    estimate = table.estimate_cost(model_id, 1000, 1000)
    assert estimate.estimated_cost_usd == pytest.approx(input_per_1k + output_per_1k)
