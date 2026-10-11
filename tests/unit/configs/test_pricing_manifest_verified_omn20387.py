# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20387: every cloud row is checked against the provider's published rate.

The expected rates below were read from each provider's published pricing page on
2026-10-06 (USD per 1M tokens, standard tier, divided by 1000 here):

- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
- OpenAI: https://developers.openai.com/api/docs/pricing
- Google: https://ai.google.dev/gemini-api/docs/pricing

A row whose rate could not be read from its provider's page is left out of the
manifest, so it resolves as unpriced rather than at a guessed rate.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.models.pricing.model_pricing_table import (
    _DEFAULT_MANIFEST_PATH,
    ModelPricingTable,
)

_RETRIEVED = "2026-10-06"
_PUBLISHED_SOURCE = "PROVIDER_PUBLISHED_PRICING"
_LOCAL_SOURCE = "LOCAL_ZERO_API_COST_POLICY"
_ANTHROPIC_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_OPENAI_URL = "https://developers.openai.com/api/docs/pricing"
_GOOGLE_URL = "https://ai.google.dev/gemini-api/docs/pricing"

# model id -> (input per 1k, output per 1k, provider page)
_PUBLISHED: dict[str, tuple[float, float, str]] = {
    "claude-opus-4-6": (0.005, 0.025, _ANTHROPIC_URL),
    "claude-opus-4-5": (0.005, 0.025, _ANTHROPIC_URL),
    "claude-sonnet-4-20250514": (0.003, 0.015, _ANTHROPIC_URL),
    "claude-sonnet-4-6": (0.003, 0.015, _ANTHROPIC_URL),
    "claude-sonnet-5-5": (0.002, 0.010, _ANTHROPIC_URL),
    "claude-haiku-4-5": (0.001, 0.005, _ANTHROPIC_URL),
    "claude-haiku-3-5": (0.0008, 0.004, _ANTHROPIC_URL),
    "gpt-6-astra": (0.010, 0.050, _OPENAI_URL),
    "gpt-6.1-sol": (0.002, 0.010, _OPENAI_URL),
    "gpt-6-sol": (0.002, 0.010, _OPENAI_URL),
    "gpt-6-luna": (0.0001, 0.0005, _OPENAI_URL),
    "gpt-5.6-sol": (0.004, 0.020, _OPENAI_URL),
    "gpt-5.6-terra": (0.002, 0.012, _OPENAI_URL),
    "gpt-5.6-luna": (0.0002, 0.0012, _OPENAI_URL),
    "gpt-5.5": (0.005, 0.030, _OPENAI_URL),
    "gpt-4o": (0.0025, 0.010, _OPENAI_URL),
    "gpt-4o-mini": (0.00015, 0.0006, _OPENAI_URL),
    "gpt-4-turbo": (0.010, 0.030, _OPENAI_URL),
    "o1": (0.015, 0.060, _OPENAI_URL),
    "o3-mini": (0.0011, 0.0044, _OPENAI_URL),
    "gemini-2.5-flash": (0.0003, 0.0025, _GOOGLE_URL),
    "gemini-2.5-flash-lite": (0.0001, 0.0004, _GOOGLE_URL),
}

# The OpenAI models ChatGPT currently serves (AC1), as named on the OpenAI page.
_CHATGPT_MODELS = (
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
)

# Rows no provider page priced on 2026-10-06: they must resolve as unpriced.
_UNPUBLISHED = ("gemini-2.0-flash", "gemini-1.5-pro", "o1-mini")


def _raw_models() -> dict[str, dict[str, object]]:
    data = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    models: dict[str, dict[str, object]] = data["models"]
    return models


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_id", ["claude-haiku-4-5", "claude-sonnet-4-6", *_CHATGPT_MODELS]
)
def test_required_models_resolve_a_price(model_id: str) -> None:
    estimate = ModelPricingTable.from_yaml().estimate_cost(model_id, 1000, 1000)
    assert estimate.estimated_cost_usd is not None, f"{model_id} is unpriced"
    assert estimate.estimated_cost_usd > 0.0


@pytest.mark.unit
def test_an_unknown_model_resolves_unpriced() -> None:
    table = ModelPricingTable.from_yaml()
    estimate = table.estimate_cost("not-a-real-model-omn20387", 1000, 1000)
    assert estimate.estimated_cost_usd is None
    assert table.get_entry("not-a-real-model-omn20387") is None


@pytest.mark.unit
@pytest.mark.parametrize("model_id", _UNPUBLISHED)
def test_rows_without_a_published_rate_resolve_unpriced(model_id: str) -> None:
    table = ModelPricingTable.from_yaml()
    assert table.get_entry(model_id) is None
    assert table.estimate_cost(model_id, 1000, 1000).estimated_cost_usd is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_id", "input_cost_per_1k", "output_cost_per_1k", "source_url"),
    [(k, *v) for k, v in _PUBLISHED.items()],
)
def test_cloud_row_matches_the_published_rate(
    model_id: str,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
    source_url: str,
) -> None:
    entry = ModelPricingTable.from_yaml().get_entry(model_id)
    assert entry is not None, f"{model_id} is not priced in the shipped manifest"
    assert entry.input_cost_per_1k == pytest.approx(input_cost_per_1k)
    assert entry.output_cost_per_1k == pytest.approx(output_cost_per_1k)
    assert entry.source == _PUBLISHED_SOURCE
    assert entry.effective_date == _RETRIEVED

    evidence = _raw_models()[model_id]["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["source_url"] == source_url
    assert evidence["retrieved_at"] == _RETRIEVED
    assert evidence["authoritative"] is False
    assert evidence["note"]


@pytest.mark.unit
def test_every_cloud_row_is_cited() -> None:
    cloud = {
        model_id: entry
        for model_id, entry in _raw_models().items()
        if entry["source"] != _LOCAL_SOURCE
    }
    assert set(cloud) == set(_PUBLISHED)
    for model_id, entry in cloud.items():
        assert entry["source"] == _PUBLISHED_SOURCE, model_id
