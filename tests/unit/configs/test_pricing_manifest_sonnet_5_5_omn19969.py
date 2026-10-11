# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19969: the shipped manifest prices the savings baseline's default model.

The savings baseline defaults to ``claude-sonnet-5-5``. Until the manifest
prices it, every savings figure resolves to BASELINE_UNRESOLVED. These tests
fail if the entry is missing, if either price differs from the provider's
published rate, or if the entry claims more confidence than a documentation
price carries.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.models.pricing.model_pricing_table import (
    _DEFAULT_MANIFEST_PATH,
    ModelPricingTable,
)

_MODEL = "claude-sonnet-5-5"


@pytest.mark.unit
def test_shipped_manifest_prices_the_default_baseline_model() -> None:
    table = ModelPricingTable.from_yaml()
    entry = table.get_entry(_MODEL)
    assert entry is not None, f"{_MODEL} is not priced in the shipped manifest"
    assert entry.input_cost_per_1k == pytest.approx(0.002)
    assert entry.output_cost_per_1k == pytest.approx(0.010)


@pytest.mark.unit
def test_default_baseline_cost_estimate_uses_the_published_rate() -> None:
    table = ModelPricingTable.from_yaml()
    # 1000 input + 1000 output tokens = 0.002 + 0.010
    estimate = table.estimate_cost(_MODEL, 1000, 1000)
    assert estimate.estimated_cost_usd == pytest.approx(0.012)


@pytest.mark.unit
def test_documentation_price_is_not_presented_as_measured() -> None:
    raw = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = raw["models"][_MODEL]
    assert entry["confidence"] == "LOW_CONFIDENCE"
    assert (
        entry["source"] == "PROVIDER_PUBLISHED_PRICING"
    )  # OMN-20387: cited provider page
    assert entry["sample_count"] == 0
    assert entry["evidence"]["authoritative"] is False
    assert entry["effective_date"]
