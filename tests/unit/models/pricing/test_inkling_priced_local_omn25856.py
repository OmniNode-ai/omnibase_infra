# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856: Inkling-2-Mini, served from an owned lab GPU, is priced at zero marginal API cost.

The dashboard builds its priced-model set from the bundled pricing manifest; with no
Inkling row every Inkling run stays unpriced.
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.pricing.model_pricing_table import (
    _DEFAULT_MANIFEST_PATH,
    ModelPricingTable,
)

INKLING = "Inkling-2-Mini"


@pytest.mark.unit
def test_inkling_2_mini_has_a_zero_marginal_cost_row() -> None:
    table = ModelPricingTable.from_yaml(_DEFAULT_MANIFEST_PATH)
    assert table.has_model(INKLING)
    entry = table.get_entry(INKLING)
    assert entry is not None
    assert entry.input_cost_per_1k == 0.0
    assert entry.output_cost_per_1k == 0.0
    assert entry.source == "LOCAL_ZERO_API_COST_POLICY"


@pytest.mark.unit
def test_unknown_inkling_model_is_refused() -> None:
    """Known-bad control: an unlisted Inkling id stays unpriced."""
    table = ModelPricingTable.from_yaml(_DEFAULT_MANIFEST_PATH)
    assert not table.has_model("Inkling-9-Nonexistent")
    assert table.get_entry("Inkling-9-Nonexistent") is None
