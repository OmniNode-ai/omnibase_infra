# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""CI schema validation for the pricing manifest YAML file.

Ensures the bundled pricing_manifest.yaml is always valid and loadable.
This test runs in CI to catch manifest corruption before merge.

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.pricing.model_pricing_entry import ModelPricingEntry
from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable

_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "configs"
    / "pricing_manifest.yaml"
)


@pytest.mark.unit
class TestPricingManifestSchema:
    """CI validation that the bundled pricing manifest is always valid."""

    def test_manifest_file_exists(self) -> None:
        """The pricing manifest YAML file must exist."""
        assert _MANIFEST_PATH.exists(), (
            f"Pricing manifest not found at {_MANIFEST_PATH}. "
            "Did you delete or move configs/pricing_manifest.yaml?"
        )

    def test_manifest_is_valid_yaml(self) -> None:
        """The manifest must be valid YAML."""
        raw = _MANIFEST_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        assert isinstance(data, dict), "Manifest root must be a YAML mapping"

    def test_manifest_has_schema_version(self) -> None:
        """The manifest must contain a schema_version field."""
        raw = _MANIFEST_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        assert "schema_version" in data, "Manifest missing 'schema_version'"
        assert isinstance(data["schema_version"], str), (
            "schema_version must be a string"
        )

    def test_manifest_has_models_section(self) -> None:
        """The manifest must contain a models section."""
        raw = _MANIFEST_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        assert "models" in data, "Manifest missing 'models' section"
        assert isinstance(data["models"], dict), "models must be a mapping"

    def test_manifest_loads_as_pricing_table(self) -> None:
        """The manifest must load successfully as a ModelPricingTable."""
        table = ModelPricingTable.from_yaml(_MANIFEST_PATH)
        assert table.schema_version == "1.0.0"
        assert len(table.models) > 0

    def test_all_entries_have_required_fields(self) -> None:
        """Every model entry must validate as a ModelPricingEntry."""
        raw = _MANIFEST_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)

        for model_id, entry_data in data["models"].items():
            entry = ModelPricingEntry(**entry_data)
            assert entry.input_cost_per_1k >= 0.0, (
                f"Model {model_id}: input_cost_per_1k must be >= 0"
            )
            assert entry.output_cost_per_1k >= 0.0, (
                f"Model {model_id}: output_cost_per_1k must be >= 0"
            )

    def test_local_models_have_zero_cost(self) -> None:
        """Models with 'Local model' in note must have zero costs."""
        table = ModelPricingTable.from_yaml(_MANIFEST_PATH)

        for model_id, entry in table.models.items():
            if "local" in entry.note.lower():
                assert entry.input_cost_per_1k == 0.0, (
                    f"Local model {model_id} must have input_cost_per_1k=0.0"
                )
                assert entry.output_cost_per_1k == 0.0, (
                    f"Local model {model_id} must have output_cost_per_1k=0.0"
                )

    def test_no_duplicate_model_ids(self) -> None:
        """YAML keys are unique by definition but verify entry count matches."""
        raw = _MANIFEST_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        model_count = len(data["models"])

        table = ModelPricingTable.from_yaml(_MANIFEST_PATH)
        assert len(table.models) == model_count

    def test_manifest_has_at_least_one_cloud_and_one_local(self) -> None:
        """Manifest must contain at least one cloud and one local model."""
        table = ModelPricingTable.from_yaml(_MANIFEST_PATH)

        has_cloud = any(
            entry.input_cost_per_1k > 0 or entry.output_cost_per_1k > 0
            for entry in table.models.values()
        )
        has_local = any(
            entry.input_cost_per_1k == 0.0 and entry.output_cost_per_1k == 0.0
            for entry in table.models.values()
        )

        assert has_cloud, (
            "Manifest must contain at least one cloud model with non-zero cost"
        )
        assert has_local, (
            "Manifest must contain at least one local model with zero cost"
        )
