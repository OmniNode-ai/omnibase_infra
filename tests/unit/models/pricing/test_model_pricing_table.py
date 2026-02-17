# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelPricingTable.

Covers YAML loading, cost estimation, unknown model handling,
local model handling, and schema validation.

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable


@pytest.fixture
def sample_manifest_data() -> dict:
    """Minimal valid pricing manifest as a dict."""
    return {
        "schema_version": "1.0.0",
        "models": {
            "claude-opus-4-6": {
                "input_cost_per_1k": 0.015,
                "output_cost_per_1k": 0.075,
                "effective_date": "2026-02-01",
            },
            "gpt-4o": {
                "input_cost_per_1k": 0.0025,
                "output_cost_per_1k": 0.01,
                "effective_date": "2025-11-01",
            },
            "qwen2.5-coder-14b": {
                "input_cost_per_1k": 0.0,
                "output_cost_per_1k": 0.0,
                "effective_date": "2026-02-01",
                "note": "Local model - zero API cost",
            },
        },
    }


@pytest.fixture
def sample_manifest_yaml(tmp_path: Path, sample_manifest_data: dict) -> Path:
    """Write a sample manifest YAML file and return its path."""
    import yaml

    manifest_path = tmp_path / "pricing_manifest.yaml"
    manifest_path.write_text(
        yaml.dump(sample_manifest_data, default_flow_style=False),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.unit
class TestModelPricingTableFromDict:
    """Tests for constructing a pricing table from a dict."""

    def test_from_dict_valid(self, sample_manifest_data: dict) -> None:
        """Valid manifest data should produce a valid table."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        assert table.schema_version == "1.0.0"
        assert len(table.models) == 3
        assert "claude-opus-4-6" in table.models
        assert "gpt-4o" in table.models
        assert "qwen2.5-coder-14b" in table.models

    def test_from_dict_missing_schema_version(self) -> None:
        """Missing schema_version should raise ValueError."""
        with pytest.raises(ValueError, match="schema_version"):
            ModelPricingTable.from_dict({"models": {}})

    def test_from_dict_empty_models(self) -> None:
        """Empty models section is valid but triggers warning."""
        table = ModelPricingTable.from_dict(
            {
                "schema_version": "1.0.0",
                "models": {},
            }
        )
        assert len(table.models) == 0

    def test_from_dict_no_models_key(self) -> None:
        """Missing models key defaults to empty dict."""
        table = ModelPricingTable.from_dict(
            {
                "schema_version": "1.0.0",
            }
        )
        assert len(table.models) == 0

    def test_from_dict_invalid_models_type(self) -> None:
        """Non-dict models section should raise ValueError."""
        with pytest.raises(ValueError, match=r"models.*mapping"):
            ModelPricingTable.from_dict(
                {
                    "schema_version": "1.0.0",
                    "models": ["not", "a", "dict"],
                }
            )

    def test_from_dict_invalid_entry_type(self) -> None:
        """Non-dict entry should raise ValueError."""
        with pytest.raises(ValueError, match="must be a mapping"):
            ModelPricingTable.from_dict(
                {
                    "schema_version": "1.0.0",
                    "models": {
                        "bad-model": "not-a-dict",
                    },
                }
            )

    def test_from_dict_rejects_unexpected_top_level_keys(self) -> None:
        """Unexpected top-level keys should raise ValueError."""
        with pytest.raises(ValueError, match="unexpected fields"):
            ModelPricingTable.from_dict(
                {
                    "schema_version": "1.0.0",
                    "models": {},
                    "typo_field": "oops",
                }
            )

    def test_from_dict_rejects_multiple_unexpected_keys(self) -> None:
        """Multiple unexpected top-level keys should all be listed."""
        with pytest.raises(ValueError, match=r"unexpected fields.*extra_a.*extra_b"):
            ModelPricingTable.from_dict(
                {
                    "schema_version": "1.0.0",
                    "models": {},
                    "extra_a": 1,
                    "extra_b": 2,
                }
            )


@pytest.mark.unit
class TestModelPricingTableFromYaml:
    """Tests for loading a pricing table from YAML files."""

    def test_from_yaml_valid(self, sample_manifest_yaml: Path) -> None:
        """Valid YAML manifest should load successfully."""
        table = ModelPricingTable.from_yaml(sample_manifest_yaml)
        assert table.schema_version == "1.0.0"
        assert len(table.models) == 3

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ModelPricingTable.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_malformed_content(self, tmp_path: Path) -> None:
        """Non-mapping YAML content should raise ValueError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- this\n- is\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            ModelPricingTable.from_yaml(bad_yaml)

    def test_from_yaml_default_manifest_exists(self) -> None:
        """The bundled default manifest should load without errors."""
        table = ModelPricingTable.from_yaml()
        assert table.schema_version == "1.0.0"
        assert len(table.models) > 0

    def test_from_yaml_string_path(self, sample_manifest_yaml: Path) -> None:
        """String path should be accepted."""
        table = ModelPricingTable.from_yaml(str(sample_manifest_yaml))
        assert len(table.models) == 3


@pytest.mark.unit
class TestModelPricingTableEstimateCost:
    """Tests for the estimate_cost method."""

    def test_known_cloud_model_cost(self, sample_manifest_data: dict) -> None:
        """Cloud model should return computed cost."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        estimate = table.estimate_cost("claude-opus-4-6", 1000, 500)

        # cost = (1000/1000 * 0.015) + (500/1000 * 0.075) = 0.015 + 0.0375 = 0.0525
        assert estimate.estimated_cost_usd is not None
        assert abs(estimate.estimated_cost_usd - 0.0525) < 1e-9
        assert estimate.model_id == "claude-opus-4-6"
        assert estimate.prompt_tokens == 1000
        assert estimate.completion_tokens == 500

    def test_known_local_model_returns_zero(self, sample_manifest_data: dict) -> None:
        """Local model should return 0.0 cost (not None)."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        estimate = table.estimate_cost("qwen2.5-coder-14b", 5000, 2000)

        assert estimate.estimated_cost_usd is not None
        assert estimate.estimated_cost_usd == 0.0

    def test_unknown_model_returns_none(self, sample_manifest_data: dict) -> None:
        """Unknown model should return None cost (not 0)."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        estimate = table.estimate_cost("totally-unknown-model", 1000, 500)

        assert estimate.estimated_cost_usd is None
        assert estimate.model_id == "totally-unknown-model"
        assert estimate.prompt_tokens == 1000
        assert estimate.completion_tokens == 500

    def test_zero_tokens_returns_zero_cost(self, sample_manifest_data: dict) -> None:
        """Zero tokens for a known model should return 0.0 cost."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        estimate = table.estimate_cost("claude-opus-4-6", 0, 0)

        assert estimate.estimated_cost_usd is not None
        assert estimate.estimated_cost_usd == 0.0

    def test_cost_formula_accuracy(self, sample_manifest_data: dict) -> None:
        """Verify cost formula: (prompt/1000 * input) + (completion/1000 * output)."""
        table = ModelPricingTable.from_dict(sample_manifest_data)

        # gpt-4o: input=0.0025, output=0.01
        estimate = table.estimate_cost("gpt-4o", 2000, 1000)

        # cost = (2000/1000 * 0.0025) + (1000/1000 * 0.01) = 0.005 + 0.01 = 0.015
        assert estimate.estimated_cost_usd is not None
        assert abs(estimate.estimated_cost_usd - 0.015) < 1e-9

    def test_large_token_counts(self, sample_manifest_data: dict) -> None:
        """Large token counts should compute correctly without overflow."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        estimate = table.estimate_cost("claude-opus-4-6", 1_000_000, 500_000)

        # cost = (1_000_000/1000 * 0.015) + (500_000/1000 * 0.075)
        #      = 15.0 + 37.5 = 52.5
        assert estimate.estimated_cost_usd is not None
        assert abs(estimate.estimated_cost_usd - 52.5) < 1e-6


@pytest.mark.unit
class TestModelPricingTableLookup:
    """Tests for has_model and get_entry methods."""

    def test_has_model_true(self, sample_manifest_data: dict) -> None:
        """has_model returns True for known model."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        assert table.has_model("claude-opus-4-6") is True

    def test_has_model_false(self, sample_manifest_data: dict) -> None:
        """has_model returns False for unknown model."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        assert table.has_model("nonexistent") is False

    def test_get_entry_found(self, sample_manifest_data: dict) -> None:
        """get_entry returns the entry for a known model."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        entry = table.get_entry("qwen2.5-coder-14b")

        assert entry is not None
        assert entry.input_cost_per_1k == 0.0
        assert entry.output_cost_per_1k == 0.0
        assert entry.note == "Local model - zero API cost"

    def test_get_entry_not_found(self, sample_manifest_data: dict) -> None:
        """get_entry returns None for unknown model."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        assert table.get_entry("nonexistent") is None


@pytest.mark.unit
class TestModelPricingTableImmutability:
    """Tests for frozen model behavior."""

    def test_table_is_frozen(self, sample_manifest_data: dict) -> None:
        """Table should be immutable (frozen=True)."""
        table = ModelPricingTable.from_dict(sample_manifest_data)
        with pytest.raises(ValidationError):
            table.schema_version = "2.0.0"  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        """Extra fields should be rejected (extra='forbid')."""
        with pytest.raises(ValidationError, match="extra"):
            ModelPricingTable(
                schema_version="1.0.0",
                models={},
                unknown="oops",  # type: ignore[call-arg]
            )
