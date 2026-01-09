# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinDictLikeAccessors.

Tests dict-like accessor methods:
- get() with defaults
- __getitem__ (bracket notation)
- __contains__ (membership testing)
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.mixins import MixinDictLikeAccessors


class ExampleModel(MixinDictLikeAccessors, BaseModel):
    """Test model using the mixin."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    count: int = 0
    optional_field: str | None = None


class TestMixinDictLikeAccessorsGet:
    """Test get() method behavior."""

    def test_get_existing_field_returns_value(self) -> None:
        """Test that get() returns field value for existing fields."""
        model = ExampleModel(name="test", count=42)
        assert model.get("name") == "test"
        assert model.get("count") == 42

    def test_get_missing_field_returns_default(self) -> None:
        """Test that get() returns default for missing fields."""
        model = ExampleModel()
        assert model.get("missing") is None
        assert model.get("missing", "default") == "default"
        assert model.get("missing", 123) == 123

    def test_get_extra_field_returns_value(self) -> None:
        """Test that get() works with extra="allow" fields."""
        model = ExampleModel.model_validate({"name": "test", "extra_field": "extra"})
        assert model.get("extra_field") == "extra"

    def test_get_none_value_returns_none(self) -> None:
        """Test that get() returns None for field with None value."""
        model = ExampleModel(optional_field=None)
        assert model.get("optional_field") is None

    def test_get_with_default_does_not_override_none(self) -> None:
        """Test that get() doesn't use default when value is explicitly None."""
        model = ExampleModel(optional_field=None)
        # hasattr returns True for fields with None values, so getattr returns None
        assert model.get("optional_field", "default") is None


class TestMixinDictLikeAccessorsGetitem:
    """Test __getitem__ method behavior."""

    def test_getitem_existing_field_returns_value(self) -> None:
        """Test that bracket notation returns field value."""
        model = ExampleModel(name="test", count=42)
        assert model["name"] == "test"
        assert model["count"] == 42

    def test_getitem_missing_field_raises_keyerror(self) -> None:
        """Test that bracket notation raises KeyError for missing fields."""
        model = ExampleModel()
        with pytest.raises(KeyError, match="missing"):
            _ = model["missing"]

    def test_getitem_extra_field_returns_value(self) -> None:
        """Test that bracket notation works with extra="allow" fields."""
        model = ExampleModel.model_validate({"extra_field": [1, 2, 3]})
        assert model["extra_field"] == [1, 2, 3]

    def test_getitem_none_value_returns_none(self) -> None:
        """Test that bracket notation returns None for fields with None value."""
        model = ExampleModel(optional_field=None)
        assert model["optional_field"] is None


class TestMixinDictLikeAccessorsContains:
    """Test __contains__ method behavior."""

    def test_contains_existing_field_with_value(self) -> None:
        """Test that 'in' returns True for fields with non-None values."""
        model = ExampleModel(name="test", count=42)
        assert "name" in model
        assert "count" in model

    def test_contains_missing_field(self) -> None:
        """Test that 'in' returns False for missing fields."""
        model = ExampleModel()
        assert "nonexistent" not in model

    def test_contains_field_with_none_value(self) -> None:
        """Test that 'in' returns False for fields with None values."""
        model = ExampleModel(optional_field=None)
        # The mixin uses presence semantics: None means "not present"
        assert "optional_field" not in model

    def test_contains_extra_field(self) -> None:
        """Test that 'in' works with extra="allow" fields."""
        model = ExampleModel.model_validate({"extra_field": "value"})
        assert "extra_field" in model

    def test_contains_empty_string_is_present(self) -> None:
        """Test that 'in' returns True for empty strings (not None)."""
        model = ExampleModel(name="")
        assert "name" in model  # Empty string is not None


class TestMixinDictLikeAccessorsIntegration:
    """Integration tests for the mixin with real model classes."""

    def test_with_model_policy_result(self) -> None:
        """Test mixin works with ModelPolicyResult."""
        from omnibase_infra.runtime.models import ModelPolicyResult

        result = ModelPolicyResult(
            should_retry=True,
            delay_seconds=4.0,
            reason="Test retry",
        )
        assert result.get("should_retry") is True
        assert result["delay_seconds"] == 4.0
        assert "reason" in result
        assert "missing" not in result

    def test_with_model_policy_context(self) -> None:
        """Test mixin works with ModelPolicyContext."""
        from omnibase_infra.runtime.models import ModelPolicyContext

        context = ModelPolicyContext(attempt=3, timestamp_ms=1000)
        assert context.get("attempt") == 3
        assert context["timestamp_ms"] == 1000
        assert "attempt" in context

    def test_with_model_plugin_context(self) -> None:
        """Test mixin works with ModelPluginContext."""
        from omnibase_infra.plugins.models import ModelPluginContext

        ctx = ModelPluginContext(correlation_id="test-123")
        assert ctx.get("correlation_id") == "test-123"
        assert ctx["correlation_id"] == "test-123"
        assert "correlation_id" in ctx

    def test_with_model_plugin_input_data(self) -> None:
        """Test mixin works with ModelPluginInputData."""
        from omnibase_infra.plugins.models import ModelPluginInputData

        data = ModelPluginInputData.model_validate({"values": [1, 2, 3]})
        assert data.get("values") == [1, 2, 3]
        assert data["values"] == [1, 2, 3]
        assert "values" in data

    def test_with_model_plugin_output_data(self) -> None:
        """Test mixin works with ModelPluginOutputData."""
        from omnibase_infra.plugins.models import ModelPluginOutputData

        output = ModelPluginOutputData.model_validate({"result": "success"})
        assert output.get("result") == "success"
        assert output["result"] == "success"
        assert "result" in output


class TestMixinDictLikeAccessorsEdgeCases:
    """Edge case tests for the mixin."""

    def test_get_with_false_value(self) -> None:
        """Test that get() correctly returns False (not default)."""
        model = ExampleModel.model_validate({"enabled": False})
        assert model.get("enabled", True) is False

    def test_get_with_zero_value(self) -> None:
        """Test that get() correctly returns 0 (not default)."""
        model = ExampleModel(count=0)
        assert model.get("count", 99) == 0

    def test_get_with_empty_list(self) -> None:
        """Test that get() correctly returns empty list."""
        model = ExampleModel.model_validate({"items": []})
        assert model.get("items", ["default"]) == []

    def test_contains_with_false_value(self) -> None:
        """Test that 'in' returns True for False values."""
        model = ExampleModel.model_validate({"enabled": False})
        # False is not None, so it should be "present"
        assert "enabled" in model

    def test_contains_with_zero_value(self) -> None:
        """Test that 'in' returns True for 0 values."""
        model = ExampleModel(count=0)
        assert "count" in model

    def test_contains_with_empty_list(self) -> None:
        """Test that 'in' returns True for empty lists."""
        model = ExampleModel.model_validate({"items": []})
        assert "items" in model


__all__: list[str] = []
