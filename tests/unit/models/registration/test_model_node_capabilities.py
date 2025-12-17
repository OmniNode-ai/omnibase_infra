# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelNodeCapabilities.

Tests validate:
- Dict-like access via __getitem__
- Membership testing via __contains__
- Safe access via get() method
- Custom capabilities via model_extra
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.registration import ModelNodeCapabilities


class TestModelNodeCapabilitiesGetItem:
    """Tests for __getitem__ dict-like access."""

    def test_getitem_known_field_returns_value(self) -> None:
        """Test that __getitem__ returns value for known fields."""
        caps = ModelNodeCapabilities(postgres=True, read=True, write=False)
        assert caps["postgres"] is True
        assert caps["read"] is True
        assert caps["write"] is False

    def test_getitem_known_field_returns_default_value(self) -> None:
        """Test that __getitem__ returns default value for unset known fields."""
        caps = ModelNodeCapabilities()
        assert caps["postgres"] is False
        assert caps["read"] is False
        assert caps["batch_size"] is None
        assert caps["supported_types"] == []

    def test_getitem_custom_capability_from_model_extra(self) -> None:
        """Test that __getitem__ returns custom capabilities from model_extra."""
        caps = ModelNodeCapabilities(
            custom_capability=True,  # type: ignore[call-arg]
            another_field="value",  # type: ignore[call-arg]
            numeric_field=42,  # type: ignore[call-arg]
        )
        assert caps["custom_capability"] is True
        assert caps["another_field"] == "value"
        assert caps["numeric_field"] == 42

    def test_getitem_unknown_key_raises_keyerror(self) -> None:
        """Test that __getitem__ raises KeyError for unknown keys."""
        caps = ModelNodeCapabilities(postgres=True)
        with pytest.raises(KeyError) as exc_info:
            _ = caps["unknown_capability"]
        assert "unknown_capability" in str(exc_info.value)

    def test_getitem_complex_custom_capability(self) -> None:
        """Test __getitem__ with complex custom capability values."""
        caps = ModelNodeCapabilities(
            nested_config={"key": "value", "num": 123},  # type: ignore[call-arg]
            list_field=["a", "b", "c"],  # type: ignore[call-arg]
        )
        assert caps["nested_config"] == {"key": "value", "num": 123}
        assert caps["list_field"] == ["a", "b", "c"]


class TestModelNodeCapabilitiesContains:
    """Tests for __contains__ membership testing."""

    def test_contains_known_field_returns_true(self) -> None:
        """Test that known fields are always 'in' capabilities."""
        caps = ModelNodeCapabilities()
        assert "postgres" in caps
        assert "read" in caps
        assert "write" in caps
        assert "database" in caps
        assert "batch_size" in caps
        assert "supported_types" in caps
        assert "config" in caps

    def test_contains_custom_capability_returns_true(self) -> None:
        """Test that custom capabilities in model_extra return True."""
        caps = ModelNodeCapabilities(
            custom_cap=True,  # type: ignore[call-arg]
            another_cap="value",  # type: ignore[call-arg]
        )
        assert "custom_cap" in caps
        assert "another_cap" in caps

    def test_contains_unknown_key_returns_false(self) -> None:
        """Test that unknown keys return False."""
        caps = ModelNodeCapabilities(postgres=True)
        assert "unknown_capability" not in caps
        assert "nonexistent" not in caps

    def test_contains_non_string_key_returns_false(self) -> None:
        """Test that non-string keys return False."""
        caps = ModelNodeCapabilities(postgres=True)
        assert 123 not in caps  # type: ignore[operator]
        assert None not in caps  # type: ignore[operator]
        assert [] not in caps  # type: ignore[operator]


class TestModelNodeCapabilitiesGet:
    """Tests for get() safe access method."""

    def test_get_known_field_returns_value(self) -> None:
        """Test that get() returns value for known fields."""
        caps = ModelNodeCapabilities(postgres=True, batch_size=100)
        assert caps.get("postgres") is True
        assert caps.get("batch_size") == 100

    def test_get_known_field_default_value(self) -> None:
        """Test that get() returns field default value for unset known fields."""
        caps = ModelNodeCapabilities()
        assert caps.get("postgres") is False
        assert caps.get("batch_size") is None

    def test_get_custom_capability_returns_value(self) -> None:
        """Test that get() returns custom capability values."""
        caps = ModelNodeCapabilities(
            custom_cap="custom_value",  # type: ignore[call-arg]
        )
        assert caps.get("custom_cap") == "custom_value"

    def test_get_unknown_key_returns_none_by_default(self) -> None:
        """Test that get() returns None for unknown keys by default."""
        caps = ModelNodeCapabilities(postgres=True)
        assert caps.get("unknown_capability") is None

    def test_get_unknown_key_returns_custom_default(self) -> None:
        """Test that get() returns custom default for unknown keys."""
        caps = ModelNodeCapabilities(postgres=True)
        assert caps.get("unknown_capability", False) is False
        assert caps.get("unknown_capability", "default") == "default"
        assert caps.get("unknown_capability", 42) == 42
        assert caps.get("unknown_capability", []) == []

    def test_get_with_none_default_explicit(self) -> None:
        """Test that get() with explicit None default works correctly."""
        caps = ModelNodeCapabilities()
        result = caps.get("unknown", None)
        assert result is None


class TestModelNodeCapabilitiesDictLikeIntegration:
    """Integration tests for dict-like access patterns."""

    def test_combined_known_and_custom_capabilities(self) -> None:
        """Test access to both known fields and custom capabilities."""
        caps = ModelNodeCapabilities(
            postgres=True,
            read=True,
            batch_size=50,
            custom_field="custom_value",  # type: ignore[call-arg]
            custom_number=100,  # type: ignore[call-arg]
        )

        # Known fields via __getitem__
        assert caps["postgres"] is True
        assert caps["read"] is True
        assert caps["batch_size"] == 50

        # Custom capabilities via __getitem__
        assert caps["custom_field"] == "custom_value"
        assert caps["custom_number"] == 100

        # Membership testing
        assert "postgres" in caps
        assert "custom_field" in caps
        assert "nonexistent" not in caps

        # Safe access
        assert caps.get("postgres") is True
        assert caps.get("custom_field") == "custom_value"
        assert caps.get("nonexistent", "fallback") == "fallback"

    def test_empty_capabilities_dict_like_access(self) -> None:
        """Test dict-like access on empty capabilities."""
        caps = ModelNodeCapabilities()

        # Known fields still accessible with default values
        assert caps["postgres"] is False
        assert caps["batch_size"] is None
        assert "postgres" in caps

        # Unknown keys raise KeyError or return default
        with pytest.raises(KeyError):
            _ = caps["unknown"]
        assert caps.get("unknown") is None
        assert caps.get("unknown", "default") == "default"

    def test_config_field_dict_like_access(self) -> None:
        """Test access to config field which is itself a dict."""
        caps = ModelNodeCapabilities(
            config={"pool_size": 10, "timeout": 30},
        )
        assert caps["config"] == {"pool_size": 10, "timeout": 30}
        assert "config" in caps
        assert caps.get("config") == {"pool_size": 10, "timeout": 30}

    def test_supported_types_list_access(self) -> None:
        """Test access to supported_types field which is a list."""
        caps = ModelNodeCapabilities(
            supported_types=["read", "write", "delete"],
        )
        assert caps["supported_types"] == ["read", "write", "delete"]
        assert "supported_types" in caps
        assert caps.get("supported_types") == ["read", "write", "delete"]


class TestModelNodeCapabilitiesEdgeCases:
    """Edge case tests for dict-like access."""

    def test_model_extra_none_handling(self) -> None:
        """Test behavior when model_extra is None or empty."""
        caps = ModelNodeCapabilities()
        # model_extra should be empty dict when no extra fields provided
        assert caps.model_extra == {}
        # Should still work for known fields
        assert "postgres" in caps
        assert caps["postgres"] is False
        # Unknown keys should behave correctly
        assert "unknown" not in caps
        with pytest.raises(KeyError):
            _ = caps["unknown"]

    def test_getitem_preserves_value_types(self) -> None:
        """Test that __getitem__ preserves value types correctly."""
        caps = ModelNodeCapabilities(
            postgres=True,
            batch_size=100,
            supported_types=["a", "b"],
            config={"key": "value"},
            custom_bool=False,  # type: ignore[call-arg]
            custom_int=0,  # type: ignore[call-arg]
            custom_str="",  # type: ignore[call-arg]
            custom_list=[],  # type: ignore[call-arg]
        )

        # Known fields
        assert caps["postgres"] is True
        assert isinstance(caps["postgres"], bool)
        assert caps["batch_size"] == 100
        assert isinstance(caps["batch_size"], int)
        assert isinstance(caps["supported_types"], list)
        assert isinstance(caps["config"], dict)

        # Custom capabilities (preserve falsy values)
        assert caps["custom_bool"] is False
        assert caps["custom_int"] == 0
        assert caps["custom_str"] == ""
        assert caps["custom_list"] == []

    def test_get_with_falsy_default(self) -> None:
        """Test get() with various falsy default values."""
        caps = ModelNodeCapabilities()

        assert caps.get("unknown", False) is False
        assert caps.get("unknown", 0) == 0
        assert caps.get("unknown", "") == ""
        assert caps.get("unknown", []) == []
        assert caps.get("unknown", {}) == {}

    def test_attribute_and_dict_access_equivalence(self) -> None:
        """Test that attribute and dict access return same values."""
        caps = ModelNodeCapabilities(
            postgres=True,
            read=False,
            batch_size=50,
            supported_types=["x", "y"],
        )

        # Verify equivalence for known fields
        assert caps.postgres == caps["postgres"]
        assert caps.read == caps["read"]
        assert caps.batch_size == caps["batch_size"]
        assert caps.supported_types == caps["supported_types"]
