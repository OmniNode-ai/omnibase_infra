# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelSecurityConfig.

Tests validate:
- Default configuration returns trusted namespaces only
- Third-party handler flag controls effective namespace behavior
- Model is frozen (immutable)
- Extra fields are forbidden
- Serialization/deserialization roundtrip

.. versionadded:: 0.2.8
    Initial test coverage for ModelSecurityConfig (OMN-1519).

Related Tickets:
    - OMN-1519: Security hardening for handler namespace configuration
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.constants_security import TRUSTED_HANDLER_NAMESPACE_PREFIXES
from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig


class TestModelSecurityConfigDefaults:
    """Tests for ModelSecurityConfig default behavior."""

    def test_default_config_returns_trusted_namespaces(self) -> None:
        """Test that default config returns only trusted namespaces."""
        config = ModelSecurityConfig()
        effective = config.get_effective_namespaces()
        assert effective == TRUSTED_HANDLER_NAMESPACE_PREFIXES
        assert "omnibase_core." in effective
        assert "omnibase_infra." in effective

    def test_default_allow_third_party_is_false(self) -> None:
        """Test that allow_third_party_handlers defaults to False."""
        config = ModelSecurityConfig()
        assert config.allow_third_party_handlers is False

    def test_default_allowed_namespaces_equals_trusted(self) -> None:
        """Test that default allowed_handler_namespaces equals trusted prefixes."""
        config = ModelSecurityConfig()
        assert config.allowed_handler_namespaces == TRUSTED_HANDLER_NAMESPACE_PREFIXES


class TestModelSecurityConfigThirdPartyDisabled:
    """Tests for behavior when allow_third_party_handlers=False."""

    def test_custom_namespaces_ignored_when_disabled(self) -> None:
        """Test that custom namespaces are ignored when third-party is disabled."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=False,
            allowed_handler_namespaces=(
                "custom.namespace.",
                "another.namespace.",
            ),
        )
        effective = config.get_effective_namespaces()
        # Should return trusted namespaces, ignoring custom list
        assert effective == TRUSTED_HANDLER_NAMESPACE_PREFIXES
        assert "custom.namespace." not in effective
        assert "another.namespace." not in effective

    def test_empty_custom_namespaces_ignored_when_disabled(self) -> None:
        """Test that empty custom namespaces still returns trusted when disabled."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=False,
            allowed_handler_namespaces=(),
        )
        effective = config.get_effective_namespaces()
        assert effective == TRUSTED_HANDLER_NAMESPACE_PREFIXES


class TestModelSecurityConfigThirdPartyEnabled:
    """Tests for behavior when allow_third_party_handlers=True."""

    def test_custom_namespaces_used_when_enabled(self) -> None:
        """Test that custom namespaces are used when third-party is enabled."""
        custom_namespaces = (
            "omnibase_core.",
            "omnibase_infra.",
            "mycompany.handlers.",
        )
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=custom_namespaces,
        )
        effective = config.get_effective_namespaces()
        assert effective == custom_namespaces
        assert "mycompany.handlers." in effective

    def test_completely_custom_namespaces_when_enabled(self) -> None:
        """Test that completely custom namespaces work when enabled."""
        custom_namespaces = ("custom.only.",)
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=custom_namespaces,
        )
        effective = config.get_effective_namespaces()
        assert effective == custom_namespaces
        assert "omnibase_core." not in effective
        assert "omnibase_infra." not in effective

    def test_empty_namespaces_when_enabled(self) -> None:
        """Test that empty namespace list is allowed when enabled."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=(),
        )
        effective = config.get_effective_namespaces()
        assert effective == ()

    def test_default_namespaces_used_when_enabled_without_custom(self) -> None:
        """Test that default namespaces are used when enabled but no custom set."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
        )
        effective = config.get_effective_namespaces()
        # Should use the default value which is TRUSTED_HANDLER_NAMESPACE_PREFIXES
        assert effective == TRUSTED_HANDLER_NAMESPACE_PREFIXES


class TestModelSecurityConfigImmutability:
    """Tests for ModelSecurityConfig immutability (frozen=True)."""

    def test_allow_third_party_is_immutable(self) -> None:
        """Test that allow_third_party_handlers cannot be modified."""
        config = ModelSecurityConfig()
        with pytest.raises(ValidationError):
            config.allow_third_party_handlers = True  # type: ignore[misc]

    def test_allowed_namespaces_is_immutable(self) -> None:
        """Test that allowed_handler_namespaces cannot be modified."""
        config = ModelSecurityConfig()
        with pytest.raises(ValidationError):
            config.allowed_handler_namespaces = ("new.namespace.",)  # type: ignore[misc]

    def test_frozen_model_is_hashable(self) -> None:
        """Test that frozen model is hashable."""
        config = ModelSecurityConfig()
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_equal_configs_have_same_hash(self) -> None:
        """Test that equal configs have the same hash."""
        config1 = ModelSecurityConfig()
        config2 = ModelSecurityConfig()
        assert hash(config1) == hash(config2)

    def test_can_be_used_in_set(self) -> None:
        """Test that frozen model can be used in sets."""
        config1 = ModelSecurityConfig()
        config2 = ModelSecurityConfig()  # Duplicate
        config3 = ModelSecurityConfig(allow_third_party_handlers=True)

        config_set = {config1, config2, config3}
        assert len(config_set) == 2  # Deduplication


class TestModelSecurityConfigValidation:
    """Tests for ModelSecurityConfig field validation."""

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelSecurityConfig(
                allow_third_party_handlers=False,
                unknown_field="unexpected",  # type: ignore[call-arg]
            )
        error_str = str(exc_info.value).lower()
        assert "unknown_field" in error_str or "extra" in error_str

    def test_strict_mode_rejects_non_bool(self) -> None:
        """Test that strict mode rejects non-bool for allow_third_party_handlers."""
        with pytest.raises(ValidationError):
            ModelSecurityConfig(
                allow_third_party_handlers="yes",  # type: ignore[arg-type]
            )

    def test_strict_mode_rejects_non_tuple(self) -> None:
        """Test that strict mode rejects non-tuple for namespaces."""
        with pytest.raises(ValidationError):
            ModelSecurityConfig(
                allowed_handler_namespaces=["list", "not", "tuple"],  # type: ignore[arg-type]
            )

    def test_strict_mode_rejects_non_string_in_tuple(self) -> None:
        """Test that strict mode rejects non-string elements in namespace tuple."""
        with pytest.raises(ValidationError):
            ModelSecurityConfig(
                allowed_handler_namespaces=(123, 456),  # type: ignore[arg-type]
            )


class TestModelSecurityConfigSerialization:
    """Tests for ModelSecurityConfig serialization."""

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("custom.",),
        )
        data = config.model_dump()
        assert data == {
            "allow_third_party_handlers": True,
            "allowed_handler_namespaces": ("custom.",),
        }

    def test_model_dump_json(self) -> None:
        """Test JSON serialization."""
        config = ModelSecurityConfig()
        json_str = config.model_dump_json()
        assert '"allow_third_party_handlers":false' in json_str

    def test_roundtrip_serialization(self) -> None:
        """Test roundtrip serialization/deserialization."""
        original = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("ns1.", "ns2."),
        )
        data = original.model_dump()
        restored = ModelSecurityConfig.model_validate(data)
        assert original == restored

    def test_model_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "allow_third_party_handlers": True,
            "allowed_handler_namespaces": ("custom.namespace.",),
        }
        config = ModelSecurityConfig.model_validate(data)
        assert config.allow_third_party_handlers is True
        assert config.allowed_handler_namespaces == ("custom.namespace.",)


class TestModelSecurityConfigEdgeCases:
    """Edge case tests for ModelSecurityConfig."""

    def test_single_namespace(self) -> None:
        """Test with single namespace in tuple."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("single.",),
        )
        assert config.get_effective_namespaces() == ("single.",)

    def test_many_namespaces(self) -> None:
        """Test with many namespaces."""
        namespaces = tuple(f"namespace{i}." for i in range(100))
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=namespaces,
        )
        assert config.get_effective_namespaces() == namespaces
        assert len(config.get_effective_namespaces()) == 100

    def test_namespace_without_trailing_dot(self) -> None:
        """Test namespace string without trailing dot (still valid)."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("no_dot",),
        )
        # Model doesn't enforce trailing dot - that's validation at loader level
        assert config.get_effective_namespaces() == ("no_dot",)

    def test_empty_string_namespace(self) -> None:
        """Test that empty string namespace is allowed by model."""
        # Model doesn't validate namespace content - that's loader responsibility
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("",),
        )
        assert config.get_effective_namespaces() == ("",)

    def test_duplicate_namespaces_preserved(self) -> None:
        """Test that duplicate namespaces are preserved (tuple, not set)."""
        config = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("dup.", "dup.", "unique."),
        )
        assert config.get_effective_namespaces() == ("dup.", "dup.", "unique.")

    def test_repr_contains_class_name(self) -> None:
        """Test that repr includes class name."""
        config = ModelSecurityConfig()
        repr_str = repr(config)
        assert "ModelSecurityConfig" in repr_str

    def test_copy_creates_equal_instance(self) -> None:
        """Test that model_copy creates an equal instance."""
        original = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("ns.",),
        )
        copied = original.model_copy()
        assert original == copied
        assert original is not copied

    def test_copy_with_update(self) -> None:
        """Test that model_copy with update creates modified instance."""
        original = ModelSecurityConfig(allow_third_party_handlers=False)
        modified = original.model_copy(update={"allow_third_party_handlers": True})
        assert modified.allow_third_party_handlers is True
        assert original.allow_third_party_handlers is False


class TestModelSecurityConfigEquality:
    """Tests for ModelSecurityConfig equality comparison."""

    def test_same_values_are_equal(self) -> None:
        """Test that configs with same values are equal."""
        config1 = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("ns.",),
        )
        config2 = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("ns.",),
        )
        assert config1 == config2

    def test_different_flag_not_equal(self) -> None:
        """Test that different allow_third_party_handlers makes configs not equal."""
        config1 = ModelSecurityConfig(allow_third_party_handlers=True)
        config2 = ModelSecurityConfig(allow_third_party_handlers=False)
        assert config1 != config2

    def test_different_namespaces_not_equal(self) -> None:
        """Test that different namespaces make configs not equal."""
        config1 = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("ns1.",),
        )
        config2 = ModelSecurityConfig(
            allow_third_party_handlers=True,
            allowed_handler_namespaces=("ns2.",),
        )
        assert config1 != config2

    def test_not_equal_to_non_model(self) -> None:
        """Test that model is not equal to non-model objects."""
        config = ModelSecurityConfig()
        assert config != {"allow_third_party_handlers": False}
        assert config is not None
