# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for Infisical scheme support in BindingConfigResolver.

Tests the infisical: config_ref scheme parsing and resolution.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.enums.enum_config_ref_scheme import EnumConfigRefScheme
from omnibase_infra.runtime.models.model_config_ref import ModelConfigRef


class TestConfigRefInfisicalParsing:
    """Test parsing of infisical: config references."""

    def test_parse_infisical_simple_path(self) -> None:
        """Test parsing a simple infisical: reference."""
        result = ModelConfigRef.parse("infisical:project/env/db")
        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.INFISICAL
        assert result.config_ref.path == "project/env/db"
        assert result.config_ref.fragment is None

    def test_parse_infisical_with_fragment(self) -> None:
        """Test parsing infisical: reference with fragment."""
        result = ModelConfigRef.parse("infisical:project/env/db#password")
        assert result.success
        assert result.config_ref is not None
        assert result.config_ref.scheme == EnumConfigRefScheme.INFISICAL
        assert result.config_ref.path == "project/env/db"
        assert result.config_ref.fragment == "password"

    def test_parse_infisical_empty_fragment(self) -> None:
        """Test infisical: with empty fragment is rejected."""
        result = ModelConfigRef.parse("infisical:path#")
        assert not result.success
        assert "Empty fragment" in (result.error_message or "")

    def test_parse_infisical_missing_path(self) -> None:
        """Test infisical: with missing path is rejected."""
        result = ModelConfigRef.parse("infisical:")
        assert not result.success

    def test_parse_infisical_to_uri(self) -> None:
        """Test round-trip from parse to URI."""
        result = ModelConfigRef.parse("infisical:secrets/db#pass")
        assert result.success
        assert result.config_ref is not None
        uri = result.config_ref.to_uri()
        assert uri == "infisical:secrets/db#pass"

    def test_parse_infisical_to_uri_no_fragment(self) -> None:
        """Test URI reconstruction without fragment."""
        result = ModelConfigRef.parse("infisical:secrets/db")
        assert result.success
        assert result.config_ref is not None
        uri = result.config_ref.to_uri()
        assert uri == "infisical:secrets/db"

    def test_enum_has_infisical(self) -> None:
        """Test EnumConfigRefScheme has INFISICAL member."""
        assert hasattr(EnumConfigRefScheme, "INFISICAL")
        assert EnumConfigRefScheme.INFISICAL.value == "infisical"

    def test_infisical_scheme_in_supported_list(self) -> None:
        """Test that infisical: is accepted as valid scheme."""
        valid_schemes = [s.value for s in EnumConfigRefScheme]
        assert "infisical" in valid_schemes
