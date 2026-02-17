# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for EnumContextSectionCategory.

Tests enum member completeness and string values.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)


@pytest.mark.unit
class TestEnumContextSectionCategory:
    """Tests for the context section category enum."""

    def test_uncategorized_is_default(self) -> None:
        """UNCATEGORIZED value is 'uncategorized'."""
        assert EnumContextSectionCategory.UNCATEGORIZED.value == "uncategorized"

    def test_all_members_have_string_values(self) -> None:
        """All enum members have non-empty string values."""
        for member in EnumContextSectionCategory:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    def test_expected_member_count(self) -> None:
        """Enum has the expected number of members."""
        assert len(EnumContextSectionCategory) == 10

    def test_semantic_categories_present(self) -> None:
        """All semantic categories are present."""
        expected = {
            "uncategorized",
            "config",
            "rules",
            "topology",
            "examples",
            "commands",
            "architecture",
            "documentation",
            "testing",
            "error_handling",
        }
        actual = {member.value for member in EnumContextSectionCategory}
        assert actual == expected

    def test_is_string_enum(self) -> None:
        """Enum members are string instances."""
        assert isinstance(EnumContextSectionCategory.CONFIG, str)
        assert EnumContextSectionCategory.CONFIG == "config"

    def test_member_lookup_by_value(self) -> None:
        """Members can be looked up by string value."""
        assert EnumContextSectionCategory("config") == EnumContextSectionCategory.CONFIG
        assert (
            EnumContextSectionCategory("error_handling")
            == EnumContextSectionCategory.ERROR_HANDLING
        )
