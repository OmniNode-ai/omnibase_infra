# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceLlmCategoryAugmenter.

Tests the optional LLM semantic category classification pass.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)
from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)
from omnibase_infra.services.observability.static_context_attribution.service_llm_category_augmenter import (
    ServiceLlmCategoryAugmenter,
)


@pytest.mark.unit
class TestServiceLlmCategoryAugmenter:
    """Tests for the LLM category augmenter."""

    @staticmethod
    def _make_section(
        heading: str = "Test", content: str = "Content"
    ) -> ModelContextSection:
        """Create a test section."""
        return ModelContextSection(
            heading=heading,
            heading_level=2,
            content=content,
            line_start=1,
            line_end=1,
        )

    @pytest.mark.asyncio
    async def test_successful_classification(self) -> None:
        """LLM returning valid category classifies section."""

        async def mock_llm(prompt: str) -> str:
            return "config"

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        sections = [self._make_section("Database Config", "POSTGRES_HOST=localhost")]
        result = await augmenter.augment(sections)
        assert result[0].category == EnumContextSectionCategory.CONFIG

    @pytest.mark.asyncio
    async def test_all_valid_categories(self) -> None:
        """All valid category strings are recognized."""
        categories = [
            "config",
            "rules",
            "topology",
            "examples",
            "commands",
            "architecture",
            "documentation",
            "testing",
            "error_handling",
        ]
        for cat in categories:

            async def mock_llm(prompt: str, c: str = cat) -> str:
                return c

            augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
            sections = [self._make_section()]
            result = await augmenter.augment(sections)
            assert result[0].category.value == cat, f"Failed for category: {cat}"

    @pytest.mark.asyncio
    async def test_unrecognized_category_falls_back(self) -> None:
        """Unrecognized LLM response falls back to UNCATEGORIZED."""

        async def mock_llm(prompt: str) -> str:
            return "completely_invalid_category"

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        sections = [self._make_section()]
        result = await augmenter.augment(sections)
        assert result[0].category == EnumContextSectionCategory.UNCATEGORIZED

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self) -> None:
        """LLM raising exception falls back to UNCATEGORIZED."""

        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=failing_llm)
        sections = [self._make_section()]
        result = await augmenter.augment(sections)
        assert result[0].category == EnumContextSectionCategory.UNCATEGORIZED

    @pytest.mark.asyncio
    async def test_multiple_sections_classified(self) -> None:
        """Multiple sections are classified independently."""
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if "Database" in prompt:
                return "config"
            return "testing"

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        sections = [
            self._make_section("Database Config", "POSTGRES_HOST=localhost"),
            self._make_section("Test Setup", "pytest -m unit"),
        ]
        result = await augmenter.augment(sections)
        assert result[0].category == EnumContextSectionCategory.CONFIG
        assert result[1].category == EnumContextSectionCategory.TESTING
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_partial_match_in_response(self) -> None:
        """LLM response containing the category keyword is matched."""

        async def mock_llm(prompt: str) -> str:
            return "I think this is about config stuff"

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        sections = [self._make_section()]
        result = await augmenter.augment(sections)
        assert result[0].category == EnumContextSectionCategory.CONFIG

    @pytest.mark.asyncio
    async def test_quoted_response_handled(self) -> None:
        """LLM response with quotes is handled."""

        async def mock_llm(prompt: str) -> str:
            return '"rules"'

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        sections = [self._make_section()]
        result = await augmenter.augment(sections)
        assert result[0].category == EnumContextSectionCategory.RULES

    @pytest.mark.asyncio
    async def test_preserves_other_fields(self) -> None:
        """Augmentation preserves all non-category fields."""

        async def mock_llm(prompt: str) -> str:
            return "testing"

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        section = ModelContextSection(
            heading="Testing",
            heading_level=2,
            content="Use pytest for all tests",
            source_file="test.md",
            line_start=5,
            line_end=10,
            token_count=42,
            has_code_block=True,
            has_table=False,
        )
        result = await augmenter.augment([section])
        augmented = result[0]
        assert augmented.heading == "Testing"
        assert augmented.heading_level == 2
        assert augmented.source_file == "test.md"
        assert augmented.line_start == 5
        assert augmented.line_end == 10
        assert augmented.token_count == 42
        assert augmented.has_code_block is True
        assert augmented.category == EnumContextSectionCategory.TESTING
