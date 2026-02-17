# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceAttributionReporter.

Tests the full attribution pipeline orchestration.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)
from omnibase_infra.services.observability.static_context_attribution.service_attribution_reporter import (
    ServiceAttributionReporter,
)
from omnibase_infra.services.observability.static_context_attribution.service_llm_category_augmenter import (
    ServiceLlmCategoryAugmenter,
)


@pytest.mark.unit
class TestServiceAttributionReporter:
    """Tests for the attribution reporter orchestrator."""

    @pytest.mark.asyncio
    async def test_basic_report_generation(self) -> None:
        """Reporter produces a valid report with sections and tokens."""
        reporter = ServiceAttributionReporter()
        context_files = {
            "CLAUDE.md": "## Overview\nThis repo uses Python 3.12 with uv for dependency management.\n## Testing\nRun pytest with coverage.",
        }
        response = (
            "I'll use Python 3.12 with uv for dependency management as specified."
        )

        report = await reporter.build_report(context_files, response)

        assert report.total_tokens > 0
        assert len(report.attributions) == 2  # Two H2 sections
        assert report.input_hash != ""
        assert report.response_hash != ""
        assert report.source_files == ("CLAUDE.md",)
        assert report.llm_augmented is False

    @pytest.mark.asyncio
    async def test_provenance_hashes_deterministic(self) -> None:
        """Same input produces same provenance hashes."""
        reporter = ServiceAttributionReporter()
        context = {"test.md": "## Section\nContent here"}
        response = "response text"

        report1 = await reporter.build_report(context, response)
        report2 = await reporter.build_report(context, response)

        assert report1.input_hash == report2.input_hash
        assert report1.response_hash == report2.response_hash

    @pytest.mark.asyncio
    async def test_different_input_different_hash(self) -> None:
        """Different input produces different provenance hashes."""
        reporter = ServiceAttributionReporter()
        response = "response"

        report1 = await reporter.build_report({"a.md": "## A\nContent A"}, response)
        report2 = await reporter.build_report({"b.md": "## B\nContent B"}, response)

        assert report1.input_hash != report2.input_hash

    @pytest.mark.asyncio
    async def test_with_llm_augmentation(self) -> None:
        """Reporter with LLM augmenter marks report as augmented."""

        async def mock_llm(prompt: str) -> str:
            return "config"

        augmenter = ServiceLlmCategoryAugmenter(llm_fn=mock_llm)
        reporter = ServiceAttributionReporter(llm_augmenter=augmenter)
        context = {"test.md": "## Database\nPOSTGRES_HOST=localhost"}
        response = "configured database"

        report = await reporter.build_report(context, response)

        assert report.llm_augmented is True
        assert (
            report.attributions[0].section.category == EnumContextSectionCategory.CONFIG
        )

    @pytest.mark.asyncio
    async def test_without_llm_augmentation(self) -> None:
        """Reporter without LLM augmenter leaves sections uncategorized."""
        reporter = ServiceAttributionReporter()
        context = {"test.md": "## Database\nPOSTGRES_HOST=localhost"}
        response = "configured database"

        report = await reporter.build_report(context, response)

        assert report.llm_augmented is False
        assert (
            report.attributions[0].section.category
            == EnumContextSectionCategory.UNCATEGORIZED
        )

    @pytest.mark.asyncio
    async def test_multiple_files(self) -> None:
        """Reporter handles multiple input files."""
        reporter = ServiceAttributionReporter()
        context = {
            "CLAUDE.md": "## Overview\nMain documentation for the project with detailed guidelines.",
            "memory.md": "## Notes\nAdditional notes for context with important details.",
        }
        response = "Following the project guidelines"

        report = await reporter.build_report(context, response)

        assert len(report.source_files) == 2
        assert "CLAUDE.md" in report.source_files
        assert "memory.md" in report.source_files
        assert len(report.attributions) == 2  # One section per file

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        """Empty response gives zero attributed tokens."""
        reporter = ServiceAttributionReporter()
        context = {"test.md": "## Section\nSome content here"}

        report = await reporter.build_report(context, "")

        assert report.total_attributed_tokens == 0
        assert report.total_tokens > 0

    @pytest.mark.asyncio
    async def test_empty_context(self) -> None:
        """Empty context produces empty report."""
        reporter = ServiceAttributionReporter()

        report = await reporter.build_report({}, "Some response")

        assert report.total_tokens == 0
        assert len(report.attributions) == 0

    @pytest.mark.asyncio
    async def test_code_version_present(self) -> None:
        """Report includes code version."""
        reporter = ServiceAttributionReporter()
        context = {"test.md": "## Section\nContent"}

        report = await reporter.build_report(context, "response")

        assert report.code_version == "0.1.0"

    @pytest.mark.asyncio
    async def test_total_tokens_sum(self) -> None:
        """total_tokens equals sum of all section token counts."""
        reporter = ServiceAttributionReporter()
        context = {
            "test.md": "## First\nContent one with multiple words.\n## Second\nContent two with other words."
        }

        report = await reporter.build_report(context, "response")

        section_total = sum(a.section.token_count for a in report.attributions)
        assert report.total_tokens == section_total
