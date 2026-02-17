# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for attribution models (ModelSectionAttribution, ModelStaticContextReport).

Tests model construction, provenance tracking, and computed properties.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)
from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)
from omnibase_infra.services.observability.static_context_attribution.model_section_attribution import (
    ModelSectionAttribution,
)
from omnibase_infra.services.observability.static_context_attribution.model_static_context_report import (
    ModelStaticContextReport,
)


@pytest.mark.unit
class TestModelContextSection:
    """Tests for ModelContextSection."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        section = ModelContextSection(content="test")
        assert section.heading == ""
        assert section.heading_level == 0
        assert section.source_file == ""
        assert section.line_start == 1
        assert section.line_end == 1
        assert section.token_count == 0
        assert section.category == EnumContextSectionCategory.UNCATEGORIZED
        assert section.has_code_block is False
        assert section.has_table is False

    def test_with_token_count(self) -> None:
        """with_token_count returns updated copy."""
        section = ModelContextSection(content="test")
        updated = section.with_token_count(42)
        assert updated.token_count == 42
        assert section.token_count == 0  # Original unchanged

    def test_with_category(self) -> None:
        """with_category returns updated copy."""
        section = ModelContextSection(content="test")
        updated = section.with_category(EnumContextSectionCategory.CONFIG)
        assert updated.category == EnumContextSectionCategory.CONFIG
        assert section.category == EnumContextSectionCategory.UNCATEGORIZED

    def test_frozen(self) -> None:
        """Model is frozen (immutable)."""
        section = ModelContextSection(content="test")
        with pytest.raises(Exception):  # ValidationError for frozen models
            section.content = "modified"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(Exception):
            ModelContextSection(content="test", unknown_field="value")  # type: ignore[call-arg]


@pytest.mark.unit
class TestModelSectionAttribution:
    """Tests for ModelSectionAttribution."""

    @staticmethod
    def _make_section(token_count: int = 100) -> ModelContextSection:
        """Create a test section."""
        return ModelContextSection(
            content="test content",
            token_count=token_count,
            line_start=1,
            line_end=1,
        )

    def test_attributed_tokens_computed(self) -> None:
        """attributed_tokens = utilization_score * token_count."""
        attr = ModelSectionAttribution(
            section=self._make_section(100),
            utilization_score=0.5,
            matched_fragments=5,
            total_fragments=10,
        )
        assert attr.attributed_tokens == 50

    def test_attributed_tokens_zero_score(self) -> None:
        """Zero utilization = zero attributed tokens."""
        attr = ModelSectionAttribution(
            section=self._make_section(100),
            utilization_score=0.0,
        )
        assert attr.attributed_tokens == 0

    def test_attributed_tokens_full_score(self) -> None:
        """Full utilization = all tokens attributed."""
        attr = ModelSectionAttribution(
            section=self._make_section(100),
            utilization_score=1.0,
            matched_fragments=10,
            total_fragments=10,
        )
        assert attr.attributed_tokens == 100

    def test_attributed_tokens_rounding(self) -> None:
        """Fractional attributed tokens are rounded."""
        attr = ModelSectionAttribution(
            section=self._make_section(100),
            utilization_score=0.333,
            matched_fragments=3,
            total_fragments=9,
        )
        # 0.333 * 100 = 33.3 -> rounds to 33
        assert attr.attributed_tokens == 33

    def test_default_values(self) -> None:
        """Default values are 0."""
        attr = ModelSectionAttribution(section=self._make_section())
        assert attr.utilization_score == 0.0
        assert attr.matched_fragments == 0
        assert attr.total_fragments == 0

    def test_score_bounds(self) -> None:
        """Score must be in [0.0, 1.0]."""
        with pytest.raises(Exception):
            ModelSectionAttribution(
                section=self._make_section(),
                utilization_score=1.5,
            )
        with pytest.raises(Exception):
            ModelSectionAttribution(
                section=self._make_section(),
                utilization_score=-0.1,
            )


@pytest.mark.unit
class TestModelStaticContextReport:
    """Tests for ModelStaticContextReport."""

    def test_compute_hash_deterministic(self) -> None:
        """Same content produces same hash."""
        h1 = ModelStaticContextReport.compute_hash("hello world")
        h2 = ModelStaticContextReport.compute_hash("hello world")
        assert h1 == h2

    def test_compute_hash_different_content(self) -> None:
        """Different content produces different hash."""
        h1 = ModelStaticContextReport.compute_hash("hello")
        h2 = ModelStaticContextReport.compute_hash("world")
        assert h1 != h2

    def test_compute_hash_is_sha256(self) -> None:
        """Hash is 64 hex characters (SHA-256)."""
        h = ModelStaticContextReport.compute_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_default_values(self) -> None:
        """Report defaults are sensible."""
        report = ModelStaticContextReport()
        assert report.attributions == ()
        assert report.total_tokens == 0
        assert report.total_attributed_tokens == 0
        assert report.input_hash == ""
        assert report.response_hash == ""
        assert report.code_version == "0.1.0"
        assert report.source_files == ()
        assert report.llm_augmented is False
        assert isinstance(report.created_at, datetime)

    def test_created_at_is_utc(self) -> None:
        """created_at has UTC timezone."""
        report = ModelStaticContextReport()
        assert report.created_at.tzinfo is not None
        assert report.created_at.tzinfo == UTC

    def test_with_attributions(self) -> None:
        """Report with attributions computes totals correctly."""
        section = ModelContextSection(
            content="test",
            token_count=50,
            line_start=1,
            line_end=1,
        )
        attr = ModelSectionAttribution(
            section=section,
            utilization_score=0.5,
            matched_fragments=1,
            total_fragments=2,
        )
        report = ModelStaticContextReport(
            attributions=(attr,),
            total_tokens=50,
            total_attributed_tokens=25,
            input_hash="abc123",
            response_hash="def456",
            source_files=("CLAUDE.md",),
        )
        assert report.total_tokens == 50
        assert report.total_attributed_tokens == 25
        assert len(report.attributions) == 1
        assert report.source_files == ("CLAUDE.md",)
