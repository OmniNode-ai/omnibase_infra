#!/usr/bin/env python3
"""
Unit tests for PatternLoader.

Tests loading, parsing, and caching patterns from YAML files.
"""

from pathlib import Path

import pytest

from src.metadata_stamping.code_gen.patterns import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternMetadata,
    PatternLoader,
)
from src.metadata_stamping.code_gen.patterns.pattern_loader import (
    PatternLoaderError,
    PatternNotFoundError,
)


class TestPatternLoader:
    """Test suite for PatternLoader."""

    @pytest.fixture
    def loader(self):
        """Create a PatternLoader instance."""
        return PatternLoader()

    def test_loader_initialization(self, loader):
        """Test PatternLoader initialization."""
        assert loader is not None
        assert loader.patterns_dir.exists()
        assert loader._cache == {}

    def test_load_all_patterns(self, loader):
        """Test loading all patterns from directory."""
        patterns = loader.load_all_patterns()

        # Should have 21 patterns
        assert (
            len(patterns) >= 20
        ), f"Expected at least 20 patterns, got {len(patterns)}"

        # All should be ModelPatternMetadata
        for pattern_id, pattern in patterns.items():
            assert isinstance(pattern, ModelPatternMetadata)
            assert pattern.pattern_id == pattern_id

        # Should have patterns from all categories
        categories = {p.category for p in patterns.values()}
        assert EnumPatternCategory.RESILIENCE in categories
        assert EnumPatternCategory.OBSERVABILITY in categories
        assert EnumPatternCategory.INTEGRATION in categories

    def test_load_pattern_by_name(self, loader):
        """Test loading a single pattern by name."""
        pattern = loader.load_pattern("error_handling")

        assert pattern is not None
        assert pattern.name == "error_handling"
        assert pattern.category == EnumPatternCategory.RESILIENCE
        assert EnumNodeType.EFFECT in pattern.applicable_to

    def test_load_pattern_not_found(self, loader):
        """Test loading a non-existent pattern."""
        with pytest.raises(PatternNotFoundError):
            loader.load_pattern("nonexistent_pattern")

    def test_pattern_caching(self, loader):
        """Test that patterns are cached after first load."""
        # First load
        patterns1 = loader.load_all_patterns()

        # Second load (should be cached)
        patterns2 = loader.load_all_patterns()

        # Should be the same objects
        assert patterns1.keys() == patterns2.keys()

        # Cache should be populated
        assert len(loader._cache) > 0

    def test_clear_cache(self, loader):
        """Test clearing the pattern cache."""
        loader.load_all_patterns()
        assert len(loader._cache) > 0

        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_get_patterns_by_category(self, loader):
        """Test filtering patterns by category."""
        resilience_patterns = loader.get_patterns_by_category(
            EnumPatternCategory.RESILIENCE
        )

        assert len(resilience_patterns) > 0
        for pattern in resilience_patterns:
            assert pattern.category == EnumPatternCategory.RESILIENCE

    def test_get_patterns_by_node_type(self, loader):
        """Test filtering patterns by node type."""
        effect_patterns = loader.get_patterns_by_node_type(EnumNodeType.EFFECT)

        assert len(effect_patterns) > 0
        for pattern in effect_patterns:
            assert EnumNodeType.EFFECT in pattern.applicable_to

    def test_load_error_handling_pattern(self, loader):
        """Test loading the error_handling pattern specifically."""
        pattern = loader.load_pattern("error_handling")

        assert pattern.name == "error_handling"
        assert pattern.category == EnumPatternCategory.RESILIENCE
        assert len(pattern.code_template) > 0
        assert "OnexError" in pattern.code_template
        assert len(pattern.prerequisites) > 0

    def test_load_structured_logging_pattern(self, loader):
        """Test loading the structured_logging pattern."""
        pattern = loader.load_pattern("structured_logging")

        assert pattern.name == "structured_logging"
        assert pattern.category == EnumPatternCategory.OBSERVABILITY
        assert len(pattern.code_template) > 0
        assert "emit_log_event" in pattern.code_template

    def test_pattern_has_required_fields(self, loader):
        """Test that loaded patterns have all required fields."""
        patterns = loader.load_all_patterns()

        for pattern in patterns.values():
            # Required fields
            assert pattern.pattern_id
            assert pattern.name
            assert pattern.version
            assert pattern.category
            assert len(pattern.applicable_to) > 0
            assert len(pattern.description) > 0
            assert len(pattern.code_template) > 0

    def test_load_registry(self, loader):
        """Test loading the pattern registry."""
        registry = loader._load_registry()

        assert registry is not None
        assert "patterns" in registry
        assert len(registry["patterns"]) >= 20

    def test_invalid_patterns_dir(self):
        """Test initialization with invalid patterns directory."""
        with pytest.raises(PatternLoaderError):
            PatternLoader(patterns_dir=Path("/nonexistent/path"))


class TestPatternYAMLParsing:
    """Test YAML parsing and conversion to models."""

    @pytest.fixture
    def loader(self):
        """Create a PatternLoader instance."""
        return PatternLoader()

    def test_yaml_to_model_conversion(self, loader):
        """Test converting YAML data to ModelPatternMetadata."""
        # Load a pattern
        pattern = loader.load_pattern("error_handling")

        # Check conversion worked
        assert isinstance(pattern, ModelPatternMetadata)
        assert pattern.name == "error_handling"

    def test_pattern_tags_normalized(self, loader):
        """Test that pattern tags are normalized."""
        patterns = loader.load_all_patterns()

        for pattern in patterns.values():
            # Tags should be lowercase
            for tag in pattern.tags:
                assert tag == tag.lower()

    def test_pattern_prerequisites_normalized(self, loader):
        """Test that prerequisites are normalized."""
        patterns = loader.load_all_patterns()

        for pattern in patterns.values():
            # Prerequisites should not have duplicates
            assert len(pattern.prerequisites) == len(set(pattern.prerequisites))


@pytest.mark.performance
class TestPatternLoaderPerformance:
    """Performance tests for PatternLoader."""

    @pytest.fixture
    def loader(self):
        """Create a PatternLoader instance."""
        return PatternLoader()

    def test_cold_load_performance(self, loader, benchmark):
        """Test cold load performance (first time loading)."""

        def load():
            loader.clear_cache()
            return loader.load_all_patterns()

        result = benchmark(load)
        assert len(result) >= 20

        # Should be < 100ms
        assert benchmark.stats.stats.median < 0.1

    def test_warm_load_performance(self, loader, benchmark):
        """Test warm load performance (cached)."""
        # Pre-populate cache
        loader.load_all_patterns()

        result = benchmark(loader.load_all_patterns)
        assert len(result) >= 20

        # Should be < 1ms
        assert benchmark.stats.stats.median < 0.001
