#!/usr/bin/env python3
"""
Unit tests for pattern validation.

Tests validation of pattern definitions and usage.
"""


import pytest

from src.metadata_stamping.code_gen.patterns import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternExample,
    ModelPatternMetadata,
    PatternLoader,
    PatternValidator,
)


class TestPatternValidation:
    """Test suite for pattern validation."""

    @pytest.fixture
    def validator(self):
        """Create a PatternValidator instance."""
        return PatternValidator()

    @pytest.fixture
    def valid_pattern(self):
        """Create a valid pattern for testing."""
        return ModelPatternMetadata(
            pattern_id="test_pattern_v1",
            name="Test Pattern",
            version="1.0.0",
            category=EnumPatternCategory.RESILIENCE,
            applicable_to=[EnumNodeType.EFFECT, EnumNodeType.ORCHESTRATOR],
            description="This is a comprehensive test pattern for validation purposes",
            code_template="async def test():\n    pass",
            tags=["async", "test", "validation"],
            prerequisites=["from omnibase_core import OnexError"],
            complexity=2,
            examples=[
                ModelPatternExample(
                    node_name="test_node",
                    node_type=EnumNodeType.EFFECT,
                    code_snippet="async def test():\n    return True",
                    description="Test example demonstrating the pattern",
                )
            ],
            configuration={"timeout": 5000},
        )

    def test_validate_pattern_structure(self, validator, valid_pattern):
        """Test validating pattern structure."""
        result = validator.validate_pattern(valid_pattern)

        # Valid pattern should pass
        assert result.is_valid
        assert len(result.errors) == 0

        # Test invalid structure - empty description (too short)
        # Use model_copy to bypass Pydantic validation
        invalid_pattern = valid_pattern.model_copy(deep=True)
        # Manually set invalid field (bypasses Pydantic validation)
        invalid_pattern.__dict__["description"] = "short"  # Too short

        result = validator.validate_pattern(invalid_pattern)
        assert not result.is_valid
        assert any("description" in error.lower() for error in result.errors)

    def test_validate_pattern_metadata(self, validator, valid_pattern):
        """Test validating pattern metadata."""
        # Test valid metadata
        result = validator.validate_pattern(valid_pattern)
        assert result.is_valid

        # Test invalid version format
        invalid_pattern = valid_pattern.model_copy(deep=True)
        invalid_pattern.version = "1.0"  # Not semver format
        result = validator.validate_pattern(invalid_pattern)
        assert not result.is_valid
        assert any("version" in error.lower() for error in result.errors)

        # Test invalid complexity
        invalid_pattern = valid_pattern.model_copy(deep=True)
        invalid_pattern.complexity = 10  # Out of range
        result = validator.validate_pattern(invalid_pattern)
        assert not result.is_valid
        assert any("complexity" in error.lower() for error in result.errors)

        # Test pattern without tags - should warn
        pattern_no_tags = valid_pattern.model_copy(deep=True)
        pattern_no_tags.tags = []
        result = validator.validate_pattern(pattern_no_tags)
        assert result.is_valid  # Valid but with warnings
        assert any("tags" in warning.lower() for warning in result.warnings)

    def test_validate_pattern_code_blocks(self, validator, valid_pattern):
        """Test validating pattern code blocks."""
        # Test valid code template
        result = validator.validate_pattern(valid_pattern)
        assert result.is_valid

        # Test invalid Jinja2 syntax
        invalid_pattern = valid_pattern.model_copy(deep=True)
        invalid_pattern.code_template = "{{ unclosed_tag"
        result = validator.validate_pattern(invalid_pattern)
        assert not result.is_valid
        assert any(
            "template" in error.lower() or "jinja" in error.lower()
            for error in result.errors
        )

        # Test very short template - should warn
        short_pattern = valid_pattern.model_copy(deep=True)
        short_pattern.code_template = "pass"
        result = validator.validate_pattern(short_pattern)
        assert any("short" in warning.lower() for warning in result.warnings)

        # Test empty template
        empty_pattern = valid_pattern.model_copy(deep=True)
        empty_pattern.code_template = ""
        result = validator.validate_pattern(empty_pattern)
        assert not result.is_valid

    def test_validate_pattern_dependencies(self, validator, valid_pattern):
        """Test validating pattern dependencies."""
        # Test valid prerequisites
        result = validator.validate_pattern(valid_pattern)
        assert result.is_valid

        # Test empty prerequisite strings
        invalid_pattern = valid_pattern.model_copy(deep=True)
        invalid_pattern.prerequisites = ["from omnibase_core import OnexError", ""]
        result = validator.validate_pattern(invalid_pattern)
        assert not result.is_valid
        assert any(
            "empty" in error.lower() and "prerequisite" in error.lower()
            for error in result.errors
        )

        # Test complex pattern without prerequisites - should warn
        complex_pattern = valid_pattern.model_copy(deep=True)
        complex_pattern.complexity = 4
        complex_pattern.prerequisites = []
        result = validator.validate_pattern(complex_pattern)
        assert any("prerequisite" in warning.lower() for warning in result.warnings)


class TestPatternCompatibility:
    """Test suite for pattern compatibility checks."""

    @pytest.fixture
    def loader(self):
        """Create a PatternLoader instance."""
        return PatternLoader()

    def test_pattern_node_type_compatibility(self, loader):
        """Test pattern compatibility with node types."""
        # Load a real pattern
        pattern = loader.load_pattern("error_handling")

        # Test node type compatibility using calculate_match_score
        # error_handling is applicable to EFFECT nodes
        score = pattern.calculate_match_score(
            node_type=EnumNodeType.EFFECT,
            required_features=set(),  # No specific features needed
        )
        # Should have positive score since node type is compatible
        assert score > 0.0
        # Base score for node type match is 0.3
        assert score >= 0.3

        # Test with incompatible node type (should fail if pattern not applicable)
        # Create a pattern specific to EFFECT nodes only
        pattern_effect_only = ModelPatternMetadata(
            pattern_id="effect_only_v1",
            name="Effect Only",
            version="1.0.0",
            category=EnumPatternCategory.RESILIENCE,
            applicable_to=[EnumNodeType.EFFECT],  # Only effect nodes
            description="Pattern only for effect nodes testing purposes here",
            code_template="async def test():\n    pass",
            tags=["effect-specific"],
        )

        # Should match EFFECT nodes
        score_effect = pattern_effect_only.calculate_match_score(
            node_type=EnumNodeType.EFFECT,
            required_features={"effect-specific"},
        )
        assert score_effect > 0.0

        # Should NOT match COMPUTE nodes (score should be 0.0)
        score_compute = pattern_effect_only.calculate_match_score(
            node_type=EnumNodeType.COMPUTE,
            required_features={"effect-specific"},
        )
        assert score_compute == 0.0

    def test_pattern_version_compatibility(self, loader):
        """Test pattern version compatibility."""
        patterns = loader.load_all_patterns()

        # All patterns should have valid semver versions
        for pattern in patterns.values():
            assert pattern.version
            parts = pattern.version.split(".")
            assert len(parts) == 3
            # All parts should be integers
            for part in parts:
                assert part.isdigit()

        # Test pattern version is properly formatted
        pattern = loader.load_pattern("error_handling")
        assert pattern.version.count(".") == 2

    def test_pattern_mixin_compatibility(self, loader):
        """Test pattern compatibility with mixins."""
        # Load patterns with prerequisites
        pattern = loader.load_pattern("error_handling")

        # Should have prerequisites defining required imports/mixins
        assert pattern.prerequisites
        assert len(pattern.prerequisites) > 0

        # Prerequisites should mention required imports
        prereq_str = " ".join(pattern.prerequisites).lower()
        assert any(keyword in prereq_str for keyword in ["import", "from", "mixin"])


class TestPatternConstraints:
    """Test suite for pattern constraint validation."""

    @pytest.fixture
    def validator(self):
        """Create a PatternValidator instance."""
        return PatternValidator()

    @pytest.fixture
    def loader(self):
        """Create a PatternLoader instance."""
        return PatternLoader()

    def test_validate_required_imports(self, validator, loader):
        """Test validating required imports in patterns."""
        # Load a pattern with prerequisites
        pattern = loader.load_pattern("error_handling")
        result = validator.validate_pattern(pattern)

        # Should have valid prerequisites
        assert pattern.prerequisites
        assert len(pattern.prerequisites) > 0

        # Validate prerequisites format
        # Prerequisites can be in format "X imported from Y" or "import X" or "from X import Y"
        for prereq in pattern.prerequisites:
            # Should not be empty
            assert prereq.strip()
            # Should contain import-related keywords or describe imports
            assert any(
                keyword in prereq.lower()
                for keyword in ["import", "from", "imported", "available"]
            )

    def test_validate_required_dependencies(self, validator, loader):
        """Test validating required dependencies."""
        # Load all patterns and validate their dependencies
        patterns = loader.load_all_patterns()

        for pattern_id, pattern in patterns.items():
            result = validator.validate_pattern(pattern)

            # If pattern has prerequisites, they should be valid
            if pattern.prerequisites:
                for prereq in pattern.prerequisites:
                    # Should not be empty or whitespace-only
                    assert prereq.strip()

            # Complex patterns should have prerequisites
            if pattern.complexity >= 3 and not pattern.prerequisites:
                # Should have a warning about missing prerequisites
                assert any("prerequisite" in w.lower() for w in result.warnings)

    def test_validate_performance_constraints(self, validator, loader):
        """Test validating performance constraints."""
        # Load patterns and check performance metadata
        patterns = loader.load_all_patterns()

        for pattern in patterns.values():
            # Performance impact should be a dict if present
            if pattern.performance_impact:
                assert isinstance(pattern.performance_impact, dict)

            # Configuration should be a dict if present
            if pattern.configuration:
                assert isinstance(pattern.configuration, dict)

        # Validate a specific pattern
        pattern = loader.load_pattern("error_handling")
        result = validator.validate_pattern(pattern)

        # Should pass validation
        assert result.is_valid


@pytest.mark.parametrize(
    "pattern_name,validity",
    [
        # Valid patterns
        ("error_handling", True),
        ("structured_logging", True),
        ("event_publishing", True),
        ("metrics_tracking", True),
        ("health_check_mode", True),
        # These pattern names should be valid if they exist
        ("consul_registration", True),
        ("kafka_client_initialization", True),
        ("standard_imports", True),
    ],
)
def test_pattern_validation_status(pattern_name, validity):
    """Test validation status of different patterns."""
    loader = PatternLoader()
    validator = PatternValidator()

    try:
        # Load the pattern
        pattern = loader.load_pattern(pattern_name)

        # Validate it
        result = validator.validate_pattern(pattern)

        # Check expected validity
        if validity:
            # Should be valid (may have warnings but no errors)
            assert (
                result.is_valid
            ), f"Pattern {pattern_name} should be valid but has errors: {result.errors}"
        else:
            # Should be invalid
            assert (
                not result.is_valid
            ), f"Pattern {pattern_name} should be invalid but passed validation"

    except Exception as e:
        # If we can't load the pattern, test should fail
        pytest.fail(f"Failed to load pattern {pattern_name}: {e}")
