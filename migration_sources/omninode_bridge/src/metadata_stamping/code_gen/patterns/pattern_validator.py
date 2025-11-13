"""
Pattern Validator for Code Generation.

This module validates production patterns for completeness and correctness.
It checks pattern YAML schema, code templates, prerequisites, and examples.

Performance Target: <5ms per pattern validation
"""

import logging
from typing import Optional

from jinja2 import Environment, TemplateSyntaxError

from .models import ModelPatternMetadata
from .pattern_loader import PatternLoader

logger = logging.getLogger(__name__)


class ValidationResult:
    """
    Result of pattern validation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        pattern_id: ID of validated pattern
    """

    def __init__(self, pattern_id: str):
        self.pattern_id = pattern_id
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)

    def __repr__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return (
            f"ValidationResult({self.pattern_id}, {status}, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


class PatternValidator:
    """
    Validate production patterns for correctness.

    This class checks patterns for:
    - Required fields present
    - Valid YAML schema
    - Correct Jinja2 template syntax
    - Prerequisites availability
    - Example node references

    Attributes:
        loader: PatternLoader for loading patterns
        jinja_env: Jinja2 environment for template validation

    Performance:
        - Single pattern validation: <5ms
        - All patterns validation: <100ms
    """

    def __init__(self, loader: Optional[PatternLoader] = None):
        """
        Initialize pattern validator.

        Args:
            loader: Optional PatternLoader instance
        """
        self.loader = loader or PatternLoader()
        self.jinja_env = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        logger.debug("PatternValidator initialized")

    def validate_pattern(self, pattern: ModelPatternMetadata) -> ValidationResult:
        """
        Validate a single pattern.

        Args:
            pattern: Pattern to validate

        Returns:
            ValidationResult with errors and warnings

        Performance: <5ms per pattern
        """
        result = ValidationResult(pattern.pattern_id)

        # Validate required fields
        self._validate_required_fields(pattern, result)

        # Validate code template
        self._validate_code_template(pattern, result)

        # Validate prerequisites
        self._validate_prerequisites(pattern, result)

        # Validate examples
        self._validate_examples(pattern, result)

        # Validate configuration
        self._validate_configuration(pattern, result)

        # Validate metadata
        self._validate_metadata(pattern, result)

        logger.debug(
            f"Validated pattern {pattern.pattern_id}: "
            f"{len(result.errors)} errors, {len(result.warnings)} warnings"
        )

        return result

    def validate_all_patterns(self) -> dict[str, ValidationResult]:
        """
        Validate all patterns in the library.

        Returns:
            Dictionary mapping pattern_id to ValidationResult

        Performance: <100ms for 21 patterns
        """
        logger.info("Validating all patterns...")

        patterns = self.loader.load_all_patterns()
        results = {}

        for pattern_id, pattern in patterns.items():
            try:
                result = self.validate_pattern(pattern)
                results[pattern_id] = result
            except Exception as e:
                # Create error result for exceptions
                result = ValidationResult(pattern_id)
                result.add_error(f"Validation exception: {e}")
                results[pattern_id] = result
                logger.error(f"Failed to validate {pattern_id}: {e}")

        # Log summary
        total = len(results)
        valid = sum(1 for r in results.values() if r.is_valid)
        invalid = total - valid

        logger.info(f"Validation complete: {valid}/{total} valid, {invalid} invalid")

        return results

    def _validate_required_fields(
        self, pattern: ModelPatternMetadata, result: ValidationResult
    ) -> None:
        """Validate required fields are present and non-empty."""
        # Required fields
        required_fields = {
            "pattern_id": pattern.pattern_id,
            "name": pattern.name,
            "version": pattern.version,
            "category": pattern.category,
            "applicable_to": pattern.applicable_to,
            "description": pattern.description,
            "code_template": pattern.code_template,
        }

        for field_name, field_value in required_fields.items():
            if not field_value:
                result.add_error(f"Required field '{field_name}' is missing or empty")

        # Check description length
        if len(pattern.description) < 20:
            result.add_error(
                f"Description too short: {len(pattern.description)} chars (min 20)"
            )

        # Check applicable_to is not empty
        if not pattern.applicable_to:
            result.add_error("Field 'applicable_to' must have at least one node type")

    def _validate_code_template(
        self, pattern: ModelPatternMetadata, result: ValidationResult
    ) -> None:
        """Validate code template Jinja2 syntax."""
        if not pattern.code_template:
            result.add_warning("Pattern has no code template")
            return

        try:
            # Try to compile the template
            self.jinja_env.from_string(pattern.code_template)
        except TemplateSyntaxError as e:
            result.add_error(
                f"Invalid Jinja2 template syntax: {e.message} at line {e.lineno}"
            )
        except Exception as e:
            result.add_error(f"Template validation error: {e}")

        # Check template length
        if len(pattern.code_template) < 10:
            result.add_warning(
                f"Code template very short: {len(pattern.code_template)} chars"
            )

    def _validate_prerequisites(
        self, pattern: ModelPatternMetadata, result: ValidationResult
    ) -> None:
        """Validate prerequisites are well-formed."""
        if not pattern.prerequisites:
            # Optional field, but warn if missing for complex patterns
            if pattern.complexity >= 3:
                result.add_warning(
                    "Complex pattern (complexity >= 3) has no prerequisites defined"
                )
            return

        for prereq in pattern.prerequisites:
            # Check prerequisite format
            if not prereq.strip():
                result.add_error("Empty prerequisite string")
                continue

            # Check for common patterns
            if "from" not in prereq and "import" not in prereq:
                result.add_warning(
                    f"Prerequisite '{prereq}' doesn't match import pattern"
                )

    def _validate_examples(
        self, pattern: ModelPatternMetadata, result: ValidationResult
    ) -> None:
        """Validate examples are well-formed."""
        if not pattern.examples:
            result.add_warning("Pattern has no examples")
            return

        for idx, example in enumerate(pattern.examples):
            # Validate example node name
            if not example.node_name:
                result.add_error(f"Example {idx} has no node_name")

            # Validate example node type is in applicable_to
            if example.node_type not in pattern.applicable_to:
                result.add_error(
                    f"Example {idx} node_type '{example.node_type}' "
                    f"not in pattern's applicable_to"
                )

            # Validate code snippet
            if not example.code_snippet or len(example.code_snippet) < 5:
                result.add_warning(f"Example {idx} has very short code snippet")

            # Validate description
            if not example.description or len(example.description) < 10:
                result.add_warning(f"Example {idx} has short description")

    def _validate_configuration(
        self, pattern: ModelPatternMetadata, result: ValidationResult
    ) -> None:
        """Validate configuration structure."""
        if not pattern.configuration:
            # Optional field
            return

        # Configuration should be a dict
        if not isinstance(pattern.configuration, dict):
            result.add_error("Configuration must be a dictionary")
            return

        # Warn if empty dict
        if not pattern.configuration:
            result.add_warning("Configuration is an empty dictionary")

    def _validate_metadata(
        self, pattern: ModelPatternMetadata, result: ValidationResult
    ) -> None:
        """Validate metadata fields."""
        # Validate version format (semver)
        if not pattern.version:
            result.add_error("Version is missing")
        elif not self._is_valid_semver(pattern.version):
            result.add_error(f"Invalid semver version: {pattern.version}")

        # Validate complexity range
        if not (1 <= pattern.complexity <= 5):
            result.add_error(f"Complexity {pattern.complexity} out of range (1-5)")

        # Validate tags
        if not pattern.tags:
            result.add_warning("Pattern has no tags (affects searchability)")

        # Check for reasonable tag count
        if len(pattern.tags) > 10:
            result.add_warning(
                f"Pattern has many tags ({len(pattern.tags)}), consider reducing"
            )

    def _is_valid_semver(self, version: str) -> bool:
        """Check if version string is valid semver format."""
        if not version:
            return False

        parts = version.split(".")
        if len(parts) != 3:
            return False

        try:
            # All parts should be integers
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False

    def print_validation_summary(self, results: dict[str, ValidationResult]) -> None:
        """
        Print a human-readable validation summary.

        Args:
            results: Dictionary of validation results
        """
        print("\n" + "=" * 70)
        print("PATTERN VALIDATION SUMMARY")
        print("=" * 70)

        total = len(results)
        valid = sum(1 for r in results.values() if r.is_valid)
        invalid = total - valid

        print(f"\nTotal patterns: {total}")
        print(f"✓ Valid: {valid}")
        print(f"✗ Invalid: {invalid}")

        # Print invalid patterns
        if invalid > 0:
            print("\n" + "-" * 70)
            print("INVALID PATTERNS:")
            print("-" * 70)

            for pattern_id, result in results.items():
                if not result.is_valid:
                    print(f"\n❌ {pattern_id}")
                    for error in result.errors:
                        print(f"   ERROR: {error}")
                    for warning in result.warnings:
                        print(f"   WARNING: {warning}")

        # Print warnings for valid patterns
        patterns_with_warnings = [
            r for r in results.values() if r.is_valid and r.warnings
        ]

        if patterns_with_warnings:
            print("\n" + "-" * 70)
            print("VALID PATTERNS WITH WARNINGS:")
            print("-" * 70)

            for result in patterns_with_warnings:
                print(f"\n⚠️  {result.pattern_id}")
                for warning in result.warnings:
                    print(f"   WARNING: {warning}")

        print("\n" + "=" * 70 + "\n")
