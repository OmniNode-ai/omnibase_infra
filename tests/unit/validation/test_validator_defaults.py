"""
Tests verifying validator default parameters are consistent across all entry points.

Validates that:
- infra_validators.py functions have correct defaults
- CLI commands use correct defaults
- Scripts use correct defaults
- Constants are properly used
"""

import inspect
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import all validators and constants
from omnibase_infra.validation.infra_validators import (
    INFRA_MAX_UNIONS,
    INFRA_MAX_VIOLATIONS,
    INFRA_NODES_PATH,
    INFRA_PATTERNS_STRICT,
    INFRA_SRC_PATH,
    INFRA_UNIONS_STRICT,
    ModelModuleImportResult,
    ValidationResult,
    validate_infra_all,
    validate_infra_architecture,
    validate_infra_circular_imports,
    validate_infra_contracts,
    validate_infra_patterns,
    validate_infra_union_usage,
)


class TestInfraValidatorConstants:
    """Test constants used across validators."""

    def test_infra_max_unions_constant(self) -> None:
        """Verify INFRA_MAX_UNIONS constant has expected value.

        OMN-983: Strict validation mode enabled.

        This threshold applies ONLY to complex type annotation unions.
        The following patterns are excluded and NOT counted toward the threshold:
        - Simple optionals: `X | None` (idiomatic nullable pattern)
        - isinstance() unions: `isinstance(x, A | B)` (runtime type checks)

        What IS counted (threshold applies to):
        - Multi-type unions in annotations: `str | int`, `A | B | C`
        - Complex patterns: unions with 3+ types
        - Type annotation unions that are not simple optionals

        What is NOT counted (excluded from threshold):
        - Simple optionals: `X | None` where X is any single type
        - isinstance() unions: `isinstance(x, str | int)` (runtime checks, not annotations)

        Threshold history (after exclusion logic):
        - 120 (2025-12-25): Initial threshold after excluding ~470 `X | None` patterns
          - ~568 total unions in codebase
          - ~468 are simple `X | None` optionals (82%)
          - ~100 non-optional unions remain
          - Buffer of 20 above baseline for codebase growth
        - 121 (2025-12-25): OMN-881 introspection feature (+1 non-optional union)
        - 121 (2025-12-25): OMN-949 DLQ, OMN-816, OMN-811, OMN-1006 merges (all used X | None patterns, excluded)
        - 121 (2025-12-26): OMN-1007 registry pattern + merge with main (X | None patterns excluded)
        - 122 (2026-01-15): OMN-1346 extract registration domain plugin (+1 non-optional union)
        - 142 (2026-01-16): OMN-1305 ruff UP038 isinstance union syntax modernization (+20 unions)
        - 121 (2026-01-16): OMN-1305 isinstance union exclusion (-21 isinstance unions now excluded)

        Target: Keep below 150 - if this grows, consider typed patterns from omnibase_core.
        """
        assert INFRA_MAX_UNIONS == 122, (
            "INFRA_MAX_UNIONS should be 122 (excluding both X | None and isinstance unions)"
        )

    def test_infra_max_violations_constant(self) -> None:
        """Verify INFRA_MAX_VIOLATIONS constant has expected value."""
        assert INFRA_MAX_VIOLATIONS == 0, "INFRA_MAX_VIOLATIONS should be 0 (strict)"

    def test_infra_patterns_strict_constant(self) -> None:
        """Verify INFRA_PATTERNS_STRICT constant has expected value.

        OMN-983: Strict validation mode enabled.

        All violations must be either:
        - Fixed (code corrected to pass validation)
        - Exempted (added to exempted_patterns list with documented rationale)

        Documented exemptions (EventBusKafka, RuntimeHostProcess, etc.) are handled
        via the exempted_patterns list in validate_infra_patterns().
        """
        assert INFRA_PATTERNS_STRICT is True, (
            "INFRA_PATTERNS_STRICT should be True (strict mode per OMN-983)"
        )

    def test_infra_unions_strict_constant(self) -> None:
        """Verify INFRA_UNIONS_STRICT constant has expected value.

        OMN-983: Strict validation mode enabled.
        The validator flags actual violations (not just counting unions).
        """
        assert INFRA_UNIONS_STRICT is True, (
            "INFRA_UNIONS_STRICT should be True (strict mode per OMN-983)"
        )

    def test_infra_src_path_constant(self) -> None:
        """Verify INFRA_SRC_PATH constant has expected value."""
        assert INFRA_SRC_PATH == "src/omnibase_infra/"

    def test_infra_nodes_path_constant(self) -> None:
        """Verify INFRA_NODES_PATH constant has expected value."""
        assert INFRA_NODES_PATH == "src/omnibase_infra/nodes/"


class TestValidateInfraArchitectureDefaults:
    """Test validate_infra_architecture function defaults."""

    def test_function_signature_defaults(self) -> None:
        """Verify function has correct default parameter values."""
        sig = inspect.signature(validate_infra_architecture)

        # Check directory default
        directory_param = sig.parameters["directory"]
        assert directory_param.default == INFRA_SRC_PATH

        # Check max_violations default (strict mode)
        max_violations_param = sig.parameters["max_violations"]
        assert max_violations_param.default == INFRA_MAX_VIOLATIONS
        assert max_violations_param.default == 0, (
            "Should default to strict mode via INFRA_MAX_VIOLATIONS (0)"
        )

    @patch("omnibase_infra.validation.infra_validators.validate_architecture")
    def test_default_parameters_passed_to_core(self, mock_validate: MagicMock) -> None:
        """Verify defaults are correctly passed to core validator."""
        from omnibase_core.models.common.model_validation_metadata import (
            ModelValidationMetadata,
        )
        from omnibase_core.validation import ModelValidationResult

        # Create a proper ModelValidationResult for the mock
        mock_validate.return_value = ModelValidationResult(
            is_valid=True,
            errors=[],
            summary="Test validation",
            details="No issues",
            metadata=ModelValidationMetadata(
                files_processed=0,
                violations_found=0,
                max_violations=0,
            ),
        )

        # Call with defaults
        validate_infra_architecture()

        # Verify core validator called with correct defaults
        mock_validate.assert_called_once_with(
            INFRA_SRC_PATH,  # Default directory
            max_violations=INFRA_MAX_VIOLATIONS,  # Strict mode (0)
        )


class TestValidateInfraContractsDefaults:
    """Test validate_infra_contracts function defaults."""

    def test_function_signature_defaults(self) -> None:
        """Verify function has correct default parameter values."""
        sig = inspect.signature(validate_infra_contracts)

        # Check directory default
        directory_param = sig.parameters["directory"]
        assert directory_param.default == INFRA_NODES_PATH

    @patch("omnibase_infra.validation.infra_validators.validate_contracts")
    def test_default_parameters_passed_to_core(self, mock_validate: MagicMock) -> None:
        """Verify defaults are correctly passed to core validator."""
        mock_validate.return_value = MagicMock(is_valid=True, errors=[])

        # Call with defaults
        validate_infra_contracts()

        # Verify core validator called with correct defaults
        mock_validate.assert_called_once_with(INFRA_NODES_PATH)  # Default directory


class TestValidateInfraPatternsDefaults:
    """Test validate_infra_patterns function defaults."""

    def test_function_signature_defaults(self) -> None:
        """Verify function has correct default parameter values."""
        sig = inspect.signature(validate_infra_patterns)

        # Check directory default
        directory_param = sig.parameters["directory"]
        assert directory_param.default == INFRA_SRC_PATH

        # Check strict default - True for strict mode (OMN-983)
        strict_param = sig.parameters["strict"]
        assert strict_param.default == INFRA_PATTERNS_STRICT
        assert strict_param.default is True, (
            "Should default to strict mode via INFRA_PATTERNS_STRICT (True) per OMN-983"
        )

    @patch("omnibase_infra.validation.infra_validators.validate_patterns")
    def test_default_parameters_passed_to_core(self, mock_validate: MagicMock) -> None:
        """Verify defaults are correctly passed to core validator."""
        # Mock validation result with proper structure for filtered result creation
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.suggestions = []
        mock_result.issues = []
        mock_result.validated_value = None
        mock_result.summary = ""
        mock_result.details = ""
        mock_result.metadata = None
        mock_validate.return_value = mock_result

        # Call with defaults
        validate_infra_patterns()

        # Verify core validator called with correct defaults
        mock_validate.assert_called_once_with(
            INFRA_SRC_PATH,  # Default directory
            strict=INFRA_PATTERNS_STRICT,  # Strict mode (True) per OMN-983
        )


class TestValidateInfraUnionUsageDefaults:
    """Test validate_infra_union_usage function defaults."""

    def test_function_signature_defaults(self) -> None:
        """Verify function has correct default parameter values."""
        sig = inspect.signature(validate_infra_union_usage)

        # Check directory default
        directory_param = sig.parameters["directory"]
        assert directory_param.default == INFRA_SRC_PATH

        # Check max_unions default
        max_unions_param = sig.parameters["max_unions"]
        assert max_unions_param.default == INFRA_MAX_UNIONS, (
            f"Should default to INFRA_MAX_UNIONS ({INFRA_MAX_UNIONS})"
        )

        # Check strict default - True for strict mode (OMN-983)
        strict_param = sig.parameters["strict"]
        assert strict_param.default == INFRA_UNIONS_STRICT
        assert strict_param.default is True, (
            "Should default to strict mode via INFRA_UNIONS_STRICT (True) per OMN-983"
        )

    @patch("omnibase_infra.validation.infra_validators._count_non_optional_unions")
    def test_default_parameters_passed_to_core(
        self, mock_count_unions: MagicMock
    ) -> None:
        """Verify defaults are correctly passed through the validation chain.

        The new implementation uses _count_non_optional_unions which calls
        validate_union_usage_file per-file. This test verifies:
        1. The default directory (INFRA_SRC_PATH) is passed to the counter
        2. The result metadata reflects default max_unions (INFRA_MAX_UNIONS)
        3. The result metadata reflects default strict mode (INFRA_UNIONS_STRICT)
        """
        from pathlib import Path

        # Mock the union counter to return controlled values
        # Returns: (threshold_count, total_count, optional_count, isinstance_count, issues)
        mock_count_unions.return_value = (0, 0, 0, 0, [])

        # Call with defaults
        result = validate_infra_union_usage()

        # Verify the counter was called with default directory
        mock_count_unions.assert_called_once_with(Path(INFRA_SRC_PATH))

        # Verify result metadata reflects default parameters
        assert result.metadata is not None, "Result should have metadata"
        # max_unions and strict_mode are typed attributes on ModelValidationMetadata
        assert result.metadata.max_unions == INFRA_MAX_UNIONS, (
            f"Should use INFRA_MAX_UNIONS ({INFRA_MAX_UNIONS}) as default"
        )
        assert result.metadata.strict_mode == INFRA_UNIONS_STRICT, (
            f"Should use INFRA_UNIONS_STRICT ({INFRA_UNIONS_STRICT}) as default"
        )


class TestValidateInfraCircularImportsDefaults:
    """Test validate_infra_circular_imports function defaults."""

    def test_function_signature_defaults(self) -> None:
        """Verify function has correct default parameter values."""
        sig = inspect.signature(validate_infra_circular_imports)

        # Check directory default
        directory_param = sig.parameters["directory"]
        assert directory_param.default == INFRA_SRC_PATH

    @patch(
        "omnibase_infra.validation.infra_validators.CircularImportValidator.validate"
    )
    @patch("omnibase_infra.validation.infra_validators.CircularImportValidator")
    def test_default_parameters_passed_to_validator(
        self, mock_validator_class: MagicMock, mock_validate: MagicMock
    ) -> None:
        """Verify defaults are correctly passed to CircularImportValidator."""
        mock_instance = MagicMock()
        mock_instance.validate.return_value = MagicMock(has_circular_imports=False)
        mock_validator_class.return_value = mock_instance

        # Call with defaults
        validate_infra_circular_imports()

        # Verify validator initialized with correct default path
        mock_validator_class.assert_called_once_with(source_path=Path(INFRA_SRC_PATH))


class TestValidateInfraAllDefaults:
    """Test validate_infra_all function defaults."""

    def test_function_signature_defaults(self) -> None:
        """Verify function has correct default parameter values."""
        sig = inspect.signature(validate_infra_all)

        # Check directory default
        directory_param = sig.parameters["directory"]
        assert directory_param.default == INFRA_SRC_PATH

        # Check nodes_directory default
        nodes_directory_param = sig.parameters["nodes_directory"]
        assert nodes_directory_param.default == INFRA_NODES_PATH

    @patch("omnibase_infra.validation.infra_validators.validate_infra_architecture")
    @patch("omnibase_infra.validation.infra_validators.validate_infra_contracts")
    @patch("omnibase_infra.validation.infra_validators.validate_infra_patterns")
    @patch("omnibase_infra.validation.infra_validators.validate_infra_union_usage")
    @patch("omnibase_infra.validation.infra_validators.validate_infra_circular_imports")
    def test_all_validators_called_with_defaults(
        self,
        mock_circular: MagicMock,
        mock_unions: MagicMock,
        mock_patterns: MagicMock,
        mock_contracts: MagicMock,
        mock_architecture: MagicMock,
    ) -> None:
        """Verify all validators called with correct defaults in validate_infra_all."""
        # Setup mocks
        mock_result = MagicMock(is_valid=True, errors=[])
        mock_circular_result = MagicMock(has_circular_imports=False)

        mock_architecture.return_value = mock_result
        mock_contracts.return_value = mock_result
        mock_patterns.return_value = mock_result
        mock_unions.return_value = mock_result
        mock_circular.return_value = mock_circular_result

        # Call with defaults
        validate_infra_all()

        # Verify each validator called with correct defaults
        mock_architecture.assert_called_once_with(INFRA_SRC_PATH)
        mock_contracts.assert_called_once_with(INFRA_NODES_PATH)
        mock_patterns.assert_called_once_with(INFRA_SRC_PATH)
        mock_unions.assert_called_once_with(INFRA_SRC_PATH)
        mock_circular.assert_called_once_with(INFRA_SRC_PATH)


class TestScriptDefaults:
    """Test scripts/validate.py uses correct defaults."""

    def test_architecture_script_defaults(self) -> None:
        """Verify architecture validation script uses correct defaults."""
        # Check the script file directly
        script_path = Path("scripts/validate.py")
        assert script_path.exists(), "validate.py script should exist"

        script_content = script_path.read_text()

        # Verify architecture validator uses validate_infra_architecture() with built-in defaults
        assert "validate_infra_architecture()" in script_content, (
            "Architecture validator should use validate_infra_architecture() with built-in defaults"
        )
        assert (
            "from omnibase_infra.validation.infra_validators import" in script_content
            and "validate_infra_architecture" in script_content
        ), "Script should import validate_infra_architecture"

    def test_contracts_script_defaults(self) -> None:
        """Verify contracts validation script uses correct defaults."""
        script_path = Path("scripts/validate.py")
        script_content = script_path.read_text()

        # Verify contracts validator uses nodes directory
        assert 'validate_contracts("src/omnibase_infra/nodes/"' in script_content

    def test_patterns_script_defaults(self) -> None:
        """Verify patterns validation script uses correct defaults."""
        script_path = Path("scripts/validate.py")
        script_content = script_path.read_text()

        # Verify patterns validator uses validate_infra_patterns() which has built-in defaults
        assert "validate_infra_patterns()" in script_content, (
            "Patterns validator should use validate_infra_patterns() with built-in defaults"
        )
        assert (
            "from omnibase_infra.validation.infra_validators import validate_infra_patterns"
            in script_content
        ), "Script should import validate_infra_patterns"

    def test_unions_script_defaults(self) -> None:
        """Verify unions validation script uses correct defaults."""
        script_path = Path("scripts/validate.py")
        script_content = script_path.read_text()

        # Verify unions validator uses INFRA_MAX_UNIONS constant
        assert "INFRA_MAX_UNIONS" in script_content, (
            "Unions validator should import and use INFRA_MAX_UNIONS constant"
        )
        assert "max_unions=INFRA_MAX_UNIONS" in script_content
        # Verify unions validator uses INFRA_UNIONS_STRICT constant
        assert "INFRA_UNIONS_STRICT" in script_content, (
            "Unions validator should import and use INFRA_UNIONS_STRICT constant"
        )
        assert "strict=INFRA_UNIONS_STRICT" in script_content


class TestCLICommandDefaults:
    """Test CLI commands use correct defaults."""

    def test_architecture_cli_defaults(self) -> None:
        """Verify architecture CLI command has correct defaults."""
        # Check default max_violations in option
        from omnibase_infra.cli.commands import validate_architecture_cmd

        # Get the Click command decorators
        for decorator in validate_architecture_cmd.params:
            if decorator.name == "max_violations":
                # CLI uses None by default and resolves to INFRA_MAX_VIOLATIONS in code
                assert decorator.default is None, (
                    "CLI max_violations should default to None (resolved to INFRA_MAX_VIOLATIONS)"
                )
            elif decorator.name == "directory":
                assert decorator.default == "src/omnibase_infra/"

    def test_contracts_cli_defaults(self) -> None:
        """Verify contracts CLI command has correct defaults."""
        from omnibase_infra.cli.commands import validate_contracts_cmd

        # Get the Click command decorators
        for decorator in validate_contracts_cmd.params:
            if decorator.name == "directory":
                assert decorator.default == "src/omnibase_infra/nodes/"

    def test_patterns_cli_defaults(self) -> None:
        """Verify patterns CLI command has correct defaults."""
        from omnibase_infra.cli.commands import validate_patterns_cmd

        # Get the Click command decorators
        for decorator in validate_patterns_cmd.params:
            if decorator.name == "strict":
                # CLI uses None by default and resolves to INFRA_PATTERNS_STRICT in code
                assert decorator.default is None, (
                    "CLI strict should default to None (resolved to INFRA_PATTERNS_STRICT)"
                )
            elif decorator.name == "directory":
                assert decorator.default == "src/omnibase_infra/"

    def test_unions_cli_defaults(self) -> None:
        """Verify unions CLI command has correct defaults."""
        from omnibase_infra.cli.commands import validate_unions_cmd

        # Get the Click command decorators
        for decorator in validate_unions_cmd.params:
            if decorator.name == "max_unions":
                # CLI uses None by default and resolves to INFRA_MAX_UNIONS in code
                assert decorator.default is None, (
                    "CLI max_unions should default to None (resolved to INFRA_MAX_UNIONS)"
                )
            elif decorator.name == "strict":
                # CLI uses None by default and resolves to INFRA_UNIONS_STRICT in code
                assert decorator.default is None, (
                    "CLI strict should default to None (resolved to INFRA_UNIONS_STRICT)"
                )
            elif decorator.name == "directory":
                assert decorator.default == "src/omnibase_infra/"

    def test_imports_cli_defaults(self) -> None:
        """Verify imports CLI command has correct defaults."""
        from omnibase_infra.cli.commands import validate_imports_cmd

        # Get the Click command decorators
        for decorator in validate_imports_cmd.params:
            if decorator.name == "directory":
                assert decorator.default == "src/omnibase_infra/"

    def test_all_cli_defaults(self) -> None:
        """Verify validate all CLI command has correct defaults."""
        from omnibase_infra.cli.commands import validate_all_cmd

        # Get the Click command decorators
        for decorator in validate_all_cmd.params:
            if decorator.name == "directory":
                assert decorator.default == "src/omnibase_infra/"
            elif decorator.name == "nodes_dir":
                assert decorator.default == "src/omnibase_infra/nodes/"


class TestUnionCountRegressionGuard:
    """Regression tests verifying union count stays within configured threshold.

    These tests call the actual validator against the real codebase (not mocked)
    to ensure that new code additions don't exceed union count thresholds.

    IMPORTANT: The validator EXCLUDES simple optional patterns (`X | None`) from
    the count. Only non-optional unions count toward the threshold.

    If these tests fail, it indicates one of:
    1. New code added complex unions without using proper typed patterns
    2. The INFRA_MAX_UNIONS threshold needs to be adjusted (with documented rationale)

    See OMN-983 for threshold documentation and migration goals.
    """

    def test_union_count_within_threshold(self) -> None:
        """Verify NON-OPTIONAL union count stays within configured threshold.

        This test acts as a regression guard - if non-optional union count exceeds
        the threshold, it indicates new code added complex unions without
        using proper typed patterns from omnibase_core.

        Current baseline (~100 non-optional unions as of 2025-12-25):
        - Total unions: ~568 (including optionals)
        - Simple optionals (`X | None`): ~468 (82%) - EXCLUDED from threshold
        - Non-optional unions: ~100 - THIS is what counts toward threshold

        Threshold: INFRA_MAX_UNIONS (120) - buffer above ~100 baseline.
        Target: Keep below 150 - if this grows, consider typed patterns.
        """
        result = validate_infra_union_usage()

        # Extract actual counts from metadata for clear error messaging
        total_count = (
            result.metadata.total_unions
            if result.metadata and hasattr(result.metadata, "total_unions")
            else "unknown"
        )
        non_optional_count = (
            result.metadata.model_extra.get("non_optional_unions", "unknown")
            if result.metadata and hasattr(result.metadata, "model_extra")
            else "unknown"
        )

        assert result.is_valid, (
            f"Non-optional union count {non_optional_count} exceeds threshold {INFRA_MAX_UNIONS}. "
            f"(Total unions: {total_count}, but X | None patterns are excluded.) "
            f"New code may have added complex unions without using typed patterns. "
            f"Errors: {result.errors[:5]}{'...' if len(result.errors) > 5 else ''}"
        )

    def test_union_validation_returns_metadata(self) -> None:
        """Verify union validation returns metadata with count information.

        The validator should return metadata containing both total and non-optional
        union counts, which are useful for monitoring and documentation purposes.
        """
        result = validate_infra_union_usage()

        # Verify metadata is present
        assert result.metadata is not None, (
            "Union validation should return metadata with count information"
        )

        # Verify total_unions is present in metadata
        assert hasattr(result.metadata, "total_unions"), (
            "Metadata should contain total_unions count for monitoring"
        )

        # Verify the total count is reasonable (positive integer)
        assert isinstance(result.metadata.total_unions, int), (
            "total_unions should be an integer"
        )
        assert result.metadata.total_unions >= 0, "total_unions should be non-negative"

        # Verify non_optional_unions is in model_extra
        assert hasattr(result.metadata, "model_extra"), (
            "Metadata should have model_extra for custom fields"
        )
        non_optional = result.metadata.model_extra.get("non_optional_unions")
        assert non_optional is not None, (
            "Metadata should contain non_optional_unions count"
        )
        assert isinstance(non_optional, int), "non_optional_unions should be an integer"
        assert non_optional >= 0, "non_optional_unions should be non-negative"

        # Non-optional count should be within threshold
        assert non_optional <= INFRA_MAX_UNIONS, (
            f"non_optional_unions ({non_optional}) should be within "
            f"threshold ({INFRA_MAX_UNIONS})"
        )

        # Verify optional_unions_excluded is present
        optional_excluded = result.metadata.model_extra.get("optional_unions_excluded")
        assert optional_excluded is not None, (
            "Metadata should contain optional_unions_excluded count"
        )

        # Verify isinstance_unions_excluded is present
        isinstance_excluded = result.metadata.model_extra.get(
            "isinstance_unions_excluded"
        )
        assert isinstance_excluded is not None, (
            "Metadata should contain isinstance_unions_excluded count"
        )

        # Verify the counts are consistent:
        # total = threshold (non_optional) + optional + isinstance
        expected_excluded = result.metadata.total_unions - non_optional
        actual_excluded = optional_excluded + isinstance_excluded
        assert actual_excluded == expected_excluded, (
            f"excluded counts should equal total_unions - non_optional_unions: "
            f"{actual_excluded} (optional: {optional_excluded} + isinstance: "
            f"{isinstance_excluded}) != {expected_excluded}"
        )


class TestUnionValidatorEdgeCases:
    """Tests for edge cases with zero or few unions.

    PR #57 review flagged that tests assume non-zero union count and should
    handle edge cases for codebases with few unions. These tests verify the
    validator behaves correctly for:
    - Empty directories (no Python files)
    - Directories with Python files but zero unions
    - Directories with very few unions (below any threshold)
    - max_unions=0 with zero actual unions

    IMPORTANT: The validator EXCLUDES simple optional patterns (`X | None`) from
    the threshold count. Total unions include all patterns, but only non-optional
    unions count toward the threshold.
    """

    def test_empty_directory_is_valid(self, tmp_path: Path) -> None:
        """Verify empty directory validates successfully with zero unions.

        An empty directory should be valid - no unions means no violations.
        """
        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        assert result.is_valid, "Empty directory should be valid"
        assert result.errors == [], "Empty directory should have no errors"

    def test_empty_python_file_zero_unions(self, tmp_path: Path) -> None:
        """Verify Python file with no unions reports zero unions correctly.

        A file with only comments or empty content should report zero unions.
        """
        (tmp_path / "empty.py").write_text("# Empty file\n")

        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        assert result.is_valid, "Directory with empty Python file should be valid"
        assert result.errors == [], "Should have no errors for zero unions"
        if result.metadata and hasattr(result.metadata, "total_unions"):
            assert result.metadata.total_unions == 0, (
                "Should report exactly zero unions"
            )

    def test_code_without_unions_is_valid(self, tmp_path: Path) -> None:
        """Verify code without any union types validates successfully.

        Python code that doesn't use union types should pass validation
        with zero unions counted.
        """
        (tmp_path / "no_unions.py").write_text(
            "def hello(name: str) -> str:\n    return f'Hello, {name}!'\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        assert result.is_valid, "Code without unions should be valid"
        assert result.errors == [], "Should have no errors for zero unions"
        if result.metadata and hasattr(result.metadata, "total_unions"):
            assert result.metadata.total_unions == 0, (
                "Should report exactly zero unions"
            )

    def test_max_unions_zero_with_zero_unions(self, tmp_path: Path) -> None:
        """Verify max_unions=0 works correctly when there are zero unions.

        Edge case: Setting max_unions=0 should pass if there are no unions
        (0 <= 0 is valid, not a violation).
        """
        (tmp_path / "no_unions.py").write_text("def hello() -> str:\n    return 'hi'\n")

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        assert result.is_valid, "Zero unions should be valid even with max_unions=0"
        assert result.errors == [], "No violation when actual count equals max"

    def test_single_optional_union_excluded_from_threshold(
        self, tmp_path: Path
    ) -> None:
        """Verify single optional union (`X | None`) is excluded from threshold.

        A file with just one optional union (e.g., `str | None`) should be valid
        even with max_unions=0 because optionals are excluded from the count.
        """
        (tmp_path / "one_union.py").write_text(
            "def greet(name: str | None = None) -> str:\n"
            "    return f'Hello, {name or \"World\"}!'\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        assert result.is_valid, (
            "Single optional union should be valid (excluded from threshold)"
        )
        assert result.errors == [], "Should have no errors for single optional union"
        if result.metadata and hasattr(result.metadata, "total_unions"):
            assert result.metadata.total_unions == 1, (
                "Should report exactly one total union"
            )
        # Non-optional should be 0 since it's an optional
        if result.metadata and hasattr(result.metadata, "model_extra"):
            non_optional = result.metadata.model_extra.get("non_optional_unions", 0)
            assert non_optional == 0, (
                "Non-optional unions should be 0 for X | None pattern"
            )

    def test_few_optionals_all_excluded_from_threshold(self, tmp_path: Path) -> None:
        """Verify few unions using optional patterns are excluded from threshold.

        Using the ONEX-preferred `X | None` pattern should not cause violations,
        even with max_unions=0 because these patterns are excluded.
        """
        (tmp_path / "few_unions.py").write_text(
            "from pydantic import BaseModel\n\n"
            "class ModelConfig(BaseModel):\n"
            "    name: str\n"
            "    value: int | None = None\n"
            "    description: str | None = None\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        assert result.is_valid, "Few optional unions should pass validation (excluded)"
        assert result.errors == [], "Valid optional patterns should not cause errors"
        if result.metadata and hasattr(result.metadata, "total_unions"):
            # Should have 2 total unions (value and description)
            assert result.metadata.total_unions == 2, (
                "Should count both unions in total"
            )
        # But non-optional should be 0
        if result.metadata and hasattr(result.metadata, "model_extra"):
            non_optional = result.metadata.model_extra.get("non_optional_unions", 0)
            assert non_optional == 0, (
                "Non-optional unions should be 0 for X | None patterns"
            )

    def test_non_optional_union_counts_toward_threshold(self, tmp_path: Path) -> None:
        """Verify non-optional unions count toward threshold.

        A union like `str | int` is NOT an optional pattern and SHOULD
        count toward the threshold.
        """
        (tmp_path / "complex_union.py").write_text(
            "def process(value: str | int) -> str:\n    return str(value)\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        # Should fail because str | int counts toward threshold
        assert not result.is_valid, "Non-optional union should count toward threshold"
        if result.metadata and hasattr(result.metadata, "model_extra"):
            non_optional = result.metadata.model_extra.get("non_optional_unions", 0)
            assert non_optional == 1, "Non-optional unions should be 1 for str | int"

    def test_no_division_by_zero_with_empty_codebase(self, tmp_path: Path) -> None:
        """Verify no division errors occur with empty or minimal codebases.

        This test guards against division by zero or similar errors that might
        occur when calculating percentages or ratios with zero counts.
        """
        # Test with truly empty directory (no files at all)
        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        # Should not raise any exceptions and should return valid result
        assert result.is_valid, "Should not crash on empty directory"
        assert isinstance(result.errors, list), "Errors should be a list"

    def test_metadata_present_even_with_zero_unions(self, tmp_path: Path) -> None:
        """Verify metadata is properly populated even with zero unions.

        The validator should always return consistent metadata structure,
        even when no unions are found.
        """
        (tmp_path / "simple.py").write_text("x: int = 42\n")

        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        assert result.is_valid, "Simple code should be valid"
        # Metadata should be present
        assert result.metadata is not None, "Metadata should be present"
        # Metadata must have total_unions attribute (consistent structure requirement)
        assert hasattr(result.metadata, "total_unions"), (
            "Metadata must have 'total_unions' attribute for consistent structure"
        )
        # total_unions should be 0 for code without unions
        assert result.metadata.total_unions == 0, (
            "total_unions should be 0 for code without unions"
        )
        # non_optional_unions should also be 0
        if hasattr(result.metadata, "model_extra"):
            non_optional = result.metadata.model_extra.get("non_optional_unions", 0)
            assert non_optional == 0, (
                "non_optional_unions should be 0 for code without unions"
            )


class TestDefaultsConsistency:
    """Test that defaults are consistent across all entry points."""

    def test_architecture_max_violations_consistency(self) -> None:
        """Verify max_violations=INFRA_MAX_VIOLATIONS across all architecture entry points."""
        # Function default uses constant
        sig = inspect.signature(validate_infra_architecture)
        assert sig.parameters["max_violations"].default == INFRA_MAX_VIOLATIONS

        # CLI uses None and resolves to constant in code
        from omnibase_infra.cli.commands import validate_architecture_cmd

        cli_default = None
        for param in validate_architecture_cmd.params:
            if param.name == "max_violations":
                cli_default = param.default
        assert cli_default is None, (
            "CLI should use None and resolve to INFRA_MAX_VIOLATIONS"
        )

        # Script uses constant (verified in test_architecture_script_defaults)

    def test_patterns_strict_consistency(self) -> None:
        """Verify strict=INFRA_PATTERNS_STRICT across all patterns entry points."""
        # Function default uses constant
        sig = inspect.signature(validate_infra_patterns)
        assert sig.parameters["strict"].default == INFRA_PATTERNS_STRICT

        # CLI uses None and resolves to constant in code
        from omnibase_infra.cli.commands import validate_patterns_cmd

        cli_default = None
        for param in validate_patterns_cmd.params:
            if param.name == "strict":
                cli_default = param.default
        assert cli_default is None, (
            "CLI should use None and resolve to INFRA_PATTERNS_STRICT"
        )

        # Script uses constant (verified in test_patterns_script_defaults)

    def test_unions_max_consistency(self) -> None:
        """Verify max_unions=INFRA_MAX_UNIONS across all union entry points."""
        # Function default
        sig = inspect.signature(validate_infra_union_usage)
        assert sig.parameters["max_unions"].default == INFRA_MAX_UNIONS

        # CLI uses None and resolves to INFRA_MAX_UNIONS (verified in code review)
        # Script imports and uses INFRA_MAX_UNIONS (verified in test_unions_script_defaults)

    def test_unions_strict_consistency(self) -> None:
        """Verify strict=INFRA_UNIONS_STRICT across all union entry points."""
        # Function default uses constant
        sig = inspect.signature(validate_infra_union_usage)
        assert sig.parameters["strict"].default == INFRA_UNIONS_STRICT

        # CLI uses None and resolves to constant in code
        from omnibase_infra.cli.commands import validate_unions_cmd

        cli_default = None
        for param in validate_unions_cmd.params:
            if param.name == "strict":
                cli_default = param.default
        assert cli_default is None, (
            "CLI should use None and resolve to INFRA_UNIONS_STRICT"
        )

        # Script uses constant (verified in test_unions_script_defaults)

    def test_directory_defaults_consistency(self) -> None:
        """Verify directory defaults are consistent across entry points."""
        # All validators using INFRA_SRC_PATH should default to same value
        validators: list[Callable[..., ValidationResult | ModelModuleImportResult]] = [
            validate_infra_architecture,
            validate_infra_patterns,
            validate_infra_union_usage,
            validate_infra_circular_imports,
        ]

        for validator in validators:
            sig = inspect.signature(validator)
            dir_param = sig.parameters["directory"]
            assert dir_param.default == INFRA_SRC_PATH, (
                f"{validator.__name__} should default to INFRA_SRC_PATH"
            )

        # Contract validator should default to INFRA_NODES_PATH
        sig = inspect.signature(validate_infra_contracts)
        assert sig.parameters["directory"].default == INFRA_NODES_PATH


class TestIsinstanceUnionExclusion:
    """Tests for isinstance() union exclusion from the threshold count.

    isinstance(x, A | B) is a runtime type check, not a type annotation.
    These patterns should NOT count toward the union complexity threshold because:
    1. They are runtime expressions, not static type hints
    2. Modern Python (PEP 604) and ruff UP038 encourage this syntax
    3. The validator's goal is to limit complex TYPE ANNOTATIONS

    See OMN-1305 for the feature implementation.
    """

    def test_isinstance_union_excluded_from_threshold(self, tmp_path: Path) -> None:
        """Verify isinstance unions are not counted toward threshold.

        A file with only isinstance unions should be valid even with max_unions=0
        because isinstance unions are excluded from the count.
        """
        (tmp_path / "isinstance_only.py").write_text(
            "def check(x) -> bool:\n"
            "    if isinstance(x, str | int):\n"
            "        return True\n"
            "    return isinstance(x, float | bool)\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        assert result.is_valid, "isinstance unions should not count toward threshold"
        assert result.errors == [], "Should have no errors for isinstance unions"

        # Verify metadata shows the exclusion
        if result.metadata:
            total = result.metadata.total_unions
            isinstance_excluded = result.metadata.model_extra.get(
                "isinstance_unions_excluded", 0
            )
            threshold_count = result.metadata.model_extra.get("non_optional_unions", 0)

            assert total == 2, "Should detect 2 total unions"
            assert isinstance_excluded == 2, "Should exclude 2 isinstance unions"
            assert threshold_count == 0, "Threshold count should be 0"

    def test_annotation_union_counts_toward_threshold(self, tmp_path: Path) -> None:
        """Verify type annotation unions DO count toward threshold.

        A file with annotation unions (not isinstance) should count toward
        the threshold and fail if max_unions=0.
        """
        (tmp_path / "annotation_union.py").write_text(
            "def process(value: str | int) -> str:\n    return str(value)\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        assert not result.is_valid, "Annotation union should count toward threshold"

        if result.metadata:
            threshold_count = result.metadata.model_extra.get("non_optional_unions", 0)
            isinstance_excluded = result.metadata.model_extra.get(
                "isinstance_unions_excluded", 0
            )

            assert threshold_count == 1, "Should count 1 annotation union"
            assert isinstance_excluded == 0, "No isinstance unions to exclude"

    def test_mixed_isinstance_and_annotation_unions(self, tmp_path: Path) -> None:
        """Verify mixed isinstance and annotation unions are counted correctly.

        isinstance unions should be excluded, annotation unions should count.
        """
        (tmp_path / "mixed_unions.py").write_text(
            "def process(value: str | int) -> str:\n"
            "    if isinstance(value, str | bytes):\n"
            "        return value\n"
            "    return str(value)\n"
        )

        # max_unions=0 should fail because annotation union counts
        result_strict = validate_infra_union_usage(
            str(tmp_path), max_unions=0, strict=True
        )
        assert not result_strict.is_valid, "Should fail with max_unions=0"

        # max_unions=1 should pass (1 annotation union, isinstance excluded)
        result_relaxed = validate_infra_union_usage(
            str(tmp_path), max_unions=1, strict=True
        )
        assert result_relaxed.is_valid, "Should pass with max_unions=1"

        if result_relaxed.metadata:
            total = result_relaxed.metadata.total_unions
            isinstance_excluded = result_relaxed.metadata.model_extra.get(
                "isinstance_unions_excluded", 0
            )
            threshold_count = result_relaxed.metadata.model_extra.get(
                "non_optional_unions", 0
            )

            assert total == 2, "Should detect 2 total unions"
            assert isinstance_excluded == 1, "Should exclude 1 isinstance union"
            assert threshold_count == 1, "Should count 1 annotation union"

    def test_isinstance_with_three_types(self, tmp_path: Path) -> None:
        """Verify isinstance with 3+ types is excluded from threshold count.

        isinstance(x, A | B | C) should be excluded regardless of type count.
        Note: We use strict=False because omnibase_core may report style issues
        for 4+ type unions, but our goal here is to verify threshold exclusion.
        """
        (tmp_path / "isinstance_multi.py").write_text(
            "def check(x) -> bool:\n"
            "    return isinstance(x, str | int | float | bool)\n"
        )

        # Use strict=False to focus on threshold check, not style issues
        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=False)

        # The threshold check should pass because isinstance unions are excluded
        # (non_optional_unions should be 0, which is <= max_unions=0)
        if result.metadata:
            threshold_count = result.metadata.model_extra.get("non_optional_unions", 0)
            isinstance_excluded = result.metadata.model_extra.get(
                "isinstance_unions_excluded", 0
            )
            assert threshold_count == 0, (
                "Multi-type isinstance union should be excluded from threshold"
            )
            assert isinstance_excluded == 1, (
                "Should count 1 isinstance union as excluded"
            )

    def test_isinstance_unions_excluded_field_in_metadata(self) -> None:
        """Verify isinstance_unions_excluded is present in actual codebase validation.

        This test validates the real codebase contains isinstance unions that
        are being properly excluded and tracked in metadata.
        """
        result = validate_infra_union_usage()

        assert result.metadata is not None, "Metadata should be present"
        isinstance_count = result.metadata.model_extra.get("isinstance_unions_excluded")
        assert isinstance_count is not None, (
            "isinstance_unions_excluded should be in metadata"
        )
        assert isinstance(isinstance_count, int), (
            "isinstance_unions_excluded should be an integer"
        )
        assert isinstance_count >= 0, (
            "isinstance_unions_excluded should be non-negative"
        )

        # Verify the math: total = threshold + optional + isinstance
        total = result.metadata.total_unions
        threshold = result.metadata.model_extra.get("non_optional_unions", 0)
        optional = result.metadata.model_extra.get("optional_unions_excluded", 0)

        assert total == threshold + optional + isinstance_count, (
            f"Union counts should sum correctly: "
            f"{total} != {threshold} + {optional} + {isinstance_count}"
        )

    def test_optional_and_isinstance_both_excluded(self, tmp_path: Path) -> None:
        """Verify both optional and isinstance patterns are excluded.

        A file with only optionals and isinstance unions should pass with
        max_unions=0 since both pattern types are excluded.
        """
        (tmp_path / "both_excluded.py").write_text(
            "def process(value: str | None = None) -> bool:\n"
            "    if isinstance(value, str | bytes):\n"
            "        return True\n"
            "    return False\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=0, strict=True)

        assert result.is_valid, "Both optional and isinstance unions should be excluded"

        if result.metadata:
            total = result.metadata.total_unions
            optional = result.metadata.model_extra.get("optional_unions_excluded", 0)
            isinstance_count = result.metadata.model_extra.get(
                "isinstance_unions_excluded", 0
            )
            threshold = result.metadata.model_extra.get("non_optional_unions", 0)

            assert total == 2, "Should detect 2 total unions"
            assert optional == 1, "Should exclude 1 optional union"
            assert isinstance_count == 1, "Should exclude 1 isinstance union"
            assert threshold == 0, "Threshold count should be 0"
