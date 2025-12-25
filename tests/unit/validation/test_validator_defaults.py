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
    CircularImportValidationResult,
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

        Current baseline (~583 unions as of 2025-12-25):
        - Most unions are legitimate `X | None` nullable patterns (ONEX-preferred)
        - These are counted but NOT flagged as violations
        - Actual violations (primitive soup, Union[X,None] syntax) are reported separately

        Threshold history:
        - 491 (2025-12-21): Initial baseline with DispatcherFunc | ContextAwareDispatcherFunc
        - 515 (2025-12-22): OMN-990 MessageDispatchEngine + OMN-947 snapshots
        - 540 (2025-12-23): OMN-950 comprehensive reducer tests
        - 544 (2025-12-23): OMN-954 effect idempotency and retry tests (PR #78)
        - 580 (2025-12-23): OMN-888 + PR #57 + OMN-954 merge
        - 585 (2025-12-25): OMN-811 ComputeRegistry + node registration orchestrator unions
        - 586 (2025-12-25): OMN-932 durable timeouts + introspection config migration
        - 588 (2025-12-25): OMN-881 Kafka integration test fixes + typing updates
        - 589 (2025-12-25): OMN-881 PR review fixes - _EventBusType conditional alias
        - 600 (2025-12-25): OMN-1006 heartbeat handler + projector update_heartbeat

        Threshold: 600 (buffer above ~595 baseline for codebase growth)
        Target: Reduce to <200 through ongoing dict[str, object] -> JsonValue migration.
        """
        assert INFRA_MAX_UNIONS == 600, (
            "INFRA_MAX_UNIONS should be 600 (OMN-1006 heartbeat handler)"
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

        Documented exemptions (KafkaEventBus, RuntimeHostProcess, etc.) are handled
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

    @patch("omnibase_infra.validation.infra_validators.validate_union_usage")
    def test_default_parameters_passed_to_core(self, mock_validate: MagicMock) -> None:
        """Verify defaults are correctly passed to core validator."""
        # Mock validation result with proper structure for filtered result creation
        # (validate_infra_union_usage now filters exempted patterns like patterns does)
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
        validate_infra_union_usage()

        # Verify core validator called with correct defaults
        mock_validate.assert_called_once_with(
            INFRA_SRC_PATH,  # Default directory
            max_unions=INFRA_MAX_UNIONS,  # Default max from constant
            strict=INFRA_UNIONS_STRICT,  # Strict mode (True) per OMN-983
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
        from click.testing import CliRunner

        runner = CliRunner()

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

    If these tests fail, it indicates one of:
    1. New code added unions without using proper typed patterns from omnibase_core
    2. The INFRA_MAX_UNIONS threshold needs to be adjusted (with documented rationale)

    See OMN-983 for threshold documentation and migration goals.
    """

    def test_union_count_within_threshold(self) -> None:
        """Verify union count stays within configured threshold.

        This test acts as a regression guard - if union count exceeds
        the threshold, it indicates new code added unions without
        using proper typed patterns from omnibase_core.

        Current baseline (~544 unions as of 2025-12-23):
        - Most unions are legitimate `X | None` nullable patterns (ONEX-preferred)
        - These are counted but NOT flagged as violations
        - Actual violations (primitive soup, Union[X,None] syntax) are reported separately

        Threshold: INFRA_MAX_UNIONS (555) - buffer above baseline.
        Target: Reduce to <200 through ongoing dict[str, object] -> JsonValue migration.
        """
        result = validate_infra_union_usage()

        # Extract actual union count from metadata for clear error messaging
        actual_count = (
            result.metadata.total_unions
            if result.metadata and hasattr(result.metadata, "total_unions")
            else "unknown"
        )

        assert result.is_valid, (
            f"Union count {actual_count} exceeds threshold {INFRA_MAX_UNIONS}. "
            f"New code may have added unions without using typed patterns. "
            f"Errors: {result.errors[:5]}{'...' if len(result.errors) > 5 else ''}"
        )

    def test_union_validation_returns_metadata(self) -> None:
        """Verify union validation returns metadata with count information.

        The validator should return metadata containing the total union count,
        which is useful for monitoring and documentation purposes.
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

        # Verify the count is reasonable (positive integer, below threshold)
        assert isinstance(result.metadata.total_unions, int), (
            "total_unions should be an integer"
        )
        assert result.metadata.total_unions >= 0, "total_unions should be non-negative"
        assert result.metadata.total_unions <= INFRA_MAX_UNIONS, (
            f"total_unions ({result.metadata.total_unions}) should be within "
            f"threshold ({INFRA_MAX_UNIONS})"
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

    def test_single_union_below_threshold(self, tmp_path: Path) -> None:
        """Verify single union counts correctly and passes validation.

        A file with just one union (e.g., `str | None`) should be valid
        when threshold is above 1.
        """
        (tmp_path / "one_union.py").write_text(
            "def greet(name: str | None = None) -> str:\n"
            "    return f'Hello, {name or \"World\"}!'\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        assert result.is_valid, "Single union should be valid when below threshold"
        assert result.errors == [], "Should have no errors for single union"
        if result.metadata and hasattr(result.metadata, "total_unions"):
            assert result.metadata.total_unions == 1, "Should report exactly one union"

    def test_few_unions_all_valid_patterns(self, tmp_path: Path) -> None:
        """Verify few unions using valid patterns pass validation.

        Using the ONEX-preferred `X | None` pattern should not cause violations,
        even with strict mode enabled.
        """
        (tmp_path / "few_unions.py").write_text(
            "from pydantic import BaseModel\n\n"
            "class ModelConfig(BaseModel):\n"
            "    name: str\n"
            "    value: int | None = None\n"
            "    description: str | None = None\n"
        )

        result = validate_infra_union_usage(str(tmp_path), max_unions=10, strict=True)

        assert result.is_valid, "Few valid unions should pass validation"
        assert result.errors == [], "Valid union patterns should not cause errors"
        if result.metadata and hasattr(result.metadata, "total_unions"):
            # Should have 2 unions (value and description)
            assert result.metadata.total_unions == 2, "Should count both unions"

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
        validators: list[
            Callable[..., ValidationResult | CircularImportValidationResult]
        ] = [
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
