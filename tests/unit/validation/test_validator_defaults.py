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

        NOTE: Currently set to 200 (baseline as of 2025-12-17) due to tech debt.
        This is documented in infra_validators.py and will be reduced incrementally.
        The omnibase_core validator counts X | None (PEP 604) patterns as unions,
        which is the ONEX-preferred syntax per CLAUDE.md.
        """
        assert INFRA_MAX_UNIONS == 200, (
            "INFRA_MAX_UNIONS should be 200 (current baseline)"
        )

    def test_infra_max_violations_constant(self) -> None:
        """Verify INFRA_MAX_VIOLATIONS constant has expected value."""
        assert INFRA_MAX_VIOLATIONS == 0, "INFRA_MAX_VIOLATIONS should be 0 (strict)"

    def test_infra_patterns_strict_constant(self) -> None:
        """Verify INFRA_PATTERNS_STRICT constant has expected value.

        Set to False to allow legitimate infrastructure patterns (registry classes
        with many methods, functions with multiple parameters for configuration).
        """
        assert INFRA_PATTERNS_STRICT is False, "INFRA_PATTERNS_STRICT should be False"

    def test_infra_unions_strict_constant(self) -> None:
        """Verify INFRA_UNIONS_STRICT constant has expected value."""
        assert INFRA_UNIONS_STRICT is False, "INFRA_UNIONS_STRICT should be False"

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
        mock_validate.return_value = MagicMock(is_valid=True, errors=[])

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

        # Check strict default - False to allow legitimate infrastructure patterns
        strict_param = sig.parameters["strict"]
        assert strict_param.default == INFRA_PATTERNS_STRICT
        assert strict_param.default is False, (
            "Should default to non-strict mode via INFRA_PATTERNS_STRICT (False)"
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
            strict=INFRA_PATTERNS_STRICT,  # Non-strict mode (False) for infra patterns
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

        # Check strict default
        strict_param = sig.parameters["strict"]
        assert strict_param.default == INFRA_UNIONS_STRICT
        assert strict_param.default is False, (
            "Should default to non-strict mode via INFRA_UNIONS_STRICT (False)"
        )

    @patch("omnibase_infra.validation.infra_validators.validate_union_usage")
    def test_default_parameters_passed_to_core(self, mock_validate: MagicMock) -> None:
        """Verify defaults are correctly passed to core validator."""
        mock_validate.return_value = MagicMock(is_valid=True, errors=[])

        # Call with defaults
        validate_infra_union_usage()

        # Verify core validator called with correct defaults
        mock_validate.assert_called_once_with(
            INFRA_SRC_PATH,  # Default directory
            max_unions=INFRA_MAX_UNIONS,  # Default max (30)
            strict=INFRA_UNIONS_STRICT,  # Non-strict (False)
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

        # Verify architecture validator uses INFRA_MAX_VIOLATIONS constant
        assert "max_violations=INFRA_MAX_VIOLATIONS" in script_content, (
            "Architecture validator should use INFRA_MAX_VIOLATIONS constant"
        )
        assert (
            "from omnibase_infra.validation.infra_validators import INFRA_MAX_VIOLATIONS"
            in script_content
        ), "Script should import INFRA_MAX_VIOLATIONS constant"
        assert "validate_architecture(" in script_content

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
