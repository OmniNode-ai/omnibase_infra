# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for ONEX Infrastructure Contract Linter.

Tests the contract_linter module for:
- Required field validation
- Type consistency checks
- YAML syntax validation
- Node type validation
- Contract version format
- Input/output model reference validation
"""

from pathlib import Path

import pytest

from omnibase_infra.validation.contract_linter import (
    ContractLinter,
    ContractRuleId,
    EnumContractViolationSeverity,
    ModelContractLintResult,
    ModelContractViolation,
    convert_violation_to_handler_error,
    lint_contract_file,
    lint_contracts_in_directory,
)


class TestModelContractViolation:
    """Tests for ModelContractViolation model."""

    def test_violation_str_format(self) -> None:
        """Test violation string formatting."""
        violation = ModelContractViolation(
            file_path="/path/to/contract.yaml",
            field_path="input_model.name",
            message="Missing required field",
            severity=EnumContractViolationSeverity.ERROR,
        )
        result = str(violation)
        assert "[ERROR]" in result
        assert "/path/to/contract.yaml:input_model.name" in result
        assert "Missing required field" in result

    def test_violation_with_suggestion(self) -> None:
        """Test violation string includes suggestion when provided."""
        violation = ModelContractViolation(
            file_path="/path/to/contract.yaml",
            field_path="name",
            message="Invalid format",
            severity=EnumContractViolationSeverity.WARNING,
            suggestion="Use snake_case",
        )
        result = str(violation)
        assert "[WARNING]" in result
        assert "(suggestion: Use snake_case)" in result


class TestModelContractLintResult:
    """Tests for ModelContractLintResult model."""

    def test_empty_result_is_valid(self) -> None:
        """Test empty result with no violations is valid."""
        result = ModelContractLintResult(
            is_valid=True,
            violations=[],
            files_checked=1,
            files_valid=1,
        )
        assert result.is_valid
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_error_count_calculation(self) -> None:
        """Test error count is calculated from violations."""
        result = ModelContractLintResult(
            is_valid=False,
            violations=[
                ModelContractViolation(
                    file_path="test.yaml",
                    field_path="field1",
                    message="Error 1",
                    severity=EnumContractViolationSeverity.ERROR,
                ),
                ModelContractViolation(
                    file_path="test.yaml",
                    field_path="field2",
                    message="Warning 1",
                    severity=EnumContractViolationSeverity.WARNING,
                ),
                ModelContractViolation(
                    file_path="test.yaml",
                    field_path="field3",
                    message="Error 2",
                    severity=EnumContractViolationSeverity.ERROR,
                ),
            ],
            files_checked=1,
            files_with_errors=1,
        )
        assert result.error_count == 2
        assert result.warning_count == 1

    def test_result_str_format(self) -> None:
        """Test result summary string format."""
        result = ModelContractLintResult(
            is_valid=True,
            violations=[],
            files_checked=3,
            files_valid=3,
        )
        summary = str(result)
        assert "PASS" in summary
        assert "3 files" in summary


class TestContractLinter:
    """Tests for ContractLinter class."""

    def test_lint_missing_file(self) -> None:
        """Test linting a file that doesn't exist."""
        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(Path("/nonexistent/contract.yaml"))

        assert not result.is_valid
        assert result.error_count == 1
        assert "not found" in result.violations[0].message.lower()

    def test_lint_invalid_yaml(self, tmp_path: Path) -> None:
        """Test linting a file with invalid YAML syntax."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("invalid: yaml: syntax: here:")

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        assert result.error_count >= 1
        assert any("yaml" in v.message.lower() for v in result.violations)

    def test_lint_non_dict_yaml(self, tmp_path: Path) -> None:
        """Test linting a YAML file that's not a dict."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("- just\n- a\n- list")

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        assert any("mapping" in v.message.lower() for v in result.violations)

    def test_lint_missing_required_fields(self, tmp_path: Path) -> None:
        """Test linting detects missing required fields."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("description: Just a description")

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        # Should report missing name, node_type, contract_version, input_model, output_model
        missing_fields = {
            "name",
            "node_type",
            "contract_version",
            "input_model",
            "output_model",
        }
        found_missing = set()
        for v in result.violations:
            if v.severity == EnumContractViolationSeverity.ERROR:
                for field in missing_fields:
                    if field in v.field_path:
                        found_missing.add(field)
        assert found_missing == missing_fields

    def test_lint_invalid_node_type(self, tmp_path: Path) -> None:
        """Test linting detects invalid node_type."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: INVALID_TYPE
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        invalid_type_errors = [
            v
            for v in result.violations
            if v.field_path == "node_type"
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(invalid_type_errors) == 1
        assert (
            "EFFECT_GENERIC" in invalid_type_errors[0].message
        )  # Should suggest valid types

    def test_lint_valid_node_types(self, tmp_path: Path) -> None:
        """Test all valid node types are accepted."""
        valid_types = [
            "EFFECT_GENERIC",
            "COMPUTE_GENERIC",
            "REDUCER_GENERIC",
            "ORCHESTRATOR_GENERIC",
        ]

        for node_type in valid_types:
            contract_file = tmp_path / f"contract_{node_type}.yaml"
            contract_file.write_text(
                f"""
name: test_node
node_type: {node_type}
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
            )

            linter = ContractLinter(check_imports=False)
            result = linter.lint_file(contract_file)

            # Should not have node_type errors
            node_type_errors = [
                v
                for v in result.violations
                if v.field_path == "node_type"
                and v.severity == EnumContractViolationSeverity.ERROR
            ]
            assert len(node_type_errors) == 0, (
                f"Unexpected error for valid node_type: {node_type}"
            )

    def test_lint_invalid_contract_version_format(self, tmp_path: Path) -> None:
        """Test linting detects invalid contract_version format."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version: "1.0.0"
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        version_errors = [
            v
            for v in result.violations
            if "contract_version" in v.field_path
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(version_errors) >= 1
        assert "dict" in version_errors[0].message.lower()

    def test_lint_missing_version_components(self, tmp_path: Path) -> None:
        """Test linting detects missing version components."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version:
  major: 1
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        missing = {"minor", "patch"}
        found = set()
        for v in result.violations:
            if v.severity == EnumContractViolationSeverity.ERROR:
                for key in missing:
                    if key in v.field_path:
                        found.add(key)
        assert found == missing

    def test_lint_invalid_model_reference_format(self, tmp_path: Path) -> None:
        """Test linting detects invalid input_model/output_model format."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model: "just a string"
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        model_errors = [
            v
            for v in result.violations
            if v.field_path == "input_model"
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(model_errors) >= 1

    def test_lint_missing_model_fields(self, tmp_path: Path) -> None:
        """Test linting detects missing name/module in model references."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
output_model:
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        # Should have errors for input_model.module and output_model.name
        errors = [
            v
            for v in result.violations
            if v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert any("input_model.module" in v.field_path for v in errors)
        assert any("output_model.name" in v.field_path for v in errors)

    def test_lint_non_model_prefix_warning(self, tmp_path: Path) -> None:
        """Test linting warns about model names not starting with 'Model'."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: InputData
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        # Should have a warning about InputData not starting with Model
        warnings = [
            v
            for v in result.violations
            if v.field_path == "input_model.name"
            and v.severity == EnumContractViolationSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert "Model" in warnings[0].message

    def test_lint_snake_case_warning(self, tmp_path: Path) -> None:
        """Test linting warns about non-snake_case names."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: TestNode
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        # Should have a warning about non-snake_case name
        warnings = [
            v
            for v in result.violations
            if v.field_path == "name"
            and v.severity == EnumContractViolationSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert "snake_case" in warnings[0].message.lower()

    def test_lint_recommended_fields_info(self, tmp_path: Path) -> None:
        """Test linting reports info about missing recommended fields."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        # Should have INFO about missing description and node_version
        infos = [
            v
            for v in result.violations
            if v.severity == EnumContractViolationSeverity.INFO
        ]
        info_fields = {v.field_path for v in infos}
        assert "description" in info_fields
        assert "node_version" in info_fields

    def test_lint_strict_mode(self, tmp_path: Path) -> None:
        """Test strict mode treats warnings as errors."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: TestNode
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        # Normal mode: valid despite warnings
        linter = ContractLinter(check_imports=False, strict_mode=False)
        normal_result = linter.lint_file(contract_file)
        assert normal_result.is_valid  # Only errors block

        # Strict mode: warnings become blocking
        strict_linter = ContractLinter(check_imports=False, strict_mode=True)
        strict_result = strict_linter.lint_file(contract_file)
        assert not strict_result.is_valid  # Warnings block in strict

    def test_lint_valid_contract(self, tmp_path: Path) -> None:
        """Test linting a fully valid contract passes."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: ORCHESTRATOR_GENERIC
description: A test node for validation
node_version: "1.0.0"
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert result.is_valid
        assert result.error_count == 0
        assert result.warning_count == 0


class TestContractLinterDirectory:
    """Tests for directory linting."""

    def test_lint_empty_directory(self, tmp_path: Path) -> None:
        """Test linting an empty directory returns valid with no files."""
        linter = ContractLinter(check_imports=False)
        result = linter.lint_directory(tmp_path)

        assert result.is_valid
        assert result.files_checked == 0

    def test_lint_nonexistent_directory(self) -> None:
        """Test linting a nonexistent directory."""
        linter = ContractLinter(check_imports=False)
        result = linter.lint_directory(Path("/nonexistent/directory"))

        assert not result.is_valid
        assert any("not found" in v.message.lower() for v in result.violations)

    def test_lint_directory_with_contracts(self, tmp_path: Path) -> None:
        """Test linting a directory with multiple contracts."""
        # Create subdirectories with contracts
        node1_dir = tmp_path / "node1"
        node1_dir.mkdir()
        (node1_dir / "contract.yaml").write_text(
            """
name: node_one
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        node2_dir = tmp_path / "node2"
        node2_dir.mkdir()
        (node2_dir / "contract.yaml").write_text(
            """
name: node_two
node_type: COMPUTE_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_directory(tmp_path, recursive=True)

        assert result.files_checked == 2
        assert result.error_count == 0

    def test_lint_directory_aggregates_errors(self, tmp_path: Path) -> None:
        """Test linting aggregates errors from multiple contracts."""
        # Create a valid contract
        node1_dir = tmp_path / "node1"
        node1_dir.mkdir()
        (node1_dir / "contract.yaml").write_text(
            """
name: node_one
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        # Create an invalid contract
        node2_dir = tmp_path / "node2"
        node2_dir.mkdir()
        (node2_dir / "contract.yaml").write_text(
            """
name: node_two
node_type: INVALID
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_directory(tmp_path, recursive=True)

        assert result.files_checked == 2
        assert result.files_valid == 1
        assert result.files_with_errors == 1
        assert not result.is_valid
        assert result.error_count > 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_lint_contract_file(self, tmp_path: Path) -> None:
        """Test lint_contract_file convenience function."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        result = lint_contract_file(contract_file, check_imports=False)
        assert result.is_valid

    def test_lint_contracts_in_directory(self, tmp_path: Path) -> None:
        """Test lint_contracts_in_directory convenience function."""
        (tmp_path / "contract.yaml").write_text(
            """
name: test_node
node_type: REDUCER_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
input_model:
  name: ModelInput
  module: some.module
output_model:
  name: ModelOutput
  module: some.module
"""
        )

        result = lint_contracts_in_directory(tmp_path, check_imports=False)
        assert result.files_checked == 1
        assert result.is_valid


class TestRealContract:
    """Tests against the real contract in the repository."""

    def test_lint_real_contract(self) -> None:
        """Test linting the actual node_registration_orchestrator contract."""
        contract_path = Path(
            "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml"
        )

        if not contract_path.exists():
            pytest.skip("Contract file not found in expected location")

        # Lint without import checking since test may not have all deps
        result = lint_contract_file(contract_path, check_imports=False)

        # The real contract should be valid
        assert result.is_valid, (
            f"Real contract has errors: {[str(v) for v in result.violations if v.severity == EnumContractViolationSeverity.ERROR]}"
        )
        assert result.error_count == 0

    def test_lint_real_nodes_directory(self) -> None:
        """Test linting all contracts in the nodes directory."""
        nodes_dir = Path("src/omnibase_infra/nodes")

        if not nodes_dir.exists():
            pytest.skip("Nodes directory not found")

        result = lint_contracts_in_directory(nodes_dir, check_imports=False)

        # All contracts should be valid
        assert result.is_valid, (
            f"Contracts have errors: {[str(v) for v in result.violations if v.severity == EnumContractViolationSeverity.ERROR]}"
        )


class TestStructuredErrorConversion:
    """Tests for structured error conversion (OMN-1091)."""

    def test_convert_yaml_parse_error(self) -> None:
        """Test converting YAML parse error to handler validation error."""
        violation = ModelContractViolation(
            file_path="nodes/registration/contract.yaml",
            field_path="",
            message="YAML parse error: invalid syntax",
            severity=EnumContractViolationSeverity.ERROR,
            suggestion="Check YAML indentation and syntax",
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.YAML_PARSE_ERROR
        assert error.handler_identity.handler_id == "registration"
        assert error.file_path == "nodes/registration/contract.yaml"
        assert error.remediation_hint == "Check YAML indentation and syntax"
        assert error.severity == "error"
        assert error.is_blocking()

    def test_convert_missing_required_field(self) -> None:
        """Test converting missing required field error."""
        violation = ModelContractViolation(
            file_path="nodes/compute/contract.yaml",
            field_path="node_type",
            message="Required field 'node_type' is missing",
            severity=EnumContractViolationSeverity.ERROR,
            suggestion="Add 'node_type:' to your contract.yaml",
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.MISSING_REQUIRED_FIELD
        assert error.handler_identity.handler_id == "compute"
        assert "node_type" in error.message

    def test_convert_invalid_node_type(self) -> None:
        """Test converting invalid node_type error."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="node_type",
            message="Invalid node_type 'INVALID'. Must be one of: EFFECT, COMPUTE",
            severity=EnumContractViolationSeverity.ERROR,
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.INVALID_NODE_TYPE
        assert "INVALID" in error.message

    def test_convert_import_error(self) -> None:
        """Test converting import error."""
        violation = ModelContractViolation(
            file_path="nodes/effect/contract.yaml",
            field_path="input_model.module",
            message="Cannot import module 'nonexistent.module': No module named 'nonexistent'",
            severity=EnumContractViolationSeverity.ERROR,
            suggestion="Verify module path and ensure it's installed",
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.IMPORT_ERROR
        assert "Cannot import" in error.message

    def test_convert_model_not_found(self) -> None:
        """Test converting model not found error."""
        violation = ModelContractViolation(
            file_path="nodes/reducer/contract.yaml",
            field_path="output_model.name",
            message="Class 'ModelMissing' not found in module 'some.module'",
            severity=EnumContractViolationSeverity.ERROR,
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.MODEL_NOT_FOUND
        assert "not found" in error.message

    def test_convert_warning_to_warning_severity(self) -> None:
        """Test converting warning severity violation."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="name",
            message="Node name 'TestNode' should be snake_case",
            severity=EnumContractViolationSeverity.WARNING,
            suggestion="Use snake_case: e.g., 'test_node'",
        )

        error = convert_violation_to_handler_error(violation)

        assert error.severity == "warning"
        assert not error.is_blocking()

    def test_convert_file_not_found(self) -> None:
        """Test converting file not found error."""
        violation = ModelContractViolation(
            file_path="/nonexistent/contract.yaml",
            field_path="",
            message="Contract file not found: /nonexistent/contract.yaml",
            severity=EnumContractViolationSeverity.ERROR,
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.FILE_NOT_FOUND
        assert "not found" in error.message.lower()

    def test_convert_encoding_error(self) -> None:
        """Test converting encoding error."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="",
            message="Contract file contains binary or non-UTF-8 content",
            severity=EnumContractViolationSeverity.ERROR,
        )

        error = convert_violation_to_handler_error(violation)

        assert error.rule_id == ContractRuleId.ENCODING_ERROR
        assert "encoding" in error.message.lower() or "binary" in error.message.lower()

    def test_result_to_handler_errors(self, tmp_path: Path) -> None:
        """Test ModelContractLintResult.to_handler_errors() method."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            """
name: test_node
node_type: INVALID_TYPE
"""
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        # Convert to handler errors
        handler_errors = result.to_handler_errors()

        assert len(handler_errors) > 0
        assert all(hasattr(error, "rule_id") for error in handler_errors)
        assert all(hasattr(error, "handler_identity") for error in handler_errors)
        assert all(hasattr(error, "remediation_hint") for error in handler_errors)

        # Verify at least one error has CONTRACT-003 (invalid node_type)
        rule_ids = {error.rule_id for error in handler_errors}
        assert ContractRuleId.INVALID_NODE_TYPE in rule_ids

    def test_handler_error_format_for_ci(self) -> None:
        """Test handler error CI formatting."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="node_type",
            message="Invalid node_type",
            severity=EnumContractViolationSeverity.ERROR,
            suggestion="Use EFFECT, COMPUTE, REDUCER, or ORCHESTRATOR",
        )

        error = convert_violation_to_handler_error(violation)
        ci_output = error.format_for_ci()

        # Should be GitHub Actions format
        assert ci_output.startswith("::error")
        assert "file=nodes/test/contract.yaml" in ci_output
        assert ContractRuleId.INVALID_NODE_TYPE in ci_output
        assert "Remediation:" in ci_output

    def test_handler_error_format_for_logging(self) -> None:
        """Test handler error logging formatting."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="input_model",
            message="Invalid model reference",
            severity=EnumContractViolationSeverity.ERROR,
            suggestion="Add 'name' and 'module' fields",
        )

        error = convert_violation_to_handler_error(violation)
        log_output = error.format_for_logging()

        # Should contain structured information
        assert "Handler Validation Error" in log_output
        assert ContractRuleId.INVALID_MODEL_REFERENCE in log_output
        assert "Type:" in log_output
        assert "Handler:" in log_output
        assert "Message:" in log_output
        assert "Remediation:" in log_output

    def test_default_remediation_hint(self) -> None:
        """Test default remediation hint when violation has no suggestion."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="node_type",
            message="Invalid node_type",
            severity=EnumContractViolationSeverity.ERROR,
            # No suggestion provided
        )

        error = convert_violation_to_handler_error(violation)

        # Should have default remediation hint
        assert (
            error.remediation_hint
            == "Review contract.yaml and fix the validation error"
        )
