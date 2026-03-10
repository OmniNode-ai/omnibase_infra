# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""
Unit tests for ONEX Infrastructure Contract Linter.

Tests the contract_linter module for:
- Required field validation
- Type consistency checks
- YAML syntax validation
- Node type validation
- Contract version format
- Input/output model reference validation
- Line-number error reporting (OMN-517)
- "Did you mean?" suggestions for unrecognized fields (OMN-517)
- Contract dependency validation (OMN-517)
- Contract version compatibility validation (OMN-517)
"""

from pathlib import Path

import pytest

from omnibase_infra.validation.linter_contract import (
    KNOWN_CONTRACT_FIELDS,
    ContractLinter,
    ContractRuleId,
    EnumContractViolationSeverity,
    ModelContractLintResult,
    ModelContractViolation,
    _get_yaml_line_numbers,
    _suggest_similar_field,
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
            message="Invalid node_type 'INVALID'. Must be one of: EFFECT_GENERIC, COMPUTE_GENERIC, REDUCER_GENERIC, ORCHESTRATOR_GENERIC",
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
            suggestion="Use EFFECT_GENERIC, COMPUTE_GENERIC, REDUCER_GENERIC, or ORCHESTRATOR_GENERIC",
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

    def test_convert_preserves_line_number(self) -> None:
        """Test that line numbers are passed through to handler errors (OMN-517)."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="node_type",
            message="Invalid node_type",
            severity=EnumContractViolationSeverity.ERROR,
            line_number=23,
        )

        error = convert_violation_to_handler_error(violation)
        assert error.line_number == 23


class TestLineNumberTracking:
    """Tests for line-number error reporting (OMN-517)."""

    def test_yaml_line_numbers_extracted(self, tmp_path: Path) -> None:
        """Test that line numbers are extracted from YAML files."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
        )

        line_map = _get_yaml_line_numbers(contract_file)

        assert line_map["name"] == 1
        assert line_map["node_type"] == 2
        assert line_map["contract_version"] == 3
        assert line_map["contract_version.major"] == 4
        assert line_map["contract_version.minor"] == 5
        assert line_map["contract_version.patch"] == 6

    def test_yaml_line_numbers_with_comments(self, tmp_path: Path) -> None:
        """Test that comments are skipped in line number extraction."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "# This is a comment\n"
            "name: test_node\n"
            "# Another comment\n"
            "node_type: EFFECT_GENERIC\n"
        )

        line_map = _get_yaml_line_numbers(contract_file)

        assert line_map["name"] == 2
        assert line_map["node_type"] == 4

    def test_violation_includes_line_number(self, tmp_path: Path) -> None:
        """Test that violations include line numbers when available."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: INVALID_TYPE\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(check_imports=False, check_unknown_fields=False)
        result = linter.lint_file(contract_file)

        # Find the node_type violation
        node_type_violations = [
            v for v in result.violations if v.field_path == "node_type"
        ]
        assert len(node_type_violations) == 1
        assert node_type_violations[0].line_number == 2

    def test_violation_str_includes_line_number(self) -> None:
        """Test that violation string includes line number when present."""
        violation = ModelContractViolation(
            file_path="/path/to/contract.yaml",
            field_path="node_type",
            message="Invalid node_type",
            severity=EnumContractViolationSeverity.ERROR,
            line_number=23,
        )
        result = str(violation)
        assert "/path/to/contract.yaml:23:node_type" in result

    def test_violation_str_without_line_number(self) -> None:
        """Test that violation string omits line when not available."""
        violation = ModelContractViolation(
            file_path="/path/to/contract.yaml",
            field_path="node_type",
            message="Invalid node_type",
            severity=EnumContractViolationSeverity.ERROR,
        )
        result = str(violation)
        assert "/path/to/contract.yaml:node_type" in result

    def test_yaml_parse_error_includes_line_number(self, tmp_path: Path) -> None:
        """Test that YAML parse errors include the error line number."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\nnode_type: EFFECT_GENERIC\nbad_yaml: [invalid: syntax\n"
        )

        linter = ContractLinter(check_imports=False)
        result = linter.lint_file(contract_file)

        assert not result.is_valid
        yaml_errors = [v for v in result.violations if "yaml" in v.message.lower()]
        assert len(yaml_errors) == 1
        # YAML parser should provide a line number
        assert yaml_errors[0].line_number is not None


class TestDidYouMeanSuggestions:
    """Tests for 'Did you mean?' suggestions for unrecognized fields (OMN-517)."""

    def test_suggest_similar_field_typo(self) -> None:
        """Test that close typos get suggestions."""
        assert _suggest_similar_field("node_typ", KNOWN_CONTRACT_FIELDS) == "node_type"
        assert (
            _suggest_similar_field("input_modle", KNOWN_CONTRACT_FIELDS)
            == "input_model"
        )
        assert (
            _suggest_similar_field("descrption", KNOWN_CONTRACT_FIELDS) == "description"
        )

    def test_suggest_no_match_for_gibberish(self) -> None:
        """Test that completely unrelated strings get no suggestion."""
        assert (
            _suggest_similar_field("zzz_nonexistent_xyz", KNOWN_CONTRACT_FIELDS) is None
        )

    def test_unknown_field_warning_with_suggestion(self, tmp_path: Path) -> None:
        """Test that unknown fields produce warnings with suggestions."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "node_typ: COMPUTE_GENERIC\n"  # Typo of node_type
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=True,
        )
        result = linter.lint_file(contract_file)

        # Find the unknown field warning
        unknown_warnings = [
            v
            for v in result.violations
            if v.field_path == "node_typ"
            and v.severity == EnumContractViolationSeverity.WARNING
        ]
        assert len(unknown_warnings) == 1
        assert "Did you mean" in unknown_warnings[0].message
        assert "node_type" in unknown_warnings[0].message

    def test_unknown_field_with_line_number(self, tmp_path: Path) -> None:
        """Test that unknown field warnings include line numbers."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "descrption: A typo\n"  # Line 3
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(check_imports=False, check_unknown_fields=True)
        result = linter.lint_file(contract_file)

        unknown_warnings = [
            v for v in result.violations if v.field_path == "descrption"
        ]
        assert len(unknown_warnings) == 1
        assert unknown_warnings[0].line_number == 3
        assert "description" in unknown_warnings[0].suggestion

    def test_no_warnings_for_known_fields(self, tmp_path: Path) -> None:
        """Test that known fields do not produce unknown-field warnings."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: A valid description\n"
            "node_version: '1.0.0'\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
            "dependencies: []\n"
            "metadata:\n"
            "  author: test\n"
        )

        linter = ContractLinter(check_imports=False, check_unknown_fields=True)
        result = linter.lint_file(contract_file)

        unknown_warnings = [
            v for v in result.violations if "unknown field" in v.message.lower()
        ]
        assert len(unknown_warnings) == 0

    def test_unknown_field_rule_id_mapping(self) -> None:
        """Test that unknown field violations map to CONTRACT-013."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="node_typ",
            message="Unknown field 'node_typ'. Did you mean 'node_type'?",
            severity=EnumContractViolationSeverity.WARNING,
        )

        error = convert_violation_to_handler_error(violation)
        assert error.rule_id == ContractRuleId.UNKNOWN_FIELD

    def test_check_unknown_fields_disabled(self, tmp_path: Path) -> None:
        """Test that unknown field checking can be disabled."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "totally_unknown_field: value\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(check_imports=False, check_unknown_fields=False)
        result = linter.lint_file(contract_file)

        unknown_warnings = [
            v for v in result.violations if "unknown field" in v.message.lower()
        ]
        assert len(unknown_warnings) == 0


class TestDependencyValidation:
    """Tests for contract dependency validation (OMN-517)."""

    def test_valid_dependencies(self, tmp_path: Path) -> None:
        """Test that valid dependencies pass validation."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: ORCHESTRATOR_GENERIC\n"
            "description: test\n"
            "node_version: '1.0.0'\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
            "dependencies:\n"
            "  - name: reducer_protocol\n"
            "    type: protocol\n"
            "    description: Protocol for reducer\n"
            "  - name: env_var\n"
            "    type: environment\n"
            "    description: Environment variable\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_version_compatibility=False,
        )
        result = linter.lint_file(contract_file)

        dep_errors = [
            v
            for v in result.violations
            if "dependencies" in v.field_path
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(dep_errors) == 0

    def test_dependencies_not_a_list(self, tmp_path: Path) -> None:
        """Test that non-list dependencies are flagged."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
            "dependencies: not_a_list\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_version_compatibility=False,
        )
        result = linter.lint_file(contract_file)

        dep_errors = [
            v
            for v in result.violations
            if v.field_path == "dependencies"
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(dep_errors) == 1
        assert "must be a list" in dep_errors[0].message

    def test_dependency_missing_name(self, tmp_path: Path) -> None:
        """Test that dependencies without name are flagged."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
            "dependencies:\n"
            "  - type: protocol\n"
            "    description: Missing name\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_version_compatibility=False,
        )
        result = linter.lint_file(contract_file)

        name_errors = [
            v
            for v in result.violations
            if "name" in v.field_path
            and "dependencies" in v.field_path
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(name_errors) == 1

    def test_dependency_invalid_type(self, tmp_path: Path) -> None:
        """Test that dependencies with invalid type are flagged."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
            "dependencies:\n"
            "  - name: bad_dep\n"
            "    type: invalid_type\n"
            "    description: Bad type\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_version_compatibility=False,
        )
        result = linter.lint_file(contract_file)

        type_errors = [
            v
            for v in result.violations
            if "type" in v.field_path
            and "dependencies" in v.field_path
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(type_errors) == 1
        assert "invalid_type" in type_errors[0].message

    def test_dependency_missing_description_info(self, tmp_path: Path) -> None:
        """Test that dependencies without description get INFO violation."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
            "dependencies:\n"
            "  - name: my_dep\n"
            "    type: protocol\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_version_compatibility=False,
        )
        result = linter.lint_file(contract_file)

        desc_infos = [
            v
            for v in result.violations
            if "description" in v.field_path
            and "dependencies" in v.field_path
            and v.severity == EnumContractViolationSeverity.INFO
        ]
        assert len(desc_infos) == 1

    def test_dependency_rule_id_mapping(self) -> None:
        """Test that dependency violations map to CONTRACT-014."""
        violation = ModelContractViolation(
            file_path="nodes/test/contract.yaml",
            field_path="dependencies[0].type",
            message="Invalid dependency type",
            severity=EnumContractViolationSeverity.ERROR,
        )

        error = convert_violation_to_handler_error(violation)
        assert error.rule_id == ContractRuleId.INVALID_DEPENDENCY


class TestVersionCompatibilityValidation:
    """Tests for contract version compatibility validation (OMN-517)."""

    def test_valid_node_version(self, tmp_path: Path) -> None:
        """Test that valid semver node_version passes."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version: '1.0.0'\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_errors = [
            v
            for v in result.violations
            if "node_version" in v.field_path
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(version_errors) == 0

    def test_invalid_node_version_format(self, tmp_path: Path) -> None:
        """Test that invalid node_version format is flagged."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version: 'not-a-version'\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_warnings = [
            v
            for v in result.violations
            if v.field_path == "node_version"
            and v.severity == EnumContractViolationSeverity.WARNING
        ]
        assert len(version_warnings) == 1
        assert "not a valid semantic version" in version_warnings[0].message

    def test_node_version_non_string_non_dict(self, tmp_path: Path) -> None:
        """Test that non-string, non-dict node_version is flagged."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version: 123\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_errors = [
            v
            for v in result.violations
            if v.field_path == "node_version"
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(version_errors) == 1
        assert "must be a string or dict" in version_errors[0].message

    def test_node_version_dict_format(self, tmp_path: Path) -> None:
        """Test that dict-style node_version is accepted."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_errors = [
            v
            for v in result.violations
            if "node_version" in v.field_path
            and v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(version_errors) == 0

    def test_semver_with_prerelease(self, tmp_path: Path) -> None:
        """Test that semver with prerelease tag is accepted."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version: '0.1.0-beta'\n"
            "contract_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_warnings = [
            v
            for v in result.violations
            if v.field_path == "node_version" and "not a valid" in v.message
        ]
        assert len(version_warnings) == 0

    def test_version_mismatch_info_string_format(self, tmp_path: Path) -> None:
        """Test version mismatch with string node_version format."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version: '1.0.0'\n"
            "contract_version:\n"
            "  major: 0\n"
            "  minor: 1\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_infos = [
            v
            for v in result.violations
            if v.field_path == "contract_version"
            and v.severity == EnumContractViolationSeverity.INFO
            and "compatibility" in v.message.lower()
        ]
        assert len(version_infos) == 1
        assert "bump" in version_infos[0].suggestion.lower()

    def test_version_mismatch_info_dict_format(self, tmp_path: Path) -> None:
        """Test version mismatch with dict node_version format."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_node\n"
            "node_type: EFFECT_GENERIC\n"
            "description: test\n"
            "node_version:\n"
            "  major: 1\n"
            "  minor: 0\n"
            "  patch: 0\n"
            "contract_version:\n"
            "  major: 0\n"
            "  minor: 1\n"
            "  patch: 0\n"
            "input_model:\n"
            "  name: ModelInput\n"
            "  module: some.module\n"
            "output_model:\n"
            "  name: ModelOutput\n"
            "  module: some.module\n"
        )

        linter = ContractLinter(
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=False,
        )
        result = linter.lint_file(contract_file)

        version_infos = [
            v
            for v in result.violations
            if v.field_path == "contract_version"
            and v.severity == EnumContractViolationSeverity.INFO
            and "compatibility" in v.message.lower()
        ]
        assert len(version_infos) == 1
        assert "bump" in version_infos[0].suggestion.lower()


class TestRealContractWithNewValidations:
    """Tests that existing real contracts pass with new validation features (OMN-517)."""

    def test_lint_real_nodes_with_all_validations(self) -> None:
        """Test that all real contracts pass with dependency and version validation."""
        nodes_dir = Path("src/omnibase_infra/nodes")

        if not nodes_dir.exists():
            pytest.skip("Nodes directory not found")

        # Run with all new validation features enabled but without import checking
        # and unknown field checking (real contracts may have domain-specific fields)
        result = lint_contracts_in_directory(
            nodes_dir,
            check_imports=False,
            check_unknown_fields=False,
            check_dependencies=True,
            check_version_compatibility=True,
        )

        # All contracts should be valid (no ERROR-level violations)
        error_violations = [
            v
            for v in result.violations
            if v.severity == EnumContractViolationSeverity.ERROR
        ]
        assert len(error_violations) == 0, (
            f"Real contracts have errors with new validations: "
            f"{[str(v) for v in error_violations]}"
        )
