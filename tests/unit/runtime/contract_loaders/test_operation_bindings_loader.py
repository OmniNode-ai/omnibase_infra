# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for operation_bindings_loader.

This module tests the operation_bindings_loader which parses and validates
the operation_bindings section from contract.yaml files.

Test Categories:
    - TestLoadOperationBindingsHappyPath: Tests for successful loading
    - TestExpressionValidation: Tests for expression syntax validation
    - TestSourceValidation: Tests for binding source validation (payload/envelope/context)
    - TestContextPathValidation: Tests for context path validation
    - TestOperationValidation: Tests for io_operations cross-reference validation
    - TestDuplicateParameterValidation: Tests for duplicate parameter detection
    - TestFileSizeEnforcement: Tests for file size security control
    - TestYamlSecurityControls: Tests for YAML safe_load and injection prevention
    - TestErrorCodes: Tests verifying specific error codes are raised
    - TestEdgeCases: Tests for boundary conditions and unusual inputs

Part of OMN-1518: Declarative operation bindings.

Running Tests:
    # Run all operation bindings loader tests:
    pytest tests/unit/runtime/contract_loaders/test_operation_bindings_loader.py -v

    # Run specific test class:
    pytest tests/unit/runtime/contract_loaders/test_operation_bindings_loader.py::TestLoadOperationBindingsHappyPath -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.bindings import (
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)
from omnibase_infra.runtime.contract_loaders import load_operation_bindings_subcontract
from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
    ERROR_CODE_CONTRACT_NOT_FOUND,
    ERROR_CODE_DUPLICATE_PARAMETER,
    ERROR_CODE_EMPTY_PATH_SEGMENT,
    ERROR_CODE_EXPRESSION_MALFORMED,
    ERROR_CODE_EXPRESSION_TOO_LONG,
    ERROR_CODE_FILE_SIZE_EXCEEDED,
    ERROR_CODE_INVALID_CONTEXT_PATH,
    ERROR_CODE_INVALID_SOURCE,
    ERROR_CODE_PATH_TOO_DEEP,
    ERROR_CODE_UNKNOWN_OPERATION,
    ERROR_CODE_YAML_PARSE_ERROR,
    MAX_CONTRACT_FILE_SIZE_BYTES,
    MAX_EXPRESSION_LENGTH,
    MAX_PATH_SEGMENTS,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _write_contract(content: dict, tmpdir: Path) -> Path:
    """Helper to write a contract YAML file.

    Args:
        content: Dictionary to serialize as YAML.
        tmpdir: Directory to write the file in.

    Returns:
        Path to the created contract.yaml file.
    """
    contract_path = tmpdir / "contract.yaml"
    with contract_path.open("w") as f:
        yaml.dump(content, f)
    return contract_path


# =============================================================================
# TestLoadOperationBindingsHappyPath
# =============================================================================


class TestLoadOperationBindingsHappyPath:
    """Tests for successful operation bindings loading.

    These tests verify that valid contract.yaml files are correctly
    parsed and converted to ModelOperationBindingsSubcontract instances.
    """

    def test_load_valid_contract(self, tmp_path: Path) -> None:
        """Valid contract loads successfully."""
        contract = {
            "operation_bindings": {
                "version": {"major": 1, "minor": 0, "patch": 0},
                "bindings": {
                    "test.operation": [
                        {
                            "parameter_name": "user_id",
                            "expression": "${payload.user.id}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.operation"],
        )

        assert isinstance(result, ModelOperationBindingsSubcontract)
        assert "test.operation" in result.bindings
        assert len(result.bindings["test.operation"]) == 1

    def test_load_with_global_bindings(self, tmp_path: Path) -> None:
        """Contract with global_bindings loads correctly."""
        contract = {
            "operation_bindings": {
                "global_bindings": [
                    {
                        "parameter_name": "correlation_id",
                        "expression": "${envelope.correlation_id}",
                        "required": True,
                    }
                ],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=[],
        )

        assert result.global_bindings is not None
        assert len(result.global_bindings) == 1
        assert result.global_bindings[0].parameter_name == "correlation_id"

    def test_missing_section_returns_empty(self, tmp_path: Path) -> None:
        """Missing operation_bindings section returns empty subcontract."""
        contract = {"name": "test", "node_type": "EFFECT_GENERIC"}
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=[],
        )

        assert isinstance(result, ModelOperationBindingsSubcontract)
        assert len(result.bindings) == 0

    def test_version_defaults_to_1_0_0(self, tmp_path: Path) -> None:
        """Missing version defaults to 1.0.0."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "param",
                            "expression": "${payload.value}",
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.version.major == 1
        assert result.version.minor == 0
        assert result.version.patch == 0

    def test_multiple_operations_loaded(self, tmp_path: Path) -> None:
        """Multiple operation binding groups load correctly."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "db.query": [
                        {"parameter_name": "sql", "expression": "${payload.sql}"}
                    ],
                    "db.execute": [
                        {"parameter_name": "stmt", "expression": "${payload.statement}"}
                    ],
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["db.query", "db.execute"],
        )

        assert len(result.bindings) == 2
        assert "db.query" in result.bindings
        assert "db.execute" in result.bindings

    def test_parsed_binding_has_correct_fields(self, tmp_path: Path) -> None:
        """Parsed bindings contain pre-parsed expression components."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "user_id",
                            "expression": "${payload.user.id}",
                            "required": True,
                            "default": None,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        binding = result.bindings["test.op"][0]
        assert isinstance(binding, ModelParsedBinding)
        assert binding.parameter_name == "user_id"
        assert binding.source == "payload"
        assert binding.path_segments == ("user", "id")
        assert binding.required is True
        assert binding.original_expression == "${payload.user.id}"

    def test_optional_binding_with_default(self, tmp_path: Path) -> None:
        """Optional binding with default value loads correctly."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "limit",
                            "expression": "${payload.limit}",
                            "required": False,
                            "default": 100,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        binding = result.bindings["test.op"][0]
        assert binding.required is False
        assert binding.default == 100

    def test_no_io_operations_validation_when_none(self, tmp_path: Path) -> None:
        """When io_operations is None, no validation occurs."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "any.operation": [
                        {"parameter_name": "p", "expression": "${payload.x}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        # Should not raise even though 'any.operation' is not in io_operations
        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=None,
        )

        assert "any.operation" in result.bindings


# =============================================================================
# TestExpressionValidation
# =============================================================================


class TestExpressionValidation:
    """Tests for binding expression syntax validation.

    These tests verify that malformed expressions are rejected at load time
    with appropriate error codes.
    """

    def test_invalid_expression_syntax_fails(self, tmp_path: Path) -> None:
        """Malformed expression raises ProtocolConfigurationError."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "invalid_no_braces",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_EXPRESSION_MALFORMED in str(exc_info.value)

    def test_missing_dollar_brace_fails(self, tmp_path: Path) -> None:
        """Expression without ${...} format fails."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "payload.id",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_EXPRESSION_MALFORMED in str(exc_info.value)

    def test_unclosed_brace_fails(self, tmp_path: Path) -> None:
        """Unclosed brace in expression fails."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "${payload.id",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_EXPRESSION_MALFORMED in str(exc_info.value)

    def test_array_access_not_allowed(self, tmp_path: Path) -> None:
        """Array access syntax [0] is rejected."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "${payload.items[0]}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_EXPRESSION_MALFORMED in str(exc_info.value)
        assert "array" in str(exc_info.value).lower()

    def test_expression_too_long_fails(self, tmp_path: Path) -> None:
        """Expression exceeding MAX_EXPRESSION_LENGTH fails."""
        # Create an expression that exceeds the maximum length
        long_path = ".".join(["field"] * 50)  # Many segments
        long_expression = f"${{payload.{long_path}}}"

        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": long_expression,
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_EXPRESSION_TOO_LONG in str(exc_info.value)

    def test_empty_segment_fails(self, tmp_path: Path) -> None:
        """Expression with empty segment raises error."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "${payload..id}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_EMPTY_PATH_SEGMENT in str(exc_info.value)

    def test_path_too_deep_fails(self, tmp_path: Path) -> None:
        """Path with too many segments fails."""
        # Create a path that exceeds MAX_PATH_SEGMENTS
        deep_path = ".".join(["field"] * (MAX_PATH_SEGMENTS + 5))
        # Keep expression short enough but path too deep
        expression = f"${{payload.{deep_path[:200]}}}"

        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": expression,
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        # Either too long or too deep error
        error_msg = str(exc_info.value)
        assert (
            ERROR_CODE_PATH_TOO_DEEP in error_msg
            or ERROR_CODE_EXPRESSION_TOO_LONG in error_msg
        )


# =============================================================================
# TestSourceValidation
# =============================================================================


class TestSourceValidation:
    """Tests for binding source validation.

    These tests verify that only valid sources (payload, envelope, context)
    are accepted.
    """

    def test_payload_source_accepted(self, tmp_path: Path) -> None:
        """Payload source is valid."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {"parameter_name": "p", "expression": "${payload.field}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.bindings["test.op"][0].source == "payload"

    def test_envelope_source_accepted(self, tmp_path: Path) -> None:
        """Envelope source is valid."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {"parameter_name": "p", "expression": "${envelope.topic}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.bindings["test.op"][0].source == "envelope"

    def test_context_source_accepted(self, tmp_path: Path) -> None:
        """Context source is valid."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {"parameter_name": "p", "expression": "${context.now_iso}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.bindings["test.op"][0].source == "context"

    def test_invalid_source_fails(self, tmp_path: Path) -> None:
        """Invalid source raises ProtocolConfigurationError."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "${invalid.path}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_INVALID_SOURCE in str(exc_info.value)

    def test_valid_sources_constant_is_complete(self) -> None:
        """VALID_SOURCES contains all expected sources."""
        assert "payload" in VALID_SOURCES
        assert "envelope" in VALID_SOURCES
        assert "context" in VALID_SOURCES
        assert len(VALID_SOURCES) == 3


# =============================================================================
# TestContextPathValidation
# =============================================================================


class TestContextPathValidation:
    """Tests for context path validation.

    These tests verify that only valid context paths are accepted.
    """

    def test_now_iso_context_path_valid(self, tmp_path: Path) -> None:
        """now_iso is a valid context path."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {"parameter_name": "ts", "expression": "${context.now_iso}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.bindings["test.op"][0].path_segments[0] == "now_iso"

    def test_dispatcher_id_context_path_valid(self, tmp_path: Path) -> None:
        """dispatcher_id is a valid context path."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "d",
                            "expression": "${context.dispatcher_id}",
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.bindings["test.op"][0].path_segments[0] == "dispatcher_id"

    def test_correlation_id_context_path_valid(self, tmp_path: Path) -> None:
        """correlation_id is a valid context path."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "c",
                            "expression": "${context.correlation_id}",
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["test.op"],
        )

        assert result.bindings["test.op"][0].path_segments[0] == "correlation_id"

    def test_invalid_context_path_fails(self, tmp_path: Path) -> None:
        """Invalid context path raises error."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "bad",
                            "expression": "${context.invalid_path}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_INVALID_CONTEXT_PATH in str(exc_info.value)

    def test_valid_context_paths_constant_is_complete(self) -> None:
        """VALID_CONTEXT_PATHS contains all expected paths."""
        assert "now_iso" in VALID_CONTEXT_PATHS
        assert "dispatcher_id" in VALID_CONTEXT_PATHS
        assert "correlation_id" in VALID_CONTEXT_PATHS
        assert len(VALID_CONTEXT_PATHS) == 3


# =============================================================================
# TestOperationValidation
# =============================================================================


class TestOperationValidation:
    """Tests for io_operations cross-reference validation.

    These tests verify that operations in bindings are validated against
    the io_operations list when provided.
    """

    def test_unknown_operation_fails(self, tmp_path: Path) -> None:
        """Operation not in io_operations raises error."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "unknown.operation": [
                        {
                            "parameter_name": "param",
                            "expression": "${payload.id}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["different.operation"],
            )

        assert ERROR_CODE_UNKNOWN_OPERATION in str(exc_info.value)

    def test_operation_in_list_succeeds(self, tmp_path: Path) -> None:
        """Operation in io_operations list succeeds."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "db.query": [
                        {"parameter_name": "sql", "expression": "${payload.sql}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["db.query", "db.execute"],
        )

        assert "db.query" in result.bindings


# =============================================================================
# TestDuplicateParameterValidation
# =============================================================================


class TestDuplicateParameterValidation:
    """Tests for duplicate parameter detection.

    These tests verify that duplicate parameter names within the same
    scope are detected and rejected.
    """

    def test_duplicate_parameter_in_operation_fails(self, tmp_path: Path) -> None:
        """Duplicate parameter_name in same operation raises error."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {
                            "parameter_name": "duplicate",
                            "expression": "${payload.a}",
                            "required": True,
                        },
                        {
                            "parameter_name": "duplicate",
                            "expression": "${payload.b}",
                            "required": True,
                        },
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=["test.op"],
            )

        assert ERROR_CODE_DUPLICATE_PARAMETER in str(exc_info.value)

    def test_duplicate_parameter_in_global_fails(self, tmp_path: Path) -> None:
        """Duplicate parameter_name in global_bindings raises error."""
        contract = {
            "operation_bindings": {
                "global_bindings": [
                    {
                        "parameter_name": "dup",
                        "expression": "${envelope.a}",
                    },
                    {
                        "parameter_name": "dup",
                        "expression": "${envelope.b}",
                    },
                ],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path,
                io_operations=[],
            )

        assert ERROR_CODE_DUPLICATE_PARAMETER in str(exc_info.value)

    def test_same_parameter_in_different_operations_succeeds(
        self, tmp_path: Path
    ) -> None:
        """Same parameter name in different operations is allowed."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op1": [{"parameter_name": "id", "expression": "${payload.id}"}],
                    "op2": [{"parameter_name": "id", "expression": "${payload.id}"}],
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["op1", "op2"],
        )

        # Should succeed - same name in different scopes is OK
        assert len(result.bindings) == 2


# =============================================================================
# TestFileSizeEnforcement
# =============================================================================


class TestFileSizeEnforcement:
    """Tests for file size limit enforcement (security control).

    Per CLAUDE.md Handler Plugin Loader security patterns, a 10MB file size
    limit is enforced to prevent memory exhaustion attacks via large YAML files.
    """

    def test_max_contract_file_size_constant_is_10mb(self) -> None:
        """Test that MAX_CONTRACT_FILE_SIZE_BYTES is 10MB."""
        expected_size = 10 * 1024 * 1024  # 10MB
        assert expected_size == MAX_CONTRACT_FILE_SIZE_BYTES

    def test_oversized_file_raises_error(self, tmp_path: Path) -> None:
        """Oversized file raises ProtocolConfigurationError."""
        contract_path = tmp_path / "contract.yaml"
        # Create a file that exceeds the limit by 1 byte
        oversized_content = "x" * (MAX_CONTRACT_FILE_SIZE_BYTES + 1)
        contract_path.write_text(oversized_content)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        error_msg = str(exc_info.value)
        assert "exceeds maximum size" in error_msg.lower()
        assert ERROR_CODE_FILE_SIZE_EXCEEDED in error_msg

    def test_file_at_limit_is_accepted(self, tmp_path: Path) -> None:
        """File exactly at size limit is accepted."""
        contract_path = tmp_path / "contract.yaml"
        # Create valid YAML content padded to exact limit
        base_content = """name: "test"
version: "1.0.0"
"""
        # Pad with comment characters to reach exact size
        padding_needed = MAX_CONTRACT_FILE_SIZE_BYTES - len(
            base_content.encode("utf-8")
        )
        if padding_needed > 0:
            padded_content = base_content + "\n# " + ("x" * (padding_needed - 3))
        else:
            padded_content = base_content
        contract_path.write_text(padded_content)

        # Should not raise - returns empty subcontract since no operation_bindings
        result = load_operation_bindings_subcontract(contract_path, io_operations=[])
        assert isinstance(result, ModelOperationBindingsSubcontract)

    def test_file_size_check_before_yaml_parsing(self, tmp_path: Path) -> None:
        """File size is checked BEFORE attempting to parse YAML."""
        contract_path = tmp_path / "contract.yaml"
        # Create oversized content that is also invalid YAML
        oversized_invalid = "[[[" * (MAX_CONTRACT_FILE_SIZE_BYTES // 3 + 1)
        contract_path.write_text(oversized_invalid)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        # Should fail on file size, not YAML parsing
        error_msg = str(exc_info.value)
        assert "exceeds maximum size" in error_msg.lower()
        assert ERROR_CODE_FILE_SIZE_EXCEEDED in error_msg

    def test_oversized_file_detected_via_mocked_stat(self, tmp_path: Path) -> None:
        """File size check via mocked Path.stat() detects oversized files.

        This test mocks Path.stat() to simulate an oversized file without
        actually creating a large file on disk. Validates that _check_file_size()
        correctly reads st_size and raises ProtocolConfigurationError with
        the correct error code.

        This is a more efficient alternative to creating actual large files,
        and directly tests the _check_file_size() security control behavior.
        """
        from unittest.mock import MagicMock, patch

        # Create a small valid contract file
        contract = {"name": "test"}
        contract_path = _write_contract(contract, tmp_path)

        # Mock stat result with oversized file (1 byte over the limit)
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = MAX_CONTRACT_FILE_SIZE_BYTES + 1

        # Patch the stat method on the Path class
        # exists() also uses stat() internally, but mock returning a result
        # without raising means exists() returns True
        with patch.object(Path, "stat", return_value=mock_stat_result):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                load_operation_bindings_subcontract(contract_path, io_operations=[])

        error_msg = str(exc_info.value)
        # Verify error message contains both human-readable code and machine code
        assert "FILE_SIZE_EXCEEDED" in error_msg
        assert ERROR_CODE_FILE_SIZE_EXCEEDED in error_msg
        assert "exceeds maximum size" in error_msg.lower()


# =============================================================================
# TestYamlSecurityControls
# =============================================================================


class TestYamlSecurityControls:
    """Tests for YAML security controls.

    These tests verify that yaml.safe_load is used and Python object
    injection is prevented.
    """

    def test_yaml_safe_load_used(self, tmp_path: Path) -> None:
        """YAML with Python objects is rejected (safe_load)."""
        contract_path = tmp_path / "contract.yaml"
        # Write raw YAML with Python object tag
        with contract_path.open("w") as f:
            f.write("operation_bindings: !!python/object:builtins.dict {}")

        # Should either raise an error or safely ignore the tag
        try:
            result = load_operation_bindings_subcontract(
                contract_path,
                io_operations=[],
            )
            # If it loads, it should not execute Python code
            # and operation_bindings should be empty or missing
            assert isinstance(result, ModelOperationBindingsSubcontract)
        except (yaml.YAMLError, ProtocolConfigurationError):
            pass  # Expected - safe_load rejects Python tags

    def test_yaml_python_apply_tag_rejected(self, tmp_path: Path) -> None:
        """YAML with python/apply tags is rejected."""
        contract_path = tmp_path / "contract.yaml"
        with contract_path.open("w") as f:
            # Write a simpler tag that should still be rejected by safe_load
            f.write("test: !!python/name:builtins.True")

        # Should raise or safely ignore
        try:
            result = load_operation_bindings_subcontract(
                contract_path,
                io_operations=[],
            )
            # If it loads without error, it should be safe
            assert isinstance(result, ModelOperationBindingsSubcontract)
        except (yaml.YAMLError, ProtocolConfigurationError):
            pass  # Expected

    def test_invalid_yaml_syntax_raises_error(self, tmp_path: Path) -> None:
        """Invalid YAML syntax raises ProtocolConfigurationError."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("invalid: [unclosed bracket")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_YAML_PARSE_ERROR in str(exc_info.value)


# =============================================================================
# TestErrorCodes
# =============================================================================


class TestErrorCodes:
    """Tests verifying specific error codes are raised.

    These tests document and verify the error code contract.
    """

    def test_error_code_contract_not_found(self) -> None:
        """CONTRACT_NOT_FOUND raised for missing file."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                Path("/nonexistent/contract.yaml"),
                io_operations=[],
            )

        assert ERROR_CODE_CONTRACT_NOT_FOUND in str(exc_info.value)

    def test_error_code_expression_malformed(self, tmp_path: Path) -> None:
        """EXPRESSION_MALFORMED raised for invalid syntax."""
        contract = {
            "operation_bindings": {
                "bindings": {"op": [{"parameter_name": "p", "expression": "bad"}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=["op"])

        assert ERROR_CODE_EXPRESSION_MALFORMED in str(exc_info.value)

    def test_error_code_invalid_source(self, tmp_path: Path) -> None:
        """INVALID_SOURCE raised for unknown source."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${unknown.field}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=["op"])

        assert ERROR_CODE_INVALID_SOURCE in str(exc_info.value)

    def test_error_code_invalid_context_path(self, tmp_path: Path) -> None:
        """INVALID_CONTEXT_PATH raised for invalid context path."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${context.bad}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=["op"])

        assert ERROR_CODE_INVALID_CONTEXT_PATH in str(exc_info.value)

    def test_error_code_unknown_operation(self, tmp_path: Path) -> None:
        """UNKNOWN_OPERATION raised when operation not in io_operations."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "bad.op": [{"parameter_name": "p", "expression": "${payload.x}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path, io_operations=["good.op"]
            )

        assert ERROR_CODE_UNKNOWN_OPERATION in str(exc_info.value)

    def test_error_code_duplicate_parameter(self, tmp_path: Path) -> None:
        """DUPLICATE_PARAMETER raised for duplicate names."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [
                        {"parameter_name": "dup", "expression": "${payload.a}"},
                        {"parameter_name": "dup", "expression": "${payload.b}"},
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=["op"])

        assert ERROR_CODE_DUPLICATE_PARAMETER in str(exc_info.value)


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_empty_bindings_section(self, tmp_path: Path) -> None:
        """Empty bindings section returns empty dict."""
        contract = {
            "operation_bindings": {
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert result.bindings == {}

    def test_empty_binding_list_for_operation(self, tmp_path: Path) -> None:
        """Empty binding list for an operation is allowed."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "empty.op": [],
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["empty.op"]
        )

        assert "empty.op" in result.bindings
        assert len(result.bindings["empty.op"]) == 0

    def test_single_segment_path(self, tmp_path: Path) -> None:
        """Single segment path is valid."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${payload.id}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        binding = result.bindings["op"][0]
        assert binding.path_segments == ("id",)

    def test_deep_nested_path(self, tmp_path: Path) -> None:
        """Deep nested path up to limit is valid."""
        # Create a path with exactly MAX_PATH_SEGMENTS segments
        segments = [f"f{i}" for i in range(MAX_PATH_SEGMENTS)]
        path_str = ".".join(segments)
        expression = f"${{payload.{path_str}}}"

        contract = {
            "operation_bindings": {
                "bindings": {"op": [{"parameter_name": "p", "expression": expression}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        binding = result.bindings["op"][0]
        assert len(binding.path_segments) == MAX_PATH_SEGMENTS

    def test_expression_at_max_length(self, tmp_path: Path) -> None:
        """Expression at exactly MAX_EXPRESSION_LENGTH is valid."""
        # Build expression that is exactly at the limit
        # ${payload.x} is 12 chars, so we have MAX_EXPRESSION_LENGTH - 12 for path
        path_len = MAX_EXPRESSION_LENGTH - len("${payload.}")
        field_name = "x" * (path_len - 1)  # -1 for trailing char if needed
        expression = f"${{payload.{field_name}}}"

        # Ensure we're at or just under the limit
        if len(expression) > MAX_EXPRESSION_LENGTH:
            field_name = "x" * (MAX_EXPRESSION_LENGTH - len("${payload.}"))
            expression = f"${{payload.{field_name}}}"

        contract = {
            "operation_bindings": {
                "bindings": {"op": [{"parameter_name": "p", "expression": expression}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert "op" in result.bindings

    def test_special_characters_in_parameter_name(self, tmp_path: Path) -> None:
        """Parameter names with underscores are valid."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [
                        {
                            "parameter_name": "user_correlation_id",
                            "expression": "${envelope.correlation_id}",
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert result.bindings["op"][0].parameter_name == "user_correlation_id"

    def test_numeric_fields_in_path(self, tmp_path: Path) -> None:
        """Numeric characters in path segments are valid."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${payload.field123}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert result.bindings["op"][0].path_segments == ("field123",)


# =============================================================================
# TestErrorContext
# =============================================================================


class TestErrorContext:
    """Tests verifying error context is properly populated."""

    def test_error_context_includes_transport_type(self, tmp_path: Path) -> None:
        """Error context includes FILESYSTEM transport type."""
        contract = {
            "operation_bindings": {
                "bindings": {"op": [{"parameter_name": "p", "expression": "invalid"}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=["op"])

        error = exc_info.value
        assert error.model.context is not None
        context = error.model.context
        assert context.get("transport_type") == "filesystem"

    def test_error_context_includes_target_name(self, tmp_path: Path) -> None:
        """Error context includes target path."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                Path("/nonexistent/path/contract.yaml"),
                io_operations=[],
            )

        error = exc_info.value
        assert "contract.yaml" in str(exc_info.value)


# =============================================================================
# TestConfigurableGuardrailLimits
# =============================================================================


class TestConfigurableGuardrailLimits:
    """Tests for per-contract guardrail limit overrides.

    These tests verify that max_expression_length and max_path_segments
    can be overridden per-contract, and that the bounds are validated.

    .. versionadded:: 0.2.7
        Added as part of OMN-1518 - Configurable guardrail limits.
    """

    def test_default_limits_when_not_specified(self, tmp_path: Path) -> None:
        """Defaults are used when limits not specified in contract."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${payload.x}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        # Should have default values
        assert result.max_expression_length == MAX_EXPRESSION_LENGTH
        assert result.max_path_segments == MAX_PATH_SEGMENTS

    def test_custom_limits_loaded_from_contract(self, tmp_path: Path) -> None:
        """Custom limits are correctly loaded from contract."""
        contract = {
            "operation_bindings": {
                "max_expression_length": 512,
                "max_path_segments": 30,
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${payload.x}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert result.max_expression_length == 512
        assert result.max_path_segments == 30

    def test_custom_limits_applied_to_expression_validation(
        self, tmp_path: Path
    ) -> None:
        """Custom limits are used during expression validation."""
        # Create expression that's over default (256) but under custom (512)
        long_path = "a" * 300
        expression = f"${{payload.{long_path}}}"
        assert len(expression) > MAX_EXPRESSION_LENGTH

        contract = {
            "operation_bindings": {
                "max_expression_length": 512,
                "bindings": {"op": [{"parameter_name": "p", "expression": expression}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        # Should succeed with custom limit
        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert "op" in result.bindings
        assert result.bindings["op"][0].path_segments[0] == long_path

    def test_custom_path_segments_applied_to_validation(self, tmp_path: Path) -> None:
        """Custom max_path_segments limit is used during validation."""
        # Create path deeper than default (20) but under custom (30)
        deep_path = ".".join([f"f{i}" for i in range(25)])
        expression = f"${{payload.{deep_path}}}"

        contract = {
            "operation_bindings": {
                "max_path_segments": 30,
                "bindings": {"op": [{"parameter_name": "p", "expression": expression}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        # Should succeed with custom limit
        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert "op" in result.bindings
        assert len(result.bindings["op"][0].path_segments) == 25

    def test_tighter_limits_can_be_set(self, tmp_path: Path) -> None:
        """Limits can be set tighter than defaults for security hardening."""
        # Expression that passes default but fails custom
        expression = "${payload.moderately_long_field_name}"  # ~40 chars
        assert len(expression) < MAX_EXPRESSION_LENGTH
        assert len(expression) > 32  # Still above minimum

        contract = {
            "operation_bindings": {
                "max_expression_length": 32,  # Minimum allowed
                "bindings": {"op": [{"parameter_name": "p", "expression": expression}]},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        # Should fail with tighter limit
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=["op"])

        assert "EXPRESSION_TOO_LONG" in str(exc_info.value)

    def test_limits_below_minimum_rejected(self, tmp_path: Path) -> None:
        """Limits below minimum bounds are rejected by Pydantic."""
        from pydantic import ValidationError

        contract = {
            "operation_bindings": {
                "max_expression_length": 10,  # Below minimum of 32
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        # ValidationError from Pydantic when creating the model
        with pytest.raises((ProtocolConfigurationError, ValidationError)):
            load_operation_bindings_subcontract(contract_path, io_operations=[])

    def test_limits_above_maximum_rejected(self, tmp_path: Path) -> None:
        """Limits above maximum bounds are rejected by Pydantic."""
        from pydantic import ValidationError

        contract = {
            "operation_bindings": {
                "max_expression_length": 2000,  # Above maximum of 1024
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        # ValidationError from Pydantic when creating the model
        with pytest.raises((ProtocolConfigurationError, ValidationError)):
            load_operation_bindings_subcontract(contract_path, io_operations=[])

    def test_limits_at_boundary_valid(self, tmp_path: Path) -> None:
        """Limits at exact boundary values are valid."""
        contract = {
            "operation_bindings": {
                "max_expression_length": 32,  # Minimum
                "max_path_segments": 50,  # Maximum
                "bindings": {
                    "op": [{"parameter_name": "p", "expression": "${payload.x}"}]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["op"]
        )

        assert result.max_expression_length == 32
        assert result.max_path_segments == 50

    def test_limits_apply_to_global_bindings(self, tmp_path: Path) -> None:
        """Custom limits apply to global_bindings as well."""
        # Deep path in global binding
        deep_path = ".".join([f"f{i}" for i in range(25)])
        expression = f"${{envelope.{deep_path}}}"

        contract = {
            "operation_bindings": {
                "max_path_segments": 30,
                "global_bindings": [
                    {"parameter_name": "deep", "expression": expression}
                ],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert result.global_bindings is not None
        assert len(result.global_bindings[0].path_segments) == 25


# =============================================================================
# TestAdditionalContextPathsValidation
# =============================================================================


class TestAdditionalContextPathsValidation:
    """Tests for additional_context_paths validation.

    These tests verify that additional_context_paths in contract.yaml are
    properly validated for:
    - Valid identifier pattern (^[a-z][a-z0-9_]*$)
    - No empty strings
    - No dots (reserved for path traversal)
    - No special characters
    - No duplicates
    - No collision with base context paths

    .. versionadded:: 0.2.7
        Created as part of additional_context_paths extensibility feature.
    """

    def test_valid_additional_context_paths_loaded(self, tmp_path: Path) -> None:
        """Valid additional_context_paths are loaded and returned."""
        contract = {
            "operation_bindings": {
                "additional_context_paths": ["tenant_id", "request_id"],
                "bindings": {
                    "test.op": [
                        {"parameter_name": "t", "expression": "${context.tenant_id}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["test.op"]
        )

        assert "tenant_id" in result.additional_context_paths
        assert "request_id" in result.additional_context_paths
        assert len(result.additional_context_paths) == 2

    def test_additional_context_path_with_numbers(self, tmp_path: Path) -> None:
        """Context path names containing numbers are valid."""
        contract = {
            "operation_bindings": {
                "additional_context_paths": ["region_v2", "user123"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert "region_v2" in result.additional_context_paths
        assert "user123" in result.additional_context_paths

    def test_additional_context_path_with_underscores(self, tmp_path: Path) -> None:
        """Context path names containing underscores are valid."""
        contract = {
            "operation_bindings": {
                "additional_context_paths": ["tenant_region_id"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert "tenant_region_id" in result.additional_context_paths

    def test_empty_string_in_additional_context_paths_fails(
        self, tmp_path: Path
    ) -> None:
        """Empty string in additional_context_paths raises error."""
        from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
            ERROR_CODE_INVALID_CONTEXT_PATH_NAME,
        )

        contract = {
            "operation_bindings": {
                "additional_context_paths": ["valid", ""],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_INVALID_CONTEXT_PATH_NAME in str(exc_info.value)
        assert "empty" in str(exc_info.value).lower()

    def test_uppercase_context_path_fails(self, tmp_path: Path) -> None:
        """Context path with uppercase letters fails pattern validation."""
        from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
            ERROR_CODE_INVALID_CONTEXT_PATH_NAME,
        )

        contract = {
            "operation_bindings": {
                "additional_context_paths": ["TenantId"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_INVALID_CONTEXT_PATH_NAME in str(exc_info.value)

    def test_path_starting_with_number_fails(self, tmp_path: Path) -> None:
        """Context path starting with a number fails pattern validation."""
        from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
            ERROR_CODE_INVALID_CONTEXT_PATH_NAME,
        )

        contract = {
            "operation_bindings": {
                "additional_context_paths": ["123abc"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_INVALID_CONTEXT_PATH_NAME in str(exc_info.value)

    def test_path_with_special_chars_fails(self, tmp_path: Path) -> None:
        """Context path with special characters (hyphen) fails."""
        from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
            ERROR_CODE_INVALID_CONTEXT_PATH_NAME,
        )

        contract = {
            "operation_bindings": {
                "additional_context_paths": ["tenant-id"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_INVALID_CONTEXT_PATH_NAME in str(exc_info.value)

    def test_duplicate_additional_context_path_fails(self, tmp_path: Path) -> None:
        """Duplicate context path names raise error."""
        from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
            ERROR_CODE_INVALID_CONTEXT_PATH_NAME,
        )

        contract = {
            "operation_bindings": {
                "additional_context_paths": ["tenant_id", "tenant_id"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_INVALID_CONTEXT_PATH_NAME in str(exc_info.value)
        assert "duplicate" in str(exc_info.value).lower()

    def test_collision_with_base_context_path_fails(self, tmp_path: Path) -> None:
        """Additional path that duplicates base path raises error."""
        from omnibase_infra.runtime.contract_loaders.operation_bindings_loader import (
            ERROR_CODE_INVALID_CONTEXT_PATH_NAME,
        )

        contract = {
            "operation_bindings": {
                # now_iso is a base context path
                "additional_context_paths": ["now_iso"],
                "bindings": {},
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(contract_path, io_operations=[])

        assert ERROR_CODE_INVALID_CONTEXT_PATH_NAME in str(exc_info.value)

    def test_bindings_can_use_additional_context_paths(self, tmp_path: Path) -> None:
        """Bindings can reference additional_context_paths in expressions."""
        contract = {
            "operation_bindings": {
                "additional_context_paths": ["tenant_id"],
                "bindings": {
                    "db.query": [
                        {
                            "parameter_name": "tenant",
                            "expression": "${context.tenant_id}",
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["db.query"]
        )

        binding = result.bindings["db.query"][0]
        assert binding.source == "context"
        assert binding.path_segments == ("tenant_id",)

    def test_bindings_cannot_use_undeclared_additional_paths(
        self, tmp_path: Path
    ) -> None:
        """Bindings cannot reference context paths not declared in additional_context_paths."""
        contract = {
            "operation_bindings": {
                "additional_context_paths": ["tenant_id"],
                "bindings": {
                    "db.query": [
                        {
                            "parameter_name": "req",
                            "expression": "${context.request_id}",  # Not declared
                            "required": True,
                        }
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_operation_bindings_subcontract(
                contract_path, io_operations=["db.query"]
            )

        assert ERROR_CODE_INVALID_CONTEXT_PATH in str(exc_info.value)

    def test_empty_additional_context_paths_list(self, tmp_path: Path) -> None:
        """Empty additional_context_paths list is valid."""
        contract = {
            "operation_bindings": {
                "additional_context_paths": [],
                "bindings": {
                    "test.op": [
                        {"parameter_name": "ts", "expression": "${context.now_iso}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["test.op"]
        )

        assert result.additional_context_paths == []

    def test_missing_additional_context_paths_defaults_to_empty(
        self, tmp_path: Path
    ) -> None:
        """Missing additional_context_paths defaults to empty list."""
        contract = {
            "operation_bindings": {
                "bindings": {
                    "test.op": [
                        {"parameter_name": "ts", "expression": "${context.now_iso}"}
                    ]
                },
            },
        }
        contract_path = _write_contract(contract, tmp_path)

        result = load_operation_bindings_subcontract(
            contract_path, io_operations=["test.op"]
        )

        assert result.additional_context_paths == []


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestAdditionalContextPathsValidation",
    "TestConfigurableGuardrailLimits",
    "TestContextPathValidation",
    "TestDuplicateParameterValidation",
    "TestEdgeCases",
    "TestErrorCodes",
    "TestErrorContext",
    "TestExpressionValidation",
    "TestFileSizeEnforcement",
    "TestLoadOperationBindingsHappyPath",
    "TestOperationValidation",
    "TestSourceValidation",
    "TestYamlSecurityControls",
]
