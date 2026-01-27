# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Loader for operation_bindings section of contract.yaml.

Validates at load time with explicit error codes. Parses expressions
into pre-compiled ModelParsedBinding instances for fast resolution.

Part of OMN-1518: Declarative operation bindings.

Contract Structure:
    The contract.yaml uses a nested structure for operation bindings::

        operation_bindings:
          version: { major: 1, minor: 0, patch: 0 }
          global_bindings:
            - parameter_name: "correlation_id"
              expression: "${envelope.correlation_id}"
          bindings:
            "db.query":
              - parameter_name: "sql"
                expression: "${payload.sql}"
              - parameter_name: "timestamp"
                expression: "${context.now_iso}"
                required: false

Usage:
    ```python
    from pathlib import Path
    from omnibase_infra.runtime.contract_loaders import (
        load_operation_bindings_subcontract,
    )

    # Load bindings from contract.yaml
    contract_path = Path("nodes/my_handler/contract.yaml")
    bindings = load_operation_bindings_subcontract(contract_path)

    # Access parsed bindings
    for operation, binding_list in bindings.bindings.items():
        for binding in binding_list:
            print(f"{operation}: {binding.parameter_name} <- {binding.original_expression}")
    ```

See Also:
    - ModelOperationBindingsSubcontract: Model for bindings configuration
    - ModelParsedBinding: Model for individual pre-parsed bindings
    - ModelOperationBinding: Raw binding entry from YAML
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final, Literal

import yaml

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.bindings import (
    MAX_EXPRESSION_LENGTH,
    MAX_PATH_SEGMENTS,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
    ModelOperationBinding,
    ModelOperationBindingsSubcontract,
    ModelParsedBinding,
)
from omnibase_infra.runtime.binding_resolver import (
    BindingExpressionParseError,
    BindingExpressionParser,
    EnumBindingParseErrorCode,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants (Loader-specific)
# =============================================================================

# Maximum allowed file size for contract.yaml files (10MB)
# Security control to prevent memory exhaustion via large YAML files
# Error code: FILE_SIZE_EXCEEDED (BINDING_LOADER_050)
MAX_CONTRACT_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB

# =============================================================================
# Error Codes
# =============================================================================

# Expression validation errors (010-019)
ERROR_CODE_EXPRESSION_MALFORMED = "BINDING_LOADER_010"
ERROR_CODE_INVALID_SOURCE = "BINDING_LOADER_011"
ERROR_CODE_PATH_TOO_DEEP = "BINDING_LOADER_012"
ERROR_CODE_EXPRESSION_TOO_LONG = "BINDING_LOADER_013"
ERROR_CODE_EMPTY_PATH_SEGMENT = "BINDING_LOADER_014"
ERROR_CODE_MISSING_PATH_SEGMENT = "BINDING_LOADER_015"
ERROR_CODE_INVALID_CONTEXT_PATH = "BINDING_LOADER_016"

# Binding validation errors (020-029)
ERROR_CODE_UNKNOWN_OPERATION = "BINDING_LOADER_020"
ERROR_CODE_DUPLICATE_PARAMETER = "BINDING_LOADER_021"
ERROR_CODE_INVALID_CONTEXT_PATH_NAME = "BINDING_LOADER_022"
"""Invalid context path name format.

Context path names must:
- Start with a lowercase letter
- Contain only lowercase letters, numbers, and underscores
- Not be empty
- Not contain dots (used for path traversal)
"""

# =============================================================================
# Context Path Validation Pattern
# =============================================================================

CONTEXT_PATH_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
"""Pattern for valid context path names.

Context paths must:
- Start with a lowercase letter
- Contain only lowercase letters (a-z), numbers (0-9), and underscores
- Be at least 1 character long

Examples:
- Valid: ``tenant_id``, ``request_id``, ``user123``
- Invalid: ``TenantId`` (uppercase), ``123abc`` (starts with number),
  ``tenant.id`` (contains dot), ``tenant-id`` (contains dash)
"""

# File/contract errors (030-039)
ERROR_CODE_CONTRACT_NOT_FOUND = "BINDING_LOADER_030"
ERROR_CODE_YAML_PARSE_ERROR = "BINDING_LOADER_031"

# Security errors (050-059)
ERROR_CODE_FILE_SIZE_EXCEEDED = "BINDING_LOADER_050"


def _check_file_size(contract_path: Path, operation: str) -> None:
    """Check that contract file does not exceed maximum allowed size.

    This is a security control to prevent memory exhaustion attacks via
    oversized YAML files. Per CLAUDE.md Handler Plugin Loader security patterns,
    a 10MB file size limit is enforced.

    Args:
        contract_path: Path to the contract.yaml file.
        operation: Name of the operation for error context.

    Raises:
        ProtocolConfigurationError: If file exceeds MAX_CONTRACT_FILE_SIZE_BYTES.
            Error code: FILE_SIZE_EXCEEDED (BINDING_LOADER_050).
    """
    try:
        file_size = contract_path.stat().st_size
    except FileNotFoundError:
        # Let the caller handle FileNotFoundError with its own error message
        return

    if file_size > MAX_CONTRACT_FILE_SIZE_BYTES:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        logger.error(
            "Contract file exceeds maximum size: %d bytes > %d bytes at %s",
            file_size,
            MAX_CONTRACT_FILE_SIZE_BYTES,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file exceeds maximum size: {file_size} bytes > "
            f"{MAX_CONTRACT_FILE_SIZE_BYTES} bytes. "
            f"Reduce the contract.yaml file size or split into multiple contracts. "
            f"Error code: FILE_SIZE_EXCEEDED ({ERROR_CODE_FILE_SIZE_EXCEEDED})",
            context=ctx,
        )


def _parse_expression(
    expression: str,
    contract_path: Path,
    max_expression_length: int | None = None,
    max_path_segments: int | None = None,
    additional_context_paths: frozenset[str] | None = None,
) -> tuple[Literal["payload", "envelope", "context"], tuple[str, ...]]:
    """Parse and validate binding expression at load time.

    Delegates to BindingExpressionParser.parse() for validation and parsing,
    then wraps any BindingExpressionParseError with ProtocolConfigurationError
    and appropriate error codes for loader-level diagnostics.

    Args:
        expression: Expression in ${source.path.to.field} format.
        contract_path: Path for error context.
        max_expression_length: Override default expression length limit.
            If None, uses the default MAX_EXPRESSION_LENGTH (256).
        max_path_segments: Override default path segment limit.
            If None, uses the default MAX_PATH_SEGMENTS (20).
        additional_context_paths: Additional valid context paths beyond
            the base VALID_CONTEXT_PATHS set.

    Returns:
        Tuple of (source, path_segments) where:
        - source: One of "payload", "envelope", "context"
        - path_segments: Tuple of field names to traverse

    Raises:
        ProtocolConfigurationError: With specific error code for:
        - BINDING_LOADER_010: Malformed expression syntax
        - BINDING_LOADER_011: Invalid source
        - BINDING_LOADER_012: Path too deep
        - BINDING_LOADER_013: Expression too long
        - BINDING_LOADER_014: Empty path segment
        - BINDING_LOADER_016: Invalid context path

    .. versionchanged:: 0.2.7
        Added max_expression_length, max_path_segments, and additional_context_paths
        parameters for per-contract guardrail overrides.
    """
    ctx = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation="parse_binding_expression",
        target_name=str(contract_path),
    )

    parser = BindingExpressionParser()

    try:
        return parser.parse(
            expression,
            max_expression_length=max_expression_length,
            max_path_segments=max_path_segments,
            additional_context_paths=additional_context_paths,
        )
    except BindingExpressionParseError as e:
        # Use typed error code from the exception - no string matching needed
        error_code = e.error_code.value
        code_name = e.error_code.name
        error_msg = e.message

        # Log with appropriate context based on error type
        if e.error_code == EnumBindingParseErrorCode.EXPRESSION_TOO_LONG:
            logger.exception(
                "Expression exceeds max length: %s in %s",
                expression,
                contract_path,
            )
            user_msg = f"{error_msg}. Error code: {code_name} ({error_code})"

        elif e.error_code == EnumBindingParseErrorCode.EXPRESSION_MALFORMED:
            # Array access or invalid syntax
            if "Array access" in error_msg:
                logger.exception(
                    "Array access not allowed in expressions: %s in %s",
                    expression,
                    contract_path,
                )
                user_msg = (
                    f"Array access not allowed in expressions: {expression}. "
                    f"Use path-based access only (e.g., ${{payload.items}} not "
                    f"${{payload.items[0]}}). "
                    f"Error code: {code_name} ({error_code})"
                )
            else:
                logger.exception(
                    "Invalid expression syntax: %s in %s. Expected ${{source.path.to.field}}",
                    expression,
                    contract_path,
                )
                user_msg = (
                    f"Invalid expression syntax: {expression}. "
                    f"Expected format: ${{source.path.to.field}}. "
                    f"Error code: {code_name} ({error_code})"
                )

        elif e.error_code == EnumBindingParseErrorCode.INVALID_SOURCE:
            logger.exception(
                "Invalid source in expression %s at %s: %s",
                expression,
                contract_path,
                error_msg,
            )
            user_msg = f"{error_msg}. Error code: {code_name} ({error_code})"

        elif e.error_code == EnumBindingParseErrorCode.EMPTY_PATH_SEGMENT:
            logger.exception(
                "Empty path segment in expression: %s at %s",
                expression,
                contract_path,
            )
            user_msg = (
                f"Empty path segment in expression: {expression}. "
                f"Path segments cannot be empty. "
                f"Error code: {code_name} ({error_code})"
            )

        elif e.error_code == EnumBindingParseErrorCode.PATH_TOO_DEEP:
            logger.exception(
                "Path exceeds max segments in expression: %s at %s",
                expression,
                contract_path,
            )
            user_msg = f"{error_msg}. Error code: {code_name} ({error_code})"

        elif e.error_code == EnumBindingParseErrorCode.INVALID_CONTEXT_PATH:
            logger.exception(
                "Invalid context path in expression %s at %s: %s",
                expression,
                contract_path,
                error_msg,
            )
            user_msg = f"{error_msg}. Error code: {code_name} ({error_code})"

        else:
            # Fallback for any unhandled error code (should not happen)
            logger.exception(
                "Expression parsing failed: %s in %s - %s",
                expression,
                contract_path,
                error_msg,
            )
            user_msg = (
                f"Expression parsing failed: {expression}. {error_msg}. "
                f"Error code: {code_name} ({error_code})"
            )

        raise ProtocolConfigurationError(user_msg, context=ctx) from e


def _parse_binding_entry(
    raw_binding: dict[str, object],
    contract_path: Path,
    max_expression_length: int | None = None,
    max_path_segments: int | None = None,
    additional_context_paths: frozenset[str] | None = None,
) -> ModelParsedBinding:
    """Parse a raw binding dict into ModelParsedBinding.

    First validates the raw YAML structure using ModelOperationBinding,
    then parses the expression into pre-compiled components.

    Args:
        raw_binding: Raw binding dict from YAML.
        contract_path: Path for error context.
        max_expression_length: Override default expression length limit.
        max_path_segments: Override default path segment limit.
        additional_context_paths: Additional valid context paths beyond
            the base VALID_CONTEXT_PATHS set.

    Returns:
        ModelParsedBinding with pre-parsed expression components.

    Raises:
        ProtocolConfigurationError: If binding or expression is invalid.
        ValidationError: If raw binding doesn't match ModelOperationBinding schema.

    .. versionchanged:: 0.2.7
        Added max_expression_length, max_path_segments, and additional_context_paths
        parameters for per-contract guardrail overrides.
    """
    # First validate as ModelOperationBinding (raw YAML structure)
    # This validates required fields and types
    operation_binding = ModelOperationBinding(**raw_binding)

    # Parse the expression into components
    source, path_segments = _parse_expression(
        operation_binding.expression,
        contract_path,
        max_expression_length=max_expression_length,
        max_path_segments=max_path_segments,
        additional_context_paths=additional_context_paths,
    )

    return ModelParsedBinding(
        parameter_name=operation_binding.parameter_name,
        source=source,
        path_segments=path_segments,
        required=operation_binding.required,
        default=operation_binding.default,
        original_expression=operation_binding.expression,
    )


def _validate_additional_context_paths(
    paths: list[str],
    contract_path: Path,
) -> frozenset[str]:
    """Validate additional context path names and return as frozenset.

    Context path names must:
    - Start with a lowercase letter
    - Contain only lowercase letters (a-z), numbers (0-9), and underscores
    - Not be empty
    - Not contain dots (reserved for path traversal)
    - Not duplicate base context paths (now_iso, dispatcher_id, correlation_id)

    Args:
        paths: List of additional context path names from contract.yaml.
        contract_path: Path for error context.

    Returns:
        Frozenset of validated context path names.

    Raises:
        ProtocolConfigurationError: If any path name is invalid.
            Error code: INVALID_CONTEXT_PATH_NAME (BINDING_LOADER_022).
    """
    if not paths:
        return frozenset()

    validated: set[str] = set()
    ctx = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation="validate_additional_context_paths",
        target_name=str(contract_path),
    )

    for path in paths:
        # Check for empty string
        if not path:
            logger.error(
                "Empty string in additional_context_paths at %s",
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Empty string in additional_context_paths. "
                f"Context path names cannot be empty. "
                f"Error code: INVALID_CONTEXT_PATH_NAME ({ERROR_CODE_INVALID_CONTEXT_PATH_NAME})",
                context=ctx,
            )

        # Check pattern
        if not CONTEXT_PATH_NAME_PATTERN.match(path):
            logger.error(
                "Invalid context path name '%s' in additional_context_paths at %s: "
                "must match pattern ^[a-z][a-z0-9_]*$",
                path,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Invalid context path name '{path}' in additional_context_paths. "
                f"Names must start with a lowercase letter and contain only "
                f"lowercase letters, numbers, and underscores. "
                f"Error code: INVALID_CONTEXT_PATH_NAME ({ERROR_CODE_INVALID_CONTEXT_PATH_NAME})",
                context=ctx,
            )

        # Check for dots (reserved for path traversal)
        if "." in path:
            logger.error(
                "Context path name '%s' contains dot at %s: dots are reserved for path traversal",
                path,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Context path name '{path}' contains a dot. "
                f"Dots are reserved for path traversal in expressions. "
                f"Use underscores instead (e.g., 'tenant_id' not 'tenant.id'). "
                f"Error code: INVALID_CONTEXT_PATH_NAME ({ERROR_CODE_INVALID_CONTEXT_PATH_NAME})",
                context=ctx,
            )

        # Check for collision with base context paths
        if path in VALID_CONTEXT_PATHS:
            logger.error(
                "Context path name '%s' duplicates base context path at %s",
                path,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Context path name '{path}' duplicates a base context path. "
                f"Base context paths ({sorted(VALID_CONTEXT_PATHS)}) are automatically available. "
                f"Error code: INVALID_CONTEXT_PATH_NAME ({ERROR_CODE_INVALID_CONTEXT_PATH_NAME})",
                context=ctx,
            )

        # Check for duplicates within the list
        if path in validated:
            logger.error(
                "Duplicate context path name '%s' in additional_context_paths at %s",
                path,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Duplicate context path name '{path}' in additional_context_paths. "
                f"Each context path name must be unique. "
                f"Error code: INVALID_CONTEXT_PATH_NAME ({ERROR_CODE_INVALID_CONTEXT_PATH_NAME})",
                context=ctx,
            )

        validated.add(path)

    return frozenset(validated)


def _check_duplicate_parameters(
    bindings: list[ModelParsedBinding],
    scope: str,
    contract_path: Path,
) -> None:
    """Check for duplicate parameter names within a binding list.

    Args:
        bindings: List of parsed bindings to check.
        scope: Description of scope for error message (e.g., "global_bindings").
        contract_path: Path for error context.

    Raises:
        ProtocolConfigurationError: If duplicate parameter name found.
            Error code: DUPLICATE_PARAMETER (BINDING_LOADER_021).
    """
    seen: set[str] = set()
    for binding in bindings:
        if binding.parameter_name in seen:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="validate_bindings",
                target_name=str(contract_path),
            )
            logger.error(
                "Duplicate parameter '%s' in %s at %s",
                binding.parameter_name,
                scope,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Duplicate parameter '{binding.parameter_name}' in {scope}. "
                f"Each parameter name must be unique within its scope. "
                f"Error code: DUPLICATE_PARAMETER ({ERROR_CODE_DUPLICATE_PARAMETER})",
                context=ctx,
            )
        seen.add(binding.parameter_name)


def load_operation_bindings_subcontract(
    contract_path: Path,
    io_operations: list[str] | None = None,
) -> ModelOperationBindingsSubcontract:
    """Load, parse, and validate operation_bindings from contract.yaml.

    Loads the operation_bindings section from a contract.yaml file
    and converts it to ModelOperationBindingsSubcontract format with
    pre-parsed expressions. All validation happens at load time.

    Validation at load time:
    - File size limit (10MB) - security control
    - YAML safe_load - security control
    - Expression syntax validation
    - Source validation (payload/envelope/context)
    - Context path validation (now_iso/dispatcher_id/correlation_id)
    - Duplicate parameter detection per scope
    - io_operations reference validation (if provided)

    Args:
        contract_path: Path to contract.yaml file.
        io_operations: Optional list of valid operation names. If provided,
            validates that all operation names in bindings exist in this list.

    Returns:
        ModelOperationBindingsSubcontract with pre-parsed bindings.
        Returns empty subcontract if operation_bindings section is missing.

    Raises:
        ProtocolConfigurationError: With specific error code for various failures:
        - BINDING_LOADER_030: Contract file not found
        - BINDING_LOADER_031: YAML parse error
        - BINDING_LOADER_050: File size exceeded
        - BINDING_LOADER_010-016: Expression validation errors
        - BINDING_LOADER_020: Unknown operation (not in io_operations)
        - BINDING_LOADER_021: Duplicate parameter name

    Example:
        ```python
        from pathlib import Path
        from omnibase_infra.runtime.contract_loaders import (
            load_operation_bindings_subcontract,
        )

        contract_path = Path(__file__).parent / "contract.yaml"
        bindings = load_operation_bindings_subcontract(
            contract_path,
            io_operations=["db.query", "db.execute"],  # Optional validation
        )

        # Access parsed bindings
        for op_name, binding_list in bindings.bindings.items():
            for binding in binding_list:
                print(f"{op_name}: {binding.parameter_name}")
        ```
    """
    operation = "load_operation_bindings"
    ctx = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation=operation,
        target_name=str(contract_path),
    )

    # Check file exists
    if not contract_path.exists():
        logger.error(
            "Contract file not found: %s - cannot load operation bindings",
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file not found: {contract_path}. "
            f"Ensure the contract.yaml exists in the handler directory. "
            f"Error code: CONTRACT_NOT_FOUND ({ERROR_CODE_CONTRACT_NOT_FOUND})",
            context=ctx,
        )

    # Check file size (security control - MUST be before yaml.safe_load)
    _check_file_size(contract_path, operation)

    # Load YAML safely
    try:
        with contract_path.open("r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # Sanitize error message - don't include raw YAML error which may contain file contents
        error_type = type(e).__name__
        logger.exception(
            "Invalid YAML syntax in contract.yaml at %s: %s",
            contract_path,
            error_type,
        )
        raise ProtocolConfigurationError(
            f"Invalid YAML syntax in contract.yaml at {contract_path}: {error_type}. "
            f"Verify the YAML syntax is correct. "
            f"Error code: YAML_PARSE_ERROR ({ERROR_CODE_YAML_PARSE_ERROR})",
            context=ctx,
        ) from e

    if contract_data is None:
        contract_data = {}

    # Get operation_bindings section (optional - return empty if missing)
    bindings_section = contract_data.get("operation_bindings", {})
    if not bindings_section:
        logger.debug(
            "No operation_bindings section in contract.yaml at %s - returning empty subcontract",
            contract_path,
        )
        return ModelOperationBindingsSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            bindings={},
            global_bindings=None,
        )

    # Parse version
    version_data = bindings_section.get("version", {"major": 1, "minor": 0, "patch": 0})
    if isinstance(version_data, dict):
        version = ModelSemVer(**version_data)
    else:
        logger.warning(
            "Invalid version format in operation_bindings section at %s: "
            "expected dict with major/minor/patch, got %s. "
            "Defaulting to 1.0.0. "
            "Use format: version: { major: 1, minor: 0, patch: 0 }",
            contract_path,
            type(version_data).__name__,
        )
        version = ModelSemVer(major=1, minor=0, patch=0)

    # Parse optional guardrail overrides
    max_expression_length_override: int | None = bindings_section.get(
        "max_expression_length"
    )
    max_path_segments_override: int | None = bindings_section.get("max_path_segments")

    # Parse and validate additional_context_paths (optional)
    raw_additional_context_paths: list[str] = bindings_section.get(
        "additional_context_paths", []
    )
    additional_context_paths: frozenset[str] = _validate_additional_context_paths(
        raw_additional_context_paths,
        contract_path,
    )

    if additional_context_paths:
        logger.debug(
            "Loaded %d additional context paths from contract.yaml at %s: %s",
            len(additional_context_paths),
            contract_path,
            sorted(additional_context_paths),
        )

    # Parse global_bindings (optional)
    global_bindings: list[ModelParsedBinding] | None = None
    raw_global = bindings_section.get("global_bindings", [])
    if raw_global:
        global_bindings = [
            _parse_binding_entry(
                b,
                contract_path,
                max_expression_length=max_expression_length_override,
                max_path_segments=max_path_segments_override,
                additional_context_paths=additional_context_paths or None,
            )
            for b in raw_global
        ]
        _check_duplicate_parameters(global_bindings, "global_bindings", contract_path)
        logger.debug(
            "Loaded %d global bindings from contract.yaml at %s",
            len(global_bindings),
            contract_path,
        )

    # Parse operation-specific bindings
    parsed_bindings: dict[str, list[ModelParsedBinding]] = {}
    raw_bindings = bindings_section.get("bindings", {})

    for operation_name, operation_binding_list in raw_bindings.items():
        # Validate operation exists in io_operations (if provided)
        if io_operations is not None and operation_name not in io_operations:
            logger.error(
                "Unknown operation '%s' in bindings - not in io_operations at %s",
                operation_name,
                contract_path,
            )
            raise ProtocolConfigurationError(
                f"Unknown operation '{operation_name}' in bindings section. "
                f"Not found in io_operations: {sorted(io_operations)}. "
                f"Error code: UNKNOWN_OPERATION ({ERROR_CODE_UNKNOWN_OPERATION})",
                context=ctx,
            )

        # Parse all bindings for this operation
        parsed_list = [
            _parse_binding_entry(
                b,
                contract_path,
                max_expression_length=max_expression_length_override,
                max_path_segments=max_path_segments_override,
                additional_context_paths=additional_context_paths or None,
            )
            for b in operation_binding_list
        ]

        # Check for duplicates within this operation's bindings
        _check_duplicate_parameters(
            parsed_list, f"operation '{operation_name}'", contract_path
        )

        parsed_bindings[operation_name] = parsed_list

    logger.debug(
        "Loaded %d operation binding groups from contract.yaml at %s",
        len(parsed_bindings),
        contract_path,
    )

    return ModelOperationBindingsSubcontract(
        version=version,
        additional_context_paths=list(additional_context_paths),
        bindings=parsed_bindings,
        global_bindings=global_bindings,
        max_expression_length=max_expression_length_override or MAX_EXPRESSION_LENGTH,
        max_path_segments=max_path_segments_override or MAX_PATH_SEGMENTS,
    )


__all__ = [
    # Loader-specific constants
    "CONTEXT_PATH_NAME_PATTERN",
    "MAX_CONTRACT_FILE_SIZE_BYTES",
    # Re-exported binding constants (canonical source: omnibase_infra.models.bindings)
    "MAX_EXPRESSION_LENGTH",
    "MAX_PATH_SEGMENTS",
    "VALID_CONTEXT_PATHS",
    "VALID_SOURCES",
    # Error codes
    "ERROR_CODE_CONTRACT_NOT_FOUND",
    "ERROR_CODE_DUPLICATE_PARAMETER",
    "ERROR_CODE_EMPTY_PATH_SEGMENT",
    "ERROR_CODE_EXPRESSION_MALFORMED",
    "ERROR_CODE_EXPRESSION_TOO_LONG",
    "ERROR_CODE_FILE_SIZE_EXCEEDED",
    "ERROR_CODE_INVALID_CONTEXT_PATH",
    "ERROR_CODE_INVALID_CONTEXT_PATH_NAME",
    "ERROR_CODE_INVALID_SOURCE",
    "ERROR_CODE_MISSING_PATH_SEGMENT",
    "ERROR_CODE_PATH_TOO_DEEP",
    "ERROR_CODE_UNKNOWN_OPERATION",
    "ERROR_CODE_YAML_PARSE_ERROR",
    # Main function
    "load_operation_bindings_subcontract",
]
