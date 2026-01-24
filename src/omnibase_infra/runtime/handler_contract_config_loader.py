# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Contract Configuration Loader.

This module provides utilities for loading handler configuration from contract
YAML files. It supports both relative and absolute paths and extracts handler-
specific configuration for use during handler initialization.

Part of the bootstrap handler contract infrastructure.

The loader validates:
- Contract file existence
- YAML syntax validity
- Required contract structure (must be a dict)

Contract File Structure:
    Handler contracts follow this schema (see contracts/handlers/*/handler_contract.yaml):

    ```yaml
    name: handler-consul
    handler_class: omnibase_infra.handlers.handler_consul.HandlerConsul
    handler_type: effect
    tags:
      - consul
      - service-discovery
    security:
      trusted_namespace: omnibase_infra.handlers
      audit_logging: true
      requires_authentication: false  # optional
    ```

Security:
    - Uses yaml.safe_load() to prevent arbitrary code execution
    - Contract files are treated as trusted configuration (see CLAUDE.md security patterns)

See Also:
    - HandlerBootstrapSource: Uses this loader for bootstrap handler contracts
    - handler_plugin_loader.py: Related handler loading infrastructure
    - docs/patterns/handler_plugin_loader.md: Security documentation

.. versionadded:: 0.6.5
    Created for bootstrap handler contract loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

if TYPE_CHECKING:
    from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Maximum contract file size (10 MB) - matches handler_plugin_loader.py
MAX_CONTRACT_SIZE_BYTES = 10 * 1024 * 1024


def load_handler_contract_config(
    contract_path: str | Path | None,
    handler_id: str,
) -> dict[str, JsonType]:
    """Load handler configuration from contract YAML file.

    Reads and parses a handler contract YAML file, returning the parsed
    dictionary for further processing. The contract path can be either
    absolute or relative (resolved against common base paths).

    Args:
        contract_path: Path to handler_contract.yaml (relative or absolute).
            If None, raises ProtocolConfigurationError.
        handler_id: Handler identifier for error messages and logging.

    Returns:
        Dict containing the parsed contract YAML content.

    Raises:
        ProtocolConfigurationError: If any of the following conditions occur:
            - contract_path is None
            - File not found
            - Permission denied (accessing, reading metadata, or reading content)
            - OS error (path too long, invalid characters, etc.)
            - File too large (exceeds MAX_CONTRACT_SIZE_BYTES)
            - YAML syntax error
            - Contract is not a dict

    Example:
        >>> contract = load_handler_contract_config(
        ...     "contracts/handlers/consul/handler_contract.yaml",
        ...     "proto.consul",
        ... )
        >>> contract["name"]
        'handler-consul'
    """
    if contract_path is None:
        raise ProtocolConfigurationError(
            f"Handler {handler_id} has no contract_path configured",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    path = Path(contract_path)

    # Resolve relative paths against common base directories
    if not path.is_absolute():
        resolved_path = _resolve_contract_path(path)
        if resolved_path is None:
            raise ProtocolConfigurationError(
                f"Contract file not found: {contract_path}",
                context=ModelInfraErrorContext.with_correlation(
                    operation="load_handler_contract",
                    target_name=handler_id,
                ),
            )
        path = resolved_path

    # Check file existence with permission error handling
    try:
        file_exists = path.exists()
    except PermissionError as e:
        raise ProtocolConfigurationError(
            f"Permission denied accessing contract file: {contract_path}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e
    except OSError as e:
        raise ProtocolConfigurationError(
            f"OS error accessing contract file: {contract_path} ({e})",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e

    if not file_exists:
        raise ProtocolConfigurationError(
            f"Contract file not found: {contract_path}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    # Check file size before reading (security: prevent memory exhaustion)
    try:
        file_size = path.stat().st_size
    except PermissionError as e:
        raise ProtocolConfigurationError(
            f"Permission denied reading contract file metadata: {contract_path}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e
    except OSError as e:
        raise ProtocolConfigurationError(
            f"OS error reading contract file metadata: {contract_path} ({e})",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e

    if file_size > MAX_CONTRACT_SIZE_BYTES:
        raise ProtocolConfigurationError(
            f"Contract file too large: {file_size} bytes (max {MAX_CONTRACT_SIZE_BYTES})",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    try:
        with path.open() as f:
            contract = yaml.safe_load(f)
    except FileNotFoundError as e:
        # Race condition: file was deleted between exists() check and open()
        raise ProtocolConfigurationError(
            f"Contract file not found (deleted after check): {contract_path}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e
    except PermissionError as e:
        raise ProtocolConfigurationError(
            f"Permission denied reading contract file: {contract_path}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e
    except yaml.YAMLError as e:
        raise ProtocolConfigurationError(
            f"Invalid YAML in contract file: {contract_path}: {e}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e
    except OSError as e:
        # Catch other I/O errors (disk full, I/O error, etc.)
        raise ProtocolConfigurationError(
            f"I/O error reading contract file: {contract_path}: {e}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        ) from e

    if not isinstance(contract, dict):
        raise ProtocolConfigurationError(
            f"Contract must be a dict, got {type(contract).__name__}",
            context=ModelInfraErrorContext.with_correlation(
                operation="load_handler_contract",
                target_name=handler_id,
            ),
        )

    logger.debug(
        "Loaded handler contract",
        extra={
            "handler_id": handler_id,
            "contract_path": str(path),
            "contract_name": contract.get("name"),
        },
    )

    return contract


def _resolve_contract_path(relative_path: Path) -> Path | None:
    """Resolve a relative contract path against common base directories.

    Tries multiple base directories to find the contract file:
    1. Current working directory
    2. Repository root (for contracts/handlers/*/handler_contract.yaml)
    3. Source directory (for src/omnibase_infra/contracts/handlers/*)

    Path Resolution Reference:
        This file: src/omnibase_infra/runtime/handler_contract_config_loader.py
        - .parent = runtime/
        - .parent.parent = omnibase_infra/
        - .parent.parent.parent = src/
        - .parent.parent.parent.parent = repo root (contains contracts/)

    Args:
        relative_path: Relative path to the contract file.

    Returns:
        Resolved absolute path if the file exists in any of the searched
        directories. Returns None if:
        - The file does not exist in any searched directory
        - Permission errors prevent checking all directories
        - OS errors (path too long, invalid characters) occur for all paths

    Note:
        Permission errors and symlink resolution issues are logged but do not
        cause failures - the function continues to the next base path. Only
        returns None after exhausting all possible base directories.
    """
    # Calculate paths with clear semantics based on this file's location:
    # This file: src/omnibase_infra/runtime/handler_contract_config_loader.py
    this_file = Path(__file__)
    runtime_dir = this_file.parent  # src/omnibase_infra/runtime/
    infra_pkg_dir = runtime_dir.parent  # src/omnibase_infra/
    src_dir = infra_pkg_dir.parent  # src/
    repo_root = src_dir.parent  # repository root (contains contracts/)

    # Base directories to try, in order of preference
    base_paths = [
        Path.cwd(),  # Current working directory (most specific)
        repo_root,  # For contracts/handlers/*/handler_contract.yaml
        src_dir,  # For src/omnibase_infra/contracts/handlers/*/handler_contract.yaml
    ]

    for base in base_paths:
        full_path = base / relative_path
        try:
            if full_path.exists():
                # Resolve symlinks and normalize the path
                try:
                    return full_path.resolve(strict=True)
                except OSError as resolve_error:
                    # Symlink resolution failed (broken symlink, etc.)
                    # Fall back to non-strict resolution
                    logger.debug(
                        "Symlink resolution failed, using non-strict resolution",
                        extra={
                            "path": str(full_path),
                            "error": str(resolve_error),
                        },
                    )
                    return full_path.resolve(strict=False)
        except PermissionError:
            logger.warning(
                "Permission denied checking contract path",
                extra={"path": str(full_path), "base": str(base)},
            )
            continue
        except OSError as e:
            # Handle other OS-level errors (e.g., path too long, invalid characters)
            logger.warning(
                "OS error checking contract path",
                extra={"path": str(full_path), "error": str(e)},
            )
            continue

    return None


def _validate_required_contract_fields(
    contract: dict[str, JsonType],
    handler_type: str,
) -> None:
    """Validate required fields are present in contract.

    Performs fail-fast validation for required contract fields. Raises
    ProtocolConfigurationError immediately if any required field is missing.

    Required Fields:
        - name: Handler name (REQUIRED)
        - handler_class: Fully qualified class path (REQUIRED)
        - handler_type OR descriptor.handler_kind: At least one (REQUIRED)

    Args:
        contract: Parsed contract dict to validate.
        handler_type: Handler type identifier for error context.

    Raises:
        ProtocolConfigurationError: If any required field is missing, with
            clear message indicating which field and proper error context.
    """
    # Validate 'name' field
    if "name" not in contract:
        raise ProtocolConfigurationError(
            f"Missing required field 'name' in handler contract for '{handler_type}'",
            context=ModelInfraErrorContext.with_correlation(
                operation="extract_handler_config",
                target_name=handler_type,
            ),
        )

    # Validate 'handler_class' field
    if "handler_class" not in contract:
        raise ProtocolConfigurationError(
            f"Missing required field 'handler_class' in handler contract for '{handler_type}'",
            context=ModelInfraErrorContext.with_correlation(
                operation="extract_handler_config",
                target_name=handler_type,
            ),
        )

    # Validate handler kind: must have either 'handler_type' or 'descriptor.handler_kind'
    has_handler_type = "handler_type" in contract
    descriptor = contract.get("descriptor", {})
    has_descriptor_handler_kind = (
        isinstance(descriptor, dict) and "handler_kind" in descriptor
    )

    if not has_handler_type and not has_descriptor_handler_kind:
        raise ProtocolConfigurationError(
            f"Missing required field 'handler_type' or 'descriptor.handler_kind' "
            f"in handler contract for '{handler_type}'",
            context=ModelInfraErrorContext.with_correlation(
                operation="extract_handler_config",
                target_name=handler_type,
            ),
        )


def extract_handler_config(
    contract: dict[str, JsonType],
    handler_type: str,
    *,
    require_basic_fields: bool = True,
) -> dict[str, JsonType]:
    """Extract handler-specific configuration from parsed contract.

    Extracts configuration values from both basic and rich contract structures
    and flattens them into a dict suitable for handler initialization.

    Fail-Fast Validation:
        When require_basic_fields=True (default), validates that required fields
        are present and raises ProtocolConfigurationError immediately if missing.

        Required fields for basic contracts:
            - name: REQUIRED
            - handler_class: REQUIRED
            - handler_type OR descriptor.handler_kind: At least one REQUIRED

    Supports Two Contract Formats:

    Basic Contract Structure (contracts/handlers/*/handler_contract.yaml):
        - name: Handler name (e.g., "handler-consul") [REQUIRED]
        - handler_class: Fully qualified class path [REQUIRED]
        - handler_type: Handler kind (effect, compute, etc.) [REQUIRED*]
        - tags: List of discovery tags
        - security: Security metadata dict
            - trusted_namespace: Required trusted import namespace
            - audit_logging: Whether to enable audit logging
            - requires_authentication: Whether auth is required (optional)

        *handler_type is required unless descriptor.handler_kind is provided.

    Rich Contract Structure (src/omnibase_infra/contracts/handlers/*/handler_contract.yaml):
        - handler_id: Unique identifier (e.g., "effect.mcp.handler")
        - name: Handler name [REQUIRED]
        - version: Semantic version
        - descriptor: Handler descriptor with timeout, retry, circuit breaker
            - handler_kind: Handler behavioral type [REQUIRED*]
            - timeout_ms: Timeout in milliseconds
            - retry_policy: Retry configuration
            - circuit_breaker: Circuit breaker configuration
        - metadata: Additional metadata
            - transport: Transport configuration
                - default_host: Default bind host
                - default_port: Default port
                - default_path: Default URL path
                - stateless: Whether handler is stateless
                - json_response: Whether to use JSON responses
            - security: Security configuration
                - tool_access: Tool access control
                    - max_tools: Maximum number of tools

        *handler_kind is required unless handler_type is provided.

    Args:
        contract: Parsed contract dict from load_handler_contract_config().
        handler_type: Handler type identifier (e.g., "consul", "db") for
            logging and context.
        require_basic_fields: If True (default), validates required fields are
            present and raises ProtocolConfigurationError if missing. Set to
            False for rich contracts that may have different requirements or
            during migration.

    Returns:
        Flat dict with extracted configuration values suitable for
        handler.initialize() or similar configuration methods:
            - name: Handler name
            - handler_class: Fully qualified class path
            - handler_kind: Handler behavioral type
            - tags: List of tags
            - trusted_namespace: Security namespace
            - audit_logging: Audit logging flag
            - requires_authentication: Auth requirement flag
            - host: Transport default host (rich contracts)
            - port: Transport default port (rich contracts)
            - path: Transport default path (rich contracts)
            - stateless: Transport stateless flag (rich contracts)
            - json_response: Transport JSON response flag (rich contracts)
            - timeout_seconds: Timeout in seconds (rich contracts)
            - max_tools: Maximum tools for MCP (rich contracts)

    Raises:
        ProtocolConfigurationError: If require_basic_fields=True and any of:
            - 'name' field is missing
            - 'handler_class' field is missing
            - Neither 'handler_type' nor 'descriptor.handler_kind' is present

    Example:
        >>> # Basic contract
        >>> contract = {
        ...     "name": "handler-consul",
        ...     "handler_class": "omnibase_infra.handlers.handler_consul.HandlerConsul",
        ...     "handler_type": "effect",
        ...     "tags": ["consul", "service-discovery"],
        ...     "security": {
        ...         "trusted_namespace": "omnibase_infra.handlers",
        ...         "audit_logging": True,
        ...     },
        ... }
        >>> config = extract_handler_config(contract, "consul")
        >>> config["name"]
        'handler-consul'
        >>> config["audit_logging"]
        True

        >>> # Rich contract with transport
        >>> rich_contract = {
        ...     "name": "MCP Handler",
        ...     "descriptor": {"handler_kind": "effect", "timeout_ms": 30000},
        ...     "metadata": {
        ...         "transport": {"default_host": "0.0.0.0", "default_port": 8090},
        ...         "security": {"tool_access": {"max_tools": 100}},
        ...     },
        ... }
        >>> config = extract_handler_config(rich_contract, "mcp")
        >>> config["port"]
        8090
        >>> config["max_tools"]
        100

        >>> # Missing required field raises error
        >>> bad_contract = {"handler_class": "some.Handler"}
        >>> extract_handler_config(bad_contract, "test")  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        ProtocolConfigurationError: Missing required field 'name' in handler contract
    """
    # Fail-fast validation for required fields
    if require_basic_fields:
        _validate_required_contract_fields(contract, handler_type)

    config: dict[str, JsonType] = {}

    # Extract top-level fields
    if "name" in contract:
        config["name"] = contract["name"]

    if "handler_class" in contract:
        config["handler_class"] = contract["handler_class"]

    # Handler kind can be in handler_type (basic) or descriptor.handler_kind (rich)
    if "handler_type" in contract:
        config["handler_kind"] = contract["handler_type"]

    if "tags" in contract:
        config["tags"] = contract["tags"]

    # Extract descriptor configuration (rich contracts)
    descriptor = contract.get("descriptor", {})
    if isinstance(descriptor, dict):
        # Handler kind from descriptor (rich contracts)
        if "handler_kind" in descriptor and "handler_kind" not in config:
            config["handler_kind"] = descriptor["handler_kind"]

        # Timeout configuration (convert ms to seconds)
        if "timeout_ms" in descriptor:
            timeout_ms = descriptor["timeout_ms"]
            if isinstance(timeout_ms, (int, float)):
                config["timeout_seconds"] = timeout_ms / 1000.0

    # Extract security configuration (basic contracts - top level)
    security = contract.get("security", {})
    if isinstance(security, dict):
        if "trusted_namespace" in security:
            config["trusted_namespace"] = security["trusted_namespace"]

        if "audit_logging" in security:
            config["audit_logging"] = security["audit_logging"]

        if "requires_authentication" in security:
            config["requires_authentication"] = security["requires_authentication"]

    # Extract metadata configuration (rich contracts)
    metadata = contract.get("metadata", {})
    if isinstance(metadata, dict):
        # Transport configuration
        transport = metadata.get("transport", {})
        if isinstance(transport, dict):
            if "default_host" in transport:
                config["host"] = transport["default_host"]

            if "default_port" in transport:
                config["port"] = transport["default_port"]

            if "default_path" in transport:
                config["path"] = transport["default_path"]

            if "stateless" in transport:
                config["stateless"] = transport["stateless"]

            if "json_response" in transport:
                config["json_response"] = transport["json_response"]

        # Security configuration (rich contracts - in metadata.security)
        metadata_security = metadata.get("security", {})
        if isinstance(metadata_security, dict):
            # Tool access configuration (for MCP)
            tool_access = metadata_security.get("tool_access", {})
            if isinstance(tool_access, dict):
                if "max_tools" in tool_access:
                    config["max_tools"] = tool_access["max_tools"]

    logger.debug(
        "Extracted handler config from contract",
        extra={
            "handler_type": handler_type,
            "config_keys": list(config.keys()),
        },
    )

    return config


__all__ = [
    "MAX_CONTRACT_SIZE_BYTES",
    "extract_handler_config",
    "load_handler_contract_config",
]
