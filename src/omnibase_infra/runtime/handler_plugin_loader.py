# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Plugin Loader for Contract-Driven Discovery.

This module provides HandlerPluginLoader, which discovers handler contracts
from the filesystem, validates handlers against protocols, and creates
ModelLoadedHandler instances for runtime registration.

Part of OMN-1132: Handler Plugin Loader implementation.

The loader implements ProtocolHandlerPluginLoader and supports:
- Single contract loading from a specific path
- Directory-based discovery with recursive scanning
- Glob pattern-based discovery for flexible matching

Error Codes:
    This module uses structured error codes from ``EnumHandlerLoaderError`` for
    precise error classification. Error codes are organized by category:

    **File-Level Errors (HANDLER_LOADER_001 - HANDLER_LOADER_009)**:
        - ``HANDLER_LOADER_001`` (FILE_NOT_FOUND): Contract file path does not exist
        - ``HANDLER_LOADER_002`` (INVALID_YAML_SYNTAX): Contract file has invalid YAML
        - ``HANDLER_LOADER_003`` (SCHEMA_VALIDATION_FAILED): Contract fails Pydantic validation
        - ``HANDLER_LOADER_004`` (MISSING_REQUIRED_FIELDS): Required contract fields missing
        - ``HANDLER_LOADER_005`` (FILE_SIZE_EXCEEDED): Contract exceeds 10MB size limit
        - ``HANDLER_LOADER_006`` (PROTOCOL_NOT_IMPLEMENTED): Handler missing protocol methods
        - ``HANDLER_LOADER_007`` (NOT_A_FILE): Path exists but is not a regular file
        - ``HANDLER_LOADER_008`` (FILE_READ_ERROR): Failed to read contract file (I/O)
        - ``HANDLER_LOADER_009`` (FILE_STAT_ERROR): Failed to stat contract file (I/O)

    **Import Errors (HANDLER_LOADER_010 - HANDLER_LOADER_013)**:
        - ``HANDLER_LOADER_010`` (MODULE_NOT_FOUND): Handler module not found
        - ``HANDLER_LOADER_011`` (CLASS_NOT_FOUND): Handler class not found in module
        - ``HANDLER_LOADER_012`` (IMPORT_ERROR): Import error (syntax/dependency)
        - ``HANDLER_LOADER_013`` (NAMESPACE_NOT_ALLOWED): Handler module namespace not allowed

    **Directory Errors (HANDLER_LOADER_020 - HANDLER_LOADER_022)**:
        - ``HANDLER_LOADER_020`` (DIRECTORY_NOT_FOUND): Directory does not exist
        - ``HANDLER_LOADER_021`` (PERMISSION_DENIED): Permission denied accessing directory
        - ``HANDLER_LOADER_022`` (NOT_A_DIRECTORY): Path exists but is not a directory

    **Pattern Errors (HANDLER_LOADER_030 - HANDLER_LOADER_031)**:
        - ``HANDLER_LOADER_030`` (EMPTY_PATTERNS_LIST): Patterns list cannot be empty
        - ``HANDLER_LOADER_031`` (INVALID_GLOB_PATTERN): Invalid glob pattern syntax
          (logged only, not raised - pattern is skipped and discovery continues)

    **Configuration Errors (HANDLER_LOADER_040)**:
        - ``HANDLER_LOADER_040`` (AMBIGUOUS_CONTRACT_CONFIGURATION): Both contract types
          exist in the same directory

    Error codes are accessible via ``error.model.context.get("loader_error")`` on
    raised exceptions. Note: HANDLER_LOADER_031 is logged but not raised as an
    exception to allow graceful continuation during discovery operations.

Concurrency Notes:
    The loader is stateless and reentrant - each load operation is independent:

    - No instance state is stored after ``__init__`` (empty constructor)
    - All method variables are local to each call
    - ``importlib.import_module()`` is thread-safe in CPython (uses GIL and import lock)
    - File operations use independent file handles per call

    **Thread Safety Guarantees**:
        - Multiple threads can safely call any loader method concurrently
        - Concurrent imports of the SAME module are serialized by Python's import lock
        - Repeated loads of the same handler are idempotent (cached by Python)

    **Thread Safety Limitations**:
        - The loader does NOT provide transactional semantics across multiple calls
        - If contracts change on disk during concurrent loading, results may be inconsistent
        - The ``discover_and_load()`` method's default ``Path.cwd()`` behavior is
          process-global; if cwd changes between calls, results will differ

    Working Directory Dependency:
        The ``discover_and_load()`` method uses ``Path.cwd()`` by default,
        which reads process-level state. For deterministic behavior when cwd
        may change between calls, provide an explicit ``base_path`` parameter.

See Also:
    - ProtocolHandlerPluginLoader: Protocol definition for plugin loaders
    - HandlerContractSource: Contract discovery and parsing
    - ModelLoadedHandler: Model representing loaded handler metadata
    - EnumHandlerLoaderError: Structured error codes for loader operations

Security Considerations:
    This loader dynamically imports Python classes specified in YAML contracts.
    Contract files should be treated as code and protected accordingly:
    - Only load contracts from trusted sources
    - Validate contract file permissions in production environments
    - Be aware that module side effects execute during import
    - Use the ``allowed_namespaces`` parameter to restrict imports to trusted packages

    Namespace Allowlisting:
        The loader supports namespace-based import restrictions via the
        ``allowed_namespaces`` parameter. When configured, only handler modules
        whose fully-qualified path starts with one of the allowed namespace
        prefixes will be imported. This provides defense-in-depth against
        malicious contract files attempting to load untrusted code.

        Example:
            >>> loader = HandlerPluginLoader(
            ...     allowed_namespaces=["omnibase_infra.", "omnibase_core.", "mycompany."]
            ... )
            >>> # This would succeed:
            >>> loader.load_from_contract(Path("contract.yaml"))  # handler_class: omnibase_infra.handlers.HandlerAuth
            >>> # This would fail with NAMESPACE_NOT_ALLOWED:
            >>> loader.load_from_contract(Path("malicious.yaml"))  # handler_class: malicious_pkg.EvilHandler

    **Error Message Sanitization**:
        Error messages are designed to be safe for end users and prevent
        information disclosure:

        - User-facing error messages use filename only (not full filesystem paths)
        - System exception details are sanitized to prevent path disclosure
        - Full paths are stored in error context for internal debugging only
        - Correlation IDs enable tracing without exposing sensitive information
        - Exception messages from underlying libraries are sanitized before inclusion

        The ``_sanitize_exception_message()`` helper strips filesystem paths from
        exception messages while preserving useful diagnostic information like
        line numbers and error types.

.. versionadded:: 0.7.0
    Created as part of OMN-1132 handler plugin loader implementation.
"""

from __future__ import annotations

import importlib
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerLoaderError, EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ProtocolConfigurationError
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.models.runtime import (
    ModelFailedPluginLoad,
    ModelHandlerContract,
    ModelLoadedHandler,
    ModelPluginLoadContext,
    ModelPluginLoadSummary,
)
from omnibase_infra.runtime.protocol_handler_plugin_loader import (
    ProtocolHandlerPluginLoader,
)

logger = logging.getLogger(__name__)

# Regex pattern for detecting filesystem paths in error messages
# Matches Unix paths (/path/to/file) and Windows paths (C:\path\to\file)
_PATH_PATTERN = re.compile(
    r"""
    (?:                             # Non-capturing group for path types
        /(?:[\w.-]+/)+[\w.-]+       # Unix absolute path: /path/to/file
        |
        [A-Za-z]:\\(?:[\w.-]+\\)*[\w.-]+  # Windows path: C:\path\to\file
        |
        \.\.?/(?:[\w.-]+/)*[\w.-]*  # Relative path: ./path or ../path
    )
    """,
    re.VERBOSE,
)


def _sanitize_exception_message(exception: BaseException) -> str:
    """Sanitize exception message to prevent information disclosure.

    Removes or masks filesystem paths from exception messages to prevent
    exposing internal directory structures in user-facing error messages.
    Preserves useful diagnostic information like line numbers and error types.

    Args:
        exception: The exception whose message should be sanitized.

    Returns:
        A sanitized version of the exception message with paths removed.

    Example:
        >>> e = OSError("[Errno 13] Permission denied: '/etc/secrets/key.pem'")
        >>> _sanitize_exception_message(e)
        "[Errno 13] Permission denied: '<path>'"

        >>> e = yaml.YAMLError("expected ... in '/home/user/config.yaml', line 10")
        >>> _sanitize_exception_message(e)
        "expected ... in '<path>', line 10"
    """
    message = str(exception)

    # Replace filesystem paths with <path> placeholder
    sanitized = _PATH_PATTERN.sub("<path>", message)

    # Also handle quoted paths that might have been missed
    # Pattern: 'path/to/file' or "path/to/file"
    sanitized = re.sub(r"['\"](?:[^'\"]*[/\\][^'\"]+)['\"]", "'<path>'", sanitized)

    return sanitized


# File pattern for handler contracts
HANDLER_CONTRACT_FILENAME = "handler_contract.yaml"
CONTRACT_YAML_FILENAME = "contract.yaml"

# Maximum contract file size (10MB) to prevent memory exhaustion
MAX_CONTRACT_SIZE = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Correlation ID Design Decision: UUID Type
# ---------------------------------------------------------------------------
# The correlation_id parameter is typed as `UUID | None` to comply with ONEX
# standards requiring typed models rather than primitives. This aligns with:
#
# 1. **ONEX Protocol Conventions**: All other protocols in the codebase use
#    `correlation_id: UUID | None` (see ProtocolIdempotencyStore,
#    ProtocolServiceDiscoveryHandler, etc.).
#
# 2. **Type Safety**: UUID type ensures valid correlation IDs at compile time.
#
# 3. **Auto-Generation Pattern**: Methods auto-generate correlation IDs via
#    `uuid4()` when not provided, ensuring all operations are traceable.
#
# For external system compatibility (OpenTelemetry, Zipkin, etc.), convert
# string-based correlation IDs to UUID at the call boundary, or use
# uuid.UUID(external_id) if the external ID is UUID-compatible.
#
# See: ONEX correlation ID conventions in omnibase_core.
# ---------------------------------------------------------------------------


class HandlerPluginLoader(ProtocolHandlerPluginLoader):
    """Load handlers as plugins from contracts.

    Discovers handler contracts, validates handlers against protocols,
    and registers them with the handler registry.

    This class implements ProtocolHandlerPluginLoader by scanning filesystem
    paths for handler_contract.yaml or contract.yaml files, parsing them,
    dynamically importing the handler classes, and creating ModelLoadedHandler
    instances.

    Protocol Compliance:
        This class explicitly implements ProtocolHandlerPluginLoader and provides
        all required methods: load_from_contract(), load_from_directory(), and
        discover_and_load(). Protocol compliance is verified via duck typing.

    Example:
        >>> # Load a single handler from contract
        >>> loader = HandlerPluginLoader()
        >>> handler = loader.load_from_contract(
        ...     Path("src/handlers/auth/handler_contract.yaml")
        ... )
        >>> print(f"Loaded: {handler.handler_name}")

        >>> # Load all handlers from a directory
        >>> handlers = loader.load_from_directory(Path("src/handlers"))
        >>> print(f"Loaded {len(handlers)} handlers")

        >>> # Discover with glob patterns
        >>> handlers = loader.discover_and_load([
        ...     "src/**/handler_contract.yaml",
        ...     "plugins/**/contract.yaml",
        ... ])

    .. versionadded:: 0.7.0
        Created as part of OMN-1132 handler plugin loader implementation.
    """

    def __init__(self, allowed_namespaces: list[str] | None = None) -> None:
        """Initialize the handler plugin loader.

        Args:
            allowed_namespaces: Optional list of allowed namespace prefixes for
                handler module imports. When provided, only handler modules whose
                fully-qualified class path starts with one of these prefixes will
                be loaded. This provides defense-in-depth security by restricting
                which packages can be dynamically imported.

                Each prefix should end with a period for explicit package boundary
                matching (e.g., "omnibase_infra." not "omnibase_infra"). Prefixes
                without a trailing period are validated at package boundaries to
                prevent unintended matches (e.g., "omnibase" matches "omnibase" or
                "omnibase.handlers" but NOT "omnibase_other").

                If None (default), no namespace restriction is applied and any
                importable module can be loaded.

                If an empty list is provided, NO namespaces are allowed, effectively
                blocking all handler imports.

        Example:
            >>> # Restrict to trusted packages
            >>> loader = HandlerPluginLoader(
            ...     allowed_namespaces=["omnibase_infra.", "omnibase_core.", "mycompany.handlers."]
            ... )
            >>>
            >>> # No restriction (default)
            >>> loader = HandlerPluginLoader()
            >>>
            >>> # Block all imports (empty list)
            >>> loader = HandlerPluginLoader(allowed_namespaces=[])

        Security Note:
            Namespace validation occurs BEFORE ``importlib.import_module()`` is
            called, preventing any module-level side effects from untrusted packages.
        """
        self._allowed_namespaces: list[str] | None = allowed_namespaces

        # Security best practice: warn if no namespace restriction is configured
        if allowed_namespaces is None:
            logger.info(
                "HandlerPluginLoader initialized without namespace restrictions. "
                "For production environments, consider setting allowed_namespaces to "
                "restrict handler imports to trusted packages (e.g., "
                "allowed_namespaces=['omnibase_infra.', 'omnibase_core.', 'mycompany.']).",
            )
        # Warn if empty list is provided - this blocks ALL handler imports
        elif len(allowed_namespaces) == 0:
            logger.warning(
                "HandlerPluginLoader initialized with empty allowed_namespaces list. "
                "This will block ALL handler imports. If this is intentional, ignore "
                "this warning. Otherwise, set allowed_namespaces=None to allow all "
                "namespaces or provide a list of allowed namespace prefixes.",
            )

    def _validate_correlation_id(
        self,
        correlation_id: UUID | None,
        operation: str,
    ) -> None:
        """Validate correlation_id is a UUID or None at runtime.

        Provides runtime type validation for correlation_id parameters at public
        API entry points. While type hints provide static checking, runtime
        validation catches cases where callers bypass type checking (e.g.,
        dynamically constructed calls, JSON deserialization without validation).

        Args:
            correlation_id: The correlation ID to validate. Must be UUID or None.
            operation: Name of the calling operation (for error context).

        Raises:
            ProtocolConfigurationError: If correlation_id is not a UUID instance
                or None. The error message includes the actual type received and
                the operation name for debugging, along with guidance on how to
                convert string IDs to UUID.

        Example:
            >>> loader = HandlerPluginLoader()
            >>> loader._validate_correlation_id(UUID("..."), "load_from_contract")  # OK
            >>> loader._validate_correlation_id(None, "load_from_contract")  # OK
            >>> loader._validate_correlation_id("not-a-uuid", "load_from_contract")
            ProtocolConfigurationError: correlation_id must be UUID or None...
        """
        if correlation_id is not None and not isinstance(correlation_id, UUID):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation=operation,
                correlation_id=None,  # Cannot use invalid correlation_id in context
            )
            raise ProtocolConfigurationError(
                f"correlation_id must be UUID or None, got {type(correlation_id).__name__} "
                f"in {operation}(). Convert string IDs using uuid.UUID(your_string).",
                context=context,
                loader_error="INVALID_CORRELATION_ID_TYPE",
                received_type=type(correlation_id).__name__,
            )

    def load_from_contract(
        self,
        contract_path: Path,
        correlation_id: UUID | None = None,
    ) -> ModelLoadedHandler:
        """Load a single handler from a contract file.

        Parses the contract YAML file at the given path, validates it,
        imports the handler class, validates protocol compliance, and
        returns a ModelLoadedHandler with the loaded metadata.

        Args:
            contract_path: Path to the handler contract YAML file.
                Must be an absolute or relative path to an existing file.
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations have traceable correlation IDs.

        Returns:
            ModelLoadedHandler containing the loaded handler metadata
            including handler class, version, and contract information.

        Raises:
            ProtocolConfigurationError: If the contract file is invalid,
                missing required fields, or fails validation. Error codes:
                - HANDLER_LOADER_001: Contract file not found (path doesn't exist)
                - HANDLER_LOADER_002: Invalid YAML syntax
                - HANDLER_LOADER_003: Schema validation failed
                - HANDLER_LOADER_004: Missing required fields
                - HANDLER_LOADER_005: Contract file exceeds size limit
                - HANDLER_LOADER_006: Handler does not implement protocol
                - HANDLER_LOADER_007: Path exists but is not a file (e.g., directory)
                - HANDLER_LOADER_008: Failed to read contract file (I/O error)
                - HANDLER_LOADER_009: Failed to stat contract file (I/O error)
            ProtocolConfigurationError: If namespace validation fails (when
                allowed_namespaces is configured).
                - HANDLER_LOADER_013: Namespace not allowed
            InfraConnectionError: If the handler class cannot be imported.
                Error codes:
                - HANDLER_LOADER_010: Module not found
                - HANDLER_LOADER_011: Class not found in module
                - HANDLER_LOADER_012: Import error (syntax/dependency)
            ProtocolConfigurationError: If correlation_id is not a UUID or None.
                Error code: INVALID_CORRELATION_ID_TYPE
        """
        # Validate correlation_id type at entry point (runtime type check)
        self._validate_correlation_id(correlation_id, "load_from_contract")

        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or uuid4()

        # Convert UUID to string for logging and error context
        correlation_id_str = str(correlation_id)

        # Start timing for performance observability
        start_time = time.perf_counter()

        logger.debug(
            "Loading handler from contract: %s",
            contract_path,
            extra={
                "contract_path": str(contract_path),
                "correlation_id": correlation_id_str,
            },
        )

        # Validate contract path exists
        # contract_path.exists() and contract_path.is_file() can raise OSError for:
        # - Permission denied when accessing the path
        # - Filesystem errors (unmounted volumes, network failures)
        # - Broken symlinks where the target cannot be resolved
        try:
            path_exists = contract_path.exists()
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            sanitized_msg = _sanitize_exception_message(e)
            raise ProtocolConfigurationError(
                f"Failed to access contract path: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                contract_path=str(contract_path),
            ) from e

        if not path_exists:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Contract file not found: {contract_path.name}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_NOT_FOUND.value,
                contract_path=str(contract_path),
            )

        try:
            is_file = contract_path.is_file()
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            sanitized_msg = _sanitize_exception_message(e)
            raise ProtocolConfigurationError(
                f"Failed to access contract path: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                contract_path=str(contract_path),
            ) from e

        if not is_file:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Contract path is not a file: {contract_path.name}",
                context=context,
                loader_error=EnumHandlerLoaderError.NOT_A_FILE.value,
                contract_path=str(contract_path),
            )

        # Validate file size (raises ProtocolConfigurationError on failure)
        self._validate_file_size(
            contract_path,
            correlation_id=correlation_id,
            operation="load_from_contract",
            raise_on_error=True,
        )

        # Parse YAML contract
        try:
            with contract_path.open("r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            raise ProtocolConfigurationError(
                f"Invalid YAML syntax in contract: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value,
                contract_path=str(contract_path),
            ) from e
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            raise ProtocolConfigurationError(
                f"Failed to read contract file: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_READ_ERROR.value,
                contract_path=str(contract_path),
            ) from e

        if raw_data is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Contract file is empty",
                context=context,
                loader_error=EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value,
                contract_path=str(contract_path),
            )

        # Validate contract using Pydantic model
        try:
            contract = ModelHandlerContract.model_validate(raw_data)
        except ValidationError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            # Convert validation errors to readable message
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            raise ProtocolConfigurationError(
                f"Contract validation failed: {error_details}",
                context=context,
                loader_error=EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value,
                contract_path=str(contract_path),
                validation_errors=[
                    {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
                    for err in e.errors()
                ],
            ) from e

        handler_name = contract.handler_name
        handler_class_path = contract.handler_class
        handler_type = contract.handler_type
        capability_tags = contract.capability_tags
        # protocol_type is the registry key (e.g., "db", "http")
        # The model_validator in ModelHandlerContract sets this from handler_name
        # if not explicitly provided (strips "handler-" prefix)
        protocol_type = contract.protocol_type
        # Should never be None after model_validator, but assert for type safety
        assert protocol_type is not None, (
            "protocol_type should be set by model_validator"
        )

        # Import and validate handler class
        handler_class = self._import_handler_class(
            handler_class_path, contract_path, correlation_id
        )

        # Validate handler implements protocol
        is_valid, missing_methods = self._validate_handler_protocol(handler_class)
        if not is_valid:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            missing_str = ", ".join(missing_methods)
            raise ProtocolConfigurationError(
                f"Handler class {handler_class_path} is missing required "
                f"ProtocolHandler methods: {missing_str}",
                context=context,
                loader_error=EnumHandlerLoaderError.PROTOCOL_NOT_IMPLEMENTED.value,
                contract_path=str(contract_path),
                handler_class=handler_class_path,
                missing_methods=missing_methods,
            )

        # Resolve the contract path, handling potential filesystem errors
        # path.resolve() can raise OSError for:
        # - Broken symlinks: symlink target no longer exists
        # - Race conditions: file deleted between validation and resolution
        # - Permission issues: lacking read permission on parent directories
        # - Filesystem errors: unmounted volumes, network filesystem failures
        try:
            resolved_contract_path = contract_path.resolve()
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
                correlation_id=correlation_id,
            )
            sanitized_msg = _sanitize_exception_message(e)
            # FILE_STAT_ERROR is used here because path resolution involves
            # filesystem metadata access similar to stat operations
            raise ProtocolConfigurationError(
                f"Failed to access contract file during path resolution: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                contract_path=str(contract_path),
            ) from e

        # Calculate load duration for performance observability
        load_duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Successfully loaded handler from contract: %s -> %s (%.2fms)",
            handler_name,
            handler_class_path,
            load_duration_ms,
            extra={
                "handler_name": handler_name,
                "handler_class": handler_class_path,
                "handler_type": handler_type.value,
                "protocol_type": protocol_type,
                "contract_path": str(resolved_contract_path),
                "correlation_id": correlation_id_str,
                "load_duration_ms": load_duration_ms,
            },
        )

        # contract.handler_version is guaranteed non-None by model_validator
        if contract.handler_version is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_contract",
            )
            raise ProtocolConfigurationError(
                "handler_version should be set by model_validator",
                context=context,
                loader_error=EnumHandlerLoaderError.MISSING_REQUIRED_FIELDS.value,
                contract_path=str(contract_path),
            )

        return ModelLoadedHandler(
            handler_name=handler_name,
            protocol_type=protocol_type,
            handler_type=handler_type,
            handler_class=handler_class_path,
            contract_path=resolved_contract_path,
            capability_tags=capability_tags,
            loaded_at=datetime.now(UTC),
            handler_version=contract.handler_version,
        )

    def load_from_directory(
        self,
        directory: Path,
        correlation_id: UUID | None = None,
        max_handlers: int | None = None,
    ) -> list[ModelLoadedHandler]:
        """Load all handlers from contract files in a directory.

        Recursively scans the given directory for handler contract files
        (handler_contract.yaml or contract.yaml), loads each handler,
        and returns a list of successfully loaded handlers.

        Failed loads are logged but do not stop processing of other handlers.
        A summary is logged at the end of the operation for observability.

        Args:
            directory: Path to the directory to scan for contract files.
                Must be an existing directory.
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations have traceable correlation IDs. The same correlation_id
                is propagated to all contract loads within the directory scan.
            max_handlers: Optional maximum number of handlers to discover and load.
                If specified, discovery stops after finding this many contract files.
                A warning is logged when the limit is reached. Set to None (default)
                for unlimited discovery. This prevents runaway resource usage when
                scanning directories with unexpectedly large numbers of handlers.

        Returns:
            List of successfully loaded handlers. May be empty if no
            contracts are found or all fail validation.

        Raises:
            ProtocolConfigurationError: If the directory does not exist
                or is not accessible. Error codes:
                - HANDLER_LOADER_020: Directory not found
                - HANDLER_LOADER_021: Permission denied
                - HANDLER_LOADER_022: Not a directory
            ProtocolConfigurationError: If correlation_id is not a UUID or None.
                Error code: INVALID_CORRELATION_ID_TYPE
        """
        # Validate correlation_id type at entry point (runtime type check)
        self._validate_correlation_id(correlation_id, "load_from_directory")

        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or uuid4()

        # Start timing for observability
        start_time = time.perf_counter()

        logger.debug(
            "Loading handlers from directory: %s",
            directory,
            extra={
                "directory": str(directory),
                "correlation_id": str(correlation_id),
                "max_handlers": max_handlers,
            },
        )

        # Validate directory exists
        # directory.exists() and directory.is_dir() can raise OSError for:
        # - Permission denied when accessing the path
        # - Filesystem errors (unmounted volumes, network failures)
        # - Broken symlinks where the target cannot be resolved
        try:
            dir_exists = directory.exists()
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
                correlation_id=correlation_id,
            )
            sanitized_msg = _sanitize_exception_message(e)
            raise ProtocolConfigurationError(
                f"Failed to access directory: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.PERMISSION_DENIED.value,
                directory=str(directory),
            ) from e

        if not dir_exists:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Directory not found: {directory.name}",
                context=context,
                loader_error=EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.value,
                directory=str(directory),
            )

        try:
            is_directory = directory.is_dir()
        except OSError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
                correlation_id=correlation_id,
            )
            sanitized_msg = _sanitize_exception_message(e)
            raise ProtocolConfigurationError(
                f"Failed to access directory: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.PERMISSION_DENIED.value,
                directory=str(directory),
            ) from e

        if not is_directory:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_directory",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Path is not a directory: {directory.name}",
                context=context,
                loader_error=EnumHandlerLoaderError.NOT_A_DIRECTORY.value,
                directory=str(directory),
            )

        # Find all contract files (with optional limit)
        contract_files = self._find_contract_files(
            directory, correlation_id, max_handlers=max_handlers
        )

        logger.debug(
            "Found %d contract files in directory: %s",
            len(contract_files),
            directory,
            extra={
                "directory": str(directory),
                "contract_count": len(contract_files),
                "correlation_id": str(correlation_id),
            },
        )

        # Load each contract (graceful mode - continue on errors)
        handlers: list[ModelLoadedHandler] = []
        failed_handlers: list[ModelFailedPluginLoad] = []

        for contract_path in contract_files:
            try:
                handler = self.load_from_contract(contract_path, correlation_id)
                handlers.append(handler)
            except (ProtocolConfigurationError, InfraConnectionError) as e:
                # Extract error code if available
                error_code: str | None = None
                if hasattr(e, "model") and hasattr(e.model, "context"):
                    loader_error = e.model.context.get("loader_error")
                    error_code = str(loader_error) if loader_error is not None else None

                failed_handlers.append(
                    ModelFailedPluginLoad(
                        contract_path=contract_path,
                        error_message=str(e),
                        error_code=error_code,
                    )
                )

                logger.warning(
                    "Failed to load handler from %s: %s",
                    contract_path,
                    str(e),
                    extra={
                        "contract_path": str(contract_path),
                        "error": str(e),
                        "error_code": error_code,
                        "correlation_id": str(correlation_id),
                    },
                )
                continue

        # Calculate duration and log summary
        duration_seconds = time.perf_counter() - start_time

        self._log_load_summary(
            ModelPluginLoadContext(
                operation="load_from_directory",
                source=str(directory),
                total_discovered=len(contract_files),
                handlers=handlers,
                failed_plugins=failed_handlers,
                duration_seconds=duration_seconds,
                correlation_id=correlation_id,
                caller_correlation_string=str(correlation_id),
            )
        )

        return handlers

    def discover_and_load(
        self,
        patterns: list[str],
        correlation_id: UUID | None = None,
        base_path: Path | None = None,
        max_handlers: int | None = None,
    ) -> list[ModelLoadedHandler]:
        """Discover contracts matching glob patterns and load handlers.

        Searches for contract files matching the given glob patterns,
        deduplicates matches, loads each handler, and returns a list
        of successfully loaded handlers.

        A summary is logged at the end of the operation for observability.

        Working Directory Dependency:
            By default, glob patterns are resolved relative to the current
            working directory (``Path.cwd()``). This means results may vary
            if the working directory changes between calls. For deterministic
            behavior in environments where cwd may change (e.g., tests,
            multi-threaded applications), provide an explicit ``base_path``
            parameter.

        Args:
            patterns: List of glob patterns to match contract files.
                Supports standard glob syntax including ** for recursive.
            correlation_id: Optional correlation ID for tracing and error context.
                If not provided, a new UUID4 is auto-generated to ensure all
                operations have traceable correlation IDs. The same correlation_id
                is propagated to all discovered contract loads.
            base_path: Optional base path for resolving glob patterns.
                If not provided, defaults to ``Path.cwd()``. Providing an
                explicit base path ensures deterministic behavior regardless
                of the current working directory.
            max_handlers: Optional maximum number of handlers to discover and load.
                If specified, discovery stops after finding this many contract files.
                A warning is logged when the limit is reached. Set to None (default)
                for unlimited discovery. This prevents runaway resource usage when
                scanning directories with unexpectedly large numbers of handlers.

        Returns:
            List of successfully loaded handlers. May be empty if no
            patterns match or all fail validation.

        Raises:
            ProtocolConfigurationError: If patterns list is empty.
                Error codes:
                - HANDLER_LOADER_030: Empty patterns list
            ProtocolConfigurationError: If correlation_id is not a UUID or None.
                Error code: INVALID_CORRELATION_ID_TYPE

        Example:
            >>> # Using default cwd-based resolution
            >>> handlers = loader.discover_and_load(["src/**/handler_contract.yaml"])
            >>>
            >>> # Using explicit base path for deterministic behavior
            >>> handlers = loader.discover_and_load(
            ...     ["src/**/handler_contract.yaml"],
            ...     base_path=Path("/app/project"),
            ... )
        """
        # Validate correlation_id type at entry point (runtime type check)
        self._validate_correlation_id(correlation_id, "discover_and_load")

        # Auto-generate correlation_id if not provided (per ONEX guidelines)
        correlation_id = correlation_id or uuid4()

        # Start timing for observability
        start_time = time.perf_counter()

        logger.debug(
            "Discovering handlers with patterns: %s",
            patterns,
            extra={
                "patterns": patterns,
                "correlation_id": str(correlation_id),
                "max_handlers": max_handlers,
            },
        )

        if not patterns:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="discover_and_load",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Patterns list cannot be empty",
                context=context,
                loader_error=EnumHandlerLoaderError.EMPTY_PATTERNS_LIST.value,
            )

        # Collect all matching contract files, deduplicated by resolved path
        discovered_paths: set[Path] = set()
        limit_reached = False

        # Use explicit base_path if provided, otherwise fall back to cwd
        # Note: Using cwd can produce different results if the working directory
        # changes between calls. For deterministic behavior, provide base_path.
        # Path.cwd() can raise OSError if:
        # - Current working directory has been deleted
        # - Permission denied accessing current directory
        if base_path is not None:
            glob_base = base_path
        else:
            try:
                glob_base = Path.cwd()
            except OSError as e:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="discover_and_load",
                    correlation_id=correlation_id,
                )
                sanitized_msg = _sanitize_exception_message(e)
                raise ProtocolConfigurationError(
                    f"Failed to access current working directory: {sanitized_msg}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.PERMISSION_DENIED.value,
                ) from e

        for pattern in patterns:
            if limit_reached:
                break

            # Path.glob() can raise:
            # - OSError: Permission denied, filesystem errors, invalid path components
            # - ValueError: Invalid glob pattern syntax (e.g., ** not at path segment boundary)
            try:
                matched_paths = list(glob_base.glob(pattern))
            except ValueError as e:
                # ValueError indicates invalid glob pattern syntax
                # Example: "foo**bar" - ** must be at path segment boundaries
                sanitized_msg = _sanitize_exception_message(e)
                logger.warning(
                    "Invalid glob pattern syntax '%s': %s",
                    pattern,
                    sanitized_msg,
                    extra={
                        "pattern": pattern,
                        "base_path": str(glob_base),
                        "error": sanitized_msg,
                        "error_code": EnumHandlerLoaderError.INVALID_GLOB_PATTERN.value,
                        "correlation_id": str(correlation_id),
                    },
                )
                continue  # Skip to next pattern
            except OSError as e:
                sanitized_msg = _sanitize_exception_message(e)
                logger.warning(
                    "Failed to evaluate glob pattern '%s': %s",
                    pattern,
                    sanitized_msg,
                    extra={
                        "pattern": pattern,
                        "base_path": str(glob_base),
                        "error": sanitized_msg,
                        "correlation_id": str(correlation_id),
                    },
                )
                continue  # Skip to next pattern

            for path in matched_paths:
                # Check if we've reached the limit
                if max_handlers is not None and len(discovered_paths) >= max_handlers:
                    limit_reached = True
                    logger.warning(
                        "Handler discovery limit reached: stopped after discovering %d "
                        "handlers (max_handlers=%d). Some handlers may not be loaded.",
                        len(discovered_paths),
                        max_handlers,
                        extra={
                            "discovered_count": len(discovered_paths),
                            "max_handlers": max_handlers,
                            "patterns": patterns,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    break

                # path.is_file() can raise OSError for:
                # - Permission denied when stat'ing the file
                # - File deleted between glob discovery and is_file() check
                # - Filesystem errors (unmounted volumes, network failures)
                try:
                    is_file = path.is_file()
                except OSError as e:
                    sanitized_msg = _sanitize_exception_message(e)
                    logger.warning(
                        "Failed to check if path is file %s: %s",
                        path.name,
                        sanitized_msg,
                        extra={
                            "path": str(path),
                            "error": sanitized_msg,
                            "error_code": EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    continue

                if is_file:
                    # Early size validation to skip oversized files before expensive operations
                    if (
                        self._validate_file_size(
                            path, correlation_id=correlation_id, raise_on_error=False
                        )
                        is None
                    ):
                        continue

                    # Early YAML syntax validation to fail fast before expensive resolve operations
                    # This catches malformed YAML immediately after discovery rather than
                    # deferring to load_from_contract, which is more efficient for batch discovery
                    if not self._validate_yaml_syntax(
                        path, correlation_id=correlation_id, raise_on_error=False
                    ):
                        continue

                    # path.resolve() can raise OSError for:
                    # - Broken symlinks: symlink target no longer exists
                    # - Race conditions: file deleted between glob discovery and resolution
                    # - Permission issues: lacking read permission on parent directories
                    # - Filesystem errors: unmounted volumes, network filesystem failures
                    try:
                        resolved = path.resolve()
                    except OSError as e:
                        sanitized_msg = _sanitize_exception_message(e)
                        logger.warning(
                            "Failed to resolve path %s: %s",
                            path.name,
                            sanitized_msg,
                            extra={
                                "path": str(path),
                                "error": sanitized_msg,
                                "correlation_id": str(correlation_id),
                            },
                        )
                        continue  # Skip to next path

                    discovered_paths.add(resolved)

        logger.debug(
            "Discovered %d unique contract files from %d patterns",
            len(discovered_paths),
            len(patterns),
            extra={
                "patterns": patterns,
                "discovered_count": len(discovered_paths),
                "limit_reached": limit_reached,
                "correlation_id": str(correlation_id),
            },
        )

        # Load each discovered contract (graceful mode)
        handlers: list[ModelLoadedHandler] = []
        failed_handlers: list[ModelFailedPluginLoad] = []

        for contract_path in sorted(discovered_paths):
            try:
                handler = self.load_from_contract(contract_path, correlation_id)
                handlers.append(handler)
            except (ProtocolConfigurationError, InfraConnectionError) as e:
                # Extract error code if available
                error_code: str | None = None
                if hasattr(e, "model") and hasattr(e.model, "context"):
                    loader_error = e.model.context.get("loader_error")
                    error_code = str(loader_error) if loader_error is not None else None

                failed_handlers.append(
                    ModelFailedPluginLoad(
                        contract_path=contract_path,
                        error_message=str(e),
                        error_code=error_code,
                    )
                )

                logger.warning(
                    "Failed to load handler from %s: %s",
                    contract_path,
                    str(e),
                    extra={
                        "contract_path": str(contract_path),
                        "error": str(e),
                        "error_code": error_code,
                        "correlation_id": str(correlation_id),
                    },
                )
                continue

        # Calculate duration and log summary
        duration_seconds = time.perf_counter() - start_time

        # Format patterns as comma-separated string for source
        patterns_str = ", ".join(patterns)

        self._log_load_summary(
            ModelPluginLoadContext(
                operation="discover_and_load",
                source=patterns_str,
                total_discovered=len(discovered_paths),
                handlers=handlers,
                failed_plugins=failed_handlers,
                duration_seconds=duration_seconds,
                correlation_id=correlation_id,
                caller_correlation_string=str(correlation_id),
            )
        )

        return handlers

    def _log_load_summary(
        self,
        context: ModelPluginLoadContext,
    ) -> ModelPluginLoadSummary:
        """Log a summary of the handler loading operation for observability.

        Creates a structured summary of the load operation and logs it at
        an appropriate level (INFO for success, WARNING if there were failures).

        The log message format is designed for easy parsing:
        - Single line summary with counts and timing
        - Detailed handler list with class names and modules
        - Failed handler details with error reasons

        Args:
            context: The load context containing operation details, handlers,
                failures, and timing information.

        Returns:
            ModelPluginLoadSummary containing the structured summary data.

        Example log output:
            Handler load complete: 5 handlers loaded in 0.23s (source: /app/handlers)
              - HandlerAuth (myapp.handlers.auth)
              - HandlerDb (myapp.handlers.db)
              ...
        """
        # Build list of loaded handler details
        loaded_handler_details = [
            {
                "name": h.handler_name,
                "class": h.handler_class.rsplit(".", 1)[-1],
                "module": h.handler_class.rsplit(".", 1)[0],
            }
            for h in context.handlers
        ]

        # Create summary model
        summary = ModelPluginLoadSummary(
            operation=context.operation,
            source=context.source,
            total_discovered=context.total_discovered,
            total_loaded=len(context.handlers),
            total_failed=len(context.failed_plugins),
            loaded_plugins=loaded_handler_details,
            failed_plugins=context.failed_plugins,
            duration_seconds=context.duration_seconds,
            correlation_id=context.correlation_id,
            completed_at=datetime.now(UTC),
        )

        # Build log message with handler details
        handler_lines = [
            f"  - {h['class']} ({h['module']})" for h in loaded_handler_details
        ]
        handler_list_str = "\n".join(handler_lines) if handler_lines else "  (none)"

        # Build failed handler message if any
        failed_lines = []
        for failed in context.failed_plugins:
            error_code_str = f" [{failed.error_code}]" if failed.error_code else ""
            failed_lines.append(f"  - {failed.contract_path}{error_code_str}")

        failed_list_str = "\n".join(failed_lines) if failed_lines else ""

        # Choose log level based on whether there were failures
        if context.failed_plugins:
            log_level = logging.WARNING
            status = "with failures"
        else:
            log_level = logging.INFO
            status = "successfully"

        # Format duration for readability
        if context.duration_seconds < 0.001:
            duration_str = f"{context.duration_seconds * 1000000:.0f}us"
        elif context.duration_seconds < 1.0:
            duration_str = f"{context.duration_seconds * 1000:.2f}ms"
        else:
            duration_str = f"{context.duration_seconds:.2f}s"

        # Log the summary
        summary_msg = (
            f"Handler load complete {status}: "
            f"{len(context.handlers)} handlers loaded in {duration_str}"
        )
        if context.failed_plugins:
            summary_msg += f" ({len(context.failed_plugins)} failed)"

        # Build detailed message
        detailed_msg = f"{summary_msg}\nLoaded handlers:\n{handler_list_str}"
        if failed_list_str:
            detailed_msg += f"\nFailed handlers:\n{failed_list_str}"

        logger.log(
            log_level,
            detailed_msg,
            extra={
                "operation": context.operation,
                "source": context.source,
                "total_discovered": context.total_discovered,
                "total_loaded": len(context.handlers),
                "total_failed": len(context.failed_plugins),
                "duration_seconds": context.duration_seconds,
                "correlation_id": context.caller_correlation_string,
                "handler_names": [h.handler_name for h in context.handlers],
                "handler_classes": [h.handler_class for h in context.handlers],
                "failed_paths": [str(f.contract_path) for f in context.failed_plugins],
            },
        )

        return summary

    def _validate_handler_protocol(self, handler_class: type) -> tuple[bool, list[str]]:
        """Validate handler implements required protocol (ProtocolHandler).

        Uses duck typing to verify the handler class has the required
        methods for ProtocolHandler compliance. Per ONEX conventions, protocol
        compliance is verified via structural typing (duck typing) rather than
        isinstance checks or explicit inheritance.

        Protocol Requirements (from omnibase_spi.protocols.handlers.protocol_handler):
            The ProtocolHandler protocol defines the following required members:

            **Required Methods (validated)**:
                - ``handler_type`` (property): Returns handler type identifier string
                - ``initialize(config)``: Async method to initialize connections/pools
                - ``shutdown(timeout_seconds)``: Async method to release resources
                - ``execute(request, operation_config)``: Async method for operations
                - ``describe()``: Sync method returning handler metadata/capabilities

            **Optional Methods (not validated)**:
                - ``health_check()``: Async method for connectivity verification.
                  While part of the ProtocolHandler protocol, this method is not
                  validated because existing handler implementations (HandlerHttp,
                  HandlerDb, HandlerVault, HandlerConsul) do not implement it.
                  Future handler implementations SHOULD include health_check().

        Validation Approach:
            This method checks for the presence and callability of all 5 required
            methods. A handler class must have ALL of these methods to pass validation.
            This prevents false positives where a class might have only ``describe()``
            but lack other essential handler functionality.

            The validation uses ``callable(getattr(...))`` for methods and
            ``hasattr()`` for the ``handler_type`` property to accommodate both
            instance properties and class-level descriptors.

        Why Duck Typing:
            ONEX uses duck typing for protocol validation to:
            1. Avoid tight coupling to specific base classes
            2. Enable flexibility in handler implementation strategies
            3. Support mixin-based handler composition
            4. Allow testing with mock handlers that satisfy the protocol

        Args:
            handler_class: The handler class to validate. Must be a class type,
                not an instance.

        Returns:
            A tuple of (is_valid, missing_methods) where:
            - is_valid: True if handler implements all required protocol methods
            - missing_methods: List of method names that are missing or not callable.
              Empty list if all methods are present.

        Example:
            >>> class ValidHandler:
            ...     @property
            ...     def handler_type(self) -> str: return "test"
            ...     async def initialize(self, config): pass
            ...     async def shutdown(self, timeout_seconds=30.0): pass
            ...     async def execute(self, request, config): pass
            ...     def describe(self): return {}
            ...
            >>> loader = HandlerPluginLoader()
            >>> loader._validate_handler_protocol(ValidHandler)
            (True, [])

            >>> class IncompleteHandler:
            ...     def describe(self): return {}
            ...
            >>> loader._validate_handler_protocol(IncompleteHandler)
            (False, ['handler_type', 'initialize', 'shutdown', 'execute'])

        See Also:
            - ``omnibase_spi.protocols.handlers.protocol_handler.ProtocolHandler``
            - ``docs/architecture/RUNTIME_HOST_IMPLEMENTATION_PLAN.md``
        """
        # Check for required ProtocolHandler methods via duck typing
        # All 5 core methods must be present for protocol compliance
        missing_methods: list[str] = []

        # 1. handler_type property - can be property or method
        if not hasattr(handler_class, "handler_type"):
            missing_methods.append("handler_type")

        # 2. initialize() - async method for connection setup
        if not callable(getattr(handler_class, "initialize", None)):
            missing_methods.append("initialize")

        # 3. shutdown() - async method for resource cleanup
        if not callable(getattr(handler_class, "shutdown", None)):
            missing_methods.append("shutdown")

        # 4. execute() - async method for operation execution
        if not callable(getattr(handler_class, "execute", None)):
            missing_methods.append("execute")

        # 5. describe() - sync method for introspection
        if not callable(getattr(handler_class, "describe", None)):
            missing_methods.append("describe")

        # Note: health_check() is part of ProtocolHandler but is NOT validated
        # because existing handlers (HandlerHttp, HandlerDb, etc.) do not
        # implement it. Future handlers SHOULD implement health_check().

        return (len(missing_methods) == 0, missing_methods)

    def _import_handler_class(
        self,
        class_path: str,
        contract_path: Path,
        correlation_id: UUID | None = None,
    ) -> type:
        """Dynamically import handler class from fully qualified path.

        This method validates the namespace (if allowed_namespaces is configured)
        BEFORE calling ``importlib.import_module()``, preventing any module-level
        side effects from untrusted packages.

        Args:
            class_path: Fully qualified class path (e.g., 'myapp.handlers.AuthHandler').
            contract_path: Path to the contract file (for error context).
            correlation_id: Optional correlation ID for tracing and error context.

        Returns:
            The imported class type.

        Raises:
            ProtocolConfigurationError: If namespace validation fails.
                - HANDLER_LOADER_013 (NAMESPACE_NOT_ALLOWED): When the class path
                  does not start with any of the allowed namespace prefixes.
            InfraConnectionError: If the module or class cannot be imported.
                Error codes include correlation_id when provided for traceability.
                - HANDLER_LOADER_010 (MODULE_NOT_FOUND): Handler module not found
                - HANDLER_LOADER_011 (CLASS_NOT_FOUND): Handler class not found in module
                - HANDLER_LOADER_012 (IMPORT_ERROR): Import error (syntax/dependency)
        """
        # Split class path into module and class name
        if "." not in class_path:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Invalid class path '{class_path}': must be fully qualified "
                "(e.g., 'myapp.handlers.AuthHandler')",
                context=context,
                loader_error=EnumHandlerLoaderError.MODULE_NOT_FOUND.value,
                class_path=class_path,
                contract_path=str(contract_path),
            )

        module_path, class_name = class_path.rsplit(".", 1)

        # Validate namespace BEFORE importing (defense-in-depth)
        # This prevents any module-level side effects from untrusted packages
        self._validate_namespace(class_path, contract_path, correlation_id)

        # Import the module
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Module not found: {module_path}",
                context=context,
                loader_error=EnumHandlerLoaderError.MODULE_NOT_FOUND.value,
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e
        except ImportError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            raise InfraConnectionError(
                f"Import error loading module {module_path}: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.IMPORT_ERROR.value,
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e
        except SyntaxError as e:
            # SyntaxError can occur during import if the handler module has syntax errors.
            # This is a subclass of Exception, not ImportError, so must be caught separately.
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            raise InfraConnectionError(
                f"Syntax error in module {module_path}: {sanitized_msg}",
                context=context,
                loader_error=EnumHandlerLoaderError.IMPORT_ERROR.value,
                module_path=module_path,
                class_path=class_path,
                contract_path=str(contract_path),
            ) from e

        # Get the class from the module
        if not hasattr(module, class_name):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"Class '{class_name}' not found in module '{module_path}'",
                context=context,
                loader_error=EnumHandlerLoaderError.CLASS_NOT_FOUND.value,
                module_path=module_path,
                class_name=class_name,
                class_path=class_path,
                contract_path=str(contract_path),
            )

        handler_class = getattr(module, class_name)

        # Verify it's actually a class
        if not isinstance(handler_class, type):
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="import_handler_class",
                correlation_id=correlation_id,
            )
            raise InfraConnectionError(
                f"'{class_path}' is not a class",
                context=context,
                loader_error=EnumHandlerLoaderError.CLASS_NOT_FOUND.value,
                class_path=class_path,
                contract_path=str(contract_path),
            )

        return handler_class

    def _validate_namespace(
        self,
        class_path: str,
        contract_path: Path,
        correlation_id: UUID | None = None,
    ) -> None:
        """Validate handler class path against allowed namespaces.

        Checks whether the handler's fully-qualified class path starts with one
        of the allowed namespace prefixes. This validation occurs BEFORE the
        module is imported, preventing any module-level side effects from
        untrusted packages.

        Args:
            class_path: Fully qualified class path (e.g., 'myapp.handlers.AuthHandler').
            contract_path: Path to the contract file (for error context).
            correlation_id: Optional correlation ID for tracing and error context.

        Raises:
            ProtocolConfigurationError: If namespace validation fails.
                - HANDLER_LOADER_013 (NAMESPACE_NOT_ALLOWED): When the class path
                  does not start with any of the allowed namespace prefixes.

        Note:
            This method is a no-op when ``allowed_namespaces`` is None, allowing
            any namespace. When ``allowed_namespaces`` is an empty list, ALL
            namespaces are blocked.

        Example:
            >>> loader = HandlerPluginLoader(
            ...     allowed_namespaces=["omnibase_infra.", "mycompany."]
            ... )
            >>> # This passes validation:
            >>> loader._validate_namespace(
            ...     "omnibase_infra.handlers.HandlerAuth",
            ...     Path("contract.yaml"),
            ... )
            >>> # This raises ProtocolConfigurationError:
            >>> loader._validate_namespace(
            ...     "malicious_pkg.EvilHandler",
            ...     Path("malicious.yaml"),
            ... )
        """
        # If no namespace restriction is configured, allow all
        if self._allowed_namespaces is None:
            return

        # Check if class_path starts with any allowed namespace with proper
        # package boundary validation. This prevents "foo" from matching "foobar.module".
        for namespace in self._allowed_namespaces:
            if class_path.startswith(namespace):
                # If namespace ends with ".", we've already matched a package boundary
                if namespace.endswith("."):
                    return
                # Otherwise, ensure we're at a package boundary (next char is ".")
                # This prevents "foo" from matching "foobar.module" - only exact
                # matches or matches followed by "." are valid.
                remaining = class_path[len(namespace) :]
                if remaining == "" or remaining.startswith("."):
                    return

        # Namespace not in allowed list - raise error
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="validate_namespace",
            correlation_id=correlation_id,
        )

        # Format allowed namespaces for error message
        if self._allowed_namespaces:
            allowed_str = ", ".join(repr(ns) for ns in self._allowed_namespaces)
        else:
            allowed_str = "(none - empty allowlist)"

        raise ProtocolConfigurationError(
            f"Handler namespace not allowed: '{class_path}' does not start with "
            f"any of the allowed namespaces: {allowed_str}",
            context=context,
            loader_error=EnumHandlerLoaderError.NAMESPACE_NOT_ALLOWED.value,
            class_path=class_path,
            contract_path=str(contract_path),
            allowed_namespaces=list(self._allowed_namespaces),
        )

    def _validate_yaml_syntax(
        self,
        path: Path,
        correlation_id: UUID | None = None,
        raise_on_error: bool = True,
    ) -> bool:
        """Validate YAML syntax of a contract file for early fail-fast behavior.

        Performs early YAML syntax validation to fail fast before expensive
        operations like path resolution and handler class loading. This method
        only validates that the file contains valid YAML syntax; it does not
        perform schema validation.

        This enables the discover_and_load method to skip malformed YAML files
        immediately after discovery, rather than deferring the error to
        load_from_contract which would be less efficient for large discovery
        operations.

        Args:
            path: Path to the YAML file to validate. Must be an existing file.
            correlation_id: Optional correlation ID for error context.
            raise_on_error: If True (default), raises ProtocolConfigurationError
                on YAML syntax errors. If False, logs a warning and returns False,
                allowing the caller to skip the file.

        Returns:
            True if YAML syntax is valid.
            False if raise_on_error is False and YAML syntax is invalid.

        Raises:
            ProtocolConfigurationError: If raise_on_error is True and:
                - INVALID_YAML_SYNTAX: File contains invalid YAML syntax
                - FILE_READ_ERROR: Failed to read file (I/O error)

        Note:
            The error message includes the YAML parser error details which
            typically contain line and column information for the syntax error.
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_yaml_syntax",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Invalid YAML syntax in contract file '{path.name}': {sanitized_msg}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value,
                    contract_path=str(path),
                ) from e
            logger.warning(
                "Skipping contract file with invalid YAML syntax %s: %s",
                path.name,
                sanitized_msg,
                extra={
                    "path": str(path),
                    "error": sanitized_msg,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )
            return False
        except OSError as e:
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_yaml_syntax",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Failed to read contract file '{path.name}': {sanitized_msg}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.FILE_READ_ERROR.value,
                    contract_path=str(path),
                ) from e
            logger.warning(
                "Failed to read contract file %s: %s",
                path.name,
                sanitized_msg,
                extra={
                    "path": str(path),
                    "error": sanitized_msg,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )
            return False

        return True

    def _validate_file_size(
        self,
        path: Path,
        correlation_id: UUID | None = None,
        operation: str = "load_from_contract",
        raise_on_error: bool = True,
    ) -> int | None:
        """Validate file size is within limits.

        Checks that the file at the given path can be stat'd and does not
        exceed MAX_CONTRACT_SIZE. Supports both strict mode (raising exceptions)
        and graceful mode (logging warnings and returning None).

        Args:
            path: Path to the file to validate. Must be an existing file.
            correlation_id: Optional correlation ID for error context.
            operation: The operation name for error context in exceptions.
            raise_on_error: If True (default), raises ProtocolConfigurationError
                on stat failure or size exceeded. If False, logs a warning
                and returns None, allowing the caller to skip the file.

        Returns:
            File size in bytes if validation passes.
            None if raise_on_error is False and validation fails (stat error
            or size exceeded).

        Raises:
            ProtocolConfigurationError: If raise_on_error is True and:
                - FILE_STAT_ERROR: Failed to stat the file (I/O error)
                - FILE_SIZE_EXCEEDED: File exceeds MAX_CONTRACT_SIZE
        """
        # Attempt to get file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            # Sanitize exception message to prevent path disclosure
            sanitized_msg = _sanitize_exception_message(e)
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=operation,
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Failed to stat contract file: {sanitized_msg}",
                    context=context,
                    loader_error=EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                    contract_path=str(path),
                ) from e
            logger.warning(
                "Failed to stat contract file %s: %s",
                path.name,
                sanitized_msg,
                extra={
                    "path": str(path),
                    "error": sanitized_msg,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )
            return None

        # Check size limit
        if file_size > MAX_CONTRACT_SIZE:
            if raise_on_error:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=operation,
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"Contract file exceeds size limit: {file_size} bytes "
                    f"(max: {MAX_CONTRACT_SIZE} bytes)",
                    context=context,
                    loader_error=EnumHandlerLoaderError.FILE_SIZE_EXCEEDED.value,
                    contract_path=str(path),
                    file_size=file_size,
                    max_size=MAX_CONTRACT_SIZE,
                )
            logger.warning(
                "Skipping oversized contract file %s: %d bytes exceeds limit of %d bytes",
                path.name,
                file_size,
                MAX_CONTRACT_SIZE,
                extra={
                    "path": str(path),
                    "file_size": file_size,
                    "max_size": MAX_CONTRACT_SIZE,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )
            return None

        return file_size

    def _find_contract_files(
        self,
        directory: Path,
        correlation_id: UUID | None = None,
        max_handlers: int | None = None,
    ) -> list[Path]:
        """Find all handler contract files under a directory.

        Searches for both handler_contract.yaml and contract.yaml files.
        Files exceeding MAX_CONTRACT_SIZE are skipped during discovery
        to fail fast before expensive path resolution and loading.

        Ambiguous Contract Detection (Fail-Fast):
            When BOTH ``handler_contract.yaml`` AND ``contract.yaml`` exist in the
            same directory, this method raises ``ProtocolConfigurationError`` with
            error code ``AMBIGUOUS_CONTRACT_CONFIGURATION``. This fail-fast behavior
            prevents:

            - Duplicate handler registrations
            - Confusion about which contract is authoritative
            - Unexpected runtime behavior from conflicting configurations

            Best practice: Use only ONE contract file per handler directory.

            See: docs/patterns/handler_plugin_loader.md#contract-file-precedence

        Args:
            directory: Directory to search recursively.
            correlation_id: Optional correlation ID for tracing and error context.
            max_handlers: Optional maximum number of handlers to discover.
                If specified, discovery stops after finding this many contract files.
                Propagated to file size validation for consistent traceability.

        Returns:
            List of paths to contract files that pass size validation.

        Raises:
            ProtocolConfigurationError: If both handler_contract.yaml and contract.yaml
                exist in the same directory. Error code: AMBIGUOUS_CONTRACT_CONFIGURATION
                (HANDLER_LOADER_040).
        """
        contract_files: list[Path] = []
        # Track if max_handlers limit was reached
        limit_reached = False

        # Search for valid contract filenames in a single scan
        # This consolidates two rglob() calls into one for better performance
        # NOTE: If both handler_contract.yaml and contract.yaml are found in the
        # same directory, we fail fast with AMBIGUOUS_CONTRACT_CONFIGURATION error
        # after discovery (see ambiguity check below).
        valid_filenames = {HANDLER_CONTRACT_FILENAME, CONTRACT_YAML_FILENAME}

        # directory.rglob() can raise OSError for:
        # - Permission denied when accessing the directory or subdirectories
        # - Filesystem errors (unmounted volumes, network failures)
        # - Directory deleted or becomes inaccessible during iteration
        try:
            rglob_iterator = directory.rglob("*.yaml")
        except OSError as e:
            # If we can't even start iterating, log warning and return empty list
            # This is graceful degradation - the caller can handle empty results
            sanitized_msg = _sanitize_exception_message(e)
            logger.warning(
                "Failed to scan directory %s for contracts: %s",
                directory.name,
                sanitized_msg,
                extra={
                    "directory": str(directory),
                    "error": sanitized_msg,
                    "error_code": EnumHandlerLoaderError.PERMISSION_DENIED.value,
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )
            return []

        # Iterate over discovered paths, handling per-path errors gracefully
        try:
            for path in rglob_iterator:
                # Check if we've reached the max_handlers limit
                if max_handlers is not None and len(contract_files) >= max_handlers:
                    limit_reached = True
                    break

                # Filter by filename first (cheap string comparison)
                if path.name not in valid_filenames:
                    continue

                # path.is_file() can raise OSError for:
                # - Permission denied when stat'ing the file
                # - File deleted between rglob discovery and is_file() check
                # - Filesystem errors (unmounted volumes, network failures)
                try:
                    is_file = path.is_file()
                except OSError as e:
                    sanitized_msg = _sanitize_exception_message(e)
                    logger.warning(
                        "Failed to check if path is file %s: %s",
                        path.name,
                        sanitized_msg,
                        extra={
                            "path": str(path),
                            "error": sanitized_msg,
                            "error_code": EnumHandlerLoaderError.FILE_STAT_ERROR.value,
                            "correlation_id": str(correlation_id)
                            if correlation_id
                            else None,
                        },
                    )
                    continue

                if not is_file:
                    continue

                # Early size validation to skip oversized files before expensive operations
                if (
                    self._validate_file_size(
                        path, correlation_id=correlation_id, raise_on_error=False
                    )
                    is None
                ):
                    continue

                contract_files.append(path)
        except OSError as e:
            # Handle errors that occur during iteration (e.g., directory becomes
            # inaccessible mid-scan). Return what we've collected so far.
            sanitized_msg = _sanitize_exception_message(e)
            logger.warning(
                "Error during directory scan of %s: %s (returning %d files found so far)",
                directory.name,
                sanitized_msg,
                len(contract_files),
                extra={
                    "directory": str(directory),
                    "error": sanitized_msg,
                    "error_code": EnumHandlerLoaderError.PERMISSION_DENIED.value,
                    "files_found": len(contract_files),
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )

        # Log warning if limit was reached
        if limit_reached:
            logger.warning(
                "Handler discovery limit reached: stopped at %d handlers. "
                "Increase max_handlers to discover more.",
                max_handlers,
                extra={
                    "max_handlers": max_handlers,
                    "directory": str(directory),
                    "correlation_id": str(correlation_id) if correlation_id else None,
                },
            )

        # Detect directories with both contract types and fail fast on ambiguity
        # This is an O(n) check after discovery, not during, to avoid overhead
        # on every file. Build a map of parent_dir -> set of contract filenames.
        dir_to_contract_types: dict[Path, set[str]] = {}
        for path in contract_files:
            parent = path.parent
            if parent not in dir_to_contract_types:
                dir_to_contract_types[parent] = set()
            dir_to_contract_types[parent].add(path.name)

        # Fail fast if any directory has both contract types (ambiguous configuration)
        for parent_dir, filenames in dir_to_contract_types.items():
            if len(filenames) > 1:
                # Use with_correlation() to ensure correlation_id is always present
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="find_contract_files",
                )
                raise ProtocolConfigurationError(
                    f"Ambiguous contract configuration in '{parent_dir.name}': "
                    f"Found both '{HANDLER_CONTRACT_FILENAME}' and '{CONTRACT_YAML_FILENAME}'. "
                    f"Use only ONE contract file per handler directory to avoid conflicts. "
                    f"Total contracts discovered so far: {len(contract_files)}.",
                    context=context,
                    loader_error=EnumHandlerLoaderError.AMBIGUOUS_CONTRACT_CONFIGURATION.value,
                    directory=str(parent_dir),
                    contract_files=sorted(filenames),
                    total_discovered=len(contract_files),
                )

        # Deduplicate by resolved path
        seen: set[Path] = set()
        deduplicated: list[Path] = []
        for path in contract_files:
            # path.resolve() can raise OSError in several scenarios:
            # - Broken symlinks: symlink target no longer exists
            # - Race conditions: file deleted between glob discovery and resolution
            # - Permission issues: lacking read permission on parent directories
            # - Filesystem errors: unmounted volumes, network filesystem failures
            try:
                resolved = path.resolve()
            except OSError as e:
                # Sanitize exception message to prevent path disclosure
                sanitized_msg = _sanitize_exception_message(e)
                logger.warning(
                    "Failed to resolve path %s: %s",
                    path.name,
                    sanitized_msg,
                    extra={
                        "path": str(path),
                        "error": sanitized_msg,
                        "correlation_id": str(correlation_id)
                        if correlation_id
                        else None,
                    },
                )
                continue
            if resolved not in seen:
                seen.add(resolved)
                deduplicated.append(path)

        return deduplicated


__all__ = [
    "CONTRACT_YAML_FILENAME",
    "HANDLER_CONTRACT_FILENAME",
    "HandlerPluginLoader",
    "MAX_CONTRACT_SIZE",
]
