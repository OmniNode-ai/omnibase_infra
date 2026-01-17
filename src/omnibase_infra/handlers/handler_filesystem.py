# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Filesystem Handler - Secure filesystem operations with path whitelisting.

Provides secure filesystem operations including read, write, list, ensure directory,
and delete operations with comprehensive security features.

Security Features:
    - Path whitelist validation: Only allowed directories can be accessed
    - File size limits: Configurable max sizes for read/write operations
    - Symlink protection: Symlinks are resolved and validated against allowed paths
    - Path traversal prevention: Prevents escape from allowed directories via ../

Supported Operations:
    - filesystem.read_file: Read file contents (text or binary)
    - filesystem.write_file: Write content to file
    - filesystem.list_directory: List directory contents with optional glob filtering
    - filesystem.ensure_directory: Create directory structure
    - filesystem.delete_file: Delete file with safety checks

Note:
    Environment variable configuration (ONEX_FS_MAX_READ_SIZE, ONEX_FS_MAX_WRITE_SIZE)
    is parsed at module import time, not at handler instantiation. This means:

    - Changes to environment variables require application restart to take effect
    - Tests should use ``unittest.mock.patch.dict(os.environ, ...)`` before importing,
      or use ``importlib.reload()`` to re-import the module after patching
    - This is an intentional design choice for startup-time validation
"""

from __future__ import annotations

import base64
import binascii
import errno
import fnmatch
import logging
import os
from pathlib import Path
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction
from omnibase_infra.utils import parse_env_int

logger = logging.getLogger(__name__)

# Default configuration from environment
_DEFAULT_MAX_READ_SIZE: int = parse_env_int(
    "ONEX_FS_MAX_READ_SIZE",
    100 * 1024 * 1024,  # 100 MB
    min_value=1024,
    max_value=1024 * 1024 * 1024,  # 1 GB
    transport_type=EnumInfraTransportType.FILESYSTEM,
    service_name="filesystem_handler",
)
_DEFAULT_MAX_WRITE_SIZE: int = parse_env_int(
    "ONEX_FS_MAX_WRITE_SIZE",
    50 * 1024 * 1024,  # 50 MB
    min_value=1024,
    max_value=500 * 1024 * 1024,  # 500 MB
    transport_type=EnumInfraTransportType.FILESYSTEM,
    service_name="filesystem_handler",
)

_SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "filesystem.read_file",
        "filesystem.write_file",
        "filesystem.list_directory",
        "filesystem.ensure_directory",
        "filesystem.delete_file",
    }
)

# Handler ID for ModelHandlerOutput
HANDLER_ID_FILESYSTEM: str = "filesystem-handler"

# Size category thresholds for sanitized logging
_SIZE_THRESHOLD_KB: int = 1024  # 1 KB
_SIZE_THRESHOLD_MB: int = 1024 * 1024  # 1 MB
_SIZE_THRESHOLD_10MB: int = 10 * 1024 * 1024  # 10 MB


def _categorize_size(size: int) -> str:
    """Categorize byte size into security-safe categories.

    This prevents exact payload sizes from being exposed in error messages
    and logs, which could help attackers probe size limits.

    Args:
        size: Size in bytes

    Returns:
        Size category: "small", "medium", "large", or "very_large"
    """
    if size < _SIZE_THRESHOLD_KB:
        return "small"
    elif size < _SIZE_THRESHOLD_MB:
        return "medium"
    elif size < _SIZE_THRESHOLD_10MB:
        return "large"
    else:
        return "very_large"


class HandlerFileSystem(MixinEnvelopeExtraction, MixinAsyncCircuitBreaker):
    """Filesystem handler with security features for ONEX infrastructure.

    Security Features:
        - Path whitelist validation to restrict file access to allowed directories
        - File size limits to prevent DoS attacks via memory exhaustion
        - Symlink resolution and validation to prevent path traversal attacks
        - All paths are resolved to absolute canonical paths before validation
        - Circuit breaker for resilient operation

    Configuration:
        Initialize with allowed_paths to define accessible directories.
        Configure max_read_size and max_write_size to control memory usage.
    """

    def __init__(self, container: ModelONEXContainer | None = None) -> None:
        """Initialize HandlerFileSystem with optional container injection.

        Args:
            container: Optional ONEX container for dependency injection.
                When provided, enables full ONEX integration. When None,
                handler operates in standalone mode for testing.

        Note:
            The container is stored for interface compliance with the standard ONEX
            handler pattern and to enable future DI-based service resolution (e.g.,
            metrics, logging, observability integration). Currently, the handler
            operates independently for filesystem operations, but storing the container
            ensures API consistency and enables future enhancements without breaking
            changes.
        """
        self._container = container
        self._allowed_paths: tuple[Path, ...] = ()
        self._max_read_size: int = _DEFAULT_MAX_READ_SIZE
        self._max_write_size: int = _DEFAULT_MAX_WRITE_SIZE
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            protocol/transport handler for filesystem operations.

        Note:
            handler_type determines lifecycle, protocol selection, and runtime
            invocation patterns. It answers "what is this handler in the architecture?"

        See Also:
            - handler_category: Behavioral classification (EFFECT/COMPUTE)
            - transport_type: Specific transport protocol (FILESYSTEM)
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (filesystem read/write). EFFECT handlers are not
            deterministic and interact with external systems.

        Note:
            handler_category determines security rules, determinism guarantees,
            replay safety, and permissions. It answers "how does this handler
            behave at runtime?"

            Categories:
            - COMPUTE: Pure, deterministic transformations (no side effects)
            - EFFECT: Side-effecting I/O (database, HTTP, filesystem)
            - NONDETERMINISTIC_COMPUTE: Pure but not deterministic (UUID, random)

        See Also:
            - handler_type: Architectural role (INFRA_HANDLER/NODE_HANDLER/etc.)
            - transport_type: Specific transport protocol (FILESYSTEM)
        """
        return EnumHandlerTypeCategory.EFFECT

    @property
    def transport_type(self) -> EnumInfraTransportType:
        """Return the transport protocol identifier for this handler.

        Returns:
            EnumInfraTransportType.FILESYSTEM - This handler uses local filesystem.

        Note:
            transport_type identifies the specific transport/protocol this handler
            uses. It is the third dimension of the handler type system, alongside
            handler_type (architectural role) and handler_category (behavioral
            classification).

            The three dimensions together form a complete handler classification:
            - handler_type: INFRA_HANDLER (what it is architecturally)
            - handler_category: EFFECT (how it behaves at runtime)
            - transport_type: FILESYSTEM (what protocol it uses)

        See Also:
            - handler_type: Architectural role
            - handler_category: Behavioral classification
        """
        return EnumInfraTransportType.FILESYSTEM

    @transport_type.setter
    def transport_type(self, value: EnumInfraTransportType) -> None:
        """Prevent modification of transport_type after initialization.

        The transport_type is immutable for this handler - it is always FILESYSTEM.
        This setter raises an AttributeError if modification is attempted.

        Args:
            value: The transport type value (assignment always raises error)

        Raises:
            AttributeError: Always raised - transport_type is read-only.
        """
        raise AttributeError(
            "transport_type is read-only; it is set during handler initialization"
        )

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize filesystem handler with path whitelist and size limits.

        Args:
            config: Configuration dict containing:
                - allowed_paths: Required list of allowed directory paths (strings)
                - max_read_size: Optional max read size in bytes (default: 100 MB)
                - max_write_size: Optional max write size in bytes (default: 50 MB)
                - correlation_id: Optional UUID or string for error tracing

        Raises:
            ProtocolConfigurationError: If allowed_paths is missing, empty, or invalid.

        Security:
            - All allowed_paths are resolved to absolute canonical paths
            - Non-existent paths are logged as warnings but not rejected
            - Empty allowed_paths list is rejected for security reasons
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="initialize",
            target_name="filesystem_handler",
            correlation_id=init_correlation_id,
        )

        # Extract and validate allowed_paths (required)
        allowed_paths_raw = config.get("allowed_paths")
        if allowed_paths_raw is None:
            raise ProtocolConfigurationError(
                "Missing required 'allowed_paths' configuration - filesystem handler "
                "requires explicit path whitelist for security",
                context=ctx,
            )

        if (
            not isinstance(allowed_paths_raw, list | tuple)
            or len(allowed_paths_raw) == 0
        ):
            raise ProtocolConfigurationError(
                "Configuration 'allowed_paths' must be a non-empty list or tuple of directory paths",
                context=ctx,
            )

        # Resolve and validate each allowed path
        resolved_paths: list[Path] = []
        for path_str in allowed_paths_raw:
            if not isinstance(path_str, str):
                raise ProtocolConfigurationError(
                    f"Invalid path in allowed_paths: expected string, got {type(path_str).__name__}",
                    context=ctx,
                )

            path = Path(path_str).resolve()

            if not path.exists():
                logger.warning(
                    "Allowed path does not exist (will be created on first use): %s",
                    path,
                    extra={
                        "path": str(path),
                        "correlation_id": str(init_correlation_id),
                    },
                )

            resolved_paths.append(path)

        # Store as immutable tuple after initialization
        self._allowed_paths = tuple(resolved_paths)

        # Extract optional size limits
        max_read_raw = config.get("max_read_size")
        if max_read_raw is not None:
            if isinstance(max_read_raw, int) and max_read_raw > 0:
                self._max_read_size = max_read_raw
            else:
                logger.warning(
                    "Invalid max_read_size config value ignored, using default",
                    extra={
                        "provided_value": max_read_raw,
                        "default_value": self._max_read_size,
                    },
                )

        max_write_raw = config.get("max_write_size")
        if max_write_raw is not None:
            if isinstance(max_write_raw, int) and max_write_raw > 0:
                self._max_write_size = max_write_raw
            else:
                logger.warning(
                    "Invalid max_write_size config value ignored, using default",
                    extra={
                        "provided_value": max_write_raw,
                        "default_value": self._max_write_size,
                    },
                )

        # Initialize circuit breaker for resilient I/O operations
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="filesystem_handler",
            transport_type=EnumInfraTransportType.FILESYSTEM,
        )

        self._initialized = True

        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "allowed_paths_count": len(self._allowed_paths),
                "max_read_size_bytes": self._max_read_size,
                "max_write_size_bytes": self._max_write_size,
                "correlation_id": str(init_correlation_id),
            },
        )

    async def shutdown(self) -> None:
        """Shutdown filesystem handler and clear configuration."""
        self._allowed_paths = ()
        self._initialized = False
        logger.info("HandlerFileSystem shutdown complete")

    def _validate_path_in_whitelist(
        self, path: Path, correlation_id: UUID, operation: str
    ) -> Path:
        """Validate that path is within allowed directories.

        This method resolves the path to its canonical form (following symlinks)
        and verifies it is within one of the allowed directories.

        Args:
            path: Path to validate
            correlation_id: Correlation ID for error context
            operation: Operation name for error context

        Returns:
            The resolved canonical path

        Raises:
            ProtocolConfigurationError: If path is outside allowed directories
                or symlink points outside allowed directories
        """
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(path),
            correlation_id=correlation_id,
        )

        # Resolve to canonical path using Path.resolve()
        # Note: In Python 3.6+, resolve() defaults to strict=False, meaning it
        # works for both existing and non-existing paths by resolving as much
        # of the path as possible without requiring the full path to exist.
        try:
            resolved = path.resolve()
        except OSError as e:
            raise ProtocolConfigurationError(
                f"Cannot resolve path: {e}",
                context=ctx,
            ) from e

        # Check if resolved path is within any allowed directory
        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue

        # Path is not within any allowed directory
        raise ProtocolConfigurationError(
            f"Path '{path}' is outside allowed directories - access denied",
            context=ctx,
        )

    def _validate_symlink_target(
        self, path: Path, correlation_id: UUID, operation: str
    ) -> None:
        """Validate symlink target is within allowed directories.

        If the path is a symlink, this method validates that the target
        is within the allowed directories to prevent symlink escape attacks.

        Args:
            path: Path that may be a symlink
            correlation_id: Correlation ID for error context
            operation: Operation name for error context

        Raises:
            ProtocolConfigurationError: If symlink target is outside allowed directories
        """
        if not path.is_symlink():
            return

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(path),
            correlation_id=correlation_id,
        )

        try:
            target = path.resolve()
        except OSError as e:
            raise ProtocolConfigurationError(
                f"Cannot resolve symlink target: {e}",
                context=ctx,
            ) from e

        # Verify symlink target is within allowed directories
        for allowed in self._allowed_paths:
            try:
                target.relative_to(allowed)
                return
            except ValueError:
                continue

        raise ProtocolConfigurationError(
            f"Symlink '{path}' points outside allowed directories - access denied",
            context=ctx,
        )

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute filesystem operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: One of the supported filesystem operations
                - payload: Operation-specific payload
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput[dict[str, object]] containing operation result

        Raises:
            RuntimeHostError: If handler not initialized
            ProtocolConfigurationError: If operation or payload is invalid
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="execute",
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HandlerFileSystem not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="execute",
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'operation' in envelope", context=ctx
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Operation '{operation}' not supported. Available: {', '.join(sorted(_SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'payload' in envelope", context=ctx
            )

        # Route to appropriate operation handler
        if operation == "filesystem.read_file":
            return await self._execute_read_file(
                payload, correlation_id, input_envelope_id
            )
        elif operation == "filesystem.write_file":
            return await self._execute_write_file(
                payload, correlation_id, input_envelope_id
            )
        elif operation == "filesystem.list_directory":
            return await self._execute_list_directory(
                payload, correlation_id, input_envelope_id
            )
        elif operation == "filesystem.ensure_directory":
            return await self._execute_ensure_directory(
                payload, correlation_id, input_envelope_id
            )
        else:  # filesystem.delete_file
            return await self._execute_delete_file(
                payload, correlation_id, input_envelope_id
            )

    async def _execute_read_file(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute filesystem.read_file operation.

        Payload:
            - path: str (required) - File path to read
            - binary: bool (optional, default False) - Read as binary (returns base64)
            - encoding: str (optional, default "utf-8") - Text encoding

        Returns:
            Result with content, size, path, and binary flag.
            For binary=True, content is base64-encoded string.
            For binary=False, content is the text string.

        Raises:
            InfraConnectionError: If file not found or read fails
            InfraUnavailableError: If file size exceeds limit or circuit breaker is open
        """
        operation = "filesystem.read_file"

        # Extract path (required)
        path_raw = payload.get("path")
        if not isinstance(path_raw, str) or not path_raw:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'path' in payload", context=ctx
            )

        path = Path(path_raw)
        resolved_path = self._validate_path_in_whitelist(
            path, correlation_id, operation
        )
        self._validate_symlink_target(resolved_path, correlation_id, operation)

        # Extract options
        binary = payload.get("binary", False)
        if not isinstance(binary, bool):
            logger.warning(
                "Invalid binary parameter type ignored, using default",
                extra={
                    "provided_value": binary,
                    "provided_type": type(binary).__name__,
                    "default_value": False,
                    "correlation_id": str(correlation_id),
                },
            )
            binary = False

        encoding = payload.get("encoding", "utf-8")
        if not isinstance(encoding, str):
            logger.warning(
                "Invalid encoding parameter type ignored, using default",
                extra={
                    "provided_value": encoding,
                    "provided_type": type(encoding).__name__,
                    "default_value": "utf-8",
                    "correlation_id": str(correlation_id),
                },
            )
            encoding = "utf-8"

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(resolved_path),
            correlation_id=correlation_id,
        )

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Check file exists
            if not resolved_path.exists():
                raise InfraConnectionError(
                    f"File not found: {resolved_path.name}",
                    context=ctx,
                )

            if not resolved_path.is_file():
                raise InfraConnectionError(
                    f"Path is not a file: {resolved_path.name}",
                    context=ctx,
                )

            # Check file size before reading
            try:
                file_size = resolved_path.stat().st_size
            except OSError as e:
                raise InfraConnectionError(
                    f"Cannot stat file: {e}",
                    context=ctx,
                ) from e

            if file_size > self._max_read_size:
                raise InfraUnavailableError(
                    f"File size ({_categorize_size(file_size)}) exceeds configured read limit",
                    context=ctx,
                )

            # Read file content
            content: str
            try:
                if binary:
                    raw_bytes = resolved_path.read_bytes()
                    # Encode bytes as base64 string for JSON safety
                    content = base64.b64encode(raw_bytes).decode("ascii")
                else:
                    content = resolved_path.read_text(encoding=encoding)
            except OSError as e:
                raise InfraConnectionError(
                    f"Failed to read file: {e}",
                    context=ctx,
                ) from e
            except UnicodeDecodeError as e:
                raise InfraConnectionError(
                    f"Failed to decode file with encoding '{encoding}': {e}",
                    context=ctx,
                ) from e

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "File read successfully",
                extra={
                    "path": str(resolved_path),
                    "size_category": _categorize_size(file_size),
                    "binary": binary,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_FILESYSTEM,
                result={
                    "status": "success",
                    "payload": {
                        "content": content,
                        "size": file_size,
                        "path": str(resolved_path),
                        "binary": binary,
                    },
                    "correlation_id": str(correlation_id),
                },
            )

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise

    async def _execute_write_file(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute filesystem.write_file operation.

        Payload:
            - path: str (required) - File path to write
            - content: str (required) - Content to write.
              For binary=True, content should be base64-encoded string.
            - binary: bool (optional, default False) - Write as binary (expects base64 content)
            - create_dirs: bool (optional, default False) - Create parent directories

        Returns:
            Result with path, bytes_written, and created flag

        Raises:
            InfraConnectionError: If write fails
            InfraUnavailableError: If content size exceeds limit or circuit breaker is open
            ProtocolConfigurationError: If base64 decoding fails for binary mode
        """
        operation = "filesystem.write_file"

        # Extract path (required)
        path_raw = payload.get("path")
        if not isinstance(path_raw, str) or not path_raw:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'path' in payload", context=ctx
            )

        # Extract content (required)
        content_raw = payload.get("content")
        if content_raw is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing 'content' in payload", context=ctx
            )

        # Extract binary flag (consistent with read_file API)
        binary = payload.get("binary", False)
        if not isinstance(binary, bool):
            logger.warning(
                "Invalid binary parameter type ignored, using default",
                extra={
                    "provided_value": binary,
                    "provided_type": type(binary).__name__,
                    "default_value": False,
                    "correlation_id": str(correlation_id),
                },
            )
            binary = False

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(path_raw),
            correlation_id=correlation_id,
        )

        # Process content based on binary flag
        content_bytes: bytes
        content_str: str

        if binary:
            # Binary mode: expect base64-encoded string or raw bytes
            if isinstance(content_raw, str):
                try:
                    content_bytes = base64.b64decode(content_raw, validate=True)
                except binascii.Error as e:
                    raise ProtocolConfigurationError(
                        f"Invalid base64 content for binary mode: {e}",
                        context=ctx,
                    ) from e
            elif isinstance(content_raw, bytes):
                content_bytes = content_raw
            else:
                raise ProtocolConfigurationError(
                    f"Invalid content type for binary mode: expected str (base64) or bytes, got {type(content_raw).__name__}",
                    context=ctx,
                )
            content_size = len(content_bytes)
        else:
            # Text mode: expect string
            if isinstance(content_raw, str):
                content_str = content_raw
            elif isinstance(content_raw, bytes):
                try:
                    content_str = content_raw.decode("utf-8")
                except UnicodeDecodeError as e:
                    raise ProtocolConfigurationError(
                        "Invalid UTF-8 bytes for text content",
                        context=ctx,
                    ) from e
            else:
                raise ProtocolConfigurationError(
                    f"Invalid content type: expected str or bytes, got {type(content_raw).__name__}",
                    context=ctx,
                )
            content_size = len(content_str.encode("utf-8"))

        create_dirs = payload.get("create_dirs", False)
        if not isinstance(create_dirs, bool):
            create_dirs = False

        path = Path(path_raw)

        # For write operations, validate parent directory is in whitelist
        parent = path.parent
        if parent.exists():
            self._validate_path_in_whitelist(parent, correlation_id, operation)
        else:
            # For non-existent parents, validate the path structure
            # Find the first existing ancestor and validate from there
            current = parent
            while not current.exists() and current != current.parent:
                current = current.parent

            if current.exists():
                self._validate_path_in_whitelist(current, correlation_id, operation)
            else:
                raise ProtocolConfigurationError(
                    f"Path '{path}' is outside allowed directories - access denied",
                    context=ctx,
                )

        # Compute resolved_path for validation and return value, but preserve
        # write_path for the actual write operation to enable O_NOFOLLOW check.
        # We use parent.resolve() / name to avoid resolving symlinks in the final component.
        resolved_parent = path.parent.resolve()
        write_path = resolved_parent / path.name
        resolved_path = path.resolve() if path.exists() else write_path

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Check content size
            if content_size > self._max_write_size:
                raise InfraUnavailableError(
                    f"Content size ({_categorize_size(content_size)}) exceeds configured write limit",
                    context=ctx,
                )

            # Check if file exists (for return value)
            file_existed = write_path.exists()

            # Create parent directories if requested
            if create_dirs and not resolved_parent.exists():
                try:
                    resolved_parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise InfraConnectionError(
                        f"Failed to create parent directories: {e}",
                        context=ctx,
                    ) from e

            # Check if symlink exists and validate target is within allowed paths.
            # This provides a helpful error message for symlinks pointing outside.
            # The O_NOFOLLOW check below will reject ALL symlinks for security.
            if write_path.is_symlink():
                self._validate_symlink_target(write_path, correlation_id, operation)

            # Write file content using O_NOFOLLOW to prevent symlink following.
            # This eliminates the TOCTOU race condition where an attacker could
            # replace the file with a symlink between validation and write.
            # We use write_path (not resolved_path) to detect symlinks.
            try:
                # O_NOFOLLOW causes the open to fail with ELOOP if the path is a symlink
                # This is atomic and cannot be raced, unlike checking is_symlink() first
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
                fd = os.open(str(write_path), flags, 0o644)
                try:
                    if binary:
                        with os.fdopen(fd, "wb") as f:
                            f.write(content_bytes)
                    else:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(content_str)
                except Exception:
                    # fdopen takes ownership of fd, but if write fails after fdopen
                    # the context manager will close it. Re-raise to outer handler.
                    raise
            except OSError as e:
                if e.errno == errno.ELOOP:
                    # ELOOP indicates the path is a symlink - reject the write
                    raise ProtocolConfigurationError(
                        f"Cannot write to symlink: {write_path.name}",
                        context=ctx,
                    ) from e
                raise InfraConnectionError(
                    f"Failed to write file: {e}",
                    context=ctx,
                ) from e

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "File written successfully",
                extra={
                    "path": str(resolved_path),
                    "size_category": _categorize_size(content_size),
                    "file_created": not file_existed,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_FILESYSTEM,
                result={
                    "status": "success",
                    "payload": {
                        "path": str(resolved_path),
                        "bytes_written": content_size,
                        "created": not file_existed,
                    },
                    "correlation_id": str(correlation_id),
                },
            )

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise

    async def _execute_list_directory(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute filesystem.list_directory operation.

        Payload:
            - path: str (required) - Directory path to list
            - recursive: bool (optional, default False) - List recursively
            - pattern: str (optional) - Glob pattern to filter entries

        Returns:
            Result with entries list, count, and path

        Raises:
            InfraConnectionError: If directory not found or list fails
            InfraUnavailableError: If circuit breaker is open
        """
        operation = "filesystem.list_directory"

        # Extract path (required)
        path_raw = payload.get("path")
        if not isinstance(path_raw, str) or not path_raw:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'path' in payload", context=ctx
            )

        path = Path(path_raw)
        resolved_path = self._validate_path_in_whitelist(
            path, correlation_id, operation
        )
        self._validate_symlink_target(resolved_path, correlation_id, operation)

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(resolved_path),
            correlation_id=correlation_id,
        )

        # Extract options
        recursive = payload.get("recursive", False)
        if not isinstance(recursive, bool):
            recursive = False

        pattern = payload.get("pattern")
        if pattern is not None and not isinstance(pattern, str):
            pattern = None

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Check directory exists
            if not resolved_path.exists():
                raise InfraConnectionError(
                    f"Directory not found: {resolved_path.name}",
                    context=ctx,
                )

            if not resolved_path.is_dir():
                raise InfraConnectionError(
                    f"Path is not a directory: {resolved_path.name}",
                    context=ctx,
                )

            # List directory contents
            entries: list[dict[str, object]] = []
            try:
                if recursive:
                    iterator = resolved_path.rglob("*")
                else:
                    iterator = resolved_path.iterdir()

                for entry in iterator:
                    # Apply pattern filter if specified
                    if pattern and not fnmatch.fnmatch(entry.name, pattern):
                        continue

                    # Get entry metadata - use lstat() to not follow symlinks
                    # This prevents exposing metadata from files outside the whitelist
                    try:
                        is_symlink = entry.is_symlink()

                        # For symlinks, check if target is within allowed paths
                        # Skip symlinks pointing outside allowed directories to prevent
                        # information disclosure about files outside the whitelist
                        if is_symlink:
                            try:
                                resolved_target = entry.resolve()
                                # Check if target is within any allowed directory
                                target_allowed = False
                                for allowed in self._allowed_paths:
                                    try:
                                        resolved_target.relative_to(allowed)
                                        target_allowed = True
                                        break
                                    except ValueError:
                                        continue
                                if not target_allowed:
                                    # Skip symlinks pointing outside allowed paths
                                    continue
                            except OSError:
                                # Skip broken or unresolvable symlinks
                                continue

                        # Use lstat() to get symlink's own metadata, not target's
                        stat_info = entry.lstat()

                        # For is_file/is_dir, report the actual entry type
                        # If it's a symlink, is_file()/is_dir() follow the link,
                        # so we report based on the symlink itself
                        entry_data: dict[str, object] = {
                            "name": entry.name,
                            "path": str(entry),
                            "is_file": entry.is_file() and not is_symlink,
                            "is_dir": entry.is_dir() and not is_symlink,
                            "is_symlink": is_symlink,
                            "size": stat_info.st_size,
                            "modified": stat_info.st_mtime,
                        }
                        entries.append(entry_data)
                    except OSError:
                        # Skip entries we can't stat
                        continue

            except OSError as e:
                raise InfraConnectionError(
                    f"Failed to list directory: {e}",
                    context=ctx,
                ) from e

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Directory listed successfully",
                extra={
                    "path": str(resolved_path),
                    "entry_count": len(entries),
                    "recursive": recursive,
                    "pattern": pattern,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_FILESYSTEM,
                result={
                    "status": "success",
                    "payload": {
                        "entries": entries,
                        "count": len(entries),
                        "path": str(resolved_path),
                    },
                    "correlation_id": str(correlation_id),
                },
            )

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise

    async def _execute_ensure_directory(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute filesystem.ensure_directory operation.

        Payload:
            - path: str (required) - Directory path to create
            - exist_ok: bool (optional, default True) - Don't error if exists

        Returns:
            Result with path, created, and already_existed flags

        Raises:
            InfraConnectionError: If directory creation fails
            InfraUnavailableError: If circuit breaker is open
        """
        operation = "filesystem.ensure_directory"

        # Extract path (required)
        path_raw = payload.get("path")
        if not isinstance(path_raw, str) or not path_raw:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'path' in payload", context=ctx
            )

        path = Path(path_raw)

        # Validate the target path is within allowed directories
        # _validate_path_in_whitelist handles non-existent paths via resolve()
        self._validate_path_in_whitelist(path, correlation_id, operation)

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(path),
            correlation_id=correlation_id,
        )

        # Extract options
        exist_ok = payload.get("exist_ok", True)
        if not isinstance(exist_ok, bool):
            exist_ok = True

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Check if already exists
            already_existed = path.exists()

            if already_existed and not path.is_dir():
                raise InfraConnectionError(
                    f"Path exists but is not a directory: {path.name}",
                    context=ctx,
                )

            # If directory exists and exist_ok=False, raise error
            if already_existed and not exist_ok:
                raise InfraConnectionError(
                    f"Directory already exists: {path.name}",
                    context=ctx,
                )

            # Create directory
            created = False
            if not already_existed:
                try:
                    # Final symlink check immediately before I/O to minimize TOCTOU window.
                    # An attacker could create a symlink at the target path between earlier
                    # validation and this point. Re-checking here reduces the window to
                    # the minimum possible (between this check and the actual mkdir).
                    # For mkdir, we also check parent directories in case a symlink was
                    # inserted in the path hierarchy.
                    if path.is_symlink():
                        self._validate_symlink_target(path, correlation_id, operation)
                    # Also check if any parent became a symlink
                    for parent in path.parents:
                        if parent.is_symlink():
                            self._validate_symlink_target(
                                parent, correlation_id, operation
                            )
                        # Stop at allowed paths boundary
                        if parent in self._allowed_paths:
                            break

                    path.mkdir(parents=True, exist_ok=exist_ok)
                    created = True
                except FileExistsError:
                    # FileExistsError is only raised when exist_ok=False
                    # (when exist_ok=True, mkdir() silently succeeds)
                    raise InfraConnectionError(
                        f"Directory already exists: {path.name}",
                        context=ctx,
                    ) from None
                except OSError as e:
                    raise InfraConnectionError(
                        f"Failed to create directory: {e}",
                        context=ctx,
                    ) from e

            resolved_path = path.resolve()

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Directory ensured",
                extra={
                    "path": str(resolved_path),
                    "dir_created": created,
                    "already_existed": already_existed,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_FILESYSTEM,
                result={
                    "status": "success",
                    "payload": {
                        "path": str(resolved_path),
                        "created": created,
                        "already_existed": already_existed,
                    },
                    "correlation_id": str(correlation_id),
                },
            )

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise

    async def _execute_delete_file(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute filesystem.delete_file operation.

        Payload:
            - path: str (required) - File path to delete
            - missing_ok: bool (optional, default False) - Don't error if missing

        Returns:
            Result with path, deleted, and was_missing flags

        Raises:
            InfraConnectionError: If delete fails or file not found (when missing_ok=False)
            InfraUnavailableError: If circuit breaker is open
        """
        operation = "filesystem.delete_file"

        # Extract path (required)
        path_raw = payload.get("path")
        if not isinstance(path_raw, str) or not path_raw:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="filesystem_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'path' in payload", context=ctx
            )

        path = Path(path_raw)

        # For delete, we need to validate the parent directory is in whitelist
        if path.parent.exists():
            self._validate_path_in_whitelist(path.parent, correlation_id, operation)
        else:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name=str(path),
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Path '{path}' is outside allowed directories - access denied",
                context=ctx,
            )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(path),
            correlation_id=correlation_id,
        )

        # Extract options
        missing_ok = payload.get("missing_ok", False)
        if not isinstance(missing_ok, bool):
            missing_ok = False

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Check if file exists
            was_missing = not path.exists()

            if was_missing:
                if not missing_ok:
                    raise InfraConnectionError(
                        f"File not found: {path.name}",
                        context=ctx,
                    )
                resolved_path = path.parent.resolve() / path.name
                deleted = False
            else:
                # Validate full path and symlink
                resolved_path = self._validate_path_in_whitelist(
                    path, correlation_id, operation
                )
                self._validate_symlink_target(resolved_path, correlation_id, operation)

                if resolved_path.is_dir():
                    raise InfraConnectionError(
                        f"Path is a directory, use rmdir for directories: {path.name}",
                        context=ctx,
                    )

                # Delete file
                try:
                    # Final symlink check immediately before I/O to minimize TOCTOU window.
                    # An attacker could replace the file with a symlink between earlier
                    # validation and this point. Re-checking here reduces the window to
                    # the minimum possible (between this check and the actual unlink).
                    if resolved_path.is_symlink():
                        self._validate_symlink_target(
                            resolved_path, correlation_id, operation
                        )

                    resolved_path.unlink()
                    deleted = True
                except OSError as e:
                    raise InfraConnectionError(
                        f"Failed to delete file: {e}",
                        context=ctx,
                    ) from e

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "File delete operation completed",
                extra={
                    "path": str(resolved_path),
                    "deleted": deleted if not was_missing else False,
                    "was_missing": was_missing,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_FILESYSTEM,
                result={
                    "status": "success",
                    "payload": {
                        "path": str(resolved_path),
                        "deleted": deleted if not was_missing else False,
                        "was_missing": was_missing,
                    },
                    "correlation_id": str(correlation_id),
                },
            )

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities for introspection.

        This method exposes the handler's three-dimensional type classification
        along with its operational configuration and capabilities.

        Returns:
            dict containing:
                - handler_type: Architectural role from handler_type property
                - handler_category: Behavioral classification from handler_category property
                - transport_type: Protocol identifier from transport_type property
                - supported_operations: List of supported operations
                - allowed_paths: List of allowed directory paths (when initialized)
                - max_read_size: Maximum read size in bytes
                - max_write_size: Maximum write size in bytes
                - initialized: Whether the handler is initialized
                - version: Handler version string
        """
        return {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "transport_type": self.transport_type.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "allowed_paths": [str(p) for p in self._allowed_paths],
            "max_read_size": self._max_read_size,
            "max_write_size": self._max_write_size,
            "initialized": self._initialized,
            "version": "0.1.0",
        }


__all__: list[str] = ["HandlerFileSystem"]
