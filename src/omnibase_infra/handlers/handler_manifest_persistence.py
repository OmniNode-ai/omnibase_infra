# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest Persistence Handler - Stores execution manifests to filesystem.

This handler persists ModelExecutionManifest objects to the filesystem with
date-based partitioning, atomic writes, and query support.

Supported Operations:
    - manifest.store: Store a manifest (idempotent by manifest_id)
    - manifest.retrieve: Retrieve a manifest by ID
    - manifest.query: Query manifests with filters (correlation_id, node_id, date range)

Storage Structure:
    manifests/
        2025/
            01/
                14/
                    {manifest_id}.json

Security Features:
    - Atomic writes using temp file + rename (prevents partial writes)
    - Idempotent storage (existing manifests are not overwritten)
    - Circuit breaker for resilient I/O operations

TOCTOU Race Condition Behavior:
    This handler has inherent Time-Of-Check-Time-Of-Use (TOCTOU) race conditions
    due to filesystem operations. These are documented for transparency:

    **manifest.store (idempotency check)**:
        The check ``file_path.exists()`` and subsequent write are not atomic.
        Between the existence check and the write, another process could:
        - Create the same file (harmless: atomic rename will overwrite or fail)
        - Delete the file (harmless: write will succeed)

        Mitigation: Atomic writes use temp file + rename. On POSIX systems, rename()
        is atomic within the same filesystem. The worst case is two concurrent writes
        for the same manifest_id both succeed, but they write identical content.

    **manifest.retrieve/query (directory scan)**:
        Directory iteration via ``iterdir()`` returns a point-in-time snapshot.
        Files may be added or removed during iteration. This is acceptable because:
        - Manifests are append-only (never deleted during normal operation)
        - Query results are best-effort snapshots, not transactional reads

    **Deployment Considerations**:
        - For multi-process deployments writing to shared storage, use a database
          backend instead of filesystem storage for strong consistency guarantees.
        - Single-process deployments (typical ONEX node) have no TOCTOU concerns.
        - NFS and network filesystems may have weaker atomicity guarantees than
          local filesystems; test rename behavior on your target storage.

Performance Characteristics (O(n) Directory Scan):
    This handler uses O(n) directory scanning for retrieve and query operations,
    where n is the total number of manifest files across all date partitions.

    **Why O(n) is acceptable for current use case**:
        - Manifest operations are low-frequency (debugging, auditing, troubleshooting)
        - Date-based partitioning enables manual pruning of old directories
        - Typical deployments have <10,000 manifests
        - Recent manifests (most common access pattern) are found quickly due to
          reverse-chronological iteration

    **Scaling recommendations for high-volume deployments**:
        - **>10k manifests**: Consider adding an index file (manifest_id -> path mapping)
        - **>100k manifests**: Consider SQLite or PostgreSQL backend with indexed queries
        - **>1M manifests**: Use dedicated manifest storage service with sharding

    **Alternative approaches not implemented**:
        - Bloom filter for fast negative lookups (adds complexity, marginal benefit)
        - In-memory manifest_id index (memory overhead, persistence complexity)
        - Filename encoding of creation date (breaks existing storage format)

Datetime Handling:
    All datetime values (created_at, created_after, created_before) should be
    timezone-aware for accurate comparisons. ISO 8601 strings with timezone info
    (e.g., "2025-01-14T12:00:00+00:00" or "2025-01-14T12:00:00Z") are parsed
    correctly. Naive datetimes may cause comparison issues when filtering.

    Timezone Awareness:
        - ISO strings with "Z" suffix are converted to UTC (+00:00)
        - ISO strings with explicit offset (e.g., "+05:00") are preserved
        - Naive datetime objects passed directly are accepted but logged as warnings
        - Comparisons between aware and naive datetimes will raise TypeError in Python 3
        - Best practice: Always use timezone-aware datetimes (e.g., datetime.now(timezone.utc))

Note:
    Environment variable configuration (ONEX_MANIFEST_MAX_FILE_SIZE) is parsed
    at module import time, not at handler instantiation. This means:

    - Changes to environment variables require application restart to take effect
    - Tests should use ``unittest.mock.patch.dict(os.environ, ...)`` before importing,
      or use ``importlib.reload()`` to re-import the module after patching
    - This is an intentional design choice for startup-time validation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumRetryErrorCategory,
)
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.models import ModelRetryState
from omnibase_infra.handlers.models.model_manifest_metadata import ModelManifestMetadata
from omnibase_infra.handlers.models.model_manifest_query_result import (
    ModelManifestQueryResult,
)
from omnibase_infra.handlers.models.model_manifest_retrieve_result import (
    ModelManifestRetrieveResult,
)
from omnibase_infra.handlers.models.model_manifest_store_result import (
    ModelManifestStoreResult,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction
from omnibase_infra.mixins.mixin_retry_execution import MixinRetryExecution
from omnibase_infra.models.model_retry_error_classification import (
    ModelRetryErrorClassification,
)
from omnibase_infra.utils import parse_env_int, warn_if_naive_datetime

logger = logging.getLogger(__name__)

# Default configuration from environment
_DEFAULT_MAX_FILE_SIZE: int = parse_env_int(
    "ONEX_MANIFEST_MAX_FILE_SIZE",
    50 * 1024 * 1024,  # 50 MB
    min_value=1024,
    max_value=500 * 1024 * 1024,  # 500 MB
    transport_type=EnumInfraTransportType.FILESYSTEM,
    service_name="manifest_persistence_handler",
)

_SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "manifest.store",
        "manifest.retrieve",
        "manifest.query",
    }
)

HANDLER_ID_MANIFEST_PERSISTENCE: str = "manifest-persistence-handler"


class HandlerManifestPersistence(
    MixinEnvelopeExtraction, MixinAsyncCircuitBreaker, MixinRetryExecution
):
    """Manifest persistence handler for storing/retrieving ModelExecutionManifest.

    This handler stores ModelExecutionManifest objects to the filesystem with:
    - Date-based partitioning (year/month/day directories)
    - Atomic writes (write to temp, then rename)
    - Idempotent storage (same manifest_id = no duplicate)
    - Query support with filters
    - Circuit breaker for resilient I/O operations
    - Retry with exponential backoff for transient I/O errors

    Storage Pattern:
        {storage_path}/{year}/{month}/{day}/{manifest_id}.json

        Example: /data/manifests/2025/01/14/550e8400-e29b-41d4-a716-446655440000.json

    Attributes:
        handler_type: Returns INFRA_HANDLER (infrastructure protocol handler)
        handler_category: Returns EFFECT (side-effecting I/O)

    Example:
        >>> handler = HandlerManifestPersistence(container)
        >>> await handler.initialize({"storage_path": "/data/manifests"})
        >>> result = await handler.execute({
        ...     "operation": "manifest.store",
        ...     "payload": {"manifest": manifest.model_dump()},
        ... })
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerManifestPersistence with required container injection.

        Args:
            container: ONEX container for dependency injection. Required per ONEX
                pattern (``def __init__(self, container: ModelONEXContainer)``).
                Enables full ONEX integration (logging, metrics, service discovery).

        See Also:
            - CLAUDE.md "Container-Based Dependency Injection" section for the
              standard ONEX container injection pattern.
            - docs/patterns/container_dependency_injection.md for detailed DI patterns.
        """
        self._container = container
        self._storage_path: Path | None = None
        self._max_file_size: int = _DEFAULT_MAX_FILE_SIZE
        self._initialized: bool = False

        # Retry configuration (populated from contract in initialize())
        self._retry_config: dict[str, float] = {
            "max_retries": 3,
            "initial_delay_seconds": 0.1,  # 100ms
            "max_delay_seconds": 5.0,  # 5000ms
            "exponential_base": 2.0,
        }

        # Required by MixinRetryExecution (no thread pool needed for async I/O)
        self._executor = None

        # Required by MixinRetryExecution for circuit breaker integration check
        # Set to True after _init_circuit_breaker() is called in initialize()
        self._circuit_breaker_initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            protocol/transport handler for manifest persistence operations.

        Note:
            handler_type determines lifecycle, protocol selection, and runtime
            invocation patterns. It answers "what is this handler in the architecture?"

        See Also:
            - handler_category: Behavioral classification (EFFECT/COMPUTE)
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
        """
        return EnumHandlerTypeCategory.EFFECT

    # =========================================================================
    # MixinRetryExecution Abstract Method Implementations
    # =========================================================================

    def _classify_error(
        self, error: Exception, operation: str
    ) -> ModelRetryErrorClassification:
        """Classify filesystem errors for retry handling.

        This method determines whether an error is retriable and how it should
        affect the circuit breaker state.

        Args:
            error: The exception to classify.
            operation: The operation name for context.

        Returns:
            ModelRetryErrorClassification with retry decision and error details.

        Error Classification:
            - TimeoutError: TIMEOUT category, retriable, records circuit failure
            - BlockingIOError: TIMEOUT category, retriable, records circuit failure
              (EAGAIN/EWOULDBLOCK - resource temporarily unavailable)
            - FileNotFoundError: NOT_FOUND category, NOT retriable, NO circuit failure
            - PermissionError: AUTHENTICATION category, NOT retriable, records circuit failure
            - OSError/IOError: CONNECTION category, retriable, records circuit failure
            - Other: UNKNOWN category, retriable, records circuit failure

        Note:
            BlockingIOError must be checked BEFORE OSError since it's a subclass.
            We classify BlockingIOError as TIMEOUT rather than CONNECTION because
            it indicates "resource temporarily unavailable" which is semantically
            closer to a timeout condition than a connection error.
        """
        if isinstance(error, TimeoutError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Filesystem operation timed out: {operation}",
            )

        # BlockingIOError indicates EAGAIN/EWOULDBLOCK (resource temporarily unavailable)
        # Must be checked before OSError since it's a subclass
        if isinstance(error, BlockingIOError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Resource temporarily unavailable: {operation}",
            )

        if isinstance(error, FileNotFoundError):
            # File not found is a user/logic error, not infrastructure failure
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.NOT_FOUND,
                should_retry=False,
                record_circuit_failure=False,
                error_message=f"File not found: {error}",
            )

        if isinstance(error, PermissionError):
            # Permission errors are not retriable and indicate config issues
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.AUTHENTICATION,
                should_retry=False,
                record_circuit_failure=True,
                error_message=f"Permission denied: {operation}",
            )

        if isinstance(error, OSError | IOError):
            # General I/O errors are retriable (disk full, temp unavailable, etc.)
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.CONNECTION,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Filesystem I/O error: {type(error).__name__}",
            )

        # Unknown errors - retry but record failure
        return ModelRetryErrorClassification(
            category=EnumRetryErrorCategory.UNKNOWN,
            should_retry=True,
            record_circuit_failure=True,
            error_message=f"Unexpected error: {type(error).__name__}",
        )

    def _get_transport_type(self) -> EnumInfraTransportType:
        """Return the transport type for error context.

        Returns:
            EnumInfraTransportType.FILESYSTEM for filesystem operations.
        """
        return EnumInfraTransportType.FILESYSTEM

    def _get_target_name(self) -> str:
        """Return the target name for error context.

        Returns:
            The handler identifier for error context and logging.
        """
        return "manifest_persistence_handler"

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize manifest persistence handler with storage path.

        Args:
            config: Configuration dict containing:
                - storage_path: Required path to manifest storage directory
                - max_file_size: Optional max file size in bytes (default: 50 MB)
                - correlation_id: Optional UUID or string for error tracing

        Raises:
            ProtocolConfigurationError: If storage_path is missing or invalid.

        Security:
            - Storage directory is created if it doesn't exist
            - Non-writable paths are logged as warnings
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
            target_name="manifest_persistence_handler",
            correlation_id=init_correlation_id,
        )

        # Extract and validate storage_path (required)
        storage_path_raw = config.get("storage_path")
        if storage_path_raw is None:
            raise ProtocolConfigurationError(
                "Missing required 'storage_path' configuration - manifest persistence "
                "handler requires a storage directory path",
                context=ctx,
            )

        if not isinstance(storage_path_raw, str) or not storage_path_raw:
            raise ProtocolConfigurationError(
                "Configuration 'storage_path' must be a non-empty string",
                context=ctx,
            )

        # Resolve to absolute path
        storage_path = Path(storage_path_raw).resolve()

        # Create storage directory if it doesn't exist
        if not storage_path.exists():
            try:
                storage_path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Created manifest storage directory: %s",
                    storage_path,
                    extra={
                        "path": str(storage_path),
                        "correlation_id": str(init_correlation_id),
                    },
                )
            except OSError as e:
                raise ProtocolConfigurationError(
                    f"Failed to create storage directory: {e}",
                    context=ctx,
                ) from e

        if not storage_path.is_dir():
            raise ProtocolConfigurationError(
                f"Storage path exists but is not a directory: {storage_path}",
                context=ctx,
            )

        self._storage_path = storage_path

        # Verify storage path is writable (fail fast on permission issues)
        test_file = self._storage_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            raise ProtocolConfigurationError(
                f"Storage path is not writable: {self._storage_path}",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.FILESYSTEM,
                    operation="initialize",
                    target_name="manifest_persistence_handler",
                    correlation_id=init_correlation_id,
                ),
            ) from e

        # Extract optional max_file_size
        max_file_size_raw = config.get("max_file_size")
        if max_file_size_raw is not None:
            if isinstance(max_file_size_raw, int) and max_file_size_raw > 0:
                self._max_file_size = max_file_size_raw
            else:
                logger.warning(
                    "Invalid max_file_size config value ignored, using default",
                    extra={
                        "provided_value": max_file_size_raw,
                        "default_value": self._max_file_size,
                    },
                )

        # Initialize circuit breaker for resilient I/O operations
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="manifest_persistence_handler",
            transport_type=EnumInfraTransportType.FILESYSTEM,
        )
        # Mark circuit breaker as initialized for MixinRetryExecution integration
        self._circuit_breaker_initialized = True

        # Parse retry_policy from configuration (matches contract defaults)
        # Fail-fast: Invalid retry config raises ProtocolConfigurationError
        retry_policy = config.get("retry_policy")
        if isinstance(retry_policy, dict):
            # max_retries (default: 3) - must be positive integer
            max_retries = retry_policy.get("max_retries")
            if max_retries is not None:
                if not isinstance(max_retries, int) or max_retries <= 0:
                    raise ProtocolConfigurationError(
                        "Invalid retry_policy.max_retries: must be a positive "
                        f"integer, got {type(max_retries).__name__}={max_retries!r}",
                        context=ctx,
                    )
                self._retry_config["max_retries"] = max_retries

            # initial_delay_ms -> convert to seconds (default: 100ms = 0.1s)
            initial_delay_ms = retry_policy.get("initial_delay_ms")
            if initial_delay_ms is not None:
                is_valid_type = isinstance(initial_delay_ms, int | float)
                if not is_valid_type or initial_delay_ms <= 0:
                    raise ProtocolConfigurationError(
                        "Invalid retry_policy.initial_delay_ms: must be a "
                        f"positive number, got "
                        f"{type(initial_delay_ms).__name__}={initial_delay_ms!r}",
                        context=ctx,
                    )
                self._retry_config["initial_delay_seconds"] = initial_delay_ms / 1000.0

            # max_delay_ms -> convert to seconds (default: 5000ms = 5.0s)
            max_delay_ms = retry_policy.get("max_delay_ms")
            if max_delay_ms is not None:
                if not isinstance(max_delay_ms, int | float) or max_delay_ms <= 0:
                    raise ProtocolConfigurationError(
                        "Invalid retry_policy.max_delay_ms: must be a positive "
                        f"number, got {type(max_delay_ms).__name__}={max_delay_ms!r}",
                        context=ctx,
                    )
                self._retry_config["max_delay_seconds"] = max_delay_ms / 1000.0

            # exponential_base (default: 2.0) - must be >= 1.0
            exponential_base = retry_policy.get("exponential_base")
            if exponential_base is not None:
                is_valid_type = isinstance(exponential_base, int | float)
                if not is_valid_type or exponential_base < 1.0:
                    raise ProtocolConfigurationError(
                        "Invalid retry_policy.exponential_base: must be a "
                        f"number >= 1.0, got "
                        f"{type(exponential_base).__name__}={exponential_base!r}",
                        context=ctx,
                    )
                self._retry_config["exponential_base"] = float(exponential_base)

            logger.debug(
                "Retry policy configured",
                extra={
                    "max_retries": self._retry_config["max_retries"],
                    "initial_delay_seconds": self._retry_config[
                        "initial_delay_seconds"
                    ],
                    "max_delay_seconds": self._retry_config["max_delay_seconds"],
                    "exponential_base": self._retry_config["exponential_base"],
                    "correlation_id": str(init_correlation_id),
                },
            )

        self._initialized = True

        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "storage_path": str(self._storage_path),
                "max_file_size_bytes": self._max_file_size,
                "correlation_id": str(init_correlation_id),
            },
        )

    async def shutdown(self) -> None:
        """Shutdown manifest persistence handler and clear configuration."""
        self._storage_path = None
        self._initialized = False
        logger.info("HandlerManifestPersistence shutdown complete")

    # =========================================================================
    # Retry Logic Helper
    # =========================================================================

    _T = TypeVar("_T")

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], Awaitable[_T]],
        correlation_id: UUID,
    ) -> _T:
        """Execute an async operation with exponential backoff retry logic.

        This method wraps I/O operations with retry logic, integrating with the
        circuit breaker for resilient operations.

        Args:
            operation: Operation name for logging and error context.
            func: Async callable to execute (returns the result).
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            The result from func().

        Raises:
            InfraTimeoutError: If operation times out after retries exhausted.
            InfraConnectionError: If connection fails after retries exhausted.
            InfraAuthenticationError: If authentication fails (not retriable).
            InfraUnavailableError: If circuit breaker is OPEN.

        Circuit Breaker Integration:
            - Checks circuit state before execution (raises if OPEN)
            - Records success/failure for circuit state management
            - Failure recorded only when retries are exhausted

        Retry Logic:
            - Uses exponential backoff with configurable parameters
            - Classifies errors to determine retry eligibility
            - Logs retry attempts with correlation tracking
        """
        # Check circuit breaker before execution
        await self._check_circuit_if_enabled(operation, correlation_id)

        # Initialize retry state from configuration
        retry_state = ModelRetryState(
            attempt=0,
            max_attempts=int(self._retry_config["max_retries"]) + 1,  # +1 for initial
            delay_seconds=float(self._retry_config["initial_delay_seconds"]),
            backoff_multiplier=float(self._retry_config["exponential_base"]),
        )

        max_delay_seconds = float(self._retry_config["max_delay_seconds"])
        last_error: Exception | None = None

        while retry_state.is_retriable():
            try:
                result = await func()
                # Reset circuit breaker on success
                await self._reset_circuit_if_enabled()
                return result

            except Exception as e:
                last_error = e
                classification = self._classify_error(e, operation)

                if not classification.should_retry:
                    # Non-retriable error - record failure and raise immediately
                    if classification.record_circuit_failure:
                        await self._record_circuit_failure_if_enabled(
                            operation, correlation_id
                        )
                    raise

                # Update retry state for next attempt
                retry_state = retry_state.next_attempt(
                    error_message=classification.error_message,
                    max_delay_seconds=max_delay_seconds,
                )

                if not retry_state.is_retriable():
                    # Retries exhausted - record failure and raise
                    if classification.record_circuit_failure:
                        await self._record_circuit_failure_if_enabled(
                            operation, correlation_id
                        )
                    raise

                # Log retry attempt
                await self._log_retry_attempt(operation, retry_state, correlation_id)

                # Wait before next attempt
                await asyncio.sleep(retry_state.delay_seconds)

        # Should never reach here, but satisfy type checker
        if last_error is not None:
            raise last_error
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name="manifest_persistence_handler",
            correlation_id=correlation_id,
        )
        raise InfraUnavailableError(
            f"Retry loop completed without result for {operation}",
            context=ctx,
        )

    def _get_manifest_path(self, manifest_id: UUID, created_at: datetime) -> Path:
        """Get the file path for a manifest based on ID and creation date.

        Args:
            manifest_id: Unique identifier of the manifest
            created_at: Creation timestamp for date partitioning

        Returns:
            Path: {storage_path}/{year}/{month:02d}/{day:02d}/{manifest_id}.json
        """
        if self._storage_path is None:
            raise RuntimeHostError(
                "Handler not initialized - storage_path is None",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.FILESYSTEM,
                    operation="get_manifest_path",
                    target_name="manifest_persistence_handler",
                ),
            )

        return (
            self._storage_path
            / str(created_at.year)
            / f"{created_at.month:02d}"
            / f"{created_at.day:02d}"
            / f"{manifest_id}.json"
        )

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute manifest persistence operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: One of the supported manifest operations
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
                target_name="manifest_persistence_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HandlerManifestPersistence not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="execute",
                target_name="manifest_persistence_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'operation' in envelope", context=ctx
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="manifest_persistence_handler",
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
                target_name="manifest_persistence_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'payload' in envelope", context=ctx
            )

        # Route to appropriate operation handler
        if operation == "manifest.store":
            return await self._execute_store(payload, correlation_id, input_envelope_id)
        elif operation == "manifest.retrieve":
            return await self._execute_retrieve(
                payload, correlation_id, input_envelope_id
            )
        else:  # manifest.query
            return await self._execute_query(payload, correlation_id, input_envelope_id)

    async def _execute_store(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute manifest.store operation with retry logic.

        Stores a manifest with atomic write (temp file + rename) and
        idempotent behavior (existing manifests are not overwritten).

        Payload:
            - manifest: dict (required) - Serialized ModelExecutionManifest

        Returns:
            Result with manifest_id, file_path, created, and bytes_written.

        Raises:
            InfraConnectionError: If write fails after retries exhausted
            InfraUnavailableError: If circuit breaker is open
        """
        operation = "manifest.store"

        # Extract manifest (required)
        manifest_raw = payload.get("manifest")
        if not isinstance(manifest_raw, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation=operation,
                target_name="manifest_persistence_handler",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'manifest' in payload - must be a dictionary",
                context=ctx,
            )

        # Extract required fields from manifest
        manifest_id_raw = manifest_raw.get("manifest_id")
        created_at_raw = manifest_raw.get("created_at")

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name="manifest_persistence_handler",
            correlation_id=correlation_id,
        )

        # Parse manifest_id
        try:
            if isinstance(manifest_id_raw, UUID):
                manifest_id = manifest_id_raw
            elif isinstance(manifest_id_raw, str):
                manifest_id = UUID(manifest_id_raw)
            else:
                raise ProtocolConfigurationError(
                    "Manifest missing required 'manifest_id' field or invalid type",
                    context=ctx,
                )
        except ValueError as e:
            raise ProtocolConfigurationError(
                f"Invalid manifest_id format: {e}",
                context=ctx,
            ) from e

        # Parse created_at
        try:
            if isinstance(created_at_raw, datetime):
                created_at = created_at_raw
                warn_if_naive_datetime(
                    created_at, field_name="created_at", correlation_id=correlation_id
                )
            elif isinstance(created_at_raw, str):
                # Try ISO format parsing (Z suffix converted to +00:00)
                created_at = datetime.fromisoformat(
                    created_at_raw.replace("Z", "+00:00")
                )
            else:
                raise ProtocolConfigurationError(
                    "Manifest missing required 'created_at' field or invalid type",
                    context=ctx,
                )
        except ValueError as e:
            raise ProtocolConfigurationError(
                f"Invalid created_at format: {e}",
                context=ctx,
            ) from e

        # Get file path
        file_path = self._get_manifest_path(manifest_id, created_at)

        async def _do_store_io() -> ModelHandlerOutput[dict[str, object]]:
            """Inner function containing I/O operations (wrapped with retry)."""
            # Check if manifest already exists (idempotent behavior)
            if file_path.exists():
                logger.debug(
                    "Manifest already exists, skipping write (idempotent)",
                    extra={
                        "manifest_id": str(manifest_id),
                        "path": str(file_path),
                        "correlation_id": str(correlation_id),
                    },
                )

                result = ModelManifestStoreResult(
                    manifest_id=manifest_id,
                    file_path=str(file_path),
                    created=False,
                    bytes_written=0,
                )

                return ModelHandlerOutput.for_compute(
                    input_envelope_id=input_envelope_id,
                    correlation_id=correlation_id,
                    handler_id=HANDLER_ID_MANIFEST_PERSISTENCE,
                    result={
                        "status": "success",
                        "payload": result.model_dump(mode="json"),
                        "correlation_id": str(correlation_id),
                    },
                )

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize manifest to JSON
            manifest_json = json.dumps(manifest_raw, indent=2, default=str)
            manifest_bytes = manifest_json.encode("utf-8")

            # Atomic write: write to temp file, then rename
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix=f"{manifest_id}_",
                dir=file_path.parent,
            )
            try:
                with os.fdopen(temp_fd, "wb") as f:
                    f.write(manifest_bytes)
                # Atomic rename
                temp_path_obj = Path(temp_path)
                temp_path_obj.rename(file_path)
            except OSError:
                # Clean up temp file on failure
                temp_path_obj = Path(temp_path)
                if temp_path_obj.exists():
                    temp_path_obj.unlink()
                raise

            bytes_written = len(manifest_bytes)

            logger.debug(
                "Manifest stored successfully",
                extra={
                    "manifest_id": str(manifest_id),
                    "path": str(file_path),
                    "bytes_written": bytes_written,
                    "correlation_id": str(correlation_id),
                },
            )

            result = ModelManifestStoreResult(
                manifest_id=manifest_id,
                file_path=str(file_path),
                created=True,
                bytes_written=bytes_written,
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_MANIFEST_PERSISTENCE,
                result={
                    "status": "success",
                    "payload": result.model_dump(mode="json"),
                    "correlation_id": str(correlation_id),
                },
            )

        # Execute with retry logic (handles circuit breaker and backoff)
        return await self._execute_with_retry(operation, _do_store_io, correlation_id)

    async def _execute_retrieve(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute manifest.retrieve operation with retry logic.

        Retrieves a manifest by scanning date directories.

        Complexity:
            O(d) where d is the number of date directories (year/month/day).
            This is a full directory scan because manifest_id does not encode
            the creation date, requiring us to search all partitions. This is
            acceptable for the current use case (low query volume, typically
            recent manifests). For high-volume retrieval patterns, consider
            maintaining a separate index file or using the query operation
            with correlation_id filter.

        Payload:
            - manifest_id: UUID or string (required) - Manifest to retrieve

        Returns:
            Result with manifest_id, manifest data, file_path, and found flag.

        Raises:
            InfraConnectionError: If read fails after retries exhausted
            InfraUnavailableError: If circuit breaker is open
        """
        operation = "manifest.retrieve"

        # Extract manifest_id (required)
        manifest_id_raw = payload.get("manifest_id")
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name="manifest_persistence_handler",
            correlation_id=correlation_id,
        )

        try:
            if isinstance(manifest_id_raw, UUID):
                manifest_id = manifest_id_raw
            elif isinstance(manifest_id_raw, str):
                manifest_id = UUID(manifest_id_raw)
            else:
                raise ProtocolConfigurationError(
                    "Missing or invalid 'manifest_id' in payload",
                    context=ctx,
                )
        except ValueError as e:
            raise ProtocolConfigurationError(
                f"Invalid manifest_id format: {e}",
                context=ctx,
            ) from e

        async def _do_retrieve_io() -> ModelHandlerOutput[dict[str, object]]:
            """Inner function containing I/O operations (wrapped with retry)."""
            # Search for manifest in date directories
            found_path: Path | None = None
            manifest_data: dict[str, object] | None = None

            if self._storage_path is None:
                raise RuntimeHostError(
                    "Handler not initialized - storage_path is None",
                    context=ctx,
                )

            # Scan year/month/day directories with performance tracking
            scan_start_time = time.monotonic()
            directories_scanned = 0

            for year_dir in sorted(self._storage_path.iterdir(), reverse=True):
                if not year_dir.is_dir():
                    continue
                directories_scanned += 1
                for month_dir in sorted(year_dir.iterdir(), reverse=True):
                    if not month_dir.is_dir():
                        continue
                    directories_scanned += 1
                    for day_dir in sorted(month_dir.iterdir(), reverse=True):
                        if not day_dir.is_dir():
                            continue
                        directories_scanned += 1
                        manifest_file = day_dir / f"{manifest_id}.json"
                        if manifest_file.exists():
                            found_path = manifest_file
                            break
                    if found_path:
                        break
                if found_path:
                    break

            scan_duration = time.monotonic() - scan_start_time
            logger.debug(
                "Directory scan completed for retrieve",
                extra={
                    "duration_seconds": round(scan_duration, 6),
                    "directories_scanned": directories_scanned,
                    "manifest_found": found_path is not None,
                    "manifest_id": str(manifest_id),
                    "correlation_id": str(correlation_id),
                },
            )

            if found_path:
                # Check file size before reading
                file_size = found_path.stat().st_size
                if file_size > self._max_file_size:
                    raise InfraUnavailableError(
                        "Manifest file size exceeds configured limit",
                        context=ctx,
                    )

                # Read and parse manifest
                manifest_json = found_path.read_text(encoding="utf-8")
                manifest_data = json.loads(manifest_json)

            result = ModelManifestRetrieveResult(
                manifest_id=manifest_id,
                manifest=manifest_data,
                file_path=str(found_path) if found_path else None,
                found=found_path is not None,
            )

            logger.debug(
                "Manifest retrieve completed",
                extra={
                    "manifest_id": str(manifest_id),
                    "found": result.found,
                    "path": str(found_path) if found_path else None,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_MANIFEST_PERSISTENCE,
                result={
                    "status": "success",
                    "payload": result.model_dump(mode="json"),
                    "correlation_id": str(correlation_id),
                },
            )

        # Execute with retry logic (handles circuit breaker and backoff)
        return await self._execute_with_retry(
            operation, _do_retrieve_io, correlation_id
        )

    async def _execute_query(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute manifest.query operation with retry logic.

        Queries manifests with filters and respects metadata_only flag.

        Complexity:
            O(n) where n is the total number of manifest files. Each file must
            be read and parsed to apply filters. The limit parameter provides
            early termination but worst case (few matches) scans all files.
            This is acceptable for the current use case where:
            - Query operations are infrequent (debugging, auditing)
            - Date-based partitioning enables manual pruning of old directories
            - Typical deployments have <10k manifests

        Payload:
            - correlation_id: UUID or string (optional) - Filter by correlation_id
            - node_id: string (optional) - Filter by node_id
            - created_after: datetime or ISO string (optional) - Filter by creation time
            - created_before: datetime or ISO string (optional) - Filter by creation time
            - metadata_only: bool (optional, default False) - Return only metadata
            - limit: int (optional, default 100) - Maximum results

        Returns:
            Result with manifests list, total_count, and metadata_only flag.

        Raises:
            InfraConnectionError: If read fails after retries exhausted
            InfraUnavailableError: If circuit breaker is open
        """
        operation = "manifest.query"

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name="manifest_persistence_handler",
            correlation_id=correlation_id,
        )

        # Extract filter parameters
        filter_correlation_id: UUID | None = None
        correlation_id_raw = payload.get("correlation_id")
        if correlation_id_raw is not None:
            try:
                if isinstance(correlation_id_raw, UUID):
                    filter_correlation_id = correlation_id_raw
                elif isinstance(correlation_id_raw, str):
                    filter_correlation_id = UUID(correlation_id_raw)
                else:
                    logger.warning(
                        "Invalid correlation_id filter type, ignoring filter",
                        extra={
                            "provided_type": type(correlation_id_raw).__name__,
                            "correlation_id": str(correlation_id),
                        },
                    )
            except ValueError as e:
                logger.warning(
                    "Invalid correlation_id filter format, ignoring filter",
                    extra={
                        "provided_value": str(correlation_id_raw)[:100],
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        filter_node_id: str | None = None
        node_id_raw = payload.get("node_id")
        if node_id_raw is not None:
            if isinstance(node_id_raw, str):
                filter_node_id = node_id_raw
            else:
                logger.warning(
                    "Invalid node_id filter type, ignoring filter",
                    extra={
                        "provided_type": type(node_id_raw).__name__,
                        "correlation_id": str(correlation_id),
                    },
                )

        filter_created_after: datetime | None = None
        created_after_raw = payload.get("created_after")
        if created_after_raw is not None:
            try:
                if isinstance(created_after_raw, datetime):
                    filter_created_after = created_after_raw
                    warn_if_naive_datetime(
                        filter_created_after,
                        field_name="created_after",
                        correlation_id=correlation_id,
                    )
                elif isinstance(created_after_raw, str):
                    filter_created_after = datetime.fromisoformat(
                        created_after_raw.replace("Z", "+00:00")
                    )
                else:
                    logger.warning(
                        "Invalid created_after filter type, ignoring filter",
                        extra={
                            "provided_type": type(created_after_raw).__name__,
                            "correlation_id": str(correlation_id),
                        },
                    )
            except ValueError as e:
                logger.warning(
                    "Invalid created_after filter format, ignoring filter",
                    extra={
                        "provided_value": str(created_after_raw)[:100],
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        filter_created_before: datetime | None = None
        created_before_raw = payload.get("created_before")
        if created_before_raw is not None:
            try:
                if isinstance(created_before_raw, datetime):
                    filter_created_before = created_before_raw
                    warn_if_naive_datetime(
                        filter_created_before,
                        field_name="created_before",
                        correlation_id=correlation_id,
                    )
                elif isinstance(created_before_raw, str):
                    filter_created_before = datetime.fromisoformat(
                        created_before_raw.replace("Z", "+00:00")
                    )
                else:
                    logger.warning(
                        "Invalid created_before filter type, ignoring filter",
                        extra={
                            "provided_type": type(created_before_raw).__name__,
                            "correlation_id": str(correlation_id),
                        },
                    )
            except ValueError as e:
                logger.warning(
                    "Invalid created_before filter format, ignoring filter",
                    extra={
                        "provided_value": str(created_before_raw)[:100],
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        metadata_only_raw = payload.get("metadata_only", False)
        if isinstance(metadata_only_raw, bool):
            metadata_only = metadata_only_raw
        else:
            logger.warning(
                "Invalid metadata_only filter type, using default False",
                extra={
                    "provided_type": type(metadata_only_raw).__name__,
                    "provided_value": str(metadata_only_raw)[:100],
                    "correlation_id": str(correlation_id),
                },
            )
            metadata_only = False

        limit_raw = payload.get("limit", 100)
        if isinstance(limit_raw, int) and limit_raw >= 1:
            limit = min(limit_raw, 10000)  # Cap at 10000
        else:
            logger.warning(
                "Invalid limit filter value, using default 100",
                extra={
                    "provided_type": type(limit_raw).__name__,
                    "provided_value": str(limit_raw)[:100],
                    "correlation_id": str(correlation_id),
                },
            )
            limit = 100

        async def _do_query_io() -> ModelHandlerOutput[dict[str, object]]:
            """Inner function containing I/O operations (wrapped with retry)."""
            if self._storage_path is None:
                raise RuntimeHostError(
                    "Handler not initialized - storage_path is None",
                    context=ctx,
                )

            manifests_metadata: list[ModelManifestMetadata] = []
            manifests_data: list[dict[str, object]] = []
            count = 0

            # Scan date directories with performance tracking
            scan_start_time = time.monotonic()
            files_scanned = 0
            directories_scanned = 0

            for year_dir in sorted(self._storage_path.iterdir(), reverse=True):
                if not year_dir.is_dir() or count >= limit:
                    continue
                directories_scanned += 1
                for month_dir in sorted(year_dir.iterdir(), reverse=True):
                    if not month_dir.is_dir() or count >= limit:
                        continue
                    directories_scanned += 1
                    for day_dir in sorted(month_dir.iterdir(), reverse=True):
                        if not day_dir.is_dir() or count >= limit:
                            continue
                        directories_scanned += 1
                        for manifest_file in sorted(
                            day_dir.glob("*.json"), reverse=True
                        ):
                            if count >= limit:
                                break
                            files_scanned += 1

                            try:
                                file_stat = manifest_file.stat()
                                file_size = file_stat.st_size

                                # Skip files that are too large
                                if file_size > self._max_file_size:
                                    continue

                                # Full deserialization required to access filter
                                # fields (correlation_id, node_id, created_at)
                                # stored within the manifest JSON.
                                #
                                # The `metadata_only` flag controls the RETURN
                                # format (full manifest vs. summary), not the
                                # read pattern. This is a limitation of
                                # filesystem storage: filter fields are not
                                # available as external file metadata.
                                manifest_json = manifest_file.read_text(
                                    encoding="utf-8"
                                )
                                manifest_data = json.loads(manifest_json)

                                # Extract fields for filtering
                                manifest_id_str = manifest_data.get("manifest_id")
                                if not manifest_id_str:
                                    continue

                                try:
                                    manifest_id = UUID(str(manifest_id_str))
                                except ValueError:
                                    continue

                                created_at_str = manifest_data.get("created_at")
                                try:
                                    if isinstance(created_at_str, str):
                                        manifest_created_at = datetime.fromisoformat(
                                            created_at_str.replace("Z", "+00:00")
                                        )
                                    else:
                                        continue
                                except ValueError:
                                    continue

                                manifest_correlation_id: UUID | None = None
                                manifest_corr_id_raw = manifest_data.get(
                                    "correlation_id"
                                )
                                if manifest_corr_id_raw:
                                    try:
                                        manifest_correlation_id = UUID(
                                            str(manifest_corr_id_raw)
                                        )
                                    except ValueError:
                                        pass

                                node_identity = manifest_data.get("node_identity", {})
                                manifest_node_id = (
                                    node_identity.get("node_id")
                                    if isinstance(node_identity, dict)
                                    else None
                                )

                                # Apply filters
                                if filter_correlation_id is not None:
                                    if manifest_correlation_id != filter_correlation_id:
                                        continue

                                if filter_node_id is not None:
                                    if manifest_node_id != filter_node_id:
                                        continue

                                if filter_created_after is not None:
                                    if manifest_created_at < filter_created_after:
                                        continue

                                if filter_created_before is not None:
                                    if manifest_created_at > filter_created_before:
                                        continue

                                # Manifest passes filters
                                if metadata_only:
                                    metadata = ModelManifestMetadata(
                                        manifest_id=manifest_id,
                                        created_at=manifest_created_at,
                                        correlation_id=manifest_correlation_id,
                                        node_id=manifest_node_id,
                                        file_path=str(manifest_file),
                                        file_size=file_size,
                                    )
                                    manifests_metadata.append(metadata)
                                else:
                                    manifests_data.append(manifest_data)

                                count += 1

                            except (OSError, json.JSONDecodeError) as e:
                                logger.warning(
                                    "Failed to read manifest file: %s - %s",
                                    manifest_file,
                                    e,
                                    extra={
                                        "path": str(manifest_file),
                                        "error": str(e),
                                        "correlation_id": str(correlation_id),
                                    },
                                )
                                continue

            scan_duration = time.monotonic() - scan_start_time
            logger.debug(
                "Directory scan completed for query",
                extra={
                    "duration_seconds": round(scan_duration, 6),
                    "directories_scanned": directories_scanned,
                    "files_scanned": files_scanned,
                    "matches_found": count,
                    "limit": limit,
                    "correlation_id": str(correlation_id),
                },
            )

            result = ModelManifestQueryResult(
                manifests=manifests_metadata,
                manifest_data=manifests_data,
                total_count=count,
                metadata_only=metadata_only,
            )

            logger.debug(
                "Manifest query completed",
                extra={
                    "total_count": count,
                    "metadata_only": metadata_only,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelHandlerOutput.for_compute(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id=HANDLER_ID_MANIFEST_PERSISTENCE,
                result={
                    "status": "success",
                    "payload": result.model_dump(mode="json"),
                    "correlation_id": str(correlation_id),
                },
            )

        # Execute with retry logic (handles circuit breaker and backoff)
        return await self._execute_with_retry(operation, _do_query_io, correlation_id)

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities for introspection.

        This method exposes the handler's type classification along with
        its operational configuration and capabilities, including detailed
        circuit breaker state for operational observability.

        Returns:
            dict containing:
                - handler_type: Architectural role from handler_type property
                - handler_category: Behavioral classification
                - supported_operations: List of supported operations
                - storage_path: Storage directory path (when initialized)
                - initialized: Whether the handler is initialized
                - version: Handler version string
                - circuit_breaker: Circuit breaker state for observability
                    - initialized: Whether circuit breaker is initialized
                    - state: Current state ("closed", "open", or "half_open")
                    - failures: Current failure count
                    - threshold: Configured failure threshold before opening
                    - reset_timeout_seconds: Configured timeout before half_open transition
                    - seconds_until_half_open: Seconds remaining until half_open (only when open)

        Circuit Breaker States:
            - **closed**: Normal operation, requests allowed. Failures tracked.
            - **open**: Circuit tripped after threshold failures. Requests blocked.
              Will transition to half_open after reset_timeout_seconds.
            - **half_open**: Recovery testing phase. Next success closes circuit,
              next failure reopens it. This state is transient and detected when
              the circuit is marked open but the reset timeout has elapsed.
        """
        # Get circuit breaker state from mixin (encapsulated access)
        circuit_breaker_info = self._get_circuit_breaker_state()

        # Override initialized with handler's own flag for precise tracking
        # (handler may track initialization more granularly than mixin detection)
        circuit_breaker_info["initialized"] = self._circuit_breaker_initialized

        result: dict[str, object] = {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "storage_path": str(self._storage_path) if self._storage_path else None,
            "initialized": self._initialized,
            "version": "0.1.0",
            "circuit_breaker": circuit_breaker_info,
        }

        return result


__all__: list[str] = ["HandlerManifestPersistence", "HANDLER_ID_MANIFEST_PERSISTENCE"]
