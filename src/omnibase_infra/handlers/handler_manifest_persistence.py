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

import json
import logging
import os
import tempfile
from datetime import datetime
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
from omnibase_infra.utils import parse_env_int

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


def _warn_if_naive_datetime(
    dt: datetime,
    field_name: str,
    correlation_id: UUID,
) -> None:
    """Log a warning if the datetime is naive (lacks timezone info).

    This helper supports the timezone awareness policy documented in the module
    docstring. Naive datetimes are accepted for backwards compatibility but
    logged as warnings to encourage migration to timezone-aware datetimes.

    Args:
        dt: The datetime to check.
        field_name: Name of the field (for logging context).
        correlation_id: Correlation ID for tracing.

    Note:
        This is a non-blocking warning. The datetime is still used, but users
        are alerted that comparisons may behave unexpectedly if mixed with
        timezone-aware datetimes from stored manifests.
    """
    if dt.tzinfo is None:
        logger.warning(
            "Naive datetime detected for '%s'. For accurate comparisons, use "
            "timezone-aware datetimes (e.g., datetime.now(timezone.utc)). "
            "See module docstring for timezone handling details.",
            field_name,
            extra={
                "field_name": field_name,
                "datetime_value": dt.isoformat(),
                "correlation_id": str(correlation_id),
            },
        )


class HandlerManifestPersistence(MixinEnvelopeExtraction, MixinAsyncCircuitBreaker):
    """Manifest persistence handler for storing/retrieving ModelExecutionManifest.

    This handler stores ModelExecutionManifest objects to the filesystem with:
    - Date-based partitioning (year/month/day directories)
    - Atomic writes (write to temp, then rename)
    - Idempotent storage (same manifest_id = no duplicate)
    - Query support with filters
    - Circuit breaker for resilient I/O operations

    Storage Pattern:
        {storage_path}/{year}/{month}/{day}/{manifest_id}.json

        Example: /data/manifests/2025/01/14/550e8400-e29b-41d4-a716-446655440000.json

    Attributes:
        handler_type: Returns INFRA_HANDLER (infrastructure protocol handler)
        handler_category: Returns EFFECT (side-effecting I/O)

    Example:
        >>> handler = HandlerManifestPersistence()
        >>> await handler.initialize({"storage_path": "/data/manifests"})
        >>> result = await handler.execute({
        ...     "operation": "manifest.store",
        ...     "payload": {"manifest": manifest.model_dump()},
        ... })
    """

    def __init__(self, container: ModelONEXContainer | None = None) -> None:
        """Initialize HandlerManifestPersistence with optional container injection.

        Note:
            ONEX Pattern Deviation: The container parameter is optional here,
            deviating from the strict ONEX guideline of required container injection
            (``def __init__(self, container: ModelONEXContainer)``). This is an
            intentional design choice because:

            1. **Standalone testing**: Allows unit tests to instantiate the handler
               without a full ONEX container, simplifying test setup.
            2. **Integration flexibility**: The handler's core filesystem operations
               do not depend on container-provided services.
            3. **Gradual integration**: Enables incremental adoption in codebases
               not fully migrated to ONEX container patterns.

        Args:
            container: Optional ONEX container for dependency injection.
                When provided, enables full ONEX integration (logging, metrics,
                service discovery). When None, handler operates in standalone
                mode suitable for testing and simple deployments.

        See Also:
            - CLAUDE.md "Container-Based Dependency Injection" section for the
              standard ONEX container injection pattern.
            - docs/patterns/container_dependency_injection.md for detailed DI patterns.
        """
        self._container = container
        self._storage_path: Path | None = None
        self._max_file_size: int = _DEFAULT_MAX_FILE_SIZE
        self._initialized: bool = False

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
        """Execute manifest.store operation.

        Stores a manifest with atomic write (temp file + rename) and
        idempotent behavior (existing manifests are not overwritten).

        Payload:
            - manifest: dict (required) - Serialized ModelExecutionManifest

        Returns:
            Result with manifest_id, file_path, created, and bytes_written.

        Raises:
            InfraConnectionError: If write fails
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
                _warn_if_naive_datetime(created_at, "created_at", correlation_id)
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

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Check if manifest already exists (idempotent behavior)
            if file_path.exists():
                # Reset circuit breaker on success
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

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

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

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

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise
        except OSError as e:
            # Record failure for circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise InfraConnectionError(
                f"Failed to write manifest: {e}",
                context=ctx,
            ) from e

    async def _execute_retrieve(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute manifest.retrieve operation.

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
            InfraConnectionError: If read fails
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

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            # Search for manifest in date directories
            found_path: Path | None = None
            manifest_data: dict[str, object] | None = None

            if self._storage_path is None:
                raise RuntimeHostError(
                    "Handler not initialized - storage_path is None",
                    context=ctx,
                )

            # Scan year/month/day directories
            for year_dir in sorted(self._storage_path.iterdir(), reverse=True):
                if not year_dir.is_dir():
                    continue
                for month_dir in sorted(year_dir.iterdir(), reverse=True):
                    if not month_dir.is_dir():
                        continue
                    for day_dir in sorted(month_dir.iterdir(), reverse=True):
                        if not day_dir.is_dir():
                            continue
                        manifest_file = day_dir / f"{manifest_id}.json"
                        if manifest_file.exists():
                            found_path = manifest_file
                            break
                    if found_path:
                        break
                if found_path:
                    break

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

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

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

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise
        except OSError as e:
            # Record failure for circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise InfraConnectionError(
                f"Failed to read manifest: {e}",
                context=ctx,
            ) from e

    async def _execute_query(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute manifest.query operation.

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
            InfraConnectionError: If read fails
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
            except ValueError:
                pass  # Invalid UUID, ignore filter

        filter_node_id: str | None = None
        node_id_raw = payload.get("node_id")
        if isinstance(node_id_raw, str):
            filter_node_id = node_id_raw

        filter_created_after: datetime | None = None
        created_after_raw = payload.get("created_after")
        if created_after_raw is not None:
            try:
                if isinstance(created_after_raw, datetime):
                    filter_created_after = created_after_raw
                    _warn_if_naive_datetime(
                        filter_created_after, "created_after", correlation_id
                    )
                elif isinstance(created_after_raw, str):
                    filter_created_after = datetime.fromisoformat(
                        created_after_raw.replace("Z", "+00:00")
                    )
            except ValueError:
                pass  # Invalid datetime, ignore filter

        filter_created_before: datetime | None = None
        created_before_raw = payload.get("created_before")
        if created_before_raw is not None:
            try:
                if isinstance(created_before_raw, datetime):
                    filter_created_before = created_before_raw
                    _warn_if_naive_datetime(
                        filter_created_before, "created_before", correlation_id
                    )
                elif isinstance(created_before_raw, str):
                    filter_created_before = datetime.fromisoformat(
                        created_before_raw.replace("Z", "+00:00")
                    )
            except ValueError:
                pass  # Invalid datetime, ignore filter

        metadata_only = payload.get("metadata_only", False)
        if not isinstance(metadata_only, bool):
            metadata_only = False

        limit = payload.get("limit", 100)
        if not isinstance(limit, int) or limit < 1:
            limit = 100
        limit = min(limit, 10000)  # Cap at 10000

        # Check circuit breaker before I/O operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation, correlation_id)

        try:
            if self._storage_path is None:
                raise RuntimeHostError(
                    "Handler not initialized - storage_path is None",
                    context=ctx,
                )

            manifests_metadata: list[ModelManifestMetadata] = []
            manifests_data: list[dict[str, object]] = []
            count = 0

            # Scan date directories
            for year_dir in sorted(self._storage_path.iterdir(), reverse=True):
                if not year_dir.is_dir() or count >= limit:
                    continue
                for month_dir in sorted(year_dir.iterdir(), reverse=True):
                    if not month_dir.is_dir() or count >= limit:
                        continue
                    for day_dir in sorted(month_dir.iterdir(), reverse=True):
                        if not day_dir.is_dir() or count >= limit:
                            continue
                        for manifest_file in sorted(
                            day_dir.glob("*.json"), reverse=True
                        ):
                            if count >= limit:
                                break

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

            # Reset circuit breaker on success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

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

        except (InfraConnectionError, InfraUnavailableError):
            # Record failure for circuit breaker (infra-level failures only)
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise
        except OSError as e:
            # Record failure for circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
            raise InfraConnectionError(
                f"Failed to query manifests: {e}",
                context=ctx,
            ) from e

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities for introspection.

        This method exposes the handler's type classification along with
        its operational configuration and capabilities.

        Returns:
            dict containing:
                - handler_type: Architectural role from handler_type property
                - handler_category: Behavioral classification
                - supported_operations: List of supported operations
                - storage_path: Storage directory path (when initialized)
                - initialized: Whether the handler is initialized
                - version: Handler version string
                - circuit_breaker: Circuit breaker state (when initialized)
                    - open: Whether circuit is currently open
                    - failures: Current failure count
                    - threshold: Configured failure threshold
                    - reset_timeout_seconds: Configured reset timeout
        """
        result: dict[str, object] = {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "storage_path": str(self._storage_path) if self._storage_path else None,
            "initialized": self._initialized,
            "version": "0.1.0",
        }

        # Include circuit breaker state only when handler is initialized
        # (CB is set up during initialize() call)
        if self._initialized:
            result["circuit_breaker"] = {
                "open": self._circuit_breaker_open,
                "failures": self._circuit_breaker_failures,
                "threshold": self.circuit_breaker_threshold,
                "reset_timeout_seconds": self.circuit_breaker_reset_timeout,
            }

        return result


__all__: list[str] = ["HandlerManifestPersistence", "HANDLER_ID_MANIFEST_PERSISTENCE"]
