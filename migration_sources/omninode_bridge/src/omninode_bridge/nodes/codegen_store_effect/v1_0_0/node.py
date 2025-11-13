#!/usr/bin/env python3
"""
NodeCodegenStoreEffect - Persist generated artifacts and metrics.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeCodegenStoreEffect
- Extends NodeEffect from omnibase_core
- Uses ModelOnexError for error handling
- Structured logging with correlation tracking
"""

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

# ONEX Core Imports
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

# Node-specific imports
from .models import ModelArtifactStorageRequest, ModelStorageResult

# Aliases
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode


class NodeCodegenStoreEffect(NodeEffect):
    """
    Store Effect for persisting generated code artifacts and metrics.

    Handles file system operations for storing generated code artifacts,
    creating necessary directories, setting permissions, and optionally
    storing metadata in PostgreSQL.

    Responsibilities:
    - Write generated code artifacts to file system
    - Create parent directories with proper permissions
    - Handle file permissions and overwrite settings
    - Optional: Store artifact metadata in PostgreSQL
    - Track storage metrics (bytes written, files created, etc.)
    - Execute intents from reducer (PERSIST_METRICS, STORE_ARTIFACT)

    ONEX v2.0 Compliance:
    - Suffix-based naming: NodeCodegenStoreEffect
    - Extends NodeEffect from omnibase_core
    - Uses ModelOnexError for error handling
    - Structured logging with correlation tracking

    Performance Targets:
    - Storage time: <1s for typical artifacts
    - Throughput: >50 files/second
    - Memory: <100MB per operation

    Example Usage:
        ```python
        container = ModelContainer(...)
        store = NodeCodegenStoreEffect(container)

        contract = ModelContractEffect(
            correlation_id=uuid4(),
            input_state={
                "storage_requests": [
                    {
                        "file_path": "/path/to/node.py",
                        "content": "def foo(): pass",
                        "artifact_type": "node_file",
                        "create_directories": True
                    }
                ]
            }
        )

        result = await store.execute_effect(contract)
        if result.success:
            print(f"Stored {result.artifacts_stored} artifacts")
        ```
    """

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize store effect with dependency injection container.

        Args:
            container: ONEX container for dependency injection

        Raises:
            ModelOnexError: If container is invalid or initialization fails
        """
        super().__init__(container)

        # Configuration - defensive pattern
        try:
            if hasattr(container.config, "get") and callable(container.config.get):
                self.base_output_dir = container.config.get(
                    "codegen_output_dir", "./generated"
                )
                self.enable_db_storage = container.config.get(
                    "enable_artifact_db_storage", False
                )
                self.default_permissions = container.config.get(
                    "default_file_permissions", "0644"
                )
            else:
                self.base_output_dir = "./generated"
                self.enable_db_storage = False
                self.default_permissions = "0644"
        except Exception:
            self.base_output_dir = "./generated"
            self.enable_db_storage = False
            self.default_permissions = "0644"

        # Get database service if enabled
        self.db_service = None
        if self.enable_db_storage:
            try:
                if hasattr(container, "get_service") and callable(
                    container.get_service
                ):
                    self.db_service = container.get_service("postgres")
            except Exception:
                self.db_service = None

        # Metrics tracking
        self._total_storage_operations = 0
        self._successful_storage_operations = 0
        self._failed_storage_operations = 0
        self._total_bytes_written = 0
        self._total_duration_ms = 0.0

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenStoreEffect initialized successfully",
            {
                "node_id": self.node_id,
                "base_output_dir": self.base_output_dir,
                "db_storage_enabled": self.enable_db_storage,
                "default_permissions": self.default_permissions,
            },
        )

    async def execute_effect(self, contract: ModelContractEffect) -> ModelStorageResult:
        """
        Execute storage operation.

        Args:
            contract: Effect contract with input_state containing:
                - storage_requests (list[dict]): List of storage requests
                - base_directory (str, optional): Override base directory

        Returns:
            ModelStorageResult with storage metrics

        Raises:
            OnexError: If storage fails or invalid input
        """
        start_time = time.perf_counter()
        correlation_id = contract.correlation_id

        emit_log_event(
            LogLevel.INFO,
            "Starting artifact storage",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Parse input from contract
            input_data = contract.input_state or {}
            storage_requests_data = input_data.get("storage_requests", [])

            if not storage_requests_data:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message="Missing required field: storage_requests",
                    details={"correlation_id": str(correlation_id)},
                )

            # Parse storage requests
            storage_requests = [
                ModelArtifactStorageRequest(**req) for req in storage_requests_data
            ]

            # Override base directory if provided
            base_directory = input_data.get("base_directory", self.base_output_dir)

            # Perform storage operations
            stored_files, storage_errors, total_bytes = await self._store_artifacts(
                storage_requests, base_directory, correlation_id
            )

            # Calculate metrics
            artifacts_stored = len(stored_files)
            success = len(storage_errors) == 0
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._total_storage_operations += len(storage_requests)
            self._successful_storage_operations += artifacts_stored
            self._failed_storage_operations += len(storage_errors)
            self._total_bytes_written += total_bytes
            self._total_duration_ms += duration_ms

            emit_log_event(
                LogLevel.INFO,
                "Artifact storage completed",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "success": success,
                    "artifacts_stored": artifacts_stored,
                    "errors": len(storage_errors),
                    "bytes_written": total_bytes,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return ModelStorageResult(
                success=success,
                artifacts_stored=artifacts_stored,
                storage_errors=storage_errors,
                storage_time_ms=duration_ms,
                stored_files=stored_files,
                total_bytes_written=total_bytes,
                metrics_stored=False,  # Will be True if DB storage is implemented
                database_record_id=None,
            )

        except OnexError:
            self._failed_storage_operations += 1
            raise

        except Exception as e:
            self._failed_storage_operations += 1

            emit_log_event(
                LogLevel.ERROR,
                f"Artifact storage failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )

            raise OnexError(
                message=f"Artifact storage failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
            ) from e

    async def _store_artifacts(
        self,
        storage_requests: list[ModelArtifactStorageRequest],
        base_directory: str,
        correlation_id: UUID,
    ) -> tuple[list[str], list[str], int]:
        """
        Store artifacts to file system.

        Args:
            storage_requests: List of storage requests
            base_directory: Base directory for storage
            correlation_id: Correlation ID for tracking

        Returns:
            Tuple of (stored_files, errors, total_bytes_written)
        """
        stored_files = []
        errors = []
        total_bytes = 0

        for request in storage_requests:
            try:
                # Resolve file path
                if os.path.isabs(request.file_path):
                    file_path = Path(request.file_path)
                else:
                    file_path = Path(base_directory) / request.file_path

                # Create parent directories if needed
                if request.create_directories:
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if file exists and overwrite setting
                if file_path.exists() and not request.overwrite:
                    errors.append(f"File exists and overwrite=False: {file_path}")
                    continue

                # Write content to file
                bytes_written = len(request.content.encode("utf-8"))
                file_path.write_text(request.content, encoding="utf-8")

                # Set file permissions
                if request.file_permissions:
                    try:
                        mode = int(request.file_permissions, 8)
                        os.chmod(file_path, mode)
                    except (ValueError, OSError) as e:
                        emit_log_event(
                            LogLevel.WARNING,
                            f"Failed to set file permissions: {e}",
                            {
                                "file_path": str(file_path),
                                "permissions": request.file_permissions,
                            },
                        )

                # Store metrics in database if enabled
                if request.store_metrics and self.db_service:
                    await self._store_artifact_metrics(
                        request, file_path, bytes_written, correlation_id
                    )

                stored_files.append(str(file_path))
                total_bytes += bytes_written

                emit_log_event(
                    LogLevel.DEBUG,
                    f"Stored artifact: {file_path}",
                    {
                        "file_path": str(file_path),
                        "bytes_written": bytes_written,
                        "artifact_type": request.artifact_type,
                    },
                )

            except Exception as e:
                error_msg = f"Failed to store {request.file_path}: {e!s}"
                errors.append(error_msg)
                emit_log_event(
                    LogLevel.ERROR,
                    error_msg,
                    {
                        "file_path": request.file_path,
                        "error": str(e),
                    },
                )

        return stored_files, errors, total_bytes

    async def _store_artifact_metrics(
        self,
        request: ModelArtifactStorageRequest,
        file_path: Path,
        bytes_written: int,
        correlation_id: UUID,
    ) -> Optional[str]:
        """
        Store artifact metadata in PostgreSQL.

        Args:
            request: Storage request
            file_path: Resolved file path
            bytes_written: Number of bytes written
            correlation_id: Correlation ID

        Returns:
            Database record ID if successful, None otherwise
        """
        if not self.db_service:
            return None

        try:
            # Build metadata record
            metadata = {
                "file_path": str(file_path),
                "artifact_type": request.artifact_type,
                "bytes_written": bytes_written,
                "correlation_id": str(correlation_id),
                "created_at": datetime.now(UTC).isoformat(),
                **request.metadata,
            }

            # Store in database (would use actual db_service here)
            # record_id = await self.db_service.store_artifact_metadata(metadata)
            # For now, just log that we would store it
            emit_log_event(
                LogLevel.DEBUG,
                "Would store artifact metrics in database",
                {"metadata": metadata},
            )

            return None  # Would return record_id

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to store artifact metrics: {e}",
                {"file_path": str(file_path), "error": str(e)},
            )
            return None

    def get_metrics(self) -> dict[str, Any]:
        """
        Get storage metrics for monitoring.

        Returns:
            Dictionary with metrics
        """
        avg_duration_ms = (
            self._total_duration_ms / self._total_storage_operations
            if self._total_storage_operations > 0
            else 0
        )

        if self._total_storage_operations > 0:
            success_rate = (
                self._successful_storage_operations / self._total_storage_operations
            )
        elif self._failed_storage_operations > 0:
            success_rate = 0.0  # Failures with no successful ops = 0% success
        else:
            success_rate = 1.0  # No ops at all = neutral 100%

        avg_bytes_per_file = (
            self._total_bytes_written / self._successful_storage_operations
            if self._successful_storage_operations > 0
            else 0
        )

        return {
            "total_storage_operations": self._total_storage_operations,
            "successful_storage_operations": self._successful_storage_operations,
            "failed_storage_operations": self._failed_storage_operations,
            "success_rate": round(success_rate, 4),
            "avg_duration_ms": round(avg_duration_ms, 2),
            "total_bytes_written": self._total_bytes_written,
            "avg_bytes_per_file": round(avg_bytes_per_file, 2),
        }

    async def startup(self) -> None:
        """Node startup lifecycle hook."""
        emit_log_event(
            LogLevel.INFO,
            "Node startup initiated",
            {"node_name": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook."""
        self._deregister_from_consul()
        emit_log_event(
            LogLevel.INFO,
            "Node shutdown completed",
            {"node_name": self.__class__.__name__},
        )

    def _register_with_consul_sync(self) -> None:
        """Register node with Consul for service discovery (synchronous)."""
        try:
            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            service_id = (
                f"omninode-bridge-{self.__class__.__name__.lower()}-{self.node_id}"
            )
            service_port = 8065  # Default port
            service_host = "localhost"

            consul_client.agent.service.register(
                name=f"omninode-bridge-{self.__class__.__name__.lower()}",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=["onex", "codegen", "effect"],
                http=f"http://{service_host}:{service_port}/health",
                interval="30s",
                timeout="5s",
            )

            self._consul_service_id = service_id

            emit_log_event(
                LogLevel.INFO,
                "Registered with Consul successfully",
                {"node_id": self.node_id, "service_id": service_id},
            )

        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "python-consul not installed - Consul registration skipped",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                "Failed to register with Consul",
                {"node_id": self.node_id, "error": str(e)},
            )

    def _deregister_from_consul(self) -> None:
        """Deregister node from Consul on shutdown (synchronous)."""
        try:
            if not hasattr(self, "_consul_service_id"):
                return

            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            consul_client.agent.service.deregister(self._consul_service_id)

            emit_log_event(
                LogLevel.INFO,
                "Deregistered from Consul successfully",
                {"node_id": self.node_id, "service_id": self._consul_service_id},
            )

        except ImportError:
            pass
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to deregister from Consul",
                {"node_id": self.node_id, "error": str(e)},
            )


def main() -> int:
    """
    Entry point for node execution.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from omnibase_core.infrastructure.node_base import NodeBase

        CONTRACT_FILENAME = "contract.yaml"
        node_base = NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"NodeCodegenStoreEffect execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
