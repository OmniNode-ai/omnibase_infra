# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: metadata_stamping_service_core
# title: MetadataStampingService Core Service
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.stamping
# kind: service
# role: core_service_coordinator
# description: |
#   Core service class that coordinates all components of the metadata stamping
#   service including engine, database, and event publishing capabilities.
# tags: [service, metadata, blake3, coordination, stamping]
# author: OmniNode Development Team
# license: MIT
# entrypoint: service.py
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: true, requires_gpu: false}
# dependencies: [{"name": "asyncpg", "version": "^0.29.0"}, {"name": "blake3", "version": "^0.4.1"}]
# environment: [python>=3.11]
# === /OmniNode:Tool_Metadata ===

"""Main service class for metadata stamping.

This module implements the main service class that coordinates all components
of the metadata stamping service.
"""

import logging
import time
import uuid
from typing import Any, Optional

from .database.client import DatabaseConfig, MetadataStampingPostgresClient
from .engine.stamping_engine import StampingEngine
from .events import EventPublisher
from .protocols.file_type_handler import ProtocolFileTypeHandler

logger = logging.getLogger(__name__)


class MetadataStampingService:
    """Main service class following bridge service patterns."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize service with configuration.

        Args:
            config: Service configuration dictionary
        """
        self.config = config or {}
        self.start_time = time.time()

        # Core components
        self.stamping_engine = StampingEngine()
        self.file_handler = ProtocolFileTypeHandler()
        self.db_client: Optional[MetadataStampingPostgresClient] = None
        self.event_publisher: Optional[EventPublisher] = None

        # Service state
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize service components (DB, clients, etc.).

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing MetadataStampingService...")

            # Initialize database if configured
            if "database" in self.config and self.config["database"] is not None:
                db_config = DatabaseConfig(**self.config["database"])
                self.db_client = MetadataStampingPostgresClient(db_config)

                if not await self.db_client.initialize():
                    logger.error("Failed to initialize database client")
                    return False

                logger.info("Database client initialized successfully")

            # Initialize event publisher if configured
            if "events" in self.config:
                event_config = self.config["events"]
                self.event_publisher = EventPublisher(
                    enable_events=event_config.get("enable_events", False),
                    secret_key=event_config.get("event_secret_key"),
                )

                if not await self.event_publisher.initialize():
                    logger.warning(
                        "Failed to initialize event publisher - continuing without events"
                    )
                    self.event_publisher = None
                else:
                    logger.info("Event publisher initialized successfully")

            self.is_initialized = True
            logger.info("MetadataStampingService initialized successfully")
            return True

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Configuration error during service initialization: {e}")
            return False
        except (ConnectionError, OSError) as e:
            logger.error(f"Connection error during service initialization: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during service initialization: {e}")
            return False

    async def cleanup(self):
        """Cleanup service resources on shutdown."""
        try:
            logger.info("Cleaning up MetadataStampingService...")

            # Cleanup stamping engine
            await self.stamping_engine.cleanup()

            # Close database connection
            if self.db_client:
                await self.db_client.close()

            # Cleanup event publisher
            if self.event_publisher:
                await self.event_publisher.cleanup()

            self.is_initialized = False
            logger.info("MetadataStampingService cleanup completed")

        except (ConnectionError, OSError) as e:
            logger.error(f"Connection error during cleanup: {e}")
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Resource cleanup error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}")

    async def stamp_content(
        self,
        content: str,
        file_path: Optional[str] = None,
        stamp_type: str = "lightweight",
        metadata: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Stamp content with metadata.

        Args:
            content: Content to stamp
            file_path: Optional file path
            stamp_type: Type of stamp to create
            metadata: Additional metadata
            correlation_id: Optional correlation ID for tracking
            actor_id: Optional actor ID for events
            session_id: Optional session ID for events

        Returns:
            Stamp result with performance metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")

        # Generate stamp
        stamp_result = await self.stamping_engine.stamp_content(
            content=content,
            file_path=file_path,
            stamp_type=stamp_type,
            metadata=metadata,
        )

        # Publish stamp created event if event publisher is available
        if self.event_publisher and stamp_result.get("success", False):
            try:
                await self.event_publisher.publish_stamp_created(
                    stamp_id=str(uuid.uuid4()),
                    file_hash=stamp_result.get("hash", ""),
                    file_path=file_path or "unknown",
                    file_size=len(content.encode("utf-8")),
                    execution_time_ms=stamp_result.get("execution_time_ms", 0.0),
                    correlation_id=correlation_id,
                    actor_id=actor_id
                    or self.config.get("events", {}).get("event_actor_id", "system"),
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to publish stamp created event: {e}")
                # Don't fail the operation if event publishing fails

        return stamp_result

    async def validate_stamp(
        self,
        content: str,
        expected_hash: Optional[str] = None,
        correlation_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Validate existing stamps in content.

        Args:
            content: Content with stamps to validate
            expected_hash: Optional expected hash
            correlation_id: Optional correlation ID for tracking
            actor_id: Optional actor ID for events
            session_id: Optional session ID for events

        Returns:
            Validation results
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")

        start_time = time.time()
        validation_result = await self.stamping_engine.validate_stamp(
            content=content, expected_hash=expected_hash
        )
        validation_time_ms = (time.time() - start_time) * 1000

        # Publish stamp validated event if event publisher is available
        if self.event_publisher:
            try:
                # Extract file hash from validation result or expected hash
                file_hash = expected_hash or validation_result.get("hash", "unknown")
                validation_success = validation_result.get("valid", False)
                error_message = (
                    validation_result.get("error") if not validation_success else None
                )

                await self.event_publisher.publish_stamp_validated(
                    file_hash=file_hash,
                    validation_result=validation_success,
                    validation_time_ms=validation_time_ms,
                    error_message=error_message,
                    correlation_id=correlation_id,
                    actor_id=actor_id
                    or self.config.get("events", {}).get("event_actor_id", "system"),
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to publish stamp validated event: {e}")
                # Don't fail the operation if event publishing fails

        return validation_result

    async def generate_hash(
        self, content: bytes, file_path: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate BLAKE3 hash for content.

        Args:
            content: Content to hash
            file_path: Optional file path

        Returns:
            Hash result with performance metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")

        return await self.stamping_engine.hash_generator.generate_hash(
            file_data=content, file_path=file_path
        )

    async def process_batch(
        self,
        files_data: list[dict[str, Any]],
        correlation_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process multiple files in a batch operation.

        Args:
            files_data: List of file data dictionaries with 'content' and optional 'file_path'
            correlation_id: Optional correlation ID for tracking
            actor_id: Optional actor ID for events
            session_id: Optional session ID for events

        Returns:
            Batch processing results with metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")

        batch_id = str(uuid.uuid4())
        start_time = time.time()
        total_files = len(files_data)
        successful_files = 0
        failed_files = 0
        results = []

        logger.info(
            f"Starting batch processing for {total_files} files with batch_id: {batch_id}"
        )

        for file_data in files_data:
            try:
                result = await self.stamp_content(
                    content=file_data.get("content", ""),
                    file_path=file_data.get("file_path"),
                    stamp_type=file_data.get("stamp_type", "lightweight"),
                    metadata=file_data.get("metadata"),
                    correlation_id=correlation_id,
                    actor_id=actor_id,
                    session_id=session_id,
                )

                if result.get("success", False):
                    successful_files += 1
                else:
                    failed_files += 1

                results.append(
                    {
                        "file_path": file_data.get("file_path", "unknown"),
                        "success": result.get("success", False),
                        "hash": result.get("hash"),
                        "execution_time_ms": result.get("execution_time_ms"),
                        "error": result.get("error"),
                    }
                )

            except Exception as e:
                failed_files += 1
                results.append(
                    {
                        "file_path": file_data.get("file_path", "unknown"),
                        "success": False,
                        "error": str(e),
                    }
                )
                logger.error(f"Error processing file in batch {batch_id}: {e}")

        total_processing_time_ms = (time.time() - start_time) * 1000

        # Publish batch processed event if event publisher is available
        if self.event_publisher:
            try:
                await self.event_publisher.publish_batch_processed(
                    batch_id=batch_id,
                    total_files=total_files,
                    successful_files=successful_files,
                    failed_files=failed_files,
                    total_processing_time_ms=total_processing_time_ms,
                    correlation_id=correlation_id,
                    actor_id=actor_id
                    or self.config.get("events", {}).get("event_actor_id", "system"),
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to publish batch processed event: {e}")

        batch_result = {
            "batch_id": batch_id,
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": successful_files / total_files if total_files > 0 else 0.0,
            "total_processing_time_ms": total_processing_time_ms,
            "average_processing_time_ms": (
                total_processing_time_ms / total_files if total_files > 0 else 0.0
            ),
            "results": results,
        }

        logger.info(
            f"Batch processing completed for {batch_id}: {successful_files}/{total_files} successful"
        )
        return batch_result

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check for service components.

        Returns:
            Health check results
        """
        health_results = {
            "status": "healthy",
            "components": {},
            "uptime_seconds": self.get_uptime(),
        }

        # Check stamping engine
        try:
            # Quick hash generation test
            test_result = await self.stamping_engine.hash_generator.generate_hash(
                b"health_check"
            )
            if test_result["execution_time_ms"] < 5:
                health_results["components"]["stamping_engine"] = {
                    "status": "healthy",
                    "response_time_ms": test_result["execution_time_ms"],
                }
            else:
                health_results["components"]["stamping_engine"] = {
                    "status": "degraded",
                    "response_time_ms": test_result["execution_time_ms"],
                }
                health_results["status"] = "degraded"
        except (MemoryError, OSError) as e:
            health_results["components"]["stamping_engine"] = {
                "status": "unhealthy",
                "error": f"Resource error: {e}",
            }
            health_results["status"] = "unhealthy"
        except (RuntimeError, AttributeError, TypeError) as e:
            health_results["components"]["stamping_engine"] = {
                "status": "unhealthy",
                "error": f"Component error: {e}",
            }
            health_results["status"] = "unhealthy"
        except Exception as e:
            health_results["components"]["stamping_engine"] = {
                "status": "unhealthy",
                "error": f"Unexpected error: {e}",
            }
            health_results["status"] = "unhealthy"

        # Check database if configured
        if self.db_client:
            try:
                db_health = await self.db_client.health_check()
                health_results["components"]["database"] = {
                    "status": db_health["status"],
                    "response_time_ms": db_health.get("response_time_ms"),
                    "details": {
                        "pool_statistics": db_health.get("pool_statistics"),
                        "prepared_statements": db_health.get("prepared_statements"),
                    },
                }
                if db_health["status"] != "healthy":
                    health_results["status"] = "degraded"
            except (ConnectionError, OSError) as e:
                health_results["components"]["database"] = {
                    "status": "unhealthy",
                    "error": f"Database connection error: {e}",
                }
                health_results["status"] = "unhealthy"
            except (RuntimeError, AttributeError, TypeError) as e:
                health_results["components"]["database"] = {
                    "status": "unhealthy",
                    "error": f"Database component error: {e}",
                }
                health_results["status"] = "unhealthy"
            except Exception as e:
                health_results["components"]["database"] = {
                    "status": "unhealthy",
                    "error": f"Unexpected database error: {e}",
                }
                health_results["status"] = "unhealthy"

        # Check file handler
        health_results["components"]["file_handler"] = {
            "status": "healthy",
            "supported_types": len(self.file_handler.supported_types),
        }

        # Check event publisher if configured
        if self.event_publisher:
            try:
                event_metrics = self.event_publisher.get_metrics()
                health_results["components"]["event_publisher"] = {
                    "status": (
                        "healthy" if event_metrics["kafka_connected"] else "degraded"
                    ),
                    "events_enabled": event_metrics["events_enabled"],
                    "events_published": event_metrics["events_published"],
                    "events_failed": event_metrics["events_failed"],
                    "kafka_connected": event_metrics["kafka_connected"],
                }
                if (
                    not event_metrics["kafka_connected"]
                    and event_metrics["events_enabled"]
                ):
                    health_results["status"] = "degraded"
            except Exception as e:
                health_results["components"]["event_publisher"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_results["status"] = "degraded"
        else:
            health_results["components"]["event_publisher"] = {
                "status": "disabled",
                "message": "Event publishing not configured",
            }

        return health_results

    def get_uptime(self) -> float:
        """Get service uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time
