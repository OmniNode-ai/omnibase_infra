"""Event publisher for metadata stamping service with Kafka integration.

This module provides async event publishing infrastructure that integrates
with the existing KafkaClient from the omninode_bridge framework.
"""

import logging
import os
from typing import Any, Optional

from ....services.kafka_client import KafkaClient
from .models import OnexEnvelopeV1

logger = logging.getLogger(__name__)


class EventPublisher:
    """Async event publisher for metadata stamping events."""

    def __init__(
        self,
        kafka_client: Optional[KafkaClient] = None,
        enable_events: bool = None,
        secret_key: str = None,
    ):
        """Initialize event publisher.

        Args:
            kafka_client: Optional KafkaClient instance
            enable_events: Whether to enable event publishing (default from env)
            secret_key: Optional secret key for HMAC signing
        """
        self.enable_events = (
            enable_events
            if enable_events is not None
            else os.getenv("METADATA_STAMPING_ENABLE_EVENTS", "false").lower() == "true"
        )

        self.secret_key = secret_key or os.getenv("METADATA_STAMPING_EVENT_SECRET_KEY")
        self.key_id = os.getenv("METADATA_STAMPING_EVENT_KEY_ID", "default")

        # Kafka client for event publishing
        self._kafka_client = kafka_client
        self._owned_kafka_client = kafka_client is None  # Track if we own the client

        # Event publishing metrics
        self.events_published = 0
        self.events_failed = 0
        self.last_publish_error: Optional[str] = None

        logger.info(f"EventPublisher initialized: events_enabled={self.enable_events}")

    async def initialize(self) -> bool:
        """Initialize the event publisher.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.enable_events:
            logger.info("Event publishing disabled by configuration")
            return True

        try:
            # Create Kafka client if not provided
            if self._kafka_client is None:
                kafka_config = {
                    "bootstrap_servers": os.getenv(
                        "METADATA_STAMPING_KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"
                    ),
                    "enable_dead_letter_queue": True,
                    "max_retry_attempts": 3,
                    "timeout_seconds": 30,
                }
                self._kafka_client = KafkaClient(**kafka_config)

            # Connect to Kafka
            await self._kafka_client.connect()

            if not self._kafka_client.is_connected:
                logger.warning(
                    "Kafka client not connected - events will be logged but not published"
                )
                return True  # Don't fail service startup if Kafka is unavailable

            logger.info("Event publisher initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize event publisher: {e}")
            self.last_publish_error = str(e)
            # Don't fail service startup if event publishing fails
            return True

    async def cleanup(self) -> None:
        """Cleanup event publisher resources."""
        if self._kafka_client and self._owned_kafka_client:
            try:
                await self._kafka_client.disconnect()
                logger.info("Event publisher cleanup completed")
            except Exception as e:
                logger.error(f"Error during event publisher cleanup: {e}")

    async def publish_event(
        self, event: OnexEnvelopeV1, actor_id: str = None, session_id: str = None
    ) -> bool:
        """Publish an event to Kafka.

        Args:
            event: Event to publish
            actor_id: Optional actor ID for security context
            session_id: Optional session ID for security context

        Returns:
            True if event was published successfully, False otherwise
        """
        if not self.enable_events:
            logger.debug(
                f"Event publishing disabled, skipping event: {event.event_type}"
            )
            return True

        try:
            # Add security context if actor_id provided
            if actor_id:
                event.add_security_context(
                    actor_id=actor_id, session_id=session_id or "system"
                )

            # Sign event if secret key is available
            if self.secret_key and event.security_context:
                event.sign_event(self.secret_key, self.key_id)

            # Ensure environment is set based on configuration
            environment = os.getenv("ENVIRONMENT", "development")
            event.environment = environment

            # Set source instance if not already set
            if not event.source_instance:
                event.source_instance = os.getenv(
                    "INSTANCE_ID", f"metadata-stamping-{os.getpid()}"
                )

            # Log event for debugging
            logger.debug(
                f"Publishing event: {event.event_type} to topic: {event.to_kafka_topic()}"
            )

            # Publish to Kafka if client is available and connected
            if self._kafka_client and self._kafka_client.is_connected:
                success = await self._kafka_client.publish_raw_event(
                    topic=event.to_kafka_topic(),
                    data=event.model_dump(),
                    key=event.get_kafka_key(),
                )

                if success:
                    self.events_published += 1
                    logger.debug(f"Successfully published event: {event.event_id}")
                    return True
                else:
                    self.events_failed += 1
                    self.last_publish_error = "Kafka publish failed"
                    logger.warning(
                        f"Failed to publish event to Kafka: {event.event_id}"
                    )
                    return False
            else:
                # Kafka not available - log event data for debugging
                logger.info(
                    f"Kafka unavailable, event logged: {event.event_type} - {event.event_id}"
                )
                logger.debug(f"Event data: {event.model_dump()}")
                return True  # Consider this a success since we're handling graceful degradation

        except Exception as e:
            self.events_failed += 1
            self.last_publish_error = str(e)
            logger.error(f"Error publishing event {event.event_id}: {e}")
            return False

    async def publish_stamp_created(
        self,
        stamp_id: str,
        file_hash: str,
        file_path: str,
        file_size: int,
        execution_time_ms: float,
        correlation_id: str = None,
        actor_id: str = None,
        session_id: str = None,
    ) -> bool:
        """Publish a metadata stamp created event.

        Args:
            stamp_id: Unique stamp identifier
            file_hash: BLAKE3 hash of the file
            file_path: Path to the stamped file
            file_size: Size of the file in bytes
            execution_time_ms: Time taken to create the stamp
            correlation_id: Optional correlation ID
            actor_id: Optional actor ID
            session_id: Optional session ID

        Returns:
            True if event was published successfully, False otherwise
        """
        from .models import MetadataStampCreatedEvent

        event = MetadataStampCreatedEvent(
            stamp_id=stamp_id,
            file_hash=file_hash,
            file_path=file_path,
            file_size=file_size,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id,
        )

        return await self.publish_event(event, actor_id, session_id)

    async def publish_stamp_validated(
        self,
        file_hash: str,
        validation_result: bool,
        validation_time_ms: float,
        error_message: str = None,
        correlation_id: str = None,
        actor_id: str = None,
        session_id: str = None,
    ) -> bool:
        """Publish a metadata stamp validated event.

        Args:
            file_hash: BLAKE3 hash that was validated
            validation_result: Whether validation succeeded
            validation_time_ms: Time taken to validate
            error_message: Optional error message if validation failed
            correlation_id: Optional correlation ID
            actor_id: Optional actor ID
            session_id: Optional session ID

        Returns:
            True if event was published successfully, False otherwise
        """
        from .models import MetadataStampValidatedEvent

        event = MetadataStampValidatedEvent(
            file_hash=file_hash,
            validation_result=validation_result,
            validation_time_ms=validation_time_ms,
            error_message=error_message,
            correlation_id=correlation_id,
        )

        return await self.publish_event(event, actor_id, session_id)

    async def publish_batch_processed(
        self,
        batch_id: str,
        total_files: int,
        successful_files: int,
        failed_files: int,
        total_processing_time_ms: float,
        correlation_id: str = None,
        actor_id: str = None,
        session_id: str = None,
    ) -> bool:
        """Publish a batch processed event.

        Args:
            batch_id: Unique batch identifier
            total_files: Total number of files in batch
            successful_files: Number of successfully processed files
            failed_files: Number of failed files
            total_processing_time_ms: Total processing time
            correlation_id: Optional correlation ID
            actor_id: Optional actor ID
            session_id: Optional session ID

        Returns:
            True if event was published successfully, False otherwise
        """
        from .models import MetadataBatchProcessedEvent

        event = MetadataBatchProcessedEvent(
            batch_id=batch_id,
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            total_processing_time_ms=total_processing_time_ms,
            correlation_id=correlation_id,
        )

        return await self.publish_event(event, actor_id, session_id)

    def get_metrics(self) -> dict[str, Any]:
        """Get event publishing metrics.

        Returns:
            Dictionary containing publishing metrics
        """
        return {
            "events_enabled": self.enable_events,
            "events_published": self.events_published,
            "events_failed": self.events_failed,
            "last_publish_error": self.last_publish_error,
            "kafka_connected": (
                self._kafka_client.is_connected if self._kafka_client else False
            ),
        }
