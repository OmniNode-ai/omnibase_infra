"""Dead Letter Queue Monitor for Code Generation Events.

Monitors DLQ topics for failed code generation events and provides threshold-based
alerting for operational visibility.

Features:
- Async Kafka consumption with aiokafka
- Threshold-based alerting (configurable per topic)
- DLQ statistics tracking
- Structured logging with correlation IDs
- Production-ready error handling

DLQ Topics Monitored:
- omninode_codegen_dlq_analyze_v1
- omninode_codegen_dlq_validate_v1
- omninode_codegen_dlq_pattern_v1
- omninode_codegen_dlq_mixin_v1

Implementation: Event Infrastructure Phase 1
"""

import asyncio
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, ClassVar, Optional
from uuid import uuid4

import structlog
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
from omnibase_core.errors.error_codes import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError

# Alias for consistency with codebase patterns
CoreErrorCode = EnumCoreErrorCode
OnexError = ModelOnexError

logger = structlog.get_logger(__name__)


class CodegenDLQMonitor:
    """Monitor and alert on dead letter queue messages for code generation.

    Tracks failed events across 4 DLQ topics and triggers alerts when
    configurable thresholds are exceeded. Provides statistics endpoint
    for operational monitoring.

    Attributes:
        kafka_config: Kafka connection configuration
        alert_threshold: Number of messages to trigger alert (default: 10)
        dlq_counts: Per-topic message counts
        consumer: AIOKafkaConsumer instance
        is_running: Monitor running state
        total_messages_processed: Total DLQ messages seen

    Usage:
        ```python
        # Initialize monitor
        monitor = CodegenDLQMonitor(
            kafka_config={"bootstrap_servers": "omninode-bridge-redpanda:9092"},
            alert_threshold=10
        )

        # Start monitoring (async)
        await monitor.start_monitoring()

        # Get statistics
        stats = await monitor.get_dlq_stats()
        print(f"Total DLQ messages: {stats['total_dlq_messages']}")

        # Cleanup
        await monitor.stop_monitoring()
        ```
    """

    # DLQ topic names (configurable via environment)
    DLQ_TOPICS: ClassVar[list[str]] = [
        "omninode_codegen_dlq_analyze_v1",
        "omninode_codegen_dlq_validate_v1",
        "omninode_codegen_dlq_pattern_v1",
        "omninode_codegen_dlq_mixin_v1",
    ]

    def __init__(
        self,
        kafka_config: Optional[dict[str, Any]] = None,
        alert_threshold: int = 10,
        alert_webhook_url: Optional[str] = None,
    ):
        """Initialize DLQ monitor with Kafka configuration.

        Args:
            kafka_config: Kafka connection configuration. If None, reads from environment.
                Expected keys: bootstrap_servers, security_protocol, etc.
            alert_threshold: Number of DLQ messages to trigger alert (default: 10)
            alert_webhook_url: Optional webhook URL for alert notifications

        Environment Variables:
            KAFKA_BOOTSTRAP_SERVERS: Kafka brokers (default: omninode-bridge-redpanda:9092)
            DLQ_ALERT_THRESHOLD: Alert threshold (default: 10)
            DLQ_ALERT_WEBHOOK_URL: Webhook URL for alerts
        """
        # Load configuration from environment or parameters
        if kafka_config is None:
            # Default to remote Redpanda infrastructure (resolves to 192.168.86.200:9092 via /etc/hosts)
            kafka_config = {
                "bootstrap_servers": os.getenv(
                    "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
                )
            }

        self.kafka_config = kafka_config
        self.alert_threshold = int(
            os.getenv("DLQ_ALERT_THRESHOLD", str(alert_threshold))
        )
        self.alert_webhook_url = alert_webhook_url or os.getenv("DLQ_ALERT_WEBHOOK_URL")

        # DLQ tracking state
        self.dlq_counts: dict[str, int] = defaultdict(int)
        self.last_alert_time: dict[str, datetime] = {}
        self.total_messages_processed = 0
        self.alert_cooldown_minutes = 15  # Prevent alert spam

        # Consumer state
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Unique consumer group ID for this instance (ensures independent monitoring)
        self._consumer_group_id = f"codegen-dlq-monitor-{uuid4()}"

        logger.info(
            "CodegenDLQMonitor initialized",
            alert_threshold=self.alert_threshold,
            topics=self.DLQ_TOPICS,
            kafka_bootstrap_servers=kafka_config.get("bootstrap_servers"),
        )

    async def start_monitoring(self) -> None:
        """Start monitoring DLQ topics for failed events.

        Creates Kafka consumer, subscribes to DLQ topics, and begins
        processing messages asynchronously. Runs until explicitly stopped.

        Raises:
            OnexError: If consumer initialization or subscription fails
        """
        if self.is_running:
            logger.warning("DLQ monitoring already running")
            return

        try:
            logger.info("Starting DLQ monitoring", topics=self.DLQ_TOPICS)

            # Create AIOKafkaConsumer
            bootstrap_servers = self.kafka_config.get("bootstrap_servers", "").split(
                ","
            )
            self.consumer = AIOKafkaConsumer(
                *self.DLQ_TOPICS,
                bootstrap_servers=bootstrap_servers,
                group_id=self._consumer_group_id,  # Unique per instance for independence
                auto_offset_reset="latest",  # Only monitor new DLQ messages
                enable_auto_commit=True,  # Auto-commit for monitoring
                value_deserializer=lambda v: (
                    json.loads(v.decode("utf-8")) if v else None
                ),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )

            # Start consumer
            await self.consumer.start()
            self.is_running = True

            logger.info(
                "DLQ monitoring started successfully",
                topics=self.DLQ_TOPICS,
                group_id=self._consumer_group_id,
            )

            # Start message processing loop
            await self._consume_dlq_messages()

        except KafkaError as e:
            logger.error("Kafka connection failed", error=str(e), exc_info=True)
            raise OnexError(
                error_code=CoreErrorCode.SERVICE_UNAVAILABLE,
                message=f"Failed to connect to Kafka for DLQ monitoring: {e!s}",
                context={"kafka_config": self.kafka_config, "error_details": str(e)},
            ) from e
        except Exception as e:
            logger.error("DLQ monitoring startup failed", error=str(e), exc_info=True)
            raise OnexError(
                error_code=CoreErrorCode.INTERNAL_ERROR,
                message=f"Failed to start DLQ monitoring: {e!s}",
            ) from e

    async def _consume_dlq_messages(self) -> None:
        """Internal message consumption loop.

        Continuously processes DLQ messages until monitoring is stopped.
        Handles errors gracefully without crashing the monitor.
        """
        if not self.consumer:
            logger.error("Cannot consume messages - consumer not initialized")
            return

        try:
            async for msg in self.consumer:
                if not self.is_running:
                    break

                try:
                    await self._process_dlq_message(msg)
                except Exception as e:
                    logger.error(
                        "Error processing DLQ message",
                        topic=msg.topic,
                        error=str(e),
                        exc_info=True,
                    )
                    # Continue processing despite individual message errors

        except asyncio.CancelledError:
            logger.info("DLQ message consumption cancelled - shutting down gracefully")
            raise
        except Exception as e:
            logger.error("DLQ consumption error", error=str(e), exc_info=True)
        finally:
            if self.consumer:
                await self.consumer.stop()

    async def _process_dlq_message(self, msg: Any) -> None:
        """Process a DLQ message and update counts.

        Args:
            msg: Kafka consumer message from DLQ topic

        Updates:
            - dlq_counts: Increments count for topic
            - total_messages_processed: Global counter
            - Triggers alert if threshold exceeded
        """
        topic = msg.topic
        payload = msg.value

        # Update count for this DLQ topic
        self.dlq_counts[topic] += 1
        self.total_messages_processed += 1

        # Extract error information from payload
        error_info = "Unknown error"
        correlation_id = None

        if isinstance(payload, dict):
            error_info = payload.get("error", "Unknown error")
            correlation_id = payload.get("correlation_id")

        # Log the failed message
        logger.warning(
            "DLQ message received",
            topic=topic,
            error=error_info,
            correlation_id=correlation_id,
            dlq_count=self.dlq_counts[topic],
            offset=msg.offset,
            partition=msg.partition,
        )

        # Check if we need to alert
        if self.dlq_counts[topic] >= self.alert_threshold:
            if self._should_send_alert(topic):
                # Update last alert time BEFORE calling _send_alert to ensure
                # cooldown tracking works even when _send_alert is mocked in tests
                self.last_alert_time[topic] = datetime.now(UTC)
                await self._send_alert(topic, self.dlq_counts[topic], error_info)

    def _should_send_alert(self, topic: str) -> bool:
        """Check if alert should be sent based on cooldown period.

        Args:
            topic: DLQ topic name

        Returns:
            True if alert should be sent, False if in cooldown period
        """
        if topic not in self.last_alert_time:
            return True

        # Check cooldown period
        last_alert = self.last_alert_time[topic]
        elapsed_minutes = (datetime.now(UTC) - last_alert).total_seconds() / 60

        return elapsed_minutes >= self.alert_cooldown_minutes

    async def _send_alert(self, topic: str, count: int, recent_error: str) -> None:
        """Send alert for high DLQ count.

        Args:
            topic: DLQ topic name
            count: Current message count
            recent_error: Most recent error message

        Side Effects:
            - Logs critical alert
            - Sends webhook notification (if configured)
            - Updates last_alert_time
        """
        alert_message = {
            "alert_type": "dlq_threshold_exceeded",
            "topic": topic,
            "count": count,
            "threshold": self.alert_threshold,
            "recent_error": recent_error,
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": "critical",
        }

        # Log critical alert
        logger.critical(
            "DLQ ALERT: Threshold exceeded",
            topic=topic,
            count=count,
            threshold=self.alert_threshold,
            recent_error=recent_error,
        )

        # Send webhook notification if configured
        if self.alert_webhook_url:
            await self._send_webhook_alert(alert_message)

        # Note: last_alert_time is now updated in _process_dlq_message before
        # calling this method to ensure cooldown works even when method is mocked

    async def _send_webhook_alert(self, alert_data: dict[str, Any]) -> None:
        """Send alert via webhook.

        Args:
            alert_data: Alert payload to send

        Note:
            Logs errors but does not raise to avoid disrupting monitoring
        """
        if not self.alert_webhook_url:
            logger.debug("No webhook URL configured - skipping webhook alert")
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.alert_webhook_url,
                    json=alert_data,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.info(
                            "Webhook alert sent successfully", topic=alert_data["topic"]
                        )
                    else:
                        logger.warning(
                            "Webhook alert failed",
                            status_code=response.status,
                            topic=alert_data["topic"],
                        )
        except Exception as e:
            logger.error(
                "Failed to send webhook alert",
                error=str(e),
                webhook_url=self.alert_webhook_url,
            )

    async def get_dlq_stats(self) -> dict[str, Any]:
        """Get current DLQ statistics.

        Returns:
            Dictionary with DLQ statistics:
            - dlq_counts: Per-topic message counts
            - total_dlq_messages: Sum of all DLQ messages
            - alert_threshold: Configured alert threshold
            - is_monitoring: Whether monitor is currently running
            - last_alert_times: Timestamps of last alerts per topic
            - timestamp: Current timestamp

        Example:
            ```python
            stats = await monitor.get_dlq_stats()
            print(f"Analyze DLQ: {stats['dlq_counts']['omninode_codegen_dlq_analyze_v1']}")
            ```
        """
        return {
            "dlq_counts": dict(self.dlq_counts),
            "total_dlq_messages": self.total_messages_processed,
            "alert_threshold": self.alert_threshold,
            "is_monitoring": self.is_running,
            "last_alert_times": {
                topic: timestamp.isoformat()
                for topic, timestamp in self.last_alert_time.items()
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def stop_monitoring(self) -> None:
        """Stop DLQ monitoring gracefully.

        Stops message consumption, commits offsets, and closes Kafka consumer.
        Safe to call multiple times.
        """
        if not self.is_running:
            logger.debug("DLQ monitoring not running")
            return

        logger.info("Stopping DLQ monitoring")
        self.is_running = False

        # Cancel monitoring task if running
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Stop consumer
        if self.consumer:
            try:
                await self.consumer.stop()
                logger.info("DLQ monitoring stopped successfully")
            except Exception as e:
                logger.error("Error stopping DLQ consumer", error=str(e))
            finally:
                self.consumer = None

    async def reset_counts(self, topic: Optional[str] = None) -> None:
        """Reset DLQ counts for monitoring.

        Args:
            topic: Specific topic to reset, or None to reset all

        Note:
            Useful for testing or manual count resets
        """
        if topic:
            self.dlq_counts[topic] = 0
            logger.info("DLQ count reset", topic=topic)
        else:
            self.dlq_counts.clear()
            self.total_messages_processed = 0
            logger.info("All DLQ counts reset")

    def __repr__(self) -> str:
        """String representation of monitor state."""
        return (
            f"CodegenDLQMonitor(running={self.is_running}, "
            f"threshold={self.alert_threshold}, "
            f"total_messages={self.total_messages_processed})"
        )
