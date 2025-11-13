"""
Alert notifiers for different channels.

Implements notifiers for logging, Kafka, Slack, etc.
"""

import json
import logging
from abc import ABC, abstractmethod

from aiokafka import AIOKafkaProducer

from omninode_bridge.agents.metrics.models import Alert, AlertEvent, AlertEventPayload

logger = logging.getLogger(__name__)


class AlertNotifier(ABC):
    """
    Abstract base class for alert notifiers.

    Subclasses must implement notify() method.
    """

    @abstractmethod
    async def notify(self, alert: Alert) -> None:
        """
        Send alert notification.

        Args:
            alert: Alert to send
        """
        pass


class LogAlertNotifier(AlertNotifier):
    """
    Log-based alert notifier.

    Logs alerts to standard logging system.
    """

    async def notify(self, alert: Alert) -> None:
        """
        Log alert.

        Args:
            alert: Alert to log
        """
        log_func = logger.warning
        if alert.severity == "CRITICAL":
            log_func = logger.critical
        elif alert.severity == "INFO":
            log_func = logger.info

        log_func(
            f"[ALERT] {alert.severity} - {alert.message} "
            f"(metric={alert.metric_name}, value={alert.actual_value}, "
            f"threshold={alert.threshold})"
        )


class KafkaAlertNotifier(AlertNotifier):
    """
    Kafka-based alert notifier.

    Publishes alerts to Kafka topic in OnexEnvelopeV1 format.
    """

    ALERT_TOPIC = "dev.agent.alerts.v1"

    def __init__(self, bootstrap_servers: str):
        """
        Initialize Kafka alert notifier.

        Args:
            bootstrap_servers: Kafka bootstrap servers
        """
        self._bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Start Kafka producer."""
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="snappy",
            )
            await self._producer.start()
            logger.info(f"Kafka alert notifier started: {self._bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to start Kafka alert notifier: {e}")
            raise

    async def stop(self) -> None:
        """Stop Kafka producer."""
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka alert notifier stopped")

    async def notify(self, alert: Alert) -> None:
        """
        Publish alert to Kafka.

        Args:
            alert: Alert to publish
        """
        if not self._producer:
            logger.warning("Kafka producer not started, skipping alert")
            return

        try:
            # Convert to OnexEnvelopeV1 format
            event = AlertEvent(
                correlation_id=alert.correlation_id,
                payload=AlertEventPayload(
                    alert_id=alert.alert_id,
                    severity=alert.severity,
                    metric_name=alert.metric_name,
                    threshold=alert.threshold,
                    actual_value=alert.actual_value,
                    message=alert.message,
                    tags=alert.tags,
                ),
            )

            # Publish
            await self._producer.send(self.ALERT_TOPIC, value=event.model_dump())
            await self._producer.flush()

        except Exception as e:
            logger.error(f"Kafka alert publish error: {e}")
