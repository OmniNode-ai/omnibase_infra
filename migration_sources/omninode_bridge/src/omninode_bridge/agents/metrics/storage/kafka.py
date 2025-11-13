"""
Kafka metrics storage with OnexEnvelopeV1 format.

Publishes metrics to 5 topics for real-time monitoring.
"""

import json
import logging
from typing import ClassVar

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from omninode_bridge.agents.metrics.models import (
    Metric,
    MetricEvent,
    MetricEventPayload,
)

logger = logging.getLogger(__name__)


class KafkaMetricsWriter:
    """
    Kafka metrics writer with batch publishing.

    Features:
    - Batch publishing for efficiency
    - Snappy compression
    - OnexEnvelopeV1 format compliance
    - Topic routing by metric category

    Performance:
    - Batch publish: <30ms for 100 metrics
    - Compression: 3-5x reduction

    Topics:
    - dev.agent.metrics.routing.v1
    - dev.agent.metrics.state-ops.v1
    - dev.agent.metrics.coordination.v1
    - dev.agent.metrics.workflow.v1
    - dev.agent.metrics.ai-quorum.v1
    """

    # Metric name prefixes to topic mapping
    TOPIC_MAPPING: ClassVar[dict[str, str]] = {
        "routing_": "dev.agent.metrics.routing.v1",
        "state_": "dev.agent.metrics.state-ops.v1",
        "coordination_": "dev.agent.metrics.coordination.v1",
        "workflow_": "dev.agent.metrics.workflow.v1",
        "quorum_": "dev.agent.metrics.ai-quorum.v1",
    }

    DEFAULT_TOPIC = "dev.agent.metrics.routing.v1"

    def __init__(self, bootstrap_servers: str):
        """
        Initialize Kafka metrics writer.

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
                acks="all",
                max_batch_size=16384,  # 16KB batch size
                linger_ms=10,  # 10ms linger for batching
            )
            await self._producer.start()
            logger.info(f"Kafka producer started: {self._bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise

    async def stop(self) -> None:
        """Stop Kafka producer."""
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def write_batch(self, metrics: list[Metric]) -> None:
        """
        Write batch of metrics to Kafka.

        Args:
            metrics: List of metrics to write

        Performance: <30ms for 100 metrics
        """
        if not self._producer:
            logger.warning("Kafka producer not started, skipping write")
            return

        if not metrics:
            return

        # Group metrics by topic
        metrics_by_topic = self._group_by_topic(metrics)

        # Publish to topics
        for topic, topic_metrics in metrics_by_topic.items():
            for metric in topic_metrics:
                try:
                    # Convert to OnexEnvelopeV1 format
                    event = MetricEvent(
                        correlation_id=metric.correlation_id,
                        payload=MetricEventPayload(
                            metric_id=metric.metric_id,
                            metric_name=metric.metric_name,
                            metric_type=metric.metric_type.value,
                            value=metric.value,
                            unit=metric.unit,
                            tags=metric.tags,
                            agent_id=metric.agent_id,
                            correlation_id=metric.correlation_id,
                        ),
                    )

                    # Publish (async, don't wait)
                    await self._producer.send(topic, value=event.model_dump())

                except KafkaError as e:
                    logger.error(f"Kafka publish error for topic {topic}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error publishing to Kafka: {e}")

        # Flush to ensure delivery
        try:
            await self._producer.flush()
        except Exception as e:
            logger.error(f"Kafka flush error: {e}")

    def _group_by_topic(self, metrics: list[Metric]) -> dict[str, list[Metric]]:
        """
        Group metrics by Kafka topic based on metric name prefix.

        Args:
            metrics: List of metrics

        Returns:
            Dictionary mapping topic to metrics list
        """
        grouped: dict[str, list[Metric]] = {}

        for metric in metrics:
            topic = self._get_topic_for_metric(metric.metric_name)
            if topic not in grouped:
                grouped[topic] = []
            grouped[topic].append(metric)

        return grouped

    def _get_topic_for_metric(self, metric_name: str) -> str:
        """
        Get Kafka topic for metric based on name prefix.

        Args:
            metric_name: Metric name

        Returns:
            Kafka topic name
        """
        for prefix, topic in self.TOPIC_MAPPING.items():
            if metric_name.startswith(prefix):
                return topic
        return self.DEFAULT_TOPIC
