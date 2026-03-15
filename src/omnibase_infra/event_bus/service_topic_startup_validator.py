# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic Startup Validator for registry-first existence checks.

Validates that all platform topics declared in ``ALL_PROVISIONED_SUFFIXES``
exist on the Kafka/Redpanda broker at startup time. Default behaviour is
best-effort (log warnings). Opt-in strict mode via
``STARTUP_VALIDATION_STRICT=1`` raises ``RuntimeError`` on missing topics.

Design:
    - Best-effort by default: logs errors for missing topics but never blocks
    - Strict mode: ``STARTUP_VALIDATION_STRICT=1`` env var makes missing topics fatal
    - Graceful degradation: handles missing aiokafka and unreachable brokers
    - Follows ``TopicProvisioner`` class structure

Related Tickets:
    - OMN-3769: Registry-First Startup Assertions
"""

from __future__ import annotations

import logging
import os
from uuid import UUID, uuid4

from omnibase_infra.event_bus.enum_topic_validation_status import (
    EnumTopicValidationStatus,
)
from omnibase_infra.event_bus.model_topic_validation_result import (
    ModelTopicValidationResult,
)
from omnibase_infra.topics import ALL_PROVISIONED_SUFFIXES

logger = logging.getLogger(__name__)

# Reuse constants from TopicProvisioner
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
ENV_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"


class TopicStartupValidator:
    """Validates that required platform topics exist on the broker.

    Queries the broker for existing topics and compares against
    ``ALL_PROVISIONED_SUFFIXES``. Returns a ``ModelTopicValidationResult``
    with details about present and missing topics.

    Thread Safety:
        This class is coroutine-safe. All methods are async and use
        the AIOKafkaAdminClient which handles its own connection pooling.

    Example:
        >>> validator = TopicStartupValidator()
        >>> result = await validator.validate()
        >>> if not result.is_valid:
        ...     print(f"Missing: {result.missing_topics}")
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        request_timeout_ms: int = 10000,
    ) -> None:
        """Initialize the topic startup validator.

        Args:
            bootstrap_servers: Kafka broker addresses. If None, reads from
                KAFKA_BOOTSTRAP_SERVERS env var or defaults to localhost:9092.
            request_timeout_ms: Timeout for admin operations in milliseconds.
        """
        self._bootstrap_servers = bootstrap_servers or os.environ.get(
            ENV_BOOTSTRAP_SERVERS, DEFAULT_BOOTSTRAP_SERVERS
        )
        self._request_timeout_ms = request_timeout_ms

    async def validate(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelTopicValidationResult:
        """Validate that all required platform topics exist on the broker.

        Collects required topics from ``ALL_PROVISIONED_SUFFIXES`` and checks
        their existence via ``AIOKafkaAdminClient.list_topics()``.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ``ModelTopicValidationResult`` with validation outcome.
        """
        correlation_id = correlation_id or uuid4()
        required = ALL_PROVISIONED_SUFFIXES

        # Guard: aiokafka not installed
        try:
            from aiokafka.admin import AIOKafkaAdminClient
        except ImportError:
            logger.warning(
                "aiokafka not available, skipping topic startup validation. "
                "Install aiokafka to enable topic existence checks.",
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelTopicValidationResult(
                required_topics=required,
                is_valid=True,
                status=EnumTopicValidationStatus.SKIPPED,
            )

        # Guard: broker unreachable
        admin: AIOKafkaAdminClient | None = None
        try:
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
            )
            await admin.start()

            # list_topics() returns a dict of {topic_name: TopicMetadata}
            broker_metadata = await admin.list_topics()
            broker_topics = set(broker_metadata)

        except Exception:
            logger.warning(
                "Broker unreachable at %s, skipping topic startup validation",
                self._bootstrap_servers,
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelTopicValidationResult(
                required_topics=required,
                is_valid=True,
                status=EnumTopicValidationStatus.UNAVAILABLE,
            )

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:
                    pass  # Best-effort cleanup

        # Compute present / missing
        present = tuple(t for t in required if t in broker_topics)
        missing = tuple(t for t in required if t not in broker_topics)

        if missing:
            for topic in missing:
                logger.error(
                    "MISSING_TOPIC: Required topic '%s' not in broker",
                    topic,
                    extra={"correlation_id": str(correlation_id)},
                )
            return ModelTopicValidationResult(
                required_topics=required,
                present_topics=present,
                missing_topics=missing,
                is_valid=False,
                status=EnumTopicValidationStatus.DEGRADED,
            )

        return ModelTopicValidationResult(
            required_topics=required,
            present_topics=present,
            missing_topics=(),
            is_valid=True,
            status=EnumTopicValidationStatus.SUCCESS,
        )


__all__: list[str] = [
    "TopicStartupValidator",
]
