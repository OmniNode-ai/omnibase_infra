# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka Topic Provisioner for automatic topic creation on startup.

Ensures that all ONEX platform topics exist before the runtime begins
consuming or producing events. Uses AIOKafkaAdminClient to create topics
that are missing, with best-effort semantics (warnings on failure, never
blocks startup).

Design:
    - Best-effort: Logs warnings but never blocks startup on failure
    - Idempotent: Safe to call multiple times (skips existing topics)
    - Compatible: Works with both Redpanda and Apache Kafka
    - Configurable: Supports custom topic configs via ModelSnapshotTopicConfig

Related Tickets:
    - OMN-1990: Kafka topic auto-creation gap
"""

from __future__ import annotations

import logging
import os
from uuid import UUID, uuid4

from omnibase_infra.models.projection.model_snapshot_topic_config import (
    ModelSnapshotTopicConfig,
)
from omnibase_infra.topics import ALL_PLATFORM_SUFFIXES, SUFFIX_REGISTRATION_SNAPSHOTS
from omnibase_infra.utils import sanitize_error_message

logger = logging.getLogger(__name__)

# Default bootstrap servers (matches event_bus_kafka.py pattern)
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
ENV_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"

# Default partition and replication settings for standard event topics
DEFAULT_EVENT_TOPIC_PARTITIONS = 6
DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1


class TopicProvisioner:
    """Provisions Kafka topics automatically on startup.

    Creates ONEX platform topics if they don't already exist, using
    AIOKafkaAdminClient. Topic creation is best-effort: failures log
    warnings but never block startup.

    The provisioner handles two categories of topics:
    1. **Standard event topics**: Created with default settings (delete cleanup)
    2. **Snapshot topics**: Created with compaction settings from ModelSnapshotTopicConfig

    Thread Safety:
        This class is coroutine-safe. All methods are async and use
        the AIOKafkaAdminClient which handles its own connection pooling.

    Example:
        >>> provisioner = TopicProvisioner()
        >>> await provisioner.ensure_platform_topics_exist()
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        request_timeout_ms: int = 30000,
    ) -> None:
        """Initialize the topic provisioner.

        Args:
            bootstrap_servers: Kafka broker addresses. If None, reads from
                KAFKA_BOOTSTRAP_SERVERS env var or defaults to localhost:9092.
            request_timeout_ms: Timeout for admin operations in milliseconds.
        """
        self._bootstrap_servers = bootstrap_servers or os.environ.get(
            ENV_BOOTSTRAP_SERVERS, DEFAULT_BOOTSTRAP_SERVERS
        )
        self._request_timeout_ms = request_timeout_ms

    async def ensure_platform_topics_exist(
        self,
        correlation_id: UUID | None = None,
    ) -> dict[str, list[str] | str]:
        """Ensure all ONEX platform topics exist.

        Creates any missing topics from ALL_PLATFORM_SUFFIXES. The snapshot
        topic gets special compaction configuration via ModelSnapshotTopicConfig.

        This method is best-effort: individual topic creation failures are
        logged as warnings but do not prevent other topics from being created.
        Unrecoverable failures (connection, authentication, etc.) are also
        logged as warnings and never block startup.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Summary dict with:
                - created: List of newly created topic names
                - existing: List of topics that already existed
                - failed: List of topics that failed to create
                - status: "success", "partial", or "unavailable"
        """
        correlation_id = correlation_id or uuid4()
        created: list[str] = []
        existing: list[str] = []
        failed: list[str] = []

        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import (
                TopicAlreadyExistsError as _TopicAlreadyExistsError,
            )
        except ImportError:
            logger.warning(
                "aiokafka not available, skipping topic auto-creation. "
                "Install aiokafka to enable automatic topic management.",
                extra={"correlation_id": str(correlation_id)},
            )
            return {
                "created": created,
                "existing": existing,
                "failed": list(ALL_PLATFORM_SUFFIXES),
                "status": "unavailable",
            }

        # Bind to local after successful import block
        TopicAlreadyExistsError = _TopicAlreadyExistsError

        admin: AIOKafkaAdminClient | None = None
        try:
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
            )
            await admin.start()

            # Load snapshot topic config
            snapshot_config = ModelSnapshotTopicConfig.default()

            for suffix in ALL_PLATFORM_SUFFIXES:
                try:
                    if suffix == SUFFIX_REGISTRATION_SNAPSHOTS:
                        # Snapshot topic with compaction config
                        new_topic = NewTopic(
                            name=suffix,
                            num_partitions=snapshot_config.partition_count,
                            replication_factor=snapshot_config.replication_factor,
                            topic_configs=snapshot_config.to_kafka_config(),
                        )
                    else:
                        # Standard event topic with defaults
                        new_topic = NewTopic(
                            name=suffix,
                            num_partitions=DEFAULT_EVENT_TOPIC_PARTITIONS,
                            replication_factor=DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR,
                        )

                    await admin.create_topics([new_topic])
                    created.append(suffix)
                    logger.info(
                        "Created topic: %s",
                        suffix,
                        extra={"correlation_id": str(correlation_id)},
                    )

                except TopicAlreadyExistsError:
                    existing.append(suffix)
                    logger.debug(
                        "Topic already exists: %s",
                        suffix,
                        extra={"correlation_id": str(correlation_id)},
                    )

                except Exception as e:
                    failed.append(suffix)
                    logger.warning(
                        "Failed to create topic %s: %s",
                        suffix,
                        type(e).__name__,
                        extra={
                            "correlation_id": str(correlation_id),
                            "error": sanitize_error_message(e),
                        },
                    )

        except Exception as e:
            logger.warning(
                "Topic auto-creation interrupted by %s. "
                "Topics may need to be created manually or via broker auto-create.",
                type(e).__name__,
                extra={
                    "bootstrap_servers": self._bootstrap_servers,
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            # Separate individually-failed topics from those never attempted
            already_resolved = set(created) | set(existing) | set(failed)
            not_attempted = [
                s for s in ALL_PLATFORM_SUFFIXES if s not in already_resolved
            ]
            if not_attempted:
                logger.warning(
                    "Topics not attempted due to early termination: %d topics",
                    len(not_attempted),
                    extra={
                        "not_attempted_count": len(not_attempted),
                        "correlation_id": str(correlation_id),
                    },
                )
            return {
                "created": created,
                "existing": existing,
                "failed": failed + not_attempted,
                "status": "unavailable",
            }

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:
                    pass  # Best-effort cleanup

        status = (
            "success"
            if not failed
            else ("partial" if created or existing else "unavailable")
        )

        logger.info(
            "Topic auto-creation complete",
            extra={
                "created_count": len(created),
                "existing_count": len(existing),
                "failed_count": len(failed),
                "status": status,
                "correlation_id": str(correlation_id),
            },
        )

        return {
            "created": created,
            "existing": existing,
            "failed": failed,
            "status": status,
        }

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: ModelSnapshotTopicConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Ensure a single topic exists with optional custom config.

        Args:
            topic_name: The topic name to create.
            config: Optional topic configuration. If None, uses default
                event topic settings.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if topic was created or already exists, False on failure.
        """
        correlation_id = correlation_id or uuid4()

        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import (
                TopicAlreadyExistsError as _TopicAlreadyExistsError,
            )
        except ImportError:
            logger.warning(
                "aiokafka not available, cannot create topic %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

        # Bind to local after successful import block
        TopicAlreadyExistsError = _TopicAlreadyExistsError

        admin: AIOKafkaAdminClient | None = None
        try:
            admin = AIOKafkaAdminClient(
                bootstrap_servers=self._bootstrap_servers,
                request_timeout_ms=self._request_timeout_ms,
            )
            await admin.start()

            if config is not None:
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=config.partition_count,
                    replication_factor=config.replication_factor,
                    topic_configs=config.to_kafka_config(),
                )
            else:
                new_topic = NewTopic(
                    name=topic_name,
                    num_partitions=DEFAULT_EVENT_TOPIC_PARTITIONS,
                    replication_factor=DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR,
                )

            await admin.create_topics([new_topic])
            logger.info(
                "Created topic: %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return True

        except TopicAlreadyExistsError:
            logger.debug(
                "Topic already exists: %s",
                topic_name,
                extra={"correlation_id": str(correlation_id)},
            )
            return True

        except Exception as e:
            logger.warning(
                "Failed to create topic %s: %s",
                topic_name,
                type(e).__name__,
                extra={
                    "correlation_id": str(correlation_id),
                    "error": sanitize_error_message(e),
                },
            )
            return False

        finally:
            if admin is not None:
                try:
                    await admin.close()
                except Exception:
                    pass


__all__ = ["TopicProvisioner"]
