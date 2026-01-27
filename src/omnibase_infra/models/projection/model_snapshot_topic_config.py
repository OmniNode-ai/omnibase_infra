# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot Topic Configuration Model.

Provides Pydantic configuration for Kafka snapshot topics. Snapshot topics use
log compaction (cleanup.policy=compact) so only the latest snapshot per entity_id
is retained, enabling efficient state reconstruction without replaying the full
event log.

Design Notes:
    Snapshot topics are distinct from event topics in several key ways:

    1. **Compaction vs Retention**:
       - Event topics: cleanup.policy=delete (immutable log, time-based retention)
       - Snapshot topics: cleanup.policy=compact (only latest per key retained)

    2. **Authoritative Source**:
       - Events are the source of truth (immutable facts)
       - Snapshots are derived, read-optimized views (can be rebuilt from events)

    3. **Key Format**:
       - Snapshot keys: "{domain}:{entity_id}" (e.g., "registration:uuid-here")
       - This enables per-entity compaction where only the latest snapshot survives

    4. **Partition Strategy**:
       - Partitioning should match the projection's entity_id partitioning
       - This ensures snapshots for a given entity are always in the same partition

Topic Naming (per ONEX Topic Taxonomy):
    ONEX Kafka format: onex.<domain>.snapshots
    Environment-Aware format: <env>.<domain>.snapshots.v<version>

    Examples:
        - onex.registration.snapshots
        - dev.registration.snapshots.v1
        - prod.discovery.snapshots.v1

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model

See Also:
    - ONEX Topic Taxonomy: docs/standards/onex_topic_taxonomy.md
    - ModelKafkaEventBusConfig: Event bus configuration patterns
    - ModelRegistrationProjection: The projection model snapshotted to this topic
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.topics import SUFFIX_REGISTRATION_SNAPSHOTS

if TYPE_CHECKING:
    from omnibase_infra.errors.error_infra import ProtocolConfigurationError
    from omnibase_infra.models.errors.model_infra_error_context import (
        ModelInfraErrorContext,
    )


def _get_error_classes() -> tuple[
    type[ProtocolConfigurationError], type[ModelInfraErrorContext]
]:
    """Lazy import of error classes to avoid circular import.

    Returns:
        Tuple of (ProtocolConfigurationError, ModelInfraErrorContext)
    """
    from omnibase_infra.errors.error_infra import ProtocolConfigurationError
    from omnibase_infra.models.errors.model_infra_error_context import (
        ModelInfraErrorContext,
    )

    return ProtocolConfigurationError, ModelInfraErrorContext


logger = logging.getLogger(__name__)


class ModelSnapshotTopicConfig(BaseModel):
    """Configuration for Kafka snapshot topics.

    Snapshot topics use log compaction (cleanup.policy=compact) so only the
    latest snapshot per entity_id is retained. This enables efficient state
    reconstruction for orchestrators and read models without replaying the
    full event log.

    Key Semantics:
        Kafka compaction uses message keys to determine which records to retain.
        For snapshot topics, keys should follow the format:

            {domain}:{entity_id}

        For example: "registration:550e8400-e29b-41d4-a716-446655440000"

        This ensures that only the latest snapshot for each entity survives
        compaction, while maintaining domain isolation.

    Compaction Timing:
        The min_compaction_lag_ms and max_compaction_lag_ms settings control
        how quickly compaction can occur:

        - min_compaction_lag_ms (default: 60s): Minimum time before a message
          can be compacted. Prevents very recent snapshots from being compacted
          immediately (useful if consumers are slightly behind).

        - max_compaction_lag_ms (default: 5min): Maximum time before compaction
          is forced. Ensures stale snapshots are cleaned up in a timely manner.

        For high-write-volume topics, consider increasing these values to reduce
        compaction overhead.

    Attributes:
        topic: Full Kafka topic name for registration snapshots
        partition_count: Number of partitions (should match projection partitioning)
        replication_factor: Replication factor for durability
        cleanup_policy: Kafka cleanup policy (must be "compact" for snapshots)
        min_compaction_lag_ms: Minimum time before a message can be compacted
        max_compaction_lag_ms: Maximum time before compaction is forced
        segment_bytes: Segment size (smaller = more frequent compaction)
        retention_ms: Retention time (-1 for infinite, required for compacted topics)
        min_insync_replicas: Minimum in-sync replicas for writes

    Example:
        >>> from omnibase_infra.models.projection import ModelSnapshotTopicConfig
        >>>
        >>> # Use defaults
        >>> config = ModelSnapshotTopicConfig.default()
        >>> config.topic
        'onex.registration.snapshots'
        >>>
        >>> # Custom configuration
        >>> config = ModelSnapshotTopicConfig(
        ...     topic="prod.registration.snapshots.v1",
        ...     partition_count=24,
        ...     replication_factor=3,
        ... )
        >>>
        >>> # From YAML
        >>> config = ModelSnapshotTopicConfig.from_yaml(
        ...     Path("config/snapshot_topic.yaml")
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Topic identity
    topic: str = Field(
        default=SUFFIX_REGISTRATION_SNAPSHOTS,
        min_length=1,
        max_length=255,
        description="Full Kafka topic name for registration snapshots",
    )

    # Partition configuration
    partition_count: int = Field(
        default=12,
        ge=1,
        le=1000,
        description="Number of partitions (should match projection partitioning)",
    )
    replication_factor: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Replication factor for durability",
    )

    # Compaction configuration (required for snapshot topics)
    cleanup_policy: str = Field(
        default="compact",
        description="Kafka cleanup policy. Must be 'compact' for snapshot topics.",
    )
    min_compaction_lag_ms: int = Field(
        default=60000,  # 1 minute
        ge=0,
        le=604800000,  # 7 days max
        description="Minimum time (ms) before a message can be compacted",
    )
    max_compaction_lag_ms: int = Field(
        default=300000,  # 5 minutes
        ge=0,
        le=604800000,  # 7 days max
        description="Maximum time (ms) before compaction is forced",
    )
    segment_bytes: int = Field(
        default=104857600,  # 100MB
        ge=1048576,  # 1MB minimum
        le=1073741824,  # 1GB maximum
        description="Segment size in bytes (smaller = more frequent compaction)",
    )

    # Retention (must be -1 or very long for compacted topics)
    retention_ms: int = Field(
        default=-1,  # Infinite retention (required for compacted topics)
        ge=-1,
        description="Retention time in ms (-1 for infinite, recommended for compaction)",
    )

    # Durability settings
    min_insync_replicas: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum in-sync replicas for writes",
    )

    @field_validator("cleanup_policy", mode="before")
    @classmethod
    def validate_cleanup_policy(cls, v: object) -> str:
        """Validate that cleanup_policy is 'compact' for snapshot topics.

        Snapshot topics MUST use log compaction to retain only the latest
        snapshot per entity_id. Other cleanup policies would either:
        - 'delete': Remove old snapshots based on time, losing state
        - 'compact,delete': Hybrid that may delete before compaction completes

        Args:
            v: Cleanup policy value (any type before Pydantic conversion)

        Returns:
            Validated cleanup policy string

        Raises:
            ProtocolConfigurationError: If cleanup_policy is not 'compact'
        """
        ProtocolConfigurationError, ModelInfraErrorContext = _get_error_classes()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_snapshot_topic_config",
            target_name="snapshot_topic",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "cleanup_policy cannot be None for snapshot topics",
                context=context,
                parameter="cleanup_policy",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"cleanup_policy must be a string, got {type(v).__name__}",
                context=context,
                parameter="cleanup_policy",
                value=type(v).__name__,
            )

        v_lower = v.strip().lower()
        if v_lower != "compact":
            raise ProtocolConfigurationError(
                f"Snapshot topics MUST use cleanup.policy=compact. "
                f"Got '{v}'. Snapshots require compaction to retain only the "
                f"latest state per entity_id. Using '{v}' would cause data loss.",
                context=context,
                parameter="cleanup_policy",
                value=v,
            )

        return v_lower

    @field_validator("topic", mode="before")
    @classmethod
    def validate_topic(cls, v: object) -> str:
        """Validate topic format and content.

        Args:
            v: Topic value (any type before Pydantic conversion)

        Returns:
            Validated topic string

        Raises:
            ProtocolConfigurationError: If topic is invalid
        """
        ProtocolConfigurationError, ModelInfraErrorContext = _get_error_classes()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="validate_snapshot_topic_config",
            target_name="snapshot_topic",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "topic cannot be None",
                context=context,
                parameter="topic",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"topic must be a string, got {type(v).__name__}",
                context=context,
                parameter="topic",
                value=type(v).__name__,
            )

        topic = v.strip()
        if not topic:
            raise ProtocolConfigurationError(
                "topic cannot be empty",
                context=context,
                parameter="topic",
                value=v,
            )

        # Validate topic contains "snapshots" suffix (per ONEX taxonomy)
        if not topic.endswith(".snapshots") and ".snapshots." not in topic:
            logger.warning(
                "Topic name '%s' does not follow ONEX snapshot topic naming "
                "convention (should end with '.snapshots' or contain '.snapshots.'). "
                "Expected patterns: 'onex.<domain>.snapshots' or "
                "'<env>.<domain>.snapshots.v<version>'",
                topic,
            )

        return topic

    @model_validator(mode="after")
    def validate_compaction_lag_order(self) -> ModelSnapshotTopicConfig:
        """Validate that min_compaction_lag_ms <= max_compaction_lag_ms.

        The min lag should not exceed max lag, otherwise compaction timing
        would be undefined. Kafka requires that minimum compaction lag is
        less than or equal to maximum compaction lag.

        Returns:
            Self if validation passes

        Raises:
            ProtocolConfigurationError: If min_compaction_lag_ms > max_compaction_lag_ms
        """
        if self.min_compaction_lag_ms > self.max_compaction_lag_ms:
            ProtocolConfigurationError, ModelInfraErrorContext = _get_error_classes()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="validate_snapshot_topic_config",
                target_name="snapshot_topic",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"min_compaction_lag_ms ({self.min_compaction_lag_ms}) must be <= "
                f"max_compaction_lag_ms ({self.max_compaction_lag_ms})",
                context=context,
                parameter="compaction_lag",
                value=f"min={self.min_compaction_lag_ms}, max={self.max_compaction_lag_ms}",
            )
        return self

    def apply_environment_overrides(self) -> ModelSnapshotTopicConfig:
        """Apply environment variable overrides to configuration.

        Environment variables are mapped as follows:
            - SNAPSHOT_TOPIC -> topic
            - SNAPSHOT_PARTITION_COUNT -> partition_count
            - SNAPSHOT_REPLICATION_FACTOR -> replication_factor
            - SNAPSHOT_MIN_COMPACTION_LAG_MS -> min_compaction_lag_ms
            - SNAPSHOT_MAX_COMPACTION_LAG_MS -> max_compaction_lag_ms
            - SNAPSHOT_SEGMENT_BYTES -> segment_bytes
            - SNAPSHOT_RETENTION_MS -> retention_ms
            - SNAPSHOT_MIN_INSYNC_REPLICAS -> min_insync_replicas

        Note: cleanup_policy is NOT overridable via environment variable
        because snapshot topics MUST use compaction.

        Returns:
            New configuration instance with environment overrides applied
        """
        overrides: dict[str, object] = {}

        env_mappings: dict[str, str] = {
            "SNAPSHOT_TOPIC": "topic",
            "SNAPSHOT_PARTITION_COUNT": "partition_count",
            "SNAPSHOT_REPLICATION_FACTOR": "replication_factor",
            "SNAPSHOT_MIN_COMPACTION_LAG_MS": "min_compaction_lag_ms",
            "SNAPSHOT_MAX_COMPACTION_LAG_MS": "max_compaction_lag_ms",
            "SNAPSHOT_SEGMENT_BYTES": "segment_bytes",
            "SNAPSHOT_RETENTION_MS": "retention_ms",
            "SNAPSHOT_MIN_INSYNC_REPLICAS": "min_insync_replicas",
        }

        # Integer fields for type conversion
        int_fields = {
            "partition_count",
            "replication_factor",
            "min_compaction_lag_ms",
            "max_compaction_lag_ms",
            "segment_bytes",
            "retention_ms",
            "min_insync_replicas",
        }

        for env_var, field_name in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                if field_name in int_fields:
                    try:
                        overrides[field_name] = int(env_value)
                    except ValueError:
                        logger.warning(
                            "Failed to parse integer environment variable %s='%s', "
                            "using default value for %s",
                            env_var,
                            env_value,
                            field_name,
                        )
                        continue
                else:
                    overrides[field_name] = env_value

        if overrides:
            current_data = self.model_dump()
            current_data.update(overrides)
            return ModelSnapshotTopicConfig(**current_data)

        return self

    @classmethod
    def default(cls) -> ModelSnapshotTopicConfig:
        """Create default configuration with environment overrides.

        Returns a canonical default configuration for development and testing,
        with environment variable overrides applied.

        Defaults are tuned for:
            - Moderate write volume (12 partitions)
            - High durability (replication_factor=3, min_insync_replicas=2)
            - Balanced compaction (1 min lag, 5 min max lag)
            - Infinite retention (required for compacted topics)

        Returns:
            Default configuration instance with environment overrides
        """
        base_config = cls(
            topic=SUFFIX_REGISTRATION_SNAPSHOTS,
            partition_count=12,
            replication_factor=3,
            cleanup_policy="compact",
            min_compaction_lag_ms=60000,  # 1 minute
            max_compaction_lag_ms=300000,  # 5 minutes
            segment_bytes=104857600,  # 100MB
            retention_ms=-1,  # Infinite
            min_insync_replicas=2,
        )
        return base_config.apply_environment_overrides()

    @classmethod
    def from_yaml(cls, path: Path) -> ModelSnapshotTopicConfig:
        """Load configuration from YAML file.

        Loads configuration from a YAML file and applies environment
        variable overrides on top.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration instance loaded from YAML with env overrides

        Raises:
            FileNotFoundError: If the YAML file does not exist
            ProtocolConfigurationError: If the YAML content is invalid

        Example YAML:
            ```yaml
            topic: "prod.registration.snapshots.v1"
            partition_count: 24
            replication_factor: 3
            cleanup_policy: "compact"
            min_compaction_lag_ms: 120000
            max_compaction_lag_ms: 600000
            segment_bytes: 52428800
            retention_ms: -1
            min_insync_replicas: 2
            ```
        """
        ProtocolConfigurationError, ModelInfraErrorContext = _get_error_classes()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="load_yaml_config",
            target_name="snapshot_topic_config",
            correlation_id=uuid4(),
        )

        if not path.exists():
            raise ProtocolConfigurationError(
                f"Configuration file not found: {path}",
                context=context,
                config_path=str(path),
            )

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ProtocolConfigurationError(
                f"Failed to parse YAML from {path}: {e}",
                context=context,
                config_path=str(path),
                error_details=str(e),
            ) from e
        except UnicodeDecodeError as e:
            raise ProtocolConfigurationError(
                f"Configuration file contains binary or non-UTF-8 content: {path}",
                context=context,
                config_path=str(path),
                error_details=f"Encoding error at position {e.start}-{e.end}: {e.reason}",
            ) from e
        except OSError as e:
            raise ProtocolConfigurationError(
                f"Failed to read configuration file: {path}: {e}",
                context=context,
                config_path=str(path),
                error_details=str(e),
            ) from e

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ProtocolConfigurationError(
                f"YAML content must be a dictionary, got {type(data)}",
                context=context,
                parameter="yaml_content",
                value=type(data).__name__,
            )

        config = cls(**data)
        return config.apply_environment_overrides()

    def to_kafka_config(self) -> dict[str, str]:
        """Convert to Kafka topic configuration dictionary.

        Returns a dictionary suitable for passing to Kafka AdminClient
        topic creation APIs or kafka-topics.sh --config options.

        Returns:
            Dictionary of Kafka topic configuration properties

        Example:
            >>> config = ModelSnapshotTopicConfig.default()
            >>> kafka_config = config.to_kafka_config()
            >>> kafka_config["cleanup.policy"]
            'compact'
            >>> kafka_config["min.compaction.lag.ms"]
            '60000'
        """
        return {
            "cleanup.policy": self.cleanup_policy,
            "min.compaction.lag.ms": str(self.min_compaction_lag_ms),
            "max.compaction.lag.ms": str(self.max_compaction_lag_ms),
            "segment.bytes": str(self.segment_bytes),
            "retention.ms": str(self.retention_ms),
            "min.insync.replicas": str(self.min_insync_replicas),
        }

    def get_snapshot_key(self, domain: str, entity_id: str) -> str:
        """Generate a snapshot key for Kafka compaction.

        Keys follow the format: {domain}:{entity_id}

        This ensures that compaction retains only the latest snapshot
        for each (domain, entity_id) combination.

        Args:
            domain: Domain namespace (e.g., "registration", "discovery")
            entity_id: Entity UUID as string

        Returns:
            Formatted snapshot key for Kafka message

        Example:
            >>> config = ModelSnapshotTopicConfig.default()
            >>> config.get_snapshot_key("registration", "550e8400-e29b-41d4-a716-446655440000")
            'registration:550e8400-e29b-41d4-a716-446655440000'
        """
        return f"{domain}:{entity_id}"


__all__: list[str] = ["ModelSnapshotTopicConfig"]
