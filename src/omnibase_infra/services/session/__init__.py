# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Session storage and event consumer services.

This module provides infrastructure for persisting Claude Code session
snapshots and consuming session events from Kafka.

Moved from omniclaude as part of OMN-1526 architectural cleanup.

Components:
    - SessionSnapshotStore: PostgreSQL storage for session snapshots
    - SessionEventConsumer: Kafka consumer for session events
    - ConfigSessionStorage: Storage configuration
    - ConfigSessionConsumer: Consumer configuration
    - ConsumerMetrics: Metrics for consumer observability
    - EnumCircuitState: Circuit breaker states

Example:
    >>> from omnibase_infra.services.session import (
    ...     SessionSnapshotStore,
    ...     ConfigSessionStorage,
    ... )
    >>> from pydantic import SecretStr
    >>>
    >>> config = ConfigSessionStorage(postgres_password=SecretStr("secret"))
    >>> store = SessionSnapshotStore(config)
    >>> await store.initialize()
"""

from omnibase_infra.services.session.config_consumer import ConfigSessionConsumer
from omnibase_infra.services.session.config_store import ConfigSessionStorage
from omnibase_infra.services.session.consumer import (
    ConsumerMetrics,
    EnumCircuitState,
    SessionEventConsumer,
)
from omnibase_infra.services.session.protocol_session_aggregator import (
    ProtocolSessionAggregator,
)
from omnibase_infra.services.session.store import (
    SessionSnapshotStore,
    SessionStoreNotInitializedError,
)

__all__ = [
    # Storage
    "SessionSnapshotStore",
    "SessionStoreNotInitializedError",
    "ConfigSessionStorage",
    # Consumer
    "SessionEventConsumer",
    "ConfigSessionConsumer",
    "ConsumerMetrics",
    "EnumCircuitState",
    "ProtocolSessionAggregator",
]
