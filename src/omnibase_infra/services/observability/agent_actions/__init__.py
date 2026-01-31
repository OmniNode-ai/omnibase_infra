# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent actions observability consumer and writer.

This module provides infrastructure for consuming agent action events from
Kafka and persisting them to PostgreSQL for observability and analytics.

Moved from omniclaude as part of OMN-1743 layer-correction cleanup.

Components:
    - AgentActionsConsumer: Async Kafka consumer with per-partition offset tracking
    - ConfigAgentActionsConsumer: Configuration for the consumer
    - WriterAgentActionsPostgres: PostgreSQL writer for observability events

Topics consumed:
    - agent-actions
    - agent-routing-decisions
    - agent-transformation-events
    - router-performance-metrics
    - agent-detection-failures
    - agent-execution-logs

Example:
    >>> from omnibase_infra.services.observability.agent_actions import (
    ...     AgentActionsConsumer,
    ...     ConfigAgentActionsConsumer,
    ...     WriterAgentActionsPostgres,
    ... )
    >>>
    >>> config = ConfigAgentActionsConsumer(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
    ... )
    >>> consumer = AgentActionsConsumer(config)
    >>>
    >>> # Run consumer
    >>> await consumer.start()
    >>> await consumer.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.agent_actions.consumer
"""

from omnibase_infra.services.observability.agent_actions.config import (
    ConfigAgentActionsConsumer,
)
from omnibase_infra.services.observability.agent_actions.consumer import (
    AgentActionsConsumer,
    ConsumerMetrics,
    EnumHealthStatus,
    mask_dsn_password,
)
from omnibase_infra.services.observability.agent_actions.writer_postgres import (
    WriterAgentActionsPostgres,
)

__all__ = [
    "AgentActionsConsumer",
    "ConfigAgentActionsConsumer",
    "ConsumerMetrics",
    "EnumHealthStatus",
    "WriterAgentActionsPostgres",
    "mask_dsn_password",
]
