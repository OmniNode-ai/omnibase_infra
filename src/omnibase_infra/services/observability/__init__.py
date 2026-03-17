# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observability services for agent telemetry and monitoring.

Infrastructure for collecting, processing, and persisting
agent observability data including actions, routing decisions, and performance
metrics.

Submodules:
    - agent_actions: Consumer and writer for agent action events

Example:
    >>> from omnibase_infra.services.observability import (
    ...     AgentActionsConsumer,
    ...     ConfigAgentActionsConsumer,
    ...     WriterAgentActionsPostgres,
    ... )
    >>>
    >>> config = ConfigAgentActionsConsumer(
    ...     kafka_bootstrap_servers="localhost:19092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> consumer = AgentActionsConsumer(config)
    >>>
    >>> # Run consumer
    >>> await consumer.start()
    >>> await consumer.run()
"""

from omnibase_infra.services.observability.agent_actions import (
    AgentActionsConsumer,
    ConfigAgentActionsConsumer,
    ConfigTTLCleanup,
    ServiceTTLCleanup,
    WriterAgentActionsPostgres,
)

__all__ = [
    "AgentActionsConsumer",
    "ConfigAgentActionsConsumer",
    "ConfigTTLCleanup",
    "ServiceTTLCleanup",
    "WriterAgentActionsPostgres",
]
