# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability services for agent telemetry and monitoring.

This module provides infrastructure for collecting, processing, and persisting
agent observability data including actions, routing decisions, and performance
metrics.

Submodules:
    - agent_actions: Consumer and writer for agent action events

Example:
    >>> from omnibase_infra.services.observability.agent_actions import (
    ...     ConfigAgentActionsConsumer,
    ... )
    >>>
    >>> config = ConfigAgentActionsConsumer(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
    ... )
"""

from omnibase_infra.services.observability.agent_actions import (
    ConfigAgentActionsConsumer,
)

__all__ = [
    "ConfigAgentActionsConsumer",
]
