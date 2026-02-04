# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Injection Effectiveness Observability Service.

This module provides Kafka consumers and PostgreSQL writers for injection
effectiveness metrics collected from omniclaude hooks.

Topics consumed:
    - onex.evt.omniclaude.context-utilization.v1
    - onex.evt.omniclaude.agent-match.v1
    - onex.evt.omniclaude.latency-breakdown.v1

Related Tickets:
    - OMN-1890: Store injection metrics with corrected schema
    - OMN-1889: Emit injection metrics + utilization signal (producer)

Example:
    >>> from omnibase_infra.services.observability.injection_effectiveness import (
    ...     InjectionEffectivenessConsumer,
    ...     ConfigInjectionEffectivenessConsumer,
    ... )
    >>>
    >>> config = ConfigInjectionEffectivenessConsumer(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omninode_bridge",
    ... )
    >>> consumer = InjectionEffectivenessConsumer(config)
    >>>
    >>> await consumer.start()
    >>> await consumer.run()
"""

from omnibase_infra.services.observability.injection_effectiveness.config import (
    ConfigInjectionEffectivenessConsumer,
)
from omnibase_infra.services.observability.injection_effectiveness.consumer import (
    TOPIC_TO_MODEL,
    TOPIC_TO_WRITER_METHOD,
    ConsumerMetrics,
    EnumHealthStatus,
    InjectionEffectivenessConsumer,
    mask_dsn_password,
)
from omnibase_infra.services.observability.injection_effectiveness.models import (
    ModelAgentMatchEvent,
    ModelContextUtilizationEvent,
    ModelLatencyBreakdownEvent,
    ModelPatternUtilization,
)
from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
    WriterInjectionEffectivenessPostgres,
)

__all__ = [
    "ConfigInjectionEffectivenessConsumer",
    "ConsumerMetrics",
    "EnumHealthStatus",
    "InjectionEffectivenessConsumer",
    "ModelAgentMatchEvent",
    "ModelContextUtilizationEvent",
    "ModelLatencyBreakdownEvent",
    "ModelPatternUtilization",
    "TOPIC_TO_MODEL",
    "TOPIC_TO_WRITER_METHOD",
    "WriterInjectionEffectivenessPostgres",
    "mask_dsn_password",
]
