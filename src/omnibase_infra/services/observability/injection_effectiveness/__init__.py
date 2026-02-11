# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Injection Effectiveness Observability Service.

This module provides Kafka consumers, PostgreSQL writers, readers, and
ledger sinks for injection effectiveness metrics collected from omniclaude hooks.

Topics consumed:
    - onex.evt.omniclaude.context-utilization.v1
    - onex.evt.omniclaude.agent-match.v1
    - onex.evt.omniclaude.latency-breakdown.v1

Related Tickets:
    - OMN-1890: Store injection metrics with corrected schema
    - OMN-1889: Emit injection metrics + utilization signal (producer)
    - OMN-2078: Golden path: injection metrics + ledger storage

Example:
    >>> from omnibase_infra.services.observability.injection_effectiveness import (
    ...     InjectionEffectivenessConsumer,
    ...     ConfigInjectionEffectivenessConsumer,
    ...     ReaderInjectionEffectivenessPostgres,
    ...     LedgerSinkInjectionEffectivenessPostgres,
    ... )
    >>>
    >>> config = ConfigInjectionEffectivenessConsumer(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
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
from omnibase_infra.services.observability.injection_effectiveness.ledger_sink_postgres import (
    LedgerEntryDict,
    LedgerSinkInjectionEffectivenessPostgres,
)
from omnibase_infra.services.observability.injection_effectiveness.models import (
    ModelAgentMatchEvent,
    ModelContextUtilizationEvent,
    ModelInjectionEffectivenessQuery,
    ModelInjectionEffectivenessQueryResult,
    ModelInjectionEffectivenessRow,
    ModelLatencyBreakdownEvent,
    ModelLatencyBreakdownRow,
    ModelPatternHitRateRow,
    ModelPatternUtilization,
)
from omnibase_infra.services.observability.injection_effectiveness.protocol_reader import (
    ProtocolInjectionEffectivenessReader,
)
from omnibase_infra.services.observability.injection_effectiveness.reader_postgres import (
    ReaderInjectionEffectivenessPostgres,
)
from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
    WriterInjectionEffectivenessPostgres,
)

__all__ = [
    "ConfigInjectionEffectivenessConsumer",
    "ConsumerMetrics",
    "EnumHealthStatus",
    "InjectionEffectivenessConsumer",
    "LedgerEntryDict",
    "LedgerSinkInjectionEffectivenessPostgres",
    "ModelAgentMatchEvent",
    "ModelContextUtilizationEvent",
    "ModelInjectionEffectivenessQuery",
    "ModelInjectionEffectivenessQueryResult",
    "ModelInjectionEffectivenessRow",
    "ModelLatencyBreakdownEvent",
    "ModelLatencyBreakdownRow",
    "ModelPatternHitRateRow",
    "ModelPatternUtilization",
    "ProtocolInjectionEffectivenessReader",
    "ReaderInjectionEffectivenessPostgres",
    "TOPIC_TO_MODEL",
    "TOPIC_TO_WRITER_METHOD",
    "WriterInjectionEffectivenessPostgres",
    "mask_dsn_password",
]
