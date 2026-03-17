# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Context integrity audit observability consumer and writer (OMN-5240).

Infrastructure for consuming context audit events from Kafka and
persisting them to PostgreSQL for enforcement tracking, CLI queries,
and omnidash visualization.

Components:
    - ContextAuditConsumer: Async Kafka consumer with per-partition offset tracking
    - ConfigContextAuditConsumer: Configuration for the consumer
    - WriterContextAuditPostgres: PostgreSQL writer for audit events

Topics consumed (OMN-5234):
    - onex.evt.omniclaude.audit-dispatch-validated.v1
    - onex.evt.omniclaude.audit-scope-violation.v1
    - onex.evt.omniclaude.audit-context-budget-exceeded.v1
    - onex.evt.omniclaude.audit-return-bounded.v1
    - onex.evt.omniclaude.audit-compression-triggered.v1

Example:
    >>> from omnibase_infra.services.observability.context_audit import (
    ...     ContextAuditConsumer,
    ...     ConfigContextAuditConsumer,
    ...     WriterContextAuditPostgres,
    ... )
    >>>
    >>> config = ConfigContextAuditConsumer(
    ...     kafka_bootstrap_servers="localhost:19092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> consumer = ContextAuditConsumer(config)
    >>>
    >>> # Run consumer
    >>> await consumer.start()
    >>> await consumer.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.context_audit.consumer
"""

from omnibase_infra.services.observability.context_audit.config import (
    ConfigContextAuditConsumer,
)
from omnibase_infra.services.observability.context_audit.consumer import (
    ConsumerMetrics,
    ContextAuditConsumer,
    EnumHealthStatus,
    mask_dsn_password,
)
from omnibase_infra.services.observability.context_audit.writer_postgres import (
    WriterContextAuditPostgres,
)

__all__ = [
    "ConfigContextAuditConsumer",
    "ConsumerMetrics",
    "ContextAuditConsumer",
    "EnumHealthStatus",
    "WriterContextAuditPostgres",
    "mask_dsn_password",
]
