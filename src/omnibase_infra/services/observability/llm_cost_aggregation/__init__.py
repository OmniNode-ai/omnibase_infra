# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM cost aggregation consumer and writer.

This module provides infrastructure for consuming LLM call completed events
from Kafka and aggregating costs into the ``llm_cost_aggregates`` table
in PostgreSQL.

Components:
    - ServiceLlmCostAggregator: Async Kafka consumer with per-partition offset tracking
    - ConfigLlmCostAggregation: Configuration for the consumer
    - WriterLlmCostAggregationPostgres: PostgreSQL writer with upsert semantics

Topics consumed:
    - onex.evt.omniintelligence.llm-call-completed.v1

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
    - OMN-2236: llm_call_metrics + llm_cost_aggregates migration 031
    - OMN-2238: Extract and normalize token usage from LLM API responses

Example:
    >>> from omnibase_infra.services.observability.llm_cost_aggregation import (
    ...     ServiceLlmCostAggregator,
    ...     ConfigLlmCostAggregation,
    ... )
    >>>
    >>> config = ConfigLlmCostAggregation(
    ...     kafka_bootstrap_servers="localhost:9092",
    ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
    ... )
    >>> service = ServiceLlmCostAggregator(config)
    >>>
    >>> # Run consumer
    >>> await service.start()
    >>> await service.run()

    # Or run as module:
    # python -m omnibase_infra.services.observability.llm_cost_aggregation.consumer
"""

from omnibase_infra.services.observability.llm_cost_aggregation.config import (
    ConfigLlmCostAggregation,
)
from omnibase_infra.services.observability.llm_cost_aggregation.consumer import (
    ServiceLlmCostAggregator,
)
from omnibase_infra.services.observability.llm_cost_aggregation.writer_postgres import (
    WriterLlmCostAggregationPostgres,
)

__all__ = [
    "ConfigLlmCostAggregation",
    "ServiceLlmCostAggregator",
    "WriterLlmCostAggregationPostgres",
]
