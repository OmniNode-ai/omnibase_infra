# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer Health read-model projection (OMN-6757).

Consumes ``onex.evt.omnibase-infra.consumer-health.v1`` events and persists
them to PostgreSQL for omnidash ``/consumer-health`` dashboard.
"""

from omnibase_infra.services.observability.consumer_health.config import (
    ConfigConsumerHealthProjection,
)
from omnibase_infra.services.observability.consumer_health.consumer import (
    ConsumerHealthProjectionConsumer,
)
from omnibase_infra.services.observability.consumer_health.writer_postgres import (
    WriterConsumerHealthPostgres,
)

__all__ = [
    "ConfigConsumerHealthProjection",
    "ConsumerHealthProjectionConsumer",
    "WriterConsumerHealthPostgres",
]
