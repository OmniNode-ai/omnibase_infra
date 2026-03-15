# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infrastructure resource providers for dependency materialization.

Each provider creates a specific infrastructure resource type from
environment-driven configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from omnibase_infra.runtime.providers.provider_http_client import ProviderHttpClient
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import (
    ProviderPostgresPool,
)

__all__ = [
    "ProviderHttpClient",
    "ProviderKafkaProducer",
    "ProviderPostgresPool",
]
