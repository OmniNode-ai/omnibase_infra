# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure resource type enumeration for dependency materialization.

Defines the resource types that the DependencyMaterializer can create
from contract.dependencies declarations. Each type maps to a provider
factory that creates the corresponding infrastructure resource.

Part of OMN-1976: Contract dependency materialization.
"""

from enum import Enum


class EnumInfraResourceType(str, Enum):
    """Infrastructure resource types for contract-driven dependency materialization.

    These represent the infrastructure resource types that can be declared
    in contract.dependencies and auto-materialized by the runtime.

    Attributes:
        POSTGRES_POOL: PostgreSQL connection pool (asyncpg.Pool)
        KAFKA_PRODUCER: Kafka message producer (AIOKafkaProducer)
        HTTP_CLIENT: HTTP client (httpx.AsyncClient)
    """

    POSTGRES_POOL = "postgres_pool"
    KAFKA_PRODUCER = "kafka_producer"
    HTTP_CLIENT = "http_client"


# Set of all resource type values for quick membership testing
INFRA_RESOURCE_TYPES: frozenset[str] = frozenset(
    member.value for member in EnumInfraResourceType
)


__all__ = ["EnumInfraResourceType", "INFRA_RESOURCE_TYPES"]
