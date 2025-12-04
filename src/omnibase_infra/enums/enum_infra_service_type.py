# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Service Type Enumeration.

Defines the canonical service types for infrastructure components.
Used for error context, protocol routing, and service identification.
"""

from enum import Enum


class EnumInfraServiceType(str, Enum):
    """Infrastructure service types for ONEX infrastructure components.

    These represent the transport/protocol layer services used in
    omnibase_infra for external service integration.

    Attributes:
        HTTP: HTTP/REST API services
        DATABASE: Database connection services (PostgreSQL, etc.)
        KAFKA: Kafka message broker services
        CONSUL: Consul service discovery services
        VAULT: HashiCorp Vault secret management services
        REDIS: Redis cache/message services
        GRPC: gRPC protocol services
    """

    HTTP = "http"
    DATABASE = "db"
    KAFKA = "kafka"
    CONSUL = "consul"
    VAULT = "vault"
    REDIS = "redis"
    GRPC = "grpc"


__all__ = ["EnumInfraServiceType"]
