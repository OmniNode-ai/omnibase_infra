# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Handler Type Enumeration.

Defines the canonical handler types for infrastructure components.
Used for error context, protocol routing, and service identification.
"""

from enum import Enum


class EnumInfraHandlerType(str, Enum):
    """Infrastructure handler types for ONEX infrastructure components.

    These represent the transport/protocol layer handlers used in
    omnibase_infra for external service integration.

    Attributes:
        HTTP: HTTP/REST API handlers
        DATABASE: Database connection handlers (PostgreSQL, etc.)
        KAFKA: Kafka message broker handlers
        CONSUL: Consul service discovery handlers
        VAULT: HashiCorp Vault secret management handlers
        REDIS: Redis cache/message handlers
        GRPC: gRPC protocol handlers
    """

    HTTP = "http"
    DATABASE = "db"
    KAFKA = "kafka"
    CONSUL = "consul"
    VAULT = "vault"
    REDIS = "redis"
    GRPC = "grpc"


__all__ = ["EnumInfraHandlerType"]
