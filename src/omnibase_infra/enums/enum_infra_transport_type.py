# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Transport Type Enumeration.

Defines the canonical transport types for infrastructure components.
Used for error context, protocol routing, and transport identification.
"""

from enum import Enum


class EnumInfraTransportType(str, Enum):
    """Infrastructure transport types for ONEX infrastructure components.

    These represent the transport/protocol layer types used in
    omnibase_infra for external integration.

    Attributes:
        HTTP: HTTP/REST API transport
        DATABASE: Database connection transport (PostgreSQL, etc.)
        KAFKA: Kafka message broker transport
        CONSUL: Consul discovery transport
        VAULT: HashiCorp Vault secret transport
        VALKEY: Valkey (Redis-compatible) cache/message transport
        GRPC: gRPC protocol transport
        RUNTIME: Runtime host process internal transport
    """

    HTTP = "http"
    DATABASE = "db"
    KAFKA = "kafka"
    CONSUL = "consul"
    VAULT = "vault"
    VALKEY = "valkey"
    GRPC = "grpc"
    RUNTIME = "runtime"


__all__ = ["EnumInfraTransportType"]
