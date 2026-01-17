# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Transport Type Enumeration.

Defines the canonical transport types for infrastructure components.
Used for error context, protocol routing, and transport identification.

Supported transport types:
    - HTTP: REST API transport
    - DATABASE: PostgreSQL and other database connections
    - KAFKA: Kafka message broker
    - CONSUL: Consul service discovery and KV operations
    - VAULT: HashiCorp Vault secret management
    - VALKEY: Valkey (Redis-compatible) cache/messaging
    - GRPC: gRPC protocol
    - RUNTIME: Runtime host internal transport
    - MCP: Model Context Protocol (AI agent tool interface)
    - FILESYSTEM: Local filesystem operations
    - INMEMORY: In-memory transport for testing/local development
    - QDRANT: Qdrant vector database operations
    - GRAPH: Graph database (Memgraph/Neo4j) operations

Each transport type has a corresponding handler implementation:
    - HandlerConsul: Service registration, health checks, KV store operations
    - HandlerVault: Secret retrieval, token management, lease renewal
    - EventBusKafka: Event publishing/subscription, consumer groups
    - EventBusInmemory: In-memory event bus for testing and local development
    - PostgresConnectionManager: Connection pooling, query execution
    - HandlerMCP: MCP server for exposing ONEX nodes as AI agent tools
    - HandlerFileSystem: File read/write, directory operations
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
        MCP: Model Context Protocol transport for AI agent integration
        FILESYSTEM: Local filesystem transport
        INMEMORY: In-memory transport for testing and local development
        QDRANT: Qdrant vector database transport
        GRAPH: Graph database (Memgraph/Neo4j) transport
    """

    HTTP = "http"
    DATABASE = "db"
    KAFKA = "kafka"
    CONSUL = "consul"
    VAULT = "vault"
    VALKEY = "valkey"
    GRPC = "grpc"
    RUNTIME = "runtime"
    MCP = "mcp"
    FILESYSTEM = "filesystem"
    INMEMORY = "inmemory"
    QDRANT = "qdrant"
    GRAPH = "graph"


__all__ = ["EnumInfraTransportType"]
