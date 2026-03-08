> **Navigation**: [Home](../index.md) > Architecture

# Architecture Documentation

Understanding how ONEX works - system design, component interactions, and architectural patterns.

> **Note**: For authoritative coding rules and standards, see [CLAUDE.md](../../CLAUDE.md). This documentation provides explanations and context that supplement those rules.

## Overview

Start here to understand the ONEX architecture:

| Document | Description |
|----------|-------------|
| [Architecture Overview](overview.md) | High-level system architecture with diagrams |
| [Current Node Architecture](CURRENT_NODE_ARCHITECTURE.md) | Detailed node architecture documentation |

## Event-Driven Architecture

| Document | Description |
|----------|-------------|
| [Event Bus Integration Guide](EVENT_BUS_INTEGRATION_GUIDE.md) | Kafka event streaming integration |
| [Event Streaming Topics](EVENT_STREAMING_TOPICS.md) | Topic catalog, schemas, and usage patterns |
| [Message Dispatch Engine](MESSAGE_DISPATCH_ENGINE.md) | Event routing internals |
| [DLQ Message Format](DLQ_MESSAGE_FORMAT.md) | Dead Letter Queue message schema |

## Handler Architecture

| Document | Description |
|----------|-------------|
| [Handler Protocol-Driven Architecture](HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md) | Handler system design |
| [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) | Classification rubric for mixin/service → handler refactoring (Epic 3, OMN-4004) |
| [Handler Classification 3.2a — Circuit Breaker + Retry](HANDLER_CLASSIFICATION_3_2A_CIRCUIT_BREAKER_RETRY.md) | Classification of MixinAsyncCircuitBreaker + MixinRetryExecution: KEEP AS MIXIN (OMN-4006) |
| [Handler Classification 3.3 — LLM HTTP Transport](HANDLER_CLASSIFICATION_3_3_LLM_HTTP_TRANSPORT.md) | MixinLlmHttpTransport: CONVERT justified (5/5), deferred — protocol interface mismatch abort (OMN-4008) |
| [Snapshot Publishing](SNAPSHOT_PUBLISHING.md) | Snapshot publication patterns |

## Registration System

| Document | Description |
|----------|-------------|
| [Registration Workflow](REGISTRATION_WORKFLOW.md) | Complete 2-way registration flow: all 4 nodes, FSM states, Kafka topics, intent construction, error paths, and E2E test coverage |
| [2-Way Registration Walkthrough](../guides/registration-example.md) | Complete 4-phase flow with code examples (Phase 1: Introspection, Phase 2: Reducer, Phase 3: Effect Execution, Phase 4: Ack Flow) |

## Configuration & Secrets

| Document | Description |
|----------|-------------|
| [Config Discovery](CONFIG_DISCOVERY.md) | Contract-driven config discovery: Infisical-backed prefetch, transport config map, bootstrap sequence |

## Topic Catalog

| Document | Description |
|----------|-------------|
| [Topic Catalog Architecture](TOPIC_CATALOG_ARCHITECTURE.md) | Topic catalog service architecture: discovery, validation, response channels |

## LLM Infrastructure

| Document | Description |
|----------|-------------|
| [LLM Infrastructure](LLM_INFRASTRUCTURE.md) | Multi-server LLM topology, endpoint selection, cost tracking SPI |

## MCP Integration

| Document | Description |
|----------|-------------|
| [MCP Service Architecture](MCP_SERVICE_ARCHITECTURE.md) | MCP (Model Context Protocol) service layer: tool registration, schema generation, skip_server testing |

## Shared Enums

| Document | Description |
|----------|-------------|
| [Shared Enum Ownership Rule](SHARED_ENUM_OWNERSHIP.md) | Canonical rule: enums defined once in `omnibase_core`, imported downstream, coercion at boundaries |

## Resilience

| Document | Description |
|----------|-------------|
| [Circuit Breaker Thread Safety](CIRCUIT_BREAKER_THREAD_SAFETY.md) | Concurrency safety implementation |

## Related Documentation

- [Pattern Documentation](../patterns/README.md) - Implementation patterns
- [Operations Runbooks](../operations/README.md) - Production operations
- [ADRs](../decisions/README.md) - Why things work this way
