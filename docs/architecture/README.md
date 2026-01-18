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
| [Snapshot Publishing](SNAPSHOT_PUBLISHING.md) | Snapshot publication patterns |

## Registration System

| Document | Description |
|----------|-------------|
| [Registration Orchestrator Architecture](REGISTRATION_ORCHESTRATOR_ARCHITECTURE.md) | Node registration orchestrator |
| [Node Registration Orchestrator Protocols](NODE_REGISTRATION_ORCHESTRATOR_PROTOCOLS.md) | Registration protocol definitions |

## Implementation Plans

| Document | Description |
|----------|-------------|
| [Runtime Host Implementation Plan](RUNTIME_HOST_IMPLEMENTATION_PLAN.md) | Runtime host design plan |
| [Declarative Effect Nodes Plan](DECLARATIVE_EFFECT_NODES_PLAN.md) | Effect node implementation plan |

## Resilience

| Document | Description |
|----------|-------------|
| [Circuit Breaker Thread Safety](CIRCUIT_BREAKER_THREAD_SAFETY.md) | Concurrency safety implementation |

## Related Documentation

- [Pattern Documentation](../patterns/README.md) - Implementation patterns
- [Operations Runbooks](../operations/README.md) - Production operations
- [ADRs](../decisions/README.md) - Why things work this way
