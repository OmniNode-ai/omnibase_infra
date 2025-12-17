# Node Registrations Database Schema

> **Canonical Documentation**: This document redirects to the authoritative schema documentation.
> See [`../schema/NODE_REGISTRATIONS_SCHEMA.md`](../schema/NODE_REGISTRATIONS_SCHEMA.md) for complete details.

## Quick Reference

**Table**: `node_registrations`

**Purpose**: Persistent storage for ONEX node registry state, enabling service discovery and registration.

**Used By**: `NodeRegistryEffect` node

## Documentation Links

| Document | Description |
|----------|-------------|
| [`docs/schema/NODE_REGISTRATIONS_SCHEMA.md`](../schema/NODE_REGISTRATIONS_SCHEMA.md) | Complete schema documentation with columns, indexes, UPSERT patterns, and design decisions |
| [`docs/schema/node_registrations.sql`](../schema/node_registrations.sql) | DDL SQL file for table creation |
| [`docs/architecture/DATABASE_SCHEMA.md`](../architecture/DATABASE_SCHEMA.md) | Architecture overview and context |

## Column Summary

| Column | Type | Description |
|--------|------|-------------|
| `node_id` | `VARCHAR(255)` | Primary key - unique node identifier |
| `node_type` | `VARCHAR(50)` | Node type: effect, compute, reducer, orchestrator |
| `node_version` | `VARCHAR(50)` | Semantic version (default: 1.0.0) |
| `capabilities` | `JSONB` | Node capabilities as JSON |
| `endpoints` | `JSONB` | Endpoint URLs as JSON |
| `metadata` | `JSONB` | Additional metadata as JSON |
| `health_endpoint` | `VARCHAR(512)` | Optional health check URL |
| `last_heartbeat` | `TIMESTAMPTZ` | Last successful health check |
| `registered_at` | `TIMESTAMPTZ` | Initial registration time |
| `updated_at` | `TIMESTAMPTZ` | Last update time |

## Related Source Files

- `src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py`
- `src/omnibase_infra/nodes/node_registry_effect/v1_0_0/models/model_node_registration.py`
