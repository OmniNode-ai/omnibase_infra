# Database Schema Architecture

## Overview

This document provides an architectural overview of the PostgreSQL database schemas used by ONEX infrastructure components. For detailed table-level documentation, see the linked schema files.

## Tables

### node_registrations

**Purpose**: Persistent storage for ONEX node registry state, enabling service discovery and registration across the infrastructure.

**Used By**: `NodeRegistryEffect` (`src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py`)

**Detailed Documentation**:
- Schema Documentation: [`docs/database/NODE_REGISTRATIONS_SCHEMA.md`](../database/NODE_REGISTRATIONS_SCHEMA.md)
- SQL DDL: [`docs/schema/node_registrations.sql`](../schema/node_registrations.sql)

#### DDL

```sql
CREATE TABLE IF NOT EXISTS node_registrations (
    -- Primary identifier (unique node ID)
    node_id VARCHAR(255) PRIMARY KEY,

    -- Node classification
    node_type VARCHAR(50) NOT NULL,
    node_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',

    -- Node capabilities and configuration (stored as JSONB for flexibility)
    capabilities JSONB NOT NULL DEFAULT '{}',
    endpoints JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Health monitoring
    health_endpoint VARCHAR(512),
    last_heartbeat TIMESTAMP WITH TIME ZONE,

    -- Audit timestamps
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for filtering by node type (common query pattern)
CREATE INDEX IF NOT EXISTS idx_node_registrations_node_type
ON node_registrations(node_type);

-- Index for filtering by node version
CREATE INDEX IF NOT EXISTS idx_node_registrations_node_version
ON node_registrations(node_version);

-- Index for finding recently updated nodes
CREATE INDEX IF NOT EXISTS idx_node_registrations_updated_at
ON node_registrations(updated_at DESC);

-- Partial index for nodes with health endpoints
CREATE INDEX IF NOT EXISTS idx_node_registrations_health_endpoint
ON node_registrations(health_endpoint)
WHERE health_endpoint IS NOT NULL;

-- GIN index for JSONB capability queries
CREATE INDEX IF NOT EXISTS idx_node_registrations_capabilities
ON node_registrations USING GIN (capabilities);
```

#### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `node_id` | `VARCHAR(255)` | Primary key. Unique identifier for the registered node. |
| `node_type` | `VARCHAR(50)` | Node type classification: `effect`, `compute`, `reducer`, `orchestrator`. Required. |
| `node_version` | `VARCHAR(50)` | Semantic version of the node (e.g., `1.0.0`). Defaults to `1.0.0`. |
| `capabilities` | `JSONB` | JSON object describing node capabilities (e.g., supported features, protocols). |
| `endpoints` | `JSONB` | JSON object mapping endpoint names to URLs (e.g., `{"grpc": "localhost:50051"}`). |
| `metadata` | `JSONB` | Additional node metadata as JSON (e.g., tags, labels, environment info). |
| `health_endpoint` | `VARCHAR(512)` | Optional URL for health check endpoint (e.g., `http://localhost:8080/health`). |
| `last_heartbeat` | `TIMESTAMPTZ` | Timestamp of last successful health check. NULL if never checked. |
| `registered_at` | `TIMESTAMPTZ` | Initial registration timestamp. Preserved on re-registration. |
| `updated_at` | `TIMESTAMPTZ` | Last update timestamp. Auto-updated on UPSERT operations. |

#### ONEX Context

The `node_registrations` table supports the Registry Effect Node's core responsibilities:

1. **Node Registration**: UPSERT pattern enables idempotent registration. Nodes can safely re-register without creating duplicates.

2. **Service Discovery**: Query nodes by type, version, or capabilities. The GIN index on `capabilities` enables efficient JSONB containment queries.

3. **Health Monitoring**: Track health endpoints and heartbeat timestamps for observability and alerting.

4. **Audit Trail**: Timestamps enable tracking of node lifecycle events.

#### Query Patterns

The Registry Effect Node uses these query patterns:

```sql
-- Register (UPSERT)
INSERT INTO node_registrations (...) VALUES (...)
ON CONFLICT (node_id) DO UPDATE SET ...

-- Deregister
DELETE FROM node_registrations WHERE node_id = $1

-- Discover by type
SELECT * FROM node_registrations WHERE node_type = $1

-- Discover by type and version
SELECT * FROM node_registrations WHERE node_type = $1 AND node_version = $2
```

#### Security Considerations

- **SQL Injection Prevention**: The `ALLOWED_FILTER_KEYS` whitelist in `node.py` restricts filter columns to `node_type`, `node_id`, and `node_version`.
- **JSONB Validation**: Application-layer validation via Pydantic models before database insertion.
- **No Foreign Keys**: Self-contained identifiers avoid cross-table dependency issues.

## Schema Management

### Initial Setup

Apply the schema using the SQL DDL file:

```bash
psql -d $DATABASE_NAME -f docs/schema/node_registrations.sql
```

### Migration Strategy

Schema changes should:

1. Use `ALTER TABLE` for additive changes (new columns, indexes)
2. Maintain backwards compatibility with existing data
3. Update both SQL files and documentation
4. Consider migration tools (Alembic, Flyway) for complex changes

## Related Documentation

- [Circuit Breaker Thread Safety](./CIRCUIT_BREAKER_THREAD_SAFETY.md) - Fault tolerance patterns
- [Current Node Architecture](./CURRENT_NODE_ARCHITECTURE.md) - Node system design
- [Runtime Host Implementation Plan](./RUNTIME_HOST_IMPLEMENTATION_PLAN.md) - Infrastructure deployment
