# Node Registrations Database Schema

## Overview

The `node_registrations` table provides persistent storage for ONEX node registry state. It is used by the `NodeRegistryEffect` node to maintain a durable record of registered nodes alongside the Consul service discovery integration.

**Related Source Files:**
- `/workspace/omnibase_infra2/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/node.py`
- `/workspace/omnibase_infra2/src/omnibase_infra/nodes/node_registry_effect/v1_0_0/models/model_node_registration.py`
- `/workspace/omnibase_infra2/docs/schema/node_registrations.sql`

## Table: node_registrations

### Columns

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `node_id` | `VARCHAR(255)` | `PRIMARY KEY` | Unique identifier for the registered node |
| `node_type` | `VARCHAR(50)` | `NOT NULL` | Node type classification: `effect`, `compute`, `reducer`, `orchestrator` |
| `node_version` | `VARCHAR(50)` | `NOT NULL DEFAULT '1.0.0'` | Semantic version of the node (e.g., `1.0.0`) |
| `capabilities` | `JSONB` | `NOT NULL DEFAULT '{}'` | JSON object describing node capabilities |
| `endpoints` | `JSONB` | `NOT NULL DEFAULT '{}'` | JSON object mapping endpoint names to URLs |
| `metadata` | `JSONB` | `NOT NULL DEFAULT '{}'` | Additional node metadata as JSON |
| `health_endpoint` | `VARCHAR(512)` | `NULL` | URL for health check endpoint |
| `last_heartbeat` | `TIMESTAMP WITH TIME ZONE` | `NULL` | Timestamp of last successful health check |
| `registered_at` | `TIMESTAMP WITH TIME ZONE` | `NOT NULL DEFAULT NOW()` | Initial registration timestamp |
| `updated_at` | `TIMESTAMP WITH TIME ZONE` | `NOT NULL DEFAULT NOW()` | Last update timestamp (auto-updated on UPSERT) |

### Primary Key

- **Column:** `node_id`
- **Type:** `VARCHAR(255)`

The `node_id` serves as the natural primary key, enabling the UPSERT pattern used for idempotent re-registration.

### DDL

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
```

### Indexes

| Index Name | Column(s) | Type | Description |
|------------|-----------|------|-------------|
| `idx_node_registrations_node_type` | `node_type` | B-tree | Supports filtering by node type |
| `idx_node_registrations_node_version` | `node_version` | B-tree | Supports filtering by node version |
| `idx_node_registrations_updated_at` | `updated_at DESC` | B-tree | Supports finding recently updated nodes |
| `idx_node_registrations_health_endpoint` | `health_endpoint` | Partial B-tree | Partial index for nodes with health endpoints (`WHERE health_endpoint IS NOT NULL`) |
| `idx_node_registrations_capabilities` | `capabilities` | GIN | Supports JSONB containment queries on capabilities |

#### Index DDL

```sql
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

### Foreign Keys

This table has no foreign key constraints. The `node_id` is a self-contained identifier that does not reference other tables.

## Usage Examples

### Register Operation (UPSERT)

The `NodeRegistryEffect._register_postgres()` method uses an UPSERT pattern for idempotent registration:

```sql
INSERT INTO node_registrations (
    node_id, node_type, node_version, capabilities,
    endpoints, metadata, health_endpoint, registered_at, updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
ON CONFLICT (node_id) DO UPDATE SET
    node_type = EXCLUDED.node_type,
    node_version = EXCLUDED.node_version,
    capabilities = EXCLUDED.capabilities,
    endpoints = EXCLUDED.endpoints,
    metadata = EXCLUDED.metadata,
    health_endpoint = EXCLUDED.health_endpoint,
    updated_at = NOW()
```

**Parameters:**
1. `$1` - `node_id` (string)
2. `$2` - `node_type` (string)
3. `$3` - `node_version` (string)
4. `$4` - `capabilities` (JSON string)
5. `$5` - `endpoints` (JSON string)
6. `$6` - `metadata` (JSON string)
7. `$7` - `health_endpoint` (string or NULL)

### Deregister Operation (DELETE)

The `NodeRegistryEffect._deregister_postgres()` method removes a node:

```sql
DELETE FROM node_registrations WHERE node_id = $1
```

**Parameters:**
1. `$1` - `node_id` (string)

### Discover Operation (SELECT with Filters)

The `NodeRegistryEffect._discover_nodes()` method queries nodes with optional filters:

```sql
-- Basic query (no filters)
SELECT * FROM node_registrations

-- With filters (dynamically built based on allowed keys)
SELECT * FROM node_registrations WHERE node_type = $1
SELECT * FROM node_registrations WHERE node_type = $1 AND node_version = $2
SELECT * FROM node_registrations WHERE node_id = $1
```

**Allowed Filter Keys** (enforced via `ALLOWED_FILTER_KEYS` constant for SQL injection prevention):
- `node_type`
- `node_id`
- `node_version`

### Additional Query Examples

```sql
-- Find all effect nodes
SELECT * FROM node_registrations WHERE node_type = 'effect';

-- Find nodes with specific capability
SELECT * FROM node_registrations WHERE capabilities @> '{"feature": "logging"}';

-- Find recently updated nodes (last hour)
SELECT * FROM node_registrations WHERE updated_at > NOW() - INTERVAL '1 hour';

-- Find nodes without recent heartbeat (potential health issues)
SELECT * FROM node_registrations
WHERE last_heartbeat < NOW() - INTERVAL '5 minutes'
  AND health_endpoint IS NOT NULL;
```

## Mapping to Pydantic Model

The `ModelNodeRegistration` class maps directly to this table:

| Database Column | Pydantic Field | Python Type |
|-----------------|----------------|-------------|
| `node_id` | `node_id` | `str` |
| `node_type` | `node_type` | `str` |
| `node_version` | `node_version` | `str` |
| `capabilities` | `capabilities` | `dict[str, object]` |
| `endpoints` | `endpoints` | `dict[str, str]` |
| `metadata` | `metadata` | `dict[str, object]` |
| `health_endpoint` | `health_endpoint` | `str \| None` |
| `last_heartbeat` | `last_heartbeat` | `datetime \| None` |
| `registered_at` | `registered_at` | `datetime` |
| `updated_at` | `updated_at` | `datetime` |

## Design Decisions

### JSONB for Flexible Data

The `capabilities`, `endpoints`, and `metadata` columns use PostgreSQL `JSONB` type to:
- Allow flexible, schema-less data storage for node-specific attributes
- Enable efficient JSON containment queries via GIN indexes
- Support future extensibility without schema migrations

### UPSERT Pattern

The registration operation uses `INSERT ... ON CONFLICT DO UPDATE` to:
- Ensure idempotent re-registration (calling register multiple times is safe)
- Automatically update `updated_at` timestamp on re-registration
- Preserve `registered_at` timestamp from initial registration

### Partial Index for Health Endpoints

The partial index on `health_endpoint` only indexes rows where the value is not NULL, reducing index size while still supporting efficient health monitoring queries.

### Allowed Filter Keys Whitelist

The `ALLOWED_FILTER_KEYS` constant in `node.py` restricts which columns can be used as filters in discovery queries, providing defense-in-depth against SQL injection:

```python
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset({"node_type", "node_id", "node_version"})
```
