# Tool Registration System - Usage Guide

## Overview

The Tool Registration System provides a database-backed registry for dynamic tool discovery and orchestration in the omninode ecosystem. Tools can register their capabilities, endpoints, and metadata, enabling runtime discovery and intelligent routing.

## Architecture

### Components

1. **Database Schema** (`tool_registrations` table)
   - UUID-based primary key
   - Unique tool_id for identification
   - ONEX node_type classification
   - JSONB fields for flexible metadata storage
   - Heartbeat tracking for health monitoring

2. **Pydantic Models** (`ModelToolRegistration`, `ModelToolRegistrationCreate`, `ModelToolRegistrationUpdate`)
   - Type-safe data validation
   - Automatic serialization/deserialization
   - Built-in validation for node types and endpoints

3. **Repository Pattern** (`ToolRegistrationRepository`)
   - CRUD operations with prepared statement caching
   - Batch operations for efficiency
   - Advanced querying (by capability, health status, etc.)

## Database Schema

```sql
CREATE TABLE tool_registrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_id VARCHAR(255) UNIQUE NOT NULL,
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN ('effect', 'compute', 'reducer', 'orchestrator')),
    capabilities JSONB DEFAULT '{}',
    endpoints JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    health_endpoint VARCHAR(500),
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_tool_registrations_tool_id ON tool_registrations(tool_id);
CREATE INDEX idx_tool_registrations_node_type ON tool_registrations(node_type);
CREATE INDEX idx_tool_registrations_last_heartbeat ON tool_registrations(last_heartbeat DESC);

-- GIN indexes for JSONB querying
CREATE INDEX idx_tool_registrations_capabilities_gin ON tool_registrations USING GIN(capabilities);
CREATE INDEX idx_tool_registrations_endpoints_gin ON tool_registrations USING GIN(endpoints);
CREATE INDEX idx_tool_registrations_metadata_gin ON tool_registrations USING GIN(metadata);
```

## Installation

### 1. Run Migration

```bash
# Apply the migration
poetry run alembic upgrade head
```

### 2. Verify Installation

```bash
# Check tables
psql -h localhost -U postgres -d omninode_bridge -c "\d tool_registrations"

# Check indexes
psql -h localhost -U postgres -d omninode_bridge -c "\di tool_registrations*"
```

## Usage Examples

### Basic CRUD Operations

```python
from omninode_bridge.services import PostgresClient, ToolRegistrationRepository
from omninode_bridge.models import (
    ModelToolRegistrationCreate,
    ModelToolRegistrationUpdate,
)

# Initialize
postgres_client = PostgresClient()
await postgres_client.connect()
repository = ToolRegistrationRepository(postgres_client)

# Create Registration
registration = await repository.create_registration(
    ModelToolRegistrationCreate(
        tool_id="metadata-stamping-v1",
        node_type="effect",
        capabilities={
            "operations": ["stamp", "validate", "hash"],
            "max_file_size": 10485760,
            "supported_formats": ["json", "yaml", "text"]
        },
        endpoints={
            "stamp": "http://metadata-service:8053/api/v1/stamp",
            "validate": "http://metadata-service:8053/api/v1/validate",
            "hash": "http://metadata-service:8053/api/v1/hash"
        },
        metadata={
            "version": "1.0.0",
            "author": "OmniNode Team",
            "description": "Cryptographic metadata stamping service"
        },
        health_endpoint="http://metadata-service:8053/health"
    )
)

print(f"Registered tool: {registration.tool_id}")
print(f"Registration ID: {registration.id}")

# Get Registration
tool = await repository.get_registration("metadata-stamping-v1")
if tool:
    print(f"Found tool: {tool.tool_id}")
    print(f"Node type: {tool.node_type}")
    print(f"Capabilities: {tool.capabilities}")

# Update Registration
updated = await repository.update_registration(
    "metadata-stamping-v1",
    ModelToolRegistrationUpdate(
        capabilities={
            "operations": ["stamp", "validate", "hash", "batch_stamp"],
            "max_file_size": 20971520,  # Doubled
            "supported_formats": ["json", "yaml", "text", "xml"]
        },
        metadata={
            "version": "1.1.0",
            "author": "OmniNode Team",
            "description": "Enhanced cryptographic metadata stamping service"
        }
    )
)

# Delete Registration
success = await repository.delete_registration("metadata-stamping-v1")
print(f"Deletion successful: {success}")
```

### Health Monitoring

```python
# Update Heartbeat
await repository.update_heartbeat("metadata-stamping-v1")

# Get Healthy Tools (heartbeat within last 5 minutes)
healthy_tools = await repository.get_healthy_tools(max_age_seconds=300)
print(f"Found {len(healthy_tools)} healthy tools")

for tool in healthy_tools:
    age_seconds = (datetime.now() - tool.last_heartbeat).total_seconds()
    print(f"  - {tool.tool_id}: last seen {age_seconds:.0f}s ago")
```

### Advanced Querying

```python
# List All Tools
all_tools = await repository.list_all_registrations()
print(f"Total registered tools: {len(all_tools)}")

# Filter by Node Type
effect_nodes = await repository.list_all_registrations(node_type="effect")
compute_nodes = await repository.list_all_registrations(node_type="compute")

print(f"Effect nodes: {len(effect_nodes)}")
print(f"Compute nodes: {len(compute_nodes)}")

# Find by Capability
# Find all tools that support "stamp" operation
stamping_tools = await repository.find_by_capability(
    "operations",
    ["stamp", "validate"]  # Must contain these operations
)

print(f"Tools with stamping capability: {len(stamping_tools)}")
for tool in stamping_tools:
    print(f"  - {tool.tool_id}: {tool.capabilities.get('operations', [])}")
```

### Integration with Tool Discovery

```python
class ToolDiscoveryService:
    """Service for discovering and routing to registered tools."""

    def __init__(self, repository: ToolRegistrationRepository):
        self.repository = repository

    async def discover_tool(self, capability: str) -> Optional[ModelToolRegistration]:
        """Discover a tool by capability."""
        tools = await self.repository.find_by_capability("operations", [capability])

        # Filter to healthy tools only
        healthy_tools = [
            tool for tool in tools
            if tool.last_heartbeat and
            (datetime.now() - tool.last_heartbeat).total_seconds() < 300
        ]

        if not healthy_tools:
            return None

        # Return first healthy tool
        return healthy_tools[0]

    async def route_request(self, operation: str, data: dict) -> dict:
        """Route a request to the appropriate tool."""
        tool = await self.discover_tool(operation)

        if not tool:
            raise RuntimeError(f"No tool found for operation: {operation}")

        endpoint = tool.endpoints.get(operation)
        if not endpoint:
            raise RuntimeError(
                f"Tool {tool.tool_id} does not have endpoint for: {operation}"
            )

        # Make request to tool endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=data)
            return response.json()

# Usage
discovery_service = ToolDiscoveryService(repository)

# Discover and route
result = await discovery_service.route_request(
    "stamp",
    {"content": "Hello World", "namespace": "test"}
)
```

### Batch Registration

```python
# Register multiple tools at once
tools_to_register = [
    ModelToolRegistrationCreate(
        tool_id="onextree-intelligence-v1",
        node_type="compute",
        capabilities={
            "operations": ["analyze", "predict"],
            "ml_models": ["decision_tree", "random_forest"]
        },
        endpoints={
            "analyze": "http://onextree:8080/api/v1/analyze",
            "predict": "http://onextree:8080/api/v1/predict"
        },
        health_endpoint="http://onextree:8080/health"
    ),
    ModelToolRegistrationCreate(
        tool_id="workflow-orchestrator-v1",
        node_type="orchestrator",
        capabilities={
            "operations": ["execute", "schedule"],
            "max_concurrent_workflows": 100
        },
        endpoints={
            "execute": "http://orchestrator:8081/api/v1/execute",
            "schedule": "http://orchestrator:8081/api/v1/schedule"
        },
        health_endpoint="http://orchestrator:8081/health"
    )
]

for tool_config in tools_to_register:
    try:
        registration = await repository.create_registration(tool_config)
        print(f"✓ Registered: {registration.tool_id}")
    except Exception as e:
        print(f"✗ Failed to register {tool_config.tool_id}: {e}")
```

## API Reference

### ToolRegistrationRepository

#### `create_registration(registration: ModelToolRegistrationCreate) -> ModelToolRegistration`
Create a new tool registration.

#### `get_registration(tool_id: str) -> Optional[ModelToolRegistration]`
Get a tool registration by tool_id.

#### `update_registration(tool_id: str, update: ModelToolRegistrationUpdate) -> Optional[ModelToolRegistration]`
Update a tool registration.

#### `update_heartbeat(tool_id: str, heartbeat_time: Optional[datetime] = None) -> bool`
Update the last heartbeat timestamp for a tool.

#### `list_all_registrations(node_type: Optional[str] = None) -> List[ModelToolRegistration]`
List all tool registrations, optionally filtered by node type.

#### `delete_registration(tool_id: str) -> bool`
Delete a tool registration.

#### `find_by_capability(capability_key: str, capability_value: Any) -> List[ModelToolRegistration]`
Find tools by a specific capability using JSONB querying.

#### `get_healthy_tools(max_age_seconds: int = 300) -> List[ModelToolRegistration]`
Get tools with recent heartbeats (default: last 5 minutes).

## Performance Considerations

1. **Prepared Statement Caching**: The repository leverages PostgresClient's prepared statement cache for frequently executed queries.

2. **JSONB Indexing**: GIN indexes on `capabilities`, `endpoints`, and `metadata` enable fast JSONB queries.

3. **Connection Pooling**: Uses PostgresClient's optimized connection pool (5-50 connections based on environment).

4. **Batch Operations**: For bulk registrations, consider using transactions to ensure atomicity.

## Security Notes

1. **Validation**: All inputs are validated through Pydantic models before database insertion.

2. **SQL Injection Prevention**: Prepared statements are used for all queries.

3. **Endpoint Validation**: URLs must start with `http://` or `https://`.

4. **Node Type Constraints**: Database-level CHECK constraint ensures only valid ONEX node types.

## Troubleshooting

### Migration Issues

```bash
# Check current migration version
poetry run alembic current

# Show migration history
poetry run alembic history

# Rollback if needed
poetry run alembic downgrade -1
```

### Query Performance

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM tool_registrations WHERE node_type = 'effect';

-- Check index usage
SELECT * FROM pg_stat_user_indexes WHERE relname = 'tool_registrations';

-- Vacuum if needed
VACUUM ANALYZE tool_registrations;
```

### Connection Issues

```python
# Check PostgreSQL connection
health = await postgres_client.health_check()
print(f"Database status: {health['status']}")

# Check pool metrics
metrics = await postgres_client.get_pool_metrics()
print(f"Pool utilization: {metrics.utilization_percent}%")
```

## Next Steps

1. **Add API Endpoints**: Create FastAPI routes for tool registration
2. **Implement Auto-Discovery**: Add service discovery via Consul/etcd
3. **Add Monitoring**: Integrate with Prometheus for metrics
4. **Implement TTL**: Add automatic cleanup for stale registrations
