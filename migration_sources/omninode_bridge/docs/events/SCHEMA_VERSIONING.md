# Event Schema Versioning Strategy

**Status:** ✅ Implemented (October 2025)
**Component:** Event Infrastructure
**Related:** POLY-12 Observability Enhancements

## Overview

This document describes the event schema versioning strategy for omninode_bridge, providing a systematic approach to evolving event schemas while maintaining backward compatibility and enabling smooth migrations between versions.

## Versioning Philosophy

### Core Principles

1. **Semantic Versioning**: Version numbers follow a semantic approach (v1, v2, v3)
2. **Backward Compatibility**: New versions should accept data from older versions when possible
3. **Forward Compatibility**: Old consumers should be able to process newer versions (when strategy allows)
4. **Explicit Migrations**: Migration paths are defined explicitly for each version transition
5. **Deprecation Lifecycle**: Clear deprecation and removal timelines for old versions

### Version Lifecycle

```
Active → Deprecated → Removed
  ↓         ↓            ↓
6 months  6 months   End of Life
```

- **Active**: Current recommended version, receives all new features
- **Deprecated**: Still supported but discouraged, no new features
- **Removed**: No longer supported, consumers must migrate

## Schema Evolution Strategies

### 1. Backward Compatible (RECOMMENDED)

New versions add optional fields while preserving all existing fields.

**Use When:**
- Adding new optional features
- Extending functionality
- Collecting additional metadata

**Example:**
```python
# V1 Schema
class ModelEventCodegenRequestedV1(BaseModel):
    correlation_id: UUID
    prompt: str
    output_directory: str

# V2 Schema (Backward Compatible)
class ModelEventCodegenRequestedV2(BaseModel):
    correlation_id: UUID
    prompt: str
    output_directory: str
    node_type: Optional[str] = None  # New optional field
    enable_intelligence: bool = True  # New optional field with default
```

**Migration:** Automatic - V1 data validates with V2 schema using defaults

### 2. Forward Compatible

Old versions can process new data by ignoring unknown fields.

**Use When:**
- Old consumers should continue working
- New fields are non-critical
- Gradual rollout required

**Example:**
```python
# V1 consumers configured with `extra="ignore"`
class ModelEventCodegenRequestedV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    correlation_id: UUID
    prompt: str

# V2 adds fields, but V1 consumers ignore them
```

### 3. Breaking Change

Incompatible changes requiring all consumers to update.

**Use When:**
- Fundamental redesign required
- Correcting critical design flaws
- Security or compliance requirements

**Example:**
```python
# V1 Schema
class ModelEventCodegenRequestedV1(BaseModel):
    node_config: str  # JSON string

# V2 Schema (Breaking Change)
class ModelEventCodegenRequestedV2(BaseModel):
    node_config: NodeConfig  # Structured object
```

**Requirements:**
- Must provide migration function
- Minimum 6 months deprecation period
- Clear communication to all consumers

## Using the Event Version Registry

### Registering Event Schemas

```python
from omninode_bridge.events.versioning import (
    EventSchemaVersion,
    EventVersionRegistry,
    SchemaEvolutionStrategy,
    event_registry,
)

# Register V1 schema
event_registry.register(
    event_type="CODEGEN_REQUESTED",
    version=EventSchemaVersion.V1,
    schema_class=ModelEventCodegenRequestedV1,
    evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
    deprecated=False,
)

# Register V2 schema
event_registry.register(
    event_type="CODEGEN_REQUESTED",
    version=EventSchemaVersion.V2,
    schema_class=ModelEventCodegenRequestedV2,
    evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
    deprecated=False,
)

# Deprecate V1 schema
event_registry.register(
    event_type="CODEGEN_REQUESTED",
    version=EventSchemaVersion.V1,
    schema_class=ModelEventCodegenRequestedV1,
    evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
    deprecated=True,
    deprecation_date="2025-10-01",
    removal_date="2026-04-01",
    migration_notes="Migrate to V2 by adding node_type and enable_intelligence fields",
)
```

### Defining Migration Functions

```python
def migrate_v1_to_v2(data: dict) -> dict:
    """Migrate CODEGEN_REQUESTED from V1 to V2."""
    # Add new optional fields with defaults
    data.setdefault("node_type", None)
    data.setdefault("enable_intelligence", True)
    data.setdefault("enable_quorum", False)

    return data

# Register migration
event_registry.register_migration(
    event_type="CODEGEN_REQUESTED",
    from_version=EventSchemaVersion.V1,
    to_version=EventSchemaVersion.V2,
    migration_func=migrate_v1_to_v2,
)
```

### Validating and Migrating Events

```python
# Automatic migration to latest version
data = {
    "correlation_id": "...",
    "prompt": "Create a database effect node",
    "output_directory": "./nodes",
}

validated = event_registry.validate_and_migrate(
    event_type="CODEGEN_REQUESTED",
    data=data,
    source_version=EventSchemaVersion.V1,
    # target_version not specified = latest version
)

# Manual migration to specific version
validated = event_registry.validate_and_migrate(
    event_type="CODEGEN_REQUESTED",
    data=data,
    source_version=EventSchemaVersion.V1,
    target_version=EventSchemaVersion.V2,
)
```

## Topic Naming Convention

### Format

```
{environment}.{service}.{domain}.{event-name}.{version}
```

### Examples

```
dev.omninode-bridge.codegen.generation-requested.v1
prod.omninode-bridge.codegen.generation-completed.v2
staging.omniarchon.intelligence.query-requested.v1
```

### Topic Creation

```python
from omninode_bridge.events.versioning import get_topic_name, EventSchemaVersion

# Generate topic name
topic = get_topic_name(
    base_name="generation-requested",
    version=EventSchemaVersion.V1,
    environment="dev",
    service="omninode-bridge",
    domain="codegen",
)
# Result: "dev.omninode-bridge.codegen.generation-requested.v1"

# Parse topic name
from omninode_bridge.events.versioning import parse_topic_name

parsed = parse_topic_name("dev.omninode-bridge.codegen.generation-requested.v1")
# Result: {
#   "environment": "dev",
#   "service": "omninode-bridge",
#   "domain": "codegen",
#   "base_name": "generation-requested",
#   "version": "v1"
# }
```

## Migration Best Practices

### 1. Planning Phase

- **Impact Assessment**: Identify all consumers and their versions
- **Migration Path**: Define clear migration steps (V1 → V2 → V3)
- **Timeline**: Plan minimum 6 months for deprecation
- **Communication**: Notify all stakeholders in advance

### 2. Implementation Phase

- **Dual Publishing**: Publish to both old and new topics during transition
- **Version Detection**: Auto-detect and migrate incoming events
- **Monitoring**: Track version usage across consumers
- **Testing**: Comprehensive tests for all migration paths

### 3. Transition Phase

- **Gradual Rollout**: Migrate consumers in phases
- **Fallback Support**: Maintain old version for emergency rollback
- **Metrics Collection**: Track migration success rates
- **Documentation**: Update all consumer documentation

### 4. Completion Phase

- **Version Removal**: Remove deprecated version after timeline
- **Cleanup**: Delete migration code for removed versions
- **Archive**: Document historical versions for reference
- **Review**: Post-mortem on migration process

## Example: Complete V1 → V2 Migration

### Step 1: Design V2 Schema

```python
# V2 adds optional intelligence features
class ModelEventCodegenRequestedV2(BaseModel):
    # V1 fields (required)
    correlation_id: UUID
    prompt: str
    output_directory: str

    # V2 additions (optional for backward compatibility)
    node_type: Optional[str] = None
    enable_intelligence: bool = True
    enable_quorum: bool = False
    user_id: Optional[str] = None
```

### Step 2: Register V2 Schema

```python
event_registry.register(
    event_type="CODEGEN_REQUESTED",
    version=EventSchemaVersion.V2,
    schema_class=ModelEventCodegenRequestedV2,
    evolution_strategy=SchemaEvolutionStrategy.BACKWARD_COMPATIBLE,
    deprecated=False,
)
```

### Step 3: Define Migration

```python
def migrate_v1_to_v2(data: dict) -> dict:
    """Migrate CODEGEN_REQUESTED V1 → V2."""
    data.setdefault("node_type", None)
    data.setdefault("enable_intelligence", True)
    data.setdefault("enable_quorum", False)
    data.setdefault("user_id", None)
    return data

event_registry.register_migration(
    "CODEGEN_REQUESTED",
    EventSchemaVersion.V1,
    EventSchemaVersion.V2,
    migrate_v1_to_v2,
)
```

### Step 4: Update Producers

```python
# Old producer (V1)
event = ModelEventCodegenRequestedV1(
    correlation_id=uuid4(),
    prompt="Create node",
    output_directory="./nodes",
)

# New producer (V2)
event = ModelEventCodegenRequestedV2(
    correlation_id=uuid4(),
    prompt="Create node",
    output_directory="./nodes",
    node_type="effect",  # New field
    enable_intelligence=True,  # New field
)
```

### Step 5: Update Consumers

```python
# Consumer with automatic migration
def consume_event(topic: str, data: dict):
    # Parse version from topic
    parsed = parse_topic_name(topic)
    source_version = EventSchemaVersion(parsed["version"])

    # Validate and migrate to latest
    event = event_registry.validate_and_migrate(
        "CODEGEN_REQUESTED",
        data,
        source_version,
    )

    # Process V2 event
    process_generation_request(event)
```

### Step 6: Deprecate V1

```python
# After 6 months of dual support
event_registry.register(
    "CODEGEN_REQUESTED",
    EventSchemaVersion.V1,
    ModelEventCodegenRequestedV1,
    deprecated=True,
    deprecation_date="2025-10-01",
    removal_date="2026-04-01",
    migration_notes="All producers should migrate to V2",
)
```

### Step 7: Remove V1 (After Timeline)

```python
# Remove V1 topic subscriptions
# Delete V1 schema and migration code
# Update documentation
```

## Schema Validation

### Pre-Deployment Validation

```python
# Validate V2 accepts V1 data
v1_data = {"correlation_id": str(uuid4()), "prompt": "test", "output_directory": "./"}
v2_instance = ModelEventCodegenRequestedV2(**v1_data)
assert v2_instance.enable_intelligence is True  # Default value

# Validate migration
migrated = event_registry.migrate(
    "CODEGEN_REQUESTED",
    v1_data,
    EventSchemaVersion.V1,
    EventSchemaVersion.V2,
)
assert "enable_intelligence" in migrated
```

### Runtime Validation

```python
from pydantic import ValidationError

try:
    event = event_registry.validate_and_migrate(
        "CODEGEN_REQUESTED",
        data,
        EventSchemaVersion.V1,
    )
except ValidationError as e:
    logger.error("Event validation failed", error=str(e))
    # Handle invalid event
```

## Monitoring and Metrics

### Version Distribution

Track which versions are actively used:

```python
from collections import Counter

version_counter = Counter()

def track_event_version(topic: str):
    parsed = parse_topic_name(topic)
    version_counter[parsed["version"]] += 1

# Metrics
# - events_by_version{version="v1"} = 1234
# - events_by_version{version="v2"} = 5678
```

### Migration Success Rate

```python
migration_success = 0
migration_failures = 0

try:
    event = event_registry.validate_and_migrate(...)
    migration_success += 1
except Exception:
    migration_failures += 1

# Metric: migration_success_rate = success / (success + failures)
```

## Deprecation Policy

### Announcement

- **6 Months Before**: Announce deprecation in release notes
- **3 Months Before**: Log warnings in applications
- **1 Month Before**: Send direct notifications to consumers
- **Removal Date**: Complete removal from codebase

### Communication Channels

- Release notes and changelogs
- Slack/email notifications
- API documentation updates
- In-application warnings

### Deprecation Warnings

```python
import warnings

def check_version_deprecation(event_type: str, version: EventSchemaVersion):
    if event_registry.is_deprecated(event_type, version):
        metadata = event_registry.get_metadata(event_type, version)
        warnings.warn(
            f"{event_type} {version} is deprecated (since {metadata.deprecation_date}). "
            f"Will be removed on {metadata.removal_date}. "
            f"Migration notes: {metadata.migration_notes}",
            DeprecationWarning,
            stacklevel=2,
        )
```

## References

- **Event Schemas**: `src/omninode_bridge/events/models/codegen_events.py`
- **Versioning Implementation**: `src/omninode_bridge/events/versioning.py`
- **Tests**: `tests/observability/test_event_versioning.py`
- **Related**: POLY-12 Observability Enhancements

## Related Documentation

- [Event Infrastructure Guide](./EVENT_INFRASTRUCTURE.md) *(planned)*
- [Kafka Topic Management](../infrastructure/KAFKA_TOPICS.md) *(planned)*
- [API Versioning](../api/API_VERSIONING.md) *(planned)*
