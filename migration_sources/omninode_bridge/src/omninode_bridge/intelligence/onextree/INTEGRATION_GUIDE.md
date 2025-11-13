# OnexTree Integration Layer Guide

**Version:** 1.1.0 (Phase 2)
**Status:** Ready for Use

## Overview

The OnexTree Integration Layer provides adapter classes for seamlessly integrating OnexTree with the metadata stamping service and event-driven architecture. This enables OnexTree to enhance file processing operations without requiring changes to existing services.

## Integration Components

### 1. OnexTreeUnifiedFileProcessorIntegration

**Purpose:** Enrich file processing operations with project structure context.

**Use Case:** When metadata stamping service processes files, OnexTree provides rich context about file location, siblings, architectural patterns, and semantic purpose.

**Example Usage:**

```python
from omninode_bridge.intelligence.onextree.generator import OnexTreeGenerator
from omninode_bridge.intelligence.onextree.query_engine import OnexTreeQueryEngine
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeUnifiedFileProcessorIntegration
)

# Setup
generator = OnexTreeGenerator(project_root)
tree_root = await generator.generate_tree()

engine = OnexTreeQueryEngine()
await engine.load_tree(tree_root)

# Create integration
integration = OnexTreeUnifiedFileProcessorIntegration(engine)

# Enrich file context before processing
context = await integration.enrich_file_context("src/services/api.py")

# Context includes:
# - file_metadata: Size, extension, last_modified
# - structural_context: Parent directory, siblings
# - semantic_context: Architectural patterns, inferred purpose
# - project_context: Total files, directories

# Validate file path before creation
validation = await integration.validate_file_path("src/new_service.py")
if validation["is_valid"]:
    # Safe to create file
    pass
```

**Benefits:**
- **Rich Context:** Provides 4 levels of context (file, structural, semantic, project)
- **Fast:** Sub-5ms lookups from in-memory indexes
- **Non-Invasive:** Works with existing file processor without changes
- **Duplicate Detection:** Prevents creating files that already exist

---

### 2. OnexTreeBatchProcessingIntegration

**Purpose:** Provide pre-indexed file lists for batch operations.

**Use Case:** When batch processing files, OnexTree eliminates filesystem traversal by providing instant file lists with metadata.

**Example Usage:**

```python
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeBatchProcessingIntegration
)

# Create integration
integration = OnexTreeBatchProcessingIntegration(engine)

# Get all Python files for batch processing
files = await integration.get_files_for_batch_processing(
    file_types=["py"],
    max_files=1000
)

# Each file includes: path, name, extension, size, last_modified

# Filter by directory
service_files = await integration.get_files_for_batch_processing(
    file_types=["py"],
    directory_filter="src/services"
)

# Get statistics for planning
stats = await integration.get_batch_statistics(file_types=["py", "js"])
print(f"Will process {stats['filtered_count']} files")
```

**Benefits:**
- **Instant:** No filesystem traversal (0ms vs 200-500ms)
- **Pre-Filtered:** By file type and directory
- **Metadata Included:** Size and modification time for optimization
- **Statistics:** Plan batch operations with accurate counts

---

### 3. OnexTreeEventPublisher

**Purpose:** Publish tree lifecycle events to Kafka for distributed coordination.

**Use Case:** When tree is generated or updated, publish events for cache invalidation, audit logging, and downstream processing.

**Example Usage:**

```python
from omninode_bridge.services.metadata_stamping.events import EventPublisher
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeEventPublisher
)

# Setup with metadata stamping event publisher
metadata_event_publisher = EventPublisher(kafka_config)
onextree_publisher = OnexTreeEventPublisher(metadata_event_publisher)

# Publish tree generated event
await onextree_publisher.publish_tree_generated_event(
    tree_root,
    generation_time_ms=123.45
)

# Publish tree updated event
await onextree_publisher.publish_tree_updated_event(
    tree_root,
    trigger="filesystem_change",
    files_changed=["src/new_file.py"]
)

# Publish query metrics
await onextree_publisher.publish_query_metrics_event(
    operation="lookup_file",
    execution_time_ms=0.5,
    result_count=1
)
```

**Event Topics:**
- `onextree.tree.generated` - Tree initially generated
- `onextree.tree.updated` - Tree updated due to changes
- `onextree.query.metrics` - Query performance metrics

**Benefits:**
- **Distributed Coordination:** Synchronize across services
- **Audit Trail:** Track all tree lifecycle events
- **Observability:** Query performance metrics
- **Cache Invalidation:** Trigger distributed cache updates

---

### 4. OnexTreeIntegrationManager

**Purpose:** Central manager for all integration components with feature flags.

**Use Case:** Simplify setup and provide unified interface for all integrations with selective enablement.

**Example Usage:**

```python
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeIntegrationManager
)

# Enable all integrations
manager = OnexTreeIntegrationManager(
    query_engine=engine,
    enable_file_processor=True,
    enable_batch_processing=True,
    enable_events=True,
    event_publisher=metadata_event_publisher
)

# Check what's enabled
if manager.is_file_processor_enabled():
    context = await manager.file_processor.enrich_file_context(file_path)

if manager.is_batch_processing_enabled():
    files = await manager.batch_processing.get_files_for_batch_processing()

if manager.is_events_enabled():
    await manager.event_publisher.publish_tree_generated_event(tree_root, 100.0)

# Get integration status
status = await manager.get_integration_status()
# Returns:
# {
#   "integrations": {
#     "file_processor": {"enabled": True, "status": "ready"},
#     "batch_processing": {"enabled": True, "status": "ready"},
#     "events": {"enabled": True, "status": "ready"}
#   },
#   "query_engine": {
#     "loaded": True,
#     "total_files": 1234,
#     "total_directories": 56
#   }
# }
```

**Benefits:**
- **Feature Flags:** Enable/disable integrations independently
- **Simplified Setup:** Single manager for all integrations
- **Status Monitoring:** Check integration health
- **Graceful Degradation:** Works with partial enablement

---

## Feature Flags (Configuration)

Integration components can be selectively enabled via configuration:

```python
from omninode_bridge.intelligence.onextree.config import OnexTreeConfig

config = OnexTreeConfig(
    # Core features (always enabled)
    enable_filesystem_watcher=True,
    enable_mcp_server=True,

    # Integration features (enable as needed)
    enable_unified_file_processor_integration=True,
    enable_batch_processing_integration=True,
    enable_kafka_event_publishing=True,
    enable_database_persistence=False,  # Future
)
```

**Environment Variables:**

```bash
# Enable/disable integrations
ONEXTREE_ENABLE_UNIFIED_FILE_PROCESSOR_INTEGRATION=true
ONEXTREE_ENABLE_BATCH_PROCESSING_INTEGRATION=true
ONEXTREE_ENABLE_KAFKA_EVENT_PUBLISHING=true
```

---

## Integration with Metadata Stamping Service

### Timeline

The integration layer is **ready for use** but depends on metadata stamping service components:

| Component | Status | Timeline |
|-----------|--------|----------|
| **File Processor Integration** | ✅ Ready | Use immediately |
| **Batch Processing Integration** | ✅ Ready | Use when batch processor implemented |
| **Event Publishing** | ✅ Ready | Use with existing EventPublisher |
| **Database Persistence** | 🔄 Future | Phase 5+ |

### Integration Points

#### With Stamping Engine

```python
# In metadata stamping service
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeUnifiedFileProcessorIntegration
)

class EnhancedStampingEngine:
    def __init__(self, onextree_integration=None):
        self.onextree = onextree_integration

    async def stamp_file(self, file_path, file_data):
        # Get OnexTree context
        if self.onextree:
            context = await self.onextree.enrich_file_context(file_path)
            # Use context to enhance stamp metadata
            # Add architectural_pattern, inferred_purpose, etc.

        # Continue with stamping...
```

#### With Batch Processor (Future)

```python
# In future batch processor
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeBatchProcessingIntegration
)

class BatchProcessor:
    def __init__(self, onextree_integration=None):
        self.onextree = onextree_integration

    async def process_batch(self, file_types):
        # Get files from OnexTree (instant, no filesystem traversal)
        if self.onextree:
            files = await self.onextree.get_files_for_batch_processing(
                file_types=file_types
            )
        else:
            # Fallback to filesystem traversal
            files = self._scan_filesystem(file_types)

        # Process files...
```

---

## Performance Characteristics

### File Processor Integration

- **Context Enrichment:** < 5ms (in-memory lookup)
- **Validation:** < 1ms (hash table lookup)
- **Memory Overhead:** Minimal (references existing indexes)

### Batch Processing Integration

- **File List Generation:** < 10ms for 10K files
- **Filtering:** < 5ms (pre-indexed)
- **vs Filesystem Traversal:** **200-500ms saved**

### Event Publishing

- **Event Serialization:** < 1ms
- **Kafka Publishing:** Async, non-blocking
- **Overhead:** Minimal (disabled mode has zero overhead)

---

## Testing

### Unit Tests

All integration components have comprehensive unit tests:

```bash
# Run integration layer tests
PYTHONPATH=src python3 -m pytest tests/intelligence/onextree/test_integration_layer.py -v

# Expected: 16 tests passed
```

### Integration Tests

Test with metadata stamping service:

```bash
# Run end-to-end tests (when metadata stamping available)
PYTHONPATH=src python3 -m pytest tests/intelligence/onextree/test_integration.py -v
```

---

## Error Handling

### Graceful Degradation

All integrations handle missing components gracefully:

```python
# If tree not loaded
context = await integration.enrich_file_context("file.py")
# Returns: {"onextree_context": {"exists_in_tree": False, "status": "file_not_found"}}

# If event publisher disabled
publisher = OnexTreeEventPublisher(event_publisher=None)
result = await publisher.publish_tree_generated_event(tree, 100.0)
# Returns: False (logged but not published)

# If query engine not loaded
stats = await integration.get_batch_statistics()
# Returns: {"available": False}
```

### Error Recovery

```python
try:
    context = await integration.enrich_file_context("file.py")
except Exception as e:
    # Log error, use default context
    context = {}
```

---

## Best Practices

### 1. Use Integration Manager

For multiple integrations, use `OnexTreeIntegrationManager` for simplified setup:

```python
manager = OnexTreeIntegrationManager(
    query_engine,
    enable_file_processor=True,
    enable_batch_processing=True
)
```

### 2. Enable Selectively

Only enable integrations you need:

```python
# Development: All integrations
enable_events = True

# Production: Selective based on infrastructure
enable_events = kafka_available and config.enable_events
```

### 3. Monitor Performance

Use event publishing to track performance:

```python
if manager.is_events_enabled():
    await manager.event_publisher.publish_query_metrics_event(
        "lookup_file",
        execution_time_ms,
        result_count
    )
```

### 4. Handle Errors Gracefully

Always check for integration availability:

```python
if manager.is_file_processor_enabled():
    try:
        context = await manager.file_processor.enrich_file_context(path)
    except Exception as e:
        logger.warning(f"OnexTree context enrichment failed: {e}")
        context = {}  # Use default
```

---

## Future Enhancements

### Phase 3: Advanced Features

- **Agent-Specific Context:** Tailored context for different agent types
- **LRU Caching:** Cache frequently accessed paths
- **Predictive Prefetching:** Anticipate file lookups based on patterns

### Phase 4: Database Integration

- **Cross-Session Persistence:** Store query history
- **Distributed Caching:** Redis integration for shared cache
- **Analytics:** Track usage patterns and optimize

---

## References

- **Implementation Plan:** `ONEXTREE_STANDALONE_IMPLEMENTATION_PLAN.md`
- **Phase 1 (Standalone):** PR #12
- **Metadata Stamping Service:** `src/omninode_bridge/services/metadata_stamping/`
- **Event Publishing:** `src/omninode_bridge/services/metadata_stamping/events/`

---

## Support

For questions or issues:
1. Check test examples in `tests/intelligence/onextree/test_integration_layer.py`
2. Review implementation in `src/omninode_bridge/intelligence/onextree/integration.py`
3. Refer to metadata stamping service documentation

---

**Last Updated:** 2025-09-29
**Version:** 1.1.0 (Phase 2: Integration Layer)
