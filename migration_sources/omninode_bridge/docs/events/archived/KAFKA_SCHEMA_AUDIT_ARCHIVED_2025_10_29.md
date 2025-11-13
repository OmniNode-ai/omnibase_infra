> **ARCHIVED**: This was a one-time audit artifact, archived on October 29, 2025.
> See: [docs/meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md](../../meta/EVENT_DOCS_CONSOLIDATION_ANALYSIS_2025_10.md) for details.

---

# Kafka Event Schema Audit - OmniNode Bridge

**Date**: October 2025
**Status**: Phase 1 Complete - Comprehensive Audit
**Purpose**: Standardize event schemas and documentation across 37+ Kafka topics

---

## Executive Summary

This document provides a comprehensive audit of all Kafka topics in the omninode_bridge repository, analyzing schema compliance, OnexEnvelopeV1 usage, and documentation gaps.

### Key Findings

**Total Topics Identified**: 37 topics across 4 categories
- **Codegen Topics**: 13 topics (100% documented, 0% OnexEnvelopeV1 compliant)
- **Bridge Orchestrator Topics**: 12 topics (100% OnexEnvelopeV1 compliant)
- **Bridge Reducer Topics**: 7 topics (100% OnexEnvelopeV1 compliant)
- **Service Lifecycle Topics**: 5 topics (0% OnexEnvelopeV1 compliant)

**Compliance Status**:
- ✅ OnexEnvelopeV1 Wrapper: 19/37 topics (51%)
- ✅ Schema Versioning: 37/37 topics (100%)
- ✅ Documentation: 32/37 topics (86%)
- ⚠️ Producer/Consumer Mapping: 20/37 topics (54%)

---

## Topic Inventory

### 1. Codegen Topics (13 topics)

**Category**: Autonomous Code Generation Infrastructure
**Namespace**: `omninode_codegen_*`
**OnexEnvelopeV1 Compliance**: ❌ Not using envelope wrapper

#### Request Topics (4)

| Topic Name | Schema | Envelope | Docs | Producer | Consumer |
|------------|--------|----------|------|----------|----------|
| `omninode_codegen_request_analyze_v1` | CodegenAnalysisRequest | ❌ | ✅ | omniclaude | omniarchon |
| `omninode_codegen_request_validate_v1` | CodegenValidationRequest | ❌ | ✅ | omniclaude | omniarchon |
| `omninode_codegen_request_pattern_v1` | CodegenPatternRequest | ❌ | ✅ | omniclaude | omniarchon |
| `omninode_codegen_request_mixin_v1` | CodegenMixinRequest | ❌ | ✅ | omniclaude | omniarchon |

#### Response Topics (4)

| Topic Name | Schema | Envelope | Docs | Producer | Consumer |
|------------|--------|----------|------|----------|----------|
| `omninode_codegen_response_analyze_v1` | CodegenAnalysisResponse | ❌ | ✅ | omniarchon | omniclaude |
| `omninode_codegen_response_validate_v1` | CodegenValidationResponse | ❌ | ✅ | omniarchon | omniclaude |
| `omninode_codegen_response_pattern_v1` | CodegenPatternResponse | ❌ | ✅ | omniarchon | omniclaude |
| `omninode_codegen_response_mixin_v1` | CodegenMixinResponse | ❌ | ✅ | omniarchon | omniclaude |

#### Status Topics (1)

| Topic Name | Schema | Envelope | Docs | Producer | Consumer |
|------------|--------|----------|------|----------|----------|
| `omninode_codegen_status_session_v1` | CodegenStatusEvent | ❌ | ✅ | Both | Monitoring |

#### DLQ Topics (4)

| Topic Name | Schema | Envelope | Docs | Producer | Consumer |
|------------|--------|----------|------|----------|----------|
| `omninode_codegen_dlq_analyze_v1` | CodegenAnalysisRequest | ❌ | ✅ | KafkaClient | DLQMonitor |
| `omninode_codegen_dlq_validate_v1` | CodegenValidationRequest | ❌ | ✅ | KafkaClient | DLQMonitor |
| `omninode_codegen_dlq_pattern_v1` | CodegenPatternRequest | ❌ | ✅ | KafkaClient | DLQMonitor |
| `omninode_codegen_dlq_mixin_v1` | CodegenMixinRequest | ❌ | ✅ | KafkaClient | DLQMonitor |

**Schema Location**: `src/omninode_bridge/events/codegen_schemas.py`
**Documentation**: `docs/events/EVENT_INFRASTRUCTURE_GUIDE.md`

**Standardization Needed**:
- ❌ Migrate to OnexEnvelopeV1 wrapper format
- ❌ Add envelope_version, event_id, correlation_id fields
- ❌ Update producers/consumers to use envelope serialization

---

### 2. Bridge Orchestrator Topics (12 topics)

**Category**: Workflow Orchestration Events
**Namespace**: `dev.omninode_bridge.onex.evt.*`
**OnexEnvelopeV1 Compliance**: ✅ Fully compliant

#### Workflow Lifecycle Topics (9)

| Topic Name | Event Type | Envelope | Docs | Producer | Consumer |
|------------|-----------|----------|------|----------|----------|
| `dev.omninode_bridge.onex.evt.stamp-workflow-started.v1` | WORKFLOW_STARTED | ✅ | ✅ | Orchestrator | Registry, Analytics |
| `dev.omninode_bridge.onex.evt.stamp-workflow-completed.v1` | WORKFLOW_COMPLETED | ✅ | ✅ | Orchestrator | Registry, Analytics |
| `dev.omninode_bridge.onex.evt.stamp-workflow-failed.v1` | WORKFLOW_FAILED | ✅ | ✅ | Orchestrator | Registry, Analytics |
| `dev.omninode_bridge.onex.evt.workflow-step-completed.v1` | STEP_COMPLETED | ✅ | ✅ | Orchestrator | Analytics |
| `dev.omninode_bridge.onex.evt.onextree-intelligence-requested.v1` | INTELLIGENCE_REQUESTED | ✅ | ✅ | Orchestrator | OnexTree |
| `dev.omninode_bridge.onex.evt.onextree-intelligence-received.v1` | INTELLIGENCE_RECEIVED | ✅ | ✅ | Orchestrator | Analytics |
| `dev.omninode_bridge.onex.evt.metadata-stamp-created.v1` | STAMP_CREATED | ✅ | ✅ | Orchestrator | Storage |
| `dev.omninode_bridge.onex.evt.blake3-hash-generated.v1` | HASH_GENERATED | ✅ | ✅ | Orchestrator | Storage |
| `dev.omninode_bridge.onex.evt.workflow-state-transition.v1` | STATE_TRANSITION | ✅ | ✅ | Orchestrator | Registry |

#### Node Introspection Topics (3)

| Topic Name | Event Type | Envelope | Docs | Producer | Consumer |
|------------|-----------|----------|------|----------|----------|
| `dev.omninode_bridge.onex.evt.node-introspection.v1` | NODE_INTROSPECTION | ✅ | ✅ | All Nodes | Registry, Monitoring |
| `dev.omninode_bridge.onex.evt.registry-request-introspection.v1` | REGISTRY_REQUEST_INTROSPECTION | ✅ | ✅ | Registry | All Nodes |
| `dev.omninode_bridge.onex.evt.node-heartbeat.v1` | NODE_HEARTBEAT | ✅ | ✅ | All Nodes | Registry, Health Monitor |

**Schema Location**: `src/omninode_bridge/nodes/orchestrator/v1_0_0/models/enum_workflow_event.py`
**Envelope Model**: `src/omninode_bridge/nodes/registry/v1_0_0/models/model_onex_envelope_v1.py`
**Documentation**: `docs/KAFKA_EVENT_PRODUCER_IMPLEMENTATION.md`, `docs/architecture/two-way-registration/INTROSPECTION_TOPICS.md`

**Status**: ✅ Fully standardized - Best practice reference

---

### 3. Bridge Reducer Topics (7 topics)

**Category**: Aggregation and State Reduction Events
**Namespace**: `dev.omninode_bridge.onex.evt.*`
**OnexEnvelopeV1 Compliance**: ✅ Fully compliant

#### Aggregation Topics (5)

| Topic Name | Event Type | Envelope | Docs | Producer | Consumer |
|------------|-----------|----------|------|----------|----------|
| `dev.omninode_bridge.onex.evt.aggregation-started.v1` | AGGREGATION_STARTED | ✅ | ✅ | Reducer | Analytics |
| `dev.omninode_bridge.onex.evt.batch-processed.v1` | BATCH_PROCESSED | ✅ | ✅ | Reducer | Analytics |
| `dev.omninode_bridge.onex.evt.state-persisted.v1` | STATE_PERSISTED | ✅ | ✅ | Reducer | Storage |
| `dev.omninode_bridge.onex.evt.aggregation-completed.v1` | AGGREGATION_COMPLETED | ✅ | ✅ | Reducer | Analytics |
| `dev.omninode_bridge.onex.evt.aggregation-failed.v1` | AGGREGATION_FAILED | ✅ | ✅ | Reducer | Analytics, Alerting |

#### FSM Topics (2)

| Topic Name | Event Type | Envelope | Docs | Producer | Consumer |
|------------|-----------|----------|------|----------|----------|
| `dev.omninode_bridge.onex.evt.fsm-state-initialized.v1` | FSM_STATE_INITIALIZED | ✅ | ✅ | Reducer | Registry |
| `dev.omninode_bridge.onex.evt.fsm-state-transitioned.v1` | FSM_STATE_TRANSITIONED | ✅ | ✅ | Reducer | Registry |

**Schema Location**: `src/omninode_bridge/nodes/reducer/v1_0_0/models/enum_reducer_event.py`
**Envelope Model**: `src/omninode_bridge/nodes/registry/v1_0_0/models/model_onex_envelope_v1.py`
**Documentation**: `docs/KAFKA_EVENT_PRODUCER_IMPLEMENTATION.md`

**Status**: ✅ Fully standardized - Best practice reference

---

### 4. Service Lifecycle Topics (5 topics)

**Category**: Service and Infrastructure Events
**Namespace**: `dev.omninode_bridge.onex.evt.*`
**OnexEnvelopeV1 Compliance**: ❌ Not using envelope wrapper

#### Lifecycle Events

| Topic Name | Event Type | Envelope | Docs | Producer | Consumer |
|------------|-----------|----------|------|----------|----------|
| `dev.omninode_bridge.onex.evt.startup.v1` | SERVICE_STARTUP | ❌ | ⚠️ | All Services | Monitoring |
| `dev.omninode_bridge.onex.evt.shutdown.v1` | SERVICE_SHUTDOWN | ❌ | ⚠️ | All Services | Monitoring |
| `dev.omninode_bridge.onex.evt.health-check.v1` | HEALTH_CHECK | ❌ | ⚠️ | All Services | Health Monitor |
| `dev.omninode_bridge.onex.evt.tool-call.v1` | TOOL_EXECUTION | ❌ | ⚠️ | MCP Services | Analytics |
| `dev.omninode_bridge.onex.met.performance.v1` | PERFORMANCE | ❌ | ⚠️ | All Services | Prometheus |

**Schema Location**: `src/omninode_bridge/models/events.py`
**Documentation**: ⚠️ Limited documentation

**Standardization Needed**:
- ❌ Migrate to OnexEnvelopeV1 wrapper format
- ❌ Comprehensive documentation needed
- ❌ Producer/consumer mapping documentation

---

## Schema Compliance Analysis

### OnexEnvelopeV1 Compliance Breakdown

#### ✅ Compliant Topics (19 topics - 51%)

**Bridge Orchestrator** (12 topics):
- All workflow lifecycle events using envelope wrapper
- All node introspection events using envelope wrapper

**Bridge Reducer** (7 topics):
- All aggregation events using envelope wrapper
- All FSM events using envelope wrapper

#### ❌ Non-Compliant Topics (18 topics - 49%)

**Codegen Topics** (13 topics):
- Using raw Pydantic models without envelope wrapper
- Missing envelope_version, event_id, source_node_id fields
- Missing correlation tracking infrastructure

**Service Lifecycle Topics** (5 topics):
- Using BaseEvent model without OnexEnvelopeV1 wrapper
- Inconsistent topic naming (mix of evt and met namespaces)

---

## Schema Versioning Analysis

### Version Field Compliance

✅ **All Topics**: 100% have schema_version field
- Codegen schemas: `schema_version: str = Field(default="1.0")`
- Bridge events: `event_version: str = Field(default="1.0")`
- Service events: Implicit v1 in topic name

### Evolution Strategy

**Codegen Topics**:
- Strategy: backward_compatible
- Documentation: ✅ Defined in codegen-topics-config.yaml
- Implementation: ✅ Schema versioning in place

**Bridge Topics**:
- Strategy: v2 topics with dual-publishing migration
- Documentation: ✅ Defined in INTROSPECTION_TOPICS.md
- Implementation: ✅ Topic versioning in place

---

## Documentation Analysis

### Comprehensive Documentation (✅ 32 topics)

**Codegen Topics** (13):
- Guide: `docs/events/EVENT_INFRASTRUCTURE_GUIDE.md` (300+ lines)
- Schemas: `docs/events/codegen_schemas.py` (270 lines)
- Config: `docs/events/codegen-topics-config.yaml` (181 lines)
- Quickstart: `docs/events/QUICKSTART.md`

**Bridge Topics** (19):
- Guide: `docs/KAFKA_EVENT_PRODUCER_IMPLEMENTATION.md` (400 lines)
- Introspection: `docs/architecture/two-way-registration/INTROSPECTION_TOPICS.md` (492 lines)
- Patterns: `docs/guides/BRIDGE_NODES_GUIDE.md`

### Limited Documentation (⚠️ 5 topics)

**Service Lifecycle Topics**:
- Missing: Dedicated documentation for service events
- Missing: Producer/consumer mapping
- Missing: Event flow diagrams

---

## Producer/Consumer Mapping

### Complete Mapping (✅ 20 topics)

**Codegen Topics** (13):
- Producers: omniclaude, omniarchon
- Consumers: omniclaude, omniarchon, DLQMonitor
- Consumer Groups: `omniclaude_codegen_consumer`, `omniarchon_codegen_intelligence`

**Bridge Introspection Topics** (3):
- Producers: All bridge nodes
- Consumers: Registry, Monitoring, Health Monitor
- Consumer Groups: `registry-introspection-collectors`, `monitoring-introspection-agents`

**Bridge Workflow Topics** (4):
- Producers: Orchestrator
- Consumers: Registry, Analytics, Storage
- Consumer Groups: TBD

### Incomplete Mapping (⚠️ 17 topics)

**Bridge Aggregation Topics** (7):
- Producers: Reducer
- Consumers: ⚠️ Unknown
- Consumer Groups: ⚠️ Undefined

**Service Lifecycle Topics** (5):
- Producers: All Services
- Consumers: ⚠️ Unknown
- Consumer Groups: ⚠️ Undefined

**Bridge State Topics** (5):
- Producers: Orchestrator, Reducer
- Consumers: ⚠️ Partially documented
- Consumer Groups: ⚠️ Undefined

---

## Recommendations

### Priority 1: OnexEnvelopeV1 Migration (CRITICAL)

**Codegen Topics Migration**:
1. Update `codegen_schemas.py` to wrap all schemas in ModelOnexEnvelopeV1
2. Modify producers (omniclaude, omniarchon) to use `publish_with_envelope()`
3. Update consumers to deserialize from envelope format
4. Add dual-format support during migration (check for envelope_version field)
5. Timeline: 2-3 days

**Service Lifecycle Topics Migration**:
1. Migrate BaseEvent to use ModelOnexEnvelopeV1 wrapper
2. Update all service lifecycle event publishers
3. Add correlation_id tracking for request/response patterns
4. Timeline: 1-2 days

### Priority 2: Documentation Completion (HIGH)

**Missing Documentation**:
1. Create `docs/events/SERVICE_LIFECYCLE_EVENTS.md` (comprehensive guide)
2. Add producer/consumer mapping for all topics
3. Create event flow diagrams (PlantUML/Mermaid)
4. Document consumer groups for bridge topics
5. Timeline: 2 days

### Priority 3: Schema Evolution Strategy (MEDIUM)

**Schema Registry**:
1. Evaluate schema registry solutions (Confluent Schema Registry vs. custom)
2. Implement schema validation at producer boundaries
3. Add schema evolution tests (backward/forward compatibility)
4. Document schema evolution workflow
5. Timeline: 3-4 days

### Priority 4: Validation Implementation (MEDIUM)

**Schema Validation**:
1. Create schema validation decorators for producers
2. Add Pydantic validation at consumer boundaries
3. Implement schema version compatibility checks
4. Add integration tests for schema evolution
5. Timeline: 2-3 days

---

## Migration Path

### Phase 1: Codegen Topics OnexEnvelopeV1 Migration (3 days)

**Day 1: Schema Updates**
- Update all 9 codegen event schemas to include envelope support
- Add `to_envelope()` and `from_envelope()` methods to each schema
- Create migration utility functions

**Day 2: Producer/Consumer Updates**
- Update omniclaude event publisher to use envelope wrapper
- Update omniarchon event consumer to handle envelopes
- Add dual-format support for backward compatibility

**Day 3: Testing & Validation**
- Integration tests for envelope serialization/deserialization
- Verify correlation_id propagation
- Performance testing for envelope overhead

### Phase 2: Service Lifecycle Migration (2 days)

**Day 1: BaseEvent Migration**
- Refactor BaseEvent to wrap in ModelOnexEnvelopeV1
- Update all service lifecycle event publishers
- Add correlation tracking for request/response events

**Day 2: Testing & Documentation**
- Integration tests for service events
- Create SERVICE_LIFECYCLE_EVENTS.md documentation
- Update producer/consumer mapping

### Phase 3: Documentation & Validation (4 days)

**Day 1-2: Documentation**
- Complete producer/consumer mapping for all topics
- Create comprehensive event flow diagrams
- Document consumer groups for all topics
- Create developer guide for event usage

**Day 3-4: Validation**
- Schema validation at producer/consumer boundaries
- Schema evolution tests
- Integration tests for all topics

---

## Success Criteria

### Completion Metrics

1. **OnexEnvelopeV1 Compliance**: 100% (37/37 topics)
2. **Documentation Coverage**: 100% (37/37 topics with comprehensive docs)
3. **Producer/Consumer Mapping**: 100% (37/37 topics with complete mapping)
4. **Schema Validation**: 100% (validation at all boundaries)
5. **Test Coverage**: >90% (integration tests for all event flows)

### Quality Gates

- ✅ All events use OnexEnvelopeV1 wrapper
- ✅ All events have schema versioning
- ✅ All topics documented with schemas
- ✅ All producer/consumer relationships mapped
- ✅ Schema validation at all boundaries
- ✅ Schema evolution strategy defined and tested

---

## Appendix

### File Locations

**Schema Definitions**:
- `src/omninode_bridge/events/codegen_schemas.py` - Codegen event schemas
- `src/omninode_bridge/models/events.py` - Service lifecycle schemas
- `src/omninode_bridge/nodes/orchestrator/v1_0_0/models/enum_workflow_event.py` - Orchestrator events
- `src/omninode_bridge/nodes/reducer/v1_0_0/models/enum_reducer_event.py` - Reducer events
- `src/omninode_bridge/nodes/registry/v1_0_0/models/model_onex_envelope_v1.py` - Envelope model

**Configuration**:
- `docs/events/codegen-topics-config.yaml` - Codegen topic configuration
- `config/development.yaml` - Development Kafka configuration
- `config/production.yaml` - Production Kafka configuration

**Documentation**:
- `docs/events/EVENT_INFRASTRUCTURE_GUIDE.md` - Codegen infrastructure guide
- `docs/KAFKA_EVENT_PRODUCER_IMPLEMENTATION.md` - Bridge event producer guide
- `docs/architecture/two-way-registration/INTROSPECTION_TOPICS.md` - Introspection topics guide
- `docs/REDPANDA_QUICK_REFERENCE.md` - Kafka/Redpanda reference

**Producer/Consumer Code**:
- `src/omninode_bridge/services/kafka_client.py` - KafkaClient with envelope support
- `src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py` - Orchestrator event publisher
- `src/omninode_bridge/nodes/reducer/v1_0_0/node.py` - Reducer event publisher
- `src/omninode_bridge/monitoring/codegen_dlq_monitor.py` - DLQ monitoring
- `src/omninode_bridge/dashboard/codegen_event_tracer.py` - Event tracing

---

**Last Updated**: October 2025
**Maintained By**: OmniNode Bridge Team
**Status**: Phase 1 Complete - Audit Documentation
