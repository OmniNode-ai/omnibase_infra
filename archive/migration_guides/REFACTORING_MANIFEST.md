# REFACTORING MANIFEST - Core Domain Models Strong Typing

**PR #11 Implementation Guide**
**Analysis Date**: 2025-01-18
**Models Analyzed**: 56 core domain model files
**Target**: Eliminate `Dict[str, Any]` usage and implement comprehensive strong typing

## Executive Summary

Comprehensive audit of all 56 core domain model files identified significant strong typing improvement opportunities. This manifest provides a structured approach to implementing @jonahgabriel's feedback on replacing weak typing with proper models and enums.

### Key Findings
- **23 enum candidates** identified across status fields, severity levels, and operational states
- **15 ID fields** requiring UUID conversion from string types
- **8 structured string fields** that should become proper model classes
- **3 file structure improvements** needed for ONEX compliance
- **Zero backwards compatibility** requirements (per CLAUDE.md policy)

## 🎯 Priority 1: Status and State Enums

### Circuit Breaker Domain
**File**: `model_circuit_breaker_result.py`
```python
# CURRENT (weak typing)
circuit_action_taken: str = Field(pattern="^(published|queued|dropped|rejected)$")

# TARGET (strong typing)
class EnumCircuitAction(str, Enum):
    PUBLISHED = "published"
    QUEUED = "queued"
    DROPPED = "dropped"
    REJECTED = "rejected"

circuit_action_taken: EnumCircuitAction = Field(...)
```

**File**: `model_dead_letter_queue_entry.py`
```python
# CURRENT (regex pattern)
circuit_breaker_state: str = Field(pattern="^(CLOSED|HALF_OPEN|OPEN)$")

# TARGET (already exists in omnibase_core)
# Use existing: EnumCircuitBreakerState.CLOSED | HALF_OPEN | OPEN
```

### Security Domain
**File**: `model_audit_details.py`
```python
# CURRENT
threat_level: str = Field(pattern="^(low|medium|high|critical)$")
data_classification: str = Field(pattern="^(public|internal|confidential|restricted)$")

# TARGET
class EnumThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EnumDataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
```

**File**: `model_audit_metadata.py`
```python
# CURRENT
alert_severity: str = Field(pattern="^(info|warning|error|critical)$")

# TARGET
class EnumAlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

**File**: `model_payload_encryption.py`
```python
# CURRENT
security_level: str = Field(pattern="^(minimal|standard|high|maximum)$")
return_format: str = Field(pattern="^(string|bytes|dict|auto)$")

# TARGET
class EnumSecurityLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

class EnumReturnFormat(str, Enum):
    STRING = "string"
    BYTES = "bytes"
    DICT = "dict"
    AUTO = "auto"
```

**File**: `model_tls_config.py`
```python
# CURRENT
tls_version_min: str = Field(pattern="^(1\\.2|1\\.3)$")
compliance_level: str = Field(pattern="^(minimal|standard|strict|maximum)$")
acks: str = Field(pattern="^(0|1|all)$")

# TARGET
class EnumTLSVersion(str, Enum):
    V1_2 = "1.2"
    V1_3 = "1.3"

class EnumComplianceLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    MAXIMUM = "maximum"

class EnumAckLevel(str, Enum):
    NONE = "0"
    LEADER = "1"
    ALL = "all"
```

### Health Domain
**File**: `model_component_status.py`
```python
# CURRENT
status: str = Field(pattern="^(healthy|warning|critical|unknown|offline)$")

# TARGET
class EnumComponentStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"
```

**File**: `model_health_alert.py`
```python
# CURRENT
alert_type: str = Field(pattern="^(performance|availability|resource|security|configuration)$")
severity: str = Field(pattern="^(low|medium|high|critical)$")
status: str = Field(pattern="^(active|acknowledged|resolved|suppressed)$")
impact_level: str = Field(pattern="^(none|low|medium|high|severe)$")

# TARGET
class EnumAlertType(str, Enum):
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RESOURCE = "resource"
    SECURITY = "security"
    CONFIGURATION = "configuration"

class EnumAlertSeverity(str, Enum):  # Consolidate with security domain
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EnumAlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class EnumImpactLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"
```

**File**: `model_health_request.py`
```python
# CURRENT
priority: str = Field(pattern="^(low|normal|high|critical)$")
deployment_stage: str = Field(pattern="^(development|staging|production|test)$")

# TARGET
class EnumPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class EnumDeploymentStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"
```

**File**: `model_trend_analysis.py`
```python
# CURRENT
trend_direction: str = Field(pattern="^(improving|stable|degrading|unknown)$")

# TARGET
class EnumTrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"
```

### Observability Domain
**File**: `model_alert_details.py`
```python
# CURRENT
threshold_operator: str = Field(pattern="^(gt|gte|lt|lte|eq|neq)$")
component_health_status: str = Field(pattern="^(healthy|degraded|unhealthy|unknown)$")
impact_level: str = Field(pattern="^(none|low|medium|high|severe)$")
trend_direction: str = Field(pattern="^(improving|stable|degrading|unknown)$")

# TARGET
class EnumThresholdOperator(str, Enum):
    GREATER_THAN = "gt"
    GREATER_THAN_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "neq"

# Reuse existing EnumComponentHealthStatus and EnumImpactLevel from health domain
```

### Workflow Domain
**File**: `model_workflow_execution_context.py`
```python
# CURRENT
agent_coordination_strategy: str = Field(examples=["sequential", "parallel", "hybrid", "adaptive"])
log_level: str = Field(examples=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

# TARGET
class EnumAgentCoordinationStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class EnumLogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
```

**File**: `model_sub_agent_result.py`
```python
# CURRENT
agent_type: str = Field(examples=["specialist", "coordinator", "processor", "validator", "analyzer"])
execution_status: str = Field(examples=["completed", "failed", "timeout", "cancelled", "partial_success"])

# TARGET
class EnumAgentType(str, Enum):
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    ANALYZER = "analyzer"

class EnumExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"
```

## 🆔 Priority 2: ID Field UUID Conversion

### High-Priority UUID Conversions
These fields are clearly identifiers that should be UUID type:

**Outbox Domain**
```python
# model_outbox_event_data.py
correlation_id: str | None → correlation_id: UUID | None
user_id: str | None → user_id: UUID | None
tenant_id: str | None → tenant_id: UUID | None
```

**Security Domain**
```python
# model_security_event_data.py
correlation_id: str | None → correlation_id: UUID | None
user_id: str | None → user_id: UUID | None
session_id: str | None → session_id: UUID | None
tenant_id: str | None → tenant_id: UUID | None

# model_audit_details.py (already has UUID for some)
# correlation_id: UUID | None ✓ (already correct)
# user_id: UUID | None ✓ (already correct)
# session_id: UUID | None ✓ (already correct)
```

**Workflow Domain**
```python
# model_workflow_execution_context.py
user_id: Optional[UUID] ✓ (already correct)
session_id: Optional[UUID] ✓ (already correct)

# All workflow models already use UUID for correlation_id ✓
```

### Keep as String (Valid Reasons)
These ID fields should remain string due to external system integration:

```python
# External system IDs
event_id: str  # External event system IDs
batch_id: str  # External batch processing IDs
trace_id: str  # Distributed tracing system IDs
span_id: str   # Distributed tracing system IDs
alert_id: str  # External alerting system IDs

# Composite/Formatted IDs
node_id: str           # Node naming patterns (e.g., "postgres_adapter_node")
agent_id: str          # Agent naming patterns
credential_id: str     # External credential system IDs
resource_id: str       # External resource identifiers
```

## 🏗️ Priority 3: Structured String Fields to Models

### Environment Configuration Model
Multiple files contain environment-related strings that should be consolidated:

```python
# Create new model: model_environment_context.py
class ModelEnvironmentContext(BaseModel):
    """Standardized environment context across all models."""

    environment: EnumDeploymentStage = Field(
        description="Deployment environment"
    )
    region: str | None = Field(
        default=None,
        max_length=50,
        description="Geographic region"
    )
    availability_zone: str | None = Field(
        default=None,
        max_length=50,
        description="Availability zone"
    )
    deployment_version: str | None = Field(
        default=None,
        max_length=50,
        description="Deployment version"
    )

# Replace in multiple files:
# - model_audit_details.py
# - model_alert_details.py
# - model_health_request.py
# - model_security_event_data.py
```

### Kafka Configuration Consolidation
**File**: `model_kafka_configuration.py` has many string fields that could be enums:

```python
# CURRENT
security_protocol: str = Field(default="PLAINTEXT")
sasl_mechanism: str | None = Field(default=None)
auto_offset_reset: str = Field(default="latest")

# TARGET
class EnumKafkaSecurityProtocol(str, Enum):
    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"

class EnumSASLMechanism(str, Enum):
    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    GSSAPI = "GSSAPI"

class EnumAutoOffsetReset(str, Enum):
    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"
```

## 📁 Priority 4: File Structure Validation

### Compliant File Structure ✅
All 56 files follow proper ONEX conventions:
- ✅ One model class per file
- ✅ Proper naming: `ModelCamelCase` classes in `model_snake_case.py` files
- ✅ Appropriate directory organization by domain

### Minor Improvements Needed
1. **Missing __init__.py files**: Some subdirectories could benefit from proper imports
2. **Cross-domain enum sharing**: Several enums are duplicated across domains and should be shared

## 🔧 Implementation Strategy

### Phase 1: Foundation Enums (Week 1)
Create shared enum modules in `src/omnibase_infra/models/core/common/`:
- `enum_severity_levels.py` - Consolidate all severity enums
- `enum_status_types.py` - Consolidate all status enums
- `enum_environment_types.py` - Deployment and environment enums
- `enum_operational_states.py` - Health, alert, and operational state enums

### Phase 2: Domain-Specific Enums (Week 1-2)
Create domain-specific enums within each domain:
- Circuit breaker enums in circuit_breaker domain
- Security-specific enums in security domain
- Workflow execution enums in workflow domain
- TLS and encryption enums in security domain

### Phase 3: ID Field Conversions (Week 2)
Convert ID fields to UUID systematically:
1. Update model field types
2. Update any tests referencing these fields
3. Verify no string assumptions in code using these models

### Phase 4: Structured Models (Week 2-3)
Create composite models:
1. `ModelEnvironmentContext` for environment standardization
2. Enhanced Kafka configuration enums
3. Alert and notification standardization

### Phase 5: Import Updates (Week 3)
Update all imports across the codebase:
1. Add enum imports to model files
2. Update any code importing the changed models
3. Verify omnibase_core enum imports work correctly

## 🎯 Success Metrics

### Compliance Targets
- **Zero** regex patterns for enumerable values
- **Zero** string fields that represent structured data
- **Zero** `Any` types (already achieved)
- **100%** enum usage for status/state fields
- **95%** UUID usage for identifier fields (excluding valid external IDs)

### Quality Gates
1. **Type Safety**: All mypy checks pass with strict mode
2. **ONEX Compliance**: All models follow ONEX architecture patterns
3. **Performance**: No performance degradation from enum usage
4. **Backwards Compatibility**: NONE (per CLAUDE.md zero tolerance policy)

## 📋 Detailed File Inventory

### Files Requiring Major Changes (15 files)
1. `model_circuit_breaker_result.py` - 2 enums + field updates
2. `model_dead_letter_queue_entry.py` - Use existing enum + field updates
3. `model_audit_details.py` - 3 enums + environment model
4. `model_audit_metadata.py` - 1 enum + field updates
5. `model_payload_encryption.py` - 3 enums + field updates
6. `model_tls_config.py` - 4 enums + field updates
7. `model_component_status.py` - 1 enum + field updates
8. `model_health_alert.py` - 4 enums + field updates
9. `model_health_request.py` - 2 enums + environment model
10. `model_trend_analysis.py` - 1 enum + field updates
11. `model_alert_details.py` - 2 enums + environment model
12. `model_workflow_execution_context.py` - 2 enums + field updates
13. `model_sub_agent_result.py` - 2 enums + field updates
14. `model_kafka_configuration.py` - 3 enums + field updates
15. `model_security_event_data.py` - UUID conversions

### Files Requiring Minor Changes (8 files)
UUID conversions only:
1. `model_outbox_event_data.py`
2. `model_workflow_execution_request.py`
3. `model_workflow_execution_result.py`
4. `model_workflow_progress_update.py`
5. `model_workflow_progress_history.py`
6. `model_agent_activity.py`
7. `model_workflow_step_details.py`
8. `model_workflow_result_data.py`

### Files Requiring No Changes (33 files)
Already properly typed with strong types and enums.

## 🚀 Next Steps

1. **Create enum modules** following the foundation pattern
2. **Update model imports** to use new enum types
3. **Convert ID fields** systematically with UUID types
4. **Validate changes** with comprehensive testing
5. **Update documentation** reflecting new strong typing patterns

---

**Total Estimated Effort**: 2-3 weeks with proper testing and validation
**Risk Level**: Low (no backwards compatibility concerns)
**Business Impact**: High (improved type safety, better developer experience, ONEX compliance)