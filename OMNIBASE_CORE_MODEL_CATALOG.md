# Omnibase Core Model Catalog for Infrastructure Integration

**Purpose**: Comprehensive catalog of available models in omnibase_core that can be used by omnibase_infra to avoid duplication and improve consistency.

**Discovery Summary**:
- **Source Repository**: `/Volumes/PRO-G40/Code/omnibase_core/src/omnibase_core/models/`
- **Total Models Available**: 361 models (all in `core/` directory)
- **Infrastructure Models in Core**: 0 (empty directory)
- **Current omnibase_infra Models**: 97 models

## 🚨 Key Finding: Significant Model Duplication Detected

Analysis reveals **direct model duplication** between omnibase_core and omnibase_infra, particularly in health and error handling domains.

## 📊 Model Category Analysis

### 1. Health & Monitoring Models (HIGH PRIORITY for Integration)

#### Available in omnibase_core:
- `ModelNodeHealthMetadata` - Comprehensive node health tracking
  - **Location**: `omnibase_core.models.core.model_node_health_metadata`
  - **Features**: Error counts, performance metrics, uptime tracking, health tags
  - **Import**: `from omnibase_core.models.core.model_node_health_metadata import ModelNodeHealthMetadata`

- `ModelHealthDetails` - Generic health check details
  - **Location**: `omnibase_core.models.core.model_health_details`
  - **Features**: Service status, database connections, disk usage, response times
  - **Import**: `from omnibase_core.models.core.model_health_details import ModelHealthDetails`

#### 🔥 DUPLICATION ALERT:
**omnibase_infra** has `ModelHealthDetails` at `/Volumes/PRO-G40/Code/omnibase_infra/src/omnibase_infra/models/core/health/model_health_details.py`

**RECOMMENDATION**:
- **IMMEDIATE ACTION REQUIRED**: Replace omnibase_infra's `ModelHealthDetails` with omnibase_core version
- **Extend omnibase_core version** if infrastructure-specific fields are needed
- **Use omnibase_core's `ModelNodeHealthMetadata`** for comprehensive health tracking

### 2. Error Handling Models (CRITICAL for Compliance)

#### Available in omnibase_core:
- `ModelOnexError` - **Canonical ONEX error model**
  - **Location**: `omnibase_core.models.core.model_onex_error`
  - **Features**: Structured error with correlation ID, timestamps, context
  - **Integration**: Uses `EnumOnexStatus` and `ModelErrorContext`
  - **Import**: `from omnibase_core.models.core.model_onex_error import ModelOnexError`

- `ModelErrorContext` - Error context information
  - **Location**: `omnibase_core.models.core.model_error_context`
  - **Features**: Additional context for error details
  - **Import**: `from omnibase_core.models.core.model_error_context import ModelErrorContext`

- `ModelErrorSummary` - Error summary information
  - **Location**: `omnibase_core.models.core.model_error_summary`

**RECOMMENDATION**:
- **MANDATORY**: All omnibase_infra error models should use `ModelOnexError` as the standard
- **Consistent error handling** across all infrastructure components

### 3. Event & Workflow Models (MODERATE PRIORITY)

#### Available in omnibase_core:
- `ModelEventType` - Dynamic event type registration
  - **Location**: `omnibase_core.models.core.model_event_type`
  - **Features**: Namespace-aware, schema versioning, plugin extensibility
  - **Integration**: Works with registry patterns
  - **Import**: `from omnibase_core.models.core.model_event_type import ModelEventType`

**RECOMMENDATION**:
- **Use for event publishing**: Replace any custom event type models in omnibase_infra
- **Consistent event handling** across infrastructure components

### 4. Configuration Models (HIGH PRIORITY)

#### Available in omnibase_core:
- `ModelPerformanceConfig` - Standardized performance configuration
  - **Location**: `omnibase_core.models.core.model_performance_config`
  - **Features**: Timeout, memory limits, concurrency, retries, caching
  - **Validation**: Built-in constraints and validation rules
  - **Import**: `from omnibase_core.models.core.model_performance_config import ModelPerformanceConfig`

**RECOMMENDATION**:
- **Infrastructure standardization**: Use for all infrastructure component configuration
- **Consistent performance tuning** across services

### 5. Connection & Security Models (MODERATE PRIORITY)

#### Available in omnibase_core:
- `ModelMaskedConnectionProperties` - Secure connection property handling
  - **Location**: `omnibase_core.models.core.model_masked_connection_properties`
  - **Features**: Connection string masking, SSL support, pool configuration
  - **Security**: Always masks passwords, configurable masking algorithm
  - **Import**: `from omnibase_core.models.core.model_masked_connection_properties import ModelMaskedConnectionProperties`

**RECOMMENDATION**:
- **Security compliance**: Use for all database and service connections
- **Consistent credential handling** across infrastructure

### 6. State Management Models (MODERATE PRIORITY)

#### Available in omnibase_core:
- `ModelOnexInputState` / `ModelOnexOutputState` - ONEX state management
- `ModelBaseState` - Base state model
- `ModelStateUpdate` - State transition tracking
- Multiple specialized state models for different contexts

**RECOMMENDATION**:
- **Workflow consistency**: Use for infrastructure state management
- **State transition tracking** for infrastructure processes

## 🏷️ Available Enums (HIGH VALUE for Infrastructure)

### Environment & Configuration Enums:
- `EnumEnvironmentType` - **HIGHLY RECOMMENDED**
  - **Location**: `omnibase_core.enums.enum_environment_type`
  - **Values**: DEVELOPMENT, STAGING, PRODUCTION, etc.
  - **Features**: Built-in methods for environment detection, timeout/retry multipliers
  - **Import**: `from omnibase_core.enums.enum_environment_type import EnumEnvironmentType`

- `EnumLoggingLevel` - Standardized logging levels
  - **Location**: `omnibase_core.enums.enum_logging_level`

**RECOMMENDATION**:
- **IMMEDIATE ADOPTION**: Replace any custom environment enums with `EnumEnvironmentType`
- **Consistent environment handling** across all infrastructure components

## 📋 Integration Priority Matrix

### 🔴 IMMEDIATE ACTION REQUIRED (Week 1)
1. **ModelHealthDetails** - Replace duplicated model
2. **ModelOnexError** - Standardize all error handling
3. **EnumEnvironmentType** - Replace custom environment handling
4. **ModelPerformanceConfig** - Standardize configuration patterns

### 🟡 HIGH PRIORITY (Week 2)
1. **ModelNodeHealthMetadata** - Enhance health monitoring
2. **ModelEventType** - Standardize event handling
3. **ModelMaskedConnectionProperties** - Improve security compliance

### 🟢 MEDIUM PRIORITY (Week 3-4)
1. **State management models** - Workflow standardization
2. **Additional specialized models** as needed for specific use cases

## 🚀 Implementation Strategy

### Phase 1: Critical Dependencies
```python
# Add to omnibase_infra dependencies
from omnibase_core.models.core.model_onex_error import ModelOnexError
from omnibase_core.models.core.model_health_details import ModelHealthDetails
from omnibase_core.enums.enum_environment_type import EnumEnvironmentType
from omnibase_core.models.core.model_performance_config import ModelPerformanceConfig
```

### Phase 2: Infrastructure Enhancement
```python
# Enhance infrastructure with standardized models
from omnibase_core.models.core.model_node_health_metadata import ModelNodeHealthMetadata
from omnibase_core.models.core.model_event_type import ModelEventType
from omnibase_core.models.core.model_masked_connection_properties import ModelMaskedConnectionProperties
```

### Phase 3: Advanced Integration
```python
# Advanced state and workflow integration
from omnibase_core.models.core.model_base_state import ModelBaseState
from omnibase_core.models.core.model_state_update import ModelStateUpdate
```

## 🎯 Expected Benefits

### Immediate Benefits:
- **Eliminate Model Duplication** - Remove 5-10 duplicate models
- **ONEX Compliance** - Consistent error handling and health monitoring
- **Enhanced Type Safety** - Replace `Any` types with structured models
- **Reduced Maintenance** - Single source of truth for common models

### Long-term Benefits:
- **Ecosystem Consistency** - Standardized patterns across all ONEX components
- **Enhanced Interoperability** - Models work seamlessly across core and infrastructure
- **Improved Testing** - Consistent validation and testing patterns
- **Future-proof Architecture** - Built on established ONEX standards

## ⚠️ Migration Considerations

### Breaking Changes:
- Some field names may differ between omnibase_core and omnibase_infra versions
- Import paths will change for affected models
- Contract files may need updates to reference omnibase_core models

### Compatibility Strategy:
- **Phased migration**: Migrate one model category at a time
- **Deprecation period**: Mark old models as deprecated before removal
- **Testing**: Comprehensive testing after each migration phase
- **Documentation**: Update all imports and usage examples

## 📚 Next Steps

1. **Audit Phase**: Compare field-by-field differences between duplicated models
2. **Migration Planning**: Create detailed migration plan for each model category
3. **Enhancement Proposals**: Identify where omnibase_core models need infrastructure-specific extensions
4. **Implementation**: Execute phased migration following ONEX standards

---

**Summary**: omnibase_core provides a rich foundation of 361 models that can significantly reduce duplication in omnibase_infra. Immediate focus should be on health monitoring, error handling, and configuration models where direct duplication exists.