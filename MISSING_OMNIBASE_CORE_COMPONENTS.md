# Missing omnibase_core Components

Based on analysis of omnibase_infra imports and dependencies, the following components are referenced but not found in omnibase_core. These should be created in the core repository:

## Missing Utility Components

### 1. Error Sanitizer Utility
**Expected Location**: `omnibase_core/utils/error_sanitizer.py`
**Referenced in**: REDUCER_NODE_TEMPLATE.md
**Import**: `from omnibase_core.utils.error_sanitizer import ErrorSanitizer`
**Purpose**: Sanitize error messages to prevent sensitive information leakage

### 2. Circuit Breaker Mixin
**Expected Location**: `omnibase_core/utils/circuit_breaker.py` 
**Referenced in**: REDUCER_NODE_TEMPLATE.md
**Import**: `from omnibase_core.utils.circuit_breaker import CircuitBreakerMixin`
**Purpose**: Mixin class for adding circuit breaker functionality to nodes
**Note**: Circuit breaker exists in `omnibase_core/core/resilience/circuit_breaker.py` but not as a mixin utility

### 3. Base Node Configuration
**Expected Location**: `omnibase_core/config/base_node_config.py`
**Referenced in**: REDUCER_NODE_TEMPLATE.md
**Import**: `from omnibase_core.config.base_node_config import BaseNodeConfig`
**Purpose**: Base configuration class for all ONEX nodes

## Missing Model Components

### 4. ONEX Error Model
**Expected Location**: `omnibase_core/models/model_onex_error.py`
**Referenced in**: REDUCER_NODE_TEMPLATE.md
**Import**: `from omnibase_core.models.model_onex_error import ModelONEXError`
**Purpose**: Standardized error model for ONEX system errors

### 5. ONEX Warning Model
**Expected Location**: `omnibase_core/models/model_onex_warning.py`
**Referenced in**: REDUCER_NODE_TEMPLATE.md
**Import**: `from omnibase_core.models.model_onex_warning import ModelONEXWarning`
**Purpose**: Standardized warning model for ONEX system warnings

## Missing Subcontract Models

### 6. Configuration Subcontract
**Expected Location**: `omnibase_core/core/subcontracts/model_configuration_subcontract.py`
**Referenced in**: CONFIGURATION_SUBCONTRACT_PLACEMENT.md, multiple files
**Import**: `from omnibase_core.core.subcontracts.model_configuration_subcontract import ModelConfigurationSubcontract`
**Purpose**: Configuration management subcontract for infrastructure nodes

## Import Path Corrections Needed

### 7. Core Error Codes Import Path
**Current Issue**: Some files import `from omnibase_core.core.core_error_codes import CoreErrorCode`
**Correct Path**: `from omnibase_core.core.errors.onex_error import CoreErrorCode`
**Files Affected**: 
- `src/omnibase_infra/.serena/memories/configuration_consolidation_specs.md`
- `tests/test_postgres_adapter.py`

### 8. Model Path Corrections
**Current Issue**: Some imports reference model paths that may not exist
**Files to verify**:
- `ModelEffectInput, ModelEffectOutput` from `omnibase_core.core.node_effect`
- Health status enums location consistency

## Already Available Components

The following components are correctly available in omnibase_core and properly imported:

✅ `omnibase_core.core.errors.onex_error.OnexError`
✅ `omnibase_core.core.errors.onex_error.CoreErrorCode`
✅ `omnibase_core.core.onex_container.ModelONEXContainer`
✅ `omnibase_core.protocol.protocol_event_bus.ProtocolEventBus`
✅ `omnibase_core.model.core.model_onex_event.ModelOnexEvent`
✅ `omnibase_core.core.node_effect_service.NodeEffectService`
✅ `omnibase_core.core.node_reducer_service.NodeReducerService`
✅ `omnibase_core.utils.generation.utility_schema_loader.UtilitySchemaLoader`
✅ Circuit breaker functionality (exists in `omnibase_core.core.resilience.circuit_breaker`)

## Recommendations

1. **Create missing utility components** in omnibase_core to support infrastructure templates
2. **Standardize error and warning models** for consistent ONEX system messaging
3. **Implement configuration subcontract** for infrastructure node configuration management
4. **Fix import path inconsistencies** in existing files
5. **Create circuit breaker mixin utility** that wraps the existing circuit breaker implementation
6. **Establish base node configuration** pattern for consistent node setup

This analysis ensures omnibase_infra can properly import all required components from omnibase_core without breaking ONEX architectural patterns.