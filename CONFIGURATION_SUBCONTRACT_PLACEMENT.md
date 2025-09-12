# Configuration Subcontract Model - Core Placement Required

## 📍 Target Location
The configuration subcontract model should be placed in:
```
omnibase_core/core/subcontracts/model_configuration_subcontract.py
```

## 🎯 Rationale
- **Foundational Pattern**: Configuration management is needed across ALL node types (AI, infrastructure, business logic)
- **Standards Consistency**: All other subcontract models are in `omnibase_core.core.subcontracts`
- **Reusability**: Infrastructure, compute, reducer, orchestrator, and gateway nodes all need standardized configuration
- **Architecture Compliance**: Follows established ONEX subcontract placement patterns

## 📦 Current Status
- ✅ **Model Created**: Complete 342-line Pydantic model with validation
- ✅ **Contract Updated**: PostgreSQL adapter references temporary local location
- ✅ **Working Implementation**: Model imports and validates successfully
- ⏳ **Migration Pending**: Awaiting omnibase_core merge for final placement

## 🔄 Migration Strategy
**Phase 1 (Current)**: Use temporary local copy
- Location: `omnibase_infra.models.infrastructure.model_configuration_subcontract`
- Status: ✅ Working and tested
- Import: `from omnibase_infra.models.infrastructure.model_configuration_subcontract import ModelConfigurationSubcontract`

**Phase 2 (After omnibase_core merge)**: Switch to core reference
- Location: `omnibase_core.core.subcontracts.model_configuration_subcontract`  
- Simple contract update: Change module path only
- Import: `from omnibase_core.core.subcontracts.model_configuration_subcontract import ModelConfigurationSubcontract`

## 🔄 Integration Pattern
Once placed in core, this subcontract will be available for:

### Infrastructure Nodes
- `postgres_adapter`, `consul_adapter`, `kafka_adapter`, `vault_adapter`
- Service discovery, message queues, secret management

### AI Processing Nodes  
- LLM processors, embedding services, model inference nodes
- Configuration for model endpoints, API keys, compute resources

### Business Logic Nodes
- Compute nodes, reducer nodes, orchestrator nodes
- Domain-specific configuration patterns

### Gateway Nodes
- API gateways, service proxies, load balancers
- Network configuration, routing rules, security policies

## 📋 File Content
The complete model is currently located at:
```
/Volumes/PRO-G40/Code/omnibase_infra/src/omnibase_infra/models/infrastructure/model_configuration_subcontract.py
```

This file contains:
- `ConfigurationSourceType` enum (container, environment, defaults, file)
- `ValidationRuleType` enum (format, range, enum, required)
- `ModelConfigurationSource` - Source priority and validation
- `ModelEnvironmentConfiguration` - Environment variable patterns
- `ModelValidationRule` - Individual validation rules with type-specific logic
- `ModelConfigurationValidation` - Validation rule collections
- `ModelConfigurationIntegration` - Container/environment integration
- `ModelConfigurationSecurity` - Security and sanitization
- `ModelConfigurationSubcontract` - Main subcontract model

## ✅ Next Steps
1. Copy the model file to `omnibase_core/core/subcontracts/`
2. Update `omnibase_core/core/subcontracts/__init__.py` to include the new model
3. Test import in PostgreSQL adapter: `from omnibase_core.core.subcontracts.model_configuration_subcontract import ModelConfigurationSubcontract`
4. Remove temporary file from `omnibase_infra/models/infrastructure/`

## 🎯 Impact
This establishes the foundational configuration management pattern for the entire ONEX architecture, ensuring consistent configuration handling across all node types with:
- Standardized environment variable prefixing (`ONEX_INFRA_{NODE_NAME}_`)
- Container service resolution with fallback
- Comprehensive validation with security sanitization
- Proper error handling and detailed messages