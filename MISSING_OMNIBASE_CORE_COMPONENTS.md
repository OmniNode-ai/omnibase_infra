# Missing omnibase_core Components

Based on analysis of imports in omnibase_infra, the following components are expected to exist in omnibase_core but are currently missing:

## Core Error Handling
- `omnibase_core.core.errors.onex_error.OnexError` - Base exception class for ONEX errors
- `omnibase_core.core.core_error_codes.CoreErrorCode` - Error code enumeration
- `omnibase_core.onex_error.OnexError` - Alternative import path for OnexError
- `omnibase_core.core_error_codes.CoreErrorCode` - Alternative import path for CoreErrorCode

## Container and Dependency Injection  
- `omnibase_core.core.onex_container.ModelONEXContainer` - ONEX dependency injection container
- `omnibase_core.onex_container.ModelONEXContainer` - Alternative import path
- `omnibase_core.onex_container.ONEXContainer` - Alternative import name

## Node Architecture
- `omnibase_core.core.node_effect_service.NodeEffectService` - Base class for EFFECT nodes
- `omnibase_core.node_effect_service.NodeEffectService` - Alternative import path
- `omnibase_core.node_effect.NodeEffect` - Base node effect functionality
- `omnibase_core.node_effect.TransactionState` - Transaction state management

## Protocol Layer
- `omnibase_core.protocol.protocol_event_bus.ProtocolEventBus` - Event bus protocol interface
- `omnibase_core.protocol.protocol_schema_loader.ProtocolSchemaLoader` - Schema loader protocol interface

## Core Models
- `omnibase_core.model.core.model_onex_event.ModelOnexEvent` - Base event model
- `omnibase_core.model.core.model_event_envelope.ModelEventEnvelope` - Event envelope model  
- `omnibase_core.model.core.model_health_status.ModelHealthStatus` - Health status model
- `omnibase_core.model.core.model_route_spec.ModelRouteSpec` - Route specification model

## Enumerations
- `omnibase_core.enums.enum_health_status.EnumHealthStatus` - Health status enumeration
- `omnibase_core.enums.enum_core_error_code.CoreErrorCode` - Alternative error code enum path

## Utilities
- `omnibase_core.utils.generation.utility_schema_loader.UtilitySchemaLoader` - Schema loading utility

## Exception Handling (Alternative Paths)
- `omnibase_core.exceptions.base_onex_error.OnexError` - Alternative exception path

## Analysis Notes

### Import Path Inconsistencies
The codebase shows multiple import paths for the same components, suggesting the core library structure is not yet standardized:

1. **OnexError**: Referenced from at least 3 different paths:
   - `omnibase_core.core.errors.onex_error.OnexError`
   - `omnibase_core.onex_error.OnexError` 
   - `omnibase_core.exceptions.base_onex_error.OnexError`

2. **CoreErrorCode**: Referenced from at least 2 different paths:
   - `omnibase_core.core.core_error_codes.CoreErrorCode`
   - `omnibase_core.core_error_codes.CoreErrorCode`

3. **Container**: Multiple names and paths:
   - `omnibase_core.core.onex_container.ModelONEXContainer`
   - `omnibase_core.onex_container.ModelONEXContainer`
   - `omnibase_core.onex_container.ONEXContainer`

### Recommendations for omnibase_core Structure

```
omnibase_core/
├── core/
│   ├── errors/
│   │   └── onex_error.py              # OnexError class
│   ├── onex_container.py              # ModelONEXContainer class
│   ├── node_effect_service.py         # NodeEffectService base class
│   └── core_error_codes.py           # CoreErrorCode enum
├── protocol/
│   ├── protocol_event_bus.py          # ProtocolEventBus interface
│   └── protocol_schema_loader.py      # ProtocolSchemaLoader interface
├── model/
│   └── core/
│       ├── model_onex_event.py        # ModelOnexEvent
│       ├── model_event_envelope.py    # ModelEventEnvelope
│       ├── model_health_status.py     # ModelHealthStatus
│       └── model_route_spec.py        # ModelRouteSpec
├── enums/
│   ├── enum_health_status.py          # EnumHealthStatus
│   └── enum_core_error_code.py        # Alternative CoreErrorCode location
├── utils/
│   └── generation/
│       └── utility_schema_loader.py   # UtilitySchemaLoader
└── node_effect.py                     # NodeEffect base functionality
```

### Priority for Implementation

**Critical (Required for basic functionality):**
1. OnexError and CoreErrorCode - Error handling foundation
2. ModelONEXContainer - Dependency injection core  
3. NodeEffectService - Node architecture base
4. ProtocolEventBus - Event communication

**High (Required for infrastructure nodes):**
5. ModelOnexEvent and ModelEventEnvelope - Event models
6. ModelHealthStatus and EnumHealthStatus - Health monitoring
7. ProtocolSchemaLoader - Schema validation

**Medium (Enhanced functionality):**
8. UtilitySchemaLoader - Utility implementations
9. ModelRouteSpec - Routing specifications
10. NodeEffect and TransactionState - Advanced node features

### Zero Backwards Compatibility Note

When implementing these components in omnibase_core, follow ONEX zero backwards compatibility policy - choose one canonical import path for each component and stick to it rather than supporting multiple import paths.