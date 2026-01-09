# Runtime Type Validation Implementation Plan

## Summary
Adding runtime type validation to registry classes to ensure registered classes implement required protocols.

## Files to Modify

### 1. `registry_protocol_binding.py`
- Add `_validate_handler_protocol()` method
- Check for `handle()` method existence and callability
- Call validation in `register()` method

### 2. `registry_event_bus_binding.py`
- Add `_validate_event_bus_protocol()` method
- Check for `publish()` or `publish_envelope()` methods
- Call validation in `register()` method

### 3. `policy_registry.py`
- Add `_validate_policy_protocol()` method
- Check for `policy_id`, `policy_type`, `evaluate()` properties/methods
- Call validation in `register()` method (in addition to existing `_validate_sync_enforcement`)

### 4. `registry_compute.py`
- Add `_validate_compute_protocol()` method
- Check for `execute()` method
- Call validation in `register()` method (in addition to existing `_validate_sync_enforcement`)

## Validation Strategy
- Use duck typing with `hasattr()` and `callable()` checks
- Provide descriptive error messages listing missing methods
- No performance impact on valid registrations (fast validation)
