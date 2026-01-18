> **Navigation**: [Home](../index.md) > [Design](README.md) > Policy Registry DI Integration

# PolicyRegistry DI Integration Design

## Executive Summary

This document designs the integration of `PolicyRegistry` with ONEX's dependency injection system using `ModelOnexContainer` from `omnibase_core`. The goal is to remove the singleton pattern and use proper container-managed dependency injection while maintaining thread-safety.

## Current Architecture

### Current Implementation (Singleton Pattern)

```python
# PolicyRegistry uses module-level singleton
_policy_registry: Optional[PolicyRegistry] = None
_singleton_lock: threading.Lock = threading.Lock()

def get_policy_registry() -> PolicyRegistry:
    """Get the singleton policy registry instance."""
    global _policy_registry
    if _policy_registry is None:
        with _singleton_lock:
            if _policy_registry is None:
                _policy_registry = PolicyRegistry()
    return _policy_registry
```

### Problems with Current Approach

1. **Not ONEX-compliant**: Violates container-based DI principles
2. **Testing difficulties**: Hard to isolate tests, requires global state cleanup
3. **Lifecycle management**: No clear initialization/shutdown hooks
4. **Configuration injection**: No way to configure registry at startup
5. **Inconsistent with other registries**: `ProtocolBindingRegistry` and `EventBusBindingRegistry` also use singletons but should migrate to container-based DI

## Target Architecture

### Container-Based DI Pattern

Following the ONEX pattern used by nodes (from documentation):

```python
# Node pattern from docs/architecture/CURRENT_NODE_ARCHITECTURE.md
class NodeVaultAdapterEffect(NodeEffectService):
    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)
        # Access container services
        self.logger = container.get_tool("LOGGER") or logging.getLogger(__name__)
```

### ModelOnexContainer API

Based on documentation analysis and import patterns:

```python
# From omnibase_core.models.container.model_onex_container
class ModelOnexContainer:
    """ONEX dependency injection container.

    Provides:
    - Service registration and resolution
    - Configuration management
    - Tool/utility access
    - Lifecycle management
    """

    # Service resolution
    def resolve(self, service_key: str) -> Any: ...
    def get_tool(self, tool_name: str) -> Any: ...
    def get_config(self, config_key: str) -> dict[str, Any]: ...

    # Service registration (typically done at startup)
    def register(self, service_key: str, instance: Any) -> None: ...
    def register_factory(self, service_key: str, factory: Callable) -> None: ...
```

## Proposed Design

### Phase 1: Container-Managed PolicyRegistry

#### 1.1 PolicyRegistry Changes

**Remove singleton pattern, add container support:**

```python
# src/omnibase_infra/runtime/policy_registry.py

class PolicyRegistry:
    """Container-managed policy registry (no singleton pattern).

    Thread-safe registry for policy plugins. Designed to be instantiated
    and managed by ModelOnexContainer.

    Thread Safety:
        All registration operations protected by threading.Lock.
        Safe for concurrent access when managed by container.

    Container Integration:
        - Instantiated during container bootstrap
        - Registered in container under "policy_registry" key
        - Resolved via container.resolve("policy_registry")
        - No global singleton required

    Lifecycle:
        - __init__: Create empty registry with thread lock
        - No initialize() needed (stateless registry)
        - No shutdown() needed (no resources to clean up)
    """

    def __init__(self) -> None:
        """Initialize empty policy registry with thread lock."""
        self._registry: dict[ModelPolicyKey, type[ProtocolPolicy]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._policy_id_index: dict[str, list[ModelPolicyKey]] = {}

    # ... rest of implementation unchanged (register, get, list_keys, etc.)
```

**Remove singleton accessor (per no-backwards-compatibility policy):**

Per CLAUDE.md, all changes are breaking changes and no backwards compatibility is maintained.
The singleton accessor should be removed entirely, not deprecated:

```python
# src/omnibase_infra/runtime/policy_registry.py

# The singleton pattern has been REMOVED (not deprecated).
# Use container-based DI exclusively:

def __init__(self, container: ModelOnexContainer):
    self.policy_registry = container.resolve("policy_registry")
```

#### 1.2 Container Registration

**Create container bootstrap/wiring module:**

```python
# src/omnibase_infra/runtime/container_wiring.py
"""Container wiring for omnibase_infra services.

This module provides functions to register infrastructure services
with ModelOnexContainer. It follows the wiring pattern established
by handler_wiring.py.

Design Principles:
- Explicit registration: All services registered explicitly
- Factory pattern: Use factories for lazy instantiation
- Lifecycle management: Services with init/shutdown hooks
- Testability: Easy to mock services via container

Example Usage:
    ```python
    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

    # Bootstrap container
    container = ModelOnexContainer()

    # Wire infrastructure services
    wire_infrastructure_services(container)

    # Resolve services
    policy_registry = container.resolve("policy_registry")
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.handler_registry import (
    ProtocolBindingRegistry,
    EventBusBindingRegistry,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelOnexContainer

logger = logging.getLogger(__name__)


def wire_infrastructure_services(container: ModelOnexContainer) -> dict[str, list[str]]:
    """Register infrastructure services with the container.

    Registers:
    - PolicyRegistry: policy_registry
    - ProtocolBindingRegistry: handler_registry
    - EventBusBindingRegistry: event_bus_registry

    Args:
        container: ONEX container instance to register services in.

    Returns:
        Summary dict with:
            - services: List of registered service keys

    Example:
        >>> from omnibase_core.models.container.model_onex_container import ModelOnexContainer
        >>> container = ModelOnexContainer()
        >>> summary = wire_infrastructure_services(container)
        >>> print(summary)
        {'services': ['event_bus_registry', 'handler_registry', 'policy_registry']}
    """
    # Register PolicyRegistry (singleton instance per container)
    policy_registry = PolicyRegistry()
    container.register("policy_registry", policy_registry)
    logger.debug("Registered PolicyRegistry in container")

    # Register ProtocolBindingRegistry (singleton instance per container)
    handler_registry = ProtocolBindingRegistry()
    container.register("handler_registry", handler_registry)
    logger.debug("Registered ProtocolBindingRegistry in container")

    # Register EventBusBindingRegistry (singleton instance per container)
    event_bus_registry = EventBusBindingRegistry()
    container.register("event_bus_registry", event_bus_registry)
    logger.debug("Registered EventBusBindingRegistry in container")

    services = ["policy_registry", "handler_registry", "event_bus_registry"]

    logger.info(
        "Infrastructure services wired",
        extra={
            "service_count": len(services),
            "services": sorted(services),
        },
    )

    return {"services": sorted(services)}


def get_or_create_policy_registry(container: ModelOnexContainer) -> PolicyRegistry:
    """Get PolicyRegistry from container, creating if not registered.

    Helper function for lazy initialization.
    Checks if PolicyRegistry is registered in container, and if not,
    creates and registers a new instance.

    Args:
        container: ONEX container instance.

    Returns:
        PolicyRegistry instance from container.

    Example:
        >>> container = ModelOnexContainer()
        >>> registry = get_or_create_policy_registry(container)
        >>> registry is get_or_create_policy_registry(container)
        True
    """
    try:
        return container.resolve("policy_registry")
    except (AttributeError, KeyError, LookupError):
        # PolicyRegistry not registered, create and register it
        policy_registry = PolicyRegistry()
        container.register("policy_registry", policy_registry)
        logger.debug("Auto-registered PolicyRegistry in container (lazy init)")
        return policy_registry


__all__ = [
    "wire_infrastructure_services",
    "get_or_create_policy_registry",
]
```

#### 1.3 Consumer Updates

**Pattern for consumers with container access:**

```python
# Example: Node using PolicyRegistry via DI
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.policy_registry import PolicyRegistry

class NodeInfrastructureOrchestratorEffect:
    """Orchestrator node using PolicyRegistry via container DI."""

    def __init__(self, container: ModelOnexContainer):
        super().__init__(container)

        # Resolve PolicyRegistry from container
        self.policy_registry: PolicyRegistry = container.resolve("policy_registry")

    async def execute_effect(self, input_data: ModelEffectInput) -> ModelEffectOutput:
        # Use policy registry
        policy_cls = self.policy_registry.get("retry_strategy")
        policy = policy_cls()
        decision = policy.evaluate(context)
        ...
```

**All consumers must use container-based DI:**

Per CLAUDE.md, no backwards compatibility is maintained. The singleton pattern has been removed.
All code must use container-based DI:

```python
# Standalone utilities must accept container parameter
from omnibase_infra.runtime.container_wiring import get_or_create_policy_registry

def register_default_policies(container: ModelOnexContainer) -> None:
    """Register default policies using container DI."""
    registry = get_or_create_policy_registry(container)

    registry.register_policy(
        policy_id="exponential_backoff",
        policy_class=ExponentialBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )
```

### Phase 2: Migration Strategy

#### 2.1 Consumers to Migrate

**Identified consumers:**

1. `src/omnibase_infra/runtime/policy_registry.py` - Convenience functions
2. `tests/unit/runtime/test_policy_registry.py` - Test fixtures
3. Any future nodes that use policies

#### 2.2 Migration Steps

**Step 1: Add container wiring (new code):**

```python
# NEW FILE: src/omnibase_infra/runtime/container_wiring.py
# (Implementation shown above in 1.2)
```

**Step 2: Update convenience functions to support container:**

```python
# src/omnibase_infra/runtime/policy_registry.py

# Add container-aware convenience functions
def get_policy_class_from_container(
    container: ModelOnexContainer,
    policy_id: str,
    policy_type: Optional[str | EnumPolicyType] = None,
    version: Optional[str] = None,
) -> type[ProtocolPolicy]:
    """Get policy class from container-managed registry.

    Preferred over get_policy_class() for container-based code.

    Args:
        container: ONEX container with registered PolicyRegistry.
        policy_id: Policy identifier.
        policy_type: Optional policy type filter.
        version: Optional version filter.

    Returns:
        Policy class registered for the configuration.

    Raises:
        PolicyRegistryError: If no matching policy is found.

    Example:
        >>> policy_cls = get_policy_class_from_container(
        ...     container, "exponential_backoff"
        ... )
    """
    registry: PolicyRegistry = container.resolve("policy_registry")
    return registry.get(policy_id, policy_type, version)


def register_policy_in_container(
    container: ModelOnexContainer,
    policy_id: str,
    policy_class: type[ProtocolPolicy],
    policy_type: str | EnumPolicyType,
    version: str = "1.0.0",
    allow_async: bool = False,
) -> None:
    """Register a policy in container-managed registry.

    Preferred over register_policy() for container-based code.

    Args:
        container: ONEX container with registered PolicyRegistry.
        policy_id: Unique identifier for the policy.
        policy_class: The policy class to register.
        policy_type: Whether this is orchestrator or reducer policy.
        version: Semantic version string (default: "1.0.0").
        allow_async: If True, allows async interface.

    Raises:
        PolicyRegistryError: If policy has async methods and
                           allow_async=False.

    Example:
        >>> register_policy_in_container(
        ...     container,
        ...     policy_id="retry_backoff",
        ...     policy_class=RetryBackoffPolicy,
        ...     policy_type="orchestrator",
        ... )
    """
    registry: PolicyRegistry = container.resolve("policy_registry")
    registry.register_policy(
        policy_id=policy_id,
        policy_class=policy_class,
        policy_type=policy_type,
        version=version,
        allow_async=allow_async,
    )


# Update __all__ exports
__all__: list[str] = [
    # ... existing exports ...
    # Container-aware convenience functions
    "get_policy_class_from_container",
    "register_policy_in_container",
]
```

**Step 3: Update tests to use container:**

```python
# tests/unit/runtime/test_policy_registry.py

import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_container():
    """Mock ModelOnexContainer for testing."""
    container = Mock(spec=["resolve", "register"])
    return container

@pytest.fixture
def registry_in_container(mock_container):
    """PolicyRegistry registered in mock container."""
    from omnibase_infra.runtime.policy_registry import PolicyRegistry

    registry = PolicyRegistry()
    mock_container.resolve.return_value = registry
    mock_container.register.return_value = None

    return registry

def test_policy_registration_via_container(mock_container, registry_in_container):
    """Test policy registration through container resolution."""
    from omnibase_infra.runtime.container_wiring import get_or_create_policy_registry

    registry = get_or_create_policy_registry(mock_container)

    registry.register_policy(
        policy_id="test_policy",
        policy_class=MockSyncPolicy,
        policy_type="orchestrator",
        version="1.0.0",
    )

    assert registry.is_registered("test_policy")
```

**Step 4: Deprecation warnings (optional):**

```python
# src/omnibase_infra/runtime/policy_registry.py

import warnings

def get_policy_registry() -> PolicyRegistry:
    """Get the module-level singleton policy registry instance.

    .. deprecated:: 0.2.0
        Use container-based DI instead.
    """
    warnings.warn(
        "get_policy_registry() is deprecated. Use container.resolve('policy_registry') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    global _policy_registry  # noqa: PLW0603
    if _policy_registry is None:
        with _singleton_lock:
            if _policy_registry is None:
                _policy_registry = PolicyRegistry()
    return _policy_registry
```

### Phase 3: Testing Strategy

#### 3.1 Unit Tests

```python
# tests/unit/runtime/test_container_wiring.py
"""Unit tests for container wiring."""

import pytest
from unittest.mock import Mock

from omnibase_infra.runtime.container_wiring import (
    wire_infrastructure_services,
    get_or_create_policy_registry,
)
from omnibase_infra.runtime.policy_registry import PolicyRegistry


def test_wire_infrastructure_services():
    """Test wiring infrastructure services in container."""
    container = Mock(spec=["register"])

    summary = wire_infrastructure_services(container)

    # Verify all services registered
    assert summary["services"] == [
        "event_bus_registry",
        "handler_registry",
        "policy_registry",
    ]

    # Verify container.register called for each service
    assert container.register.call_count == 3


def test_get_or_create_policy_registry_existing():
    """Test retrieving existing PolicyRegistry from container."""
    container = Mock(spec=["resolve"])
    registry = PolicyRegistry()
    container.resolve.return_value = registry

    result = get_or_create_policy_registry(container)

    assert result is registry
    container.resolve.assert_called_once_with("policy_registry")


def test_get_or_create_policy_registry_auto_create():
    """Test auto-creating PolicyRegistry if not in container."""
    container = Mock(spec=["resolve", "register"])
    container.resolve.side_effect = KeyError("policy_registry")

    result = get_or_create_policy_registry(container)

    assert isinstance(result, PolicyRegistry)
    container.register.assert_called_once()
```

#### 3.2 Integration Tests

```python
# tests/integration/test_policy_registry_container_integration.py
"""Integration tests for PolicyRegistry with container."""

import pytest


@pytest.mark.integration
def test_policy_registry_lifecycle_with_container():
    """Test PolicyRegistry lifecycle managed by container."""
    # This test requires actual ModelOnexContainer from omnibase_core
    # Skip if omnibase_core not available in test environment
    pytest.importorskip("omnibase_core")

    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
    from omnibase_infra.enums import EnumPolicyType

    # Bootstrap container
    container = ModelOnexContainer()
    wire_infrastructure_services(container)

    # Resolve registry
    registry = container.resolve("policy_registry")

    # Register policy
    registry.register_policy(
        policy_id="test_policy",
        policy_class=MockSyncPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )

    # Verify resolution
    policy_cls = registry.get("test_policy")
    assert policy_cls is MockSyncPolicy

    # Verify singleton per container
    registry2 = container.resolve("policy_registry")
    assert registry is registry2
```

### Phase 4: Documentation Updates

#### 4.1 CLAUDE.md Updates

```markdown
## 🎯 Core ONEX Principles

### Container-Based Dependency Injection

- **ModelOnexContainer** - All services managed by container
- **No singletons** - Use container.resolve() instead of get_*_registry()
- **Constructor injection** - `def __init__(self, container: ModelOnexContainer)`
- **Service registration** - Bootstrap via container_wiring.wire_infrastructure_services()

### Registry Naming Conventions

**Standalone Registries** (in runtime directories):
- File: `registry_<purpose>.py`
- Class: `Registry<Purpose>`
- Examples:
  - `registry_handler.py` → `ProtocolBindingRegistry` (registered as "handler_registry")
  - `registry_policy.py` → `PolicyRegistry` (registered as "policy_registry")
  - Container registration: `container.register("policy_registry", PolicyRegistry())`
```

#### 4.2 README Updates

```markdown
## Quick Start

### Container Bootstrap

```python
from omnibase_core.models.container.model_onex_container import ModelOnexContainer
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

# Create container
container = ModelOnexContainer()

# Wire infrastructure services
wire_infrastructure_services(container)

# Resolve services
policy_registry = container.resolve("policy_registry")
```

### Registering Policies

```python
# Container-based (preferred)
registry = container.resolve("policy_registry")
registry.register_policy(
    policy_id="exponential_backoff",
    policy_class=ExponentialBackoffPolicy,
    policy_type=EnumPolicyType.ORCHESTRATOR,
    version="1.0.0",
)

# NOTE: Singleton pattern has been REMOVED (not deprecated).
# Per CLAUDE.md, no backwards compatibility is maintained.
# All code must use container-based DI.
```

## Benefits of Container-Based Approach

### 1. **Testability**

- **Isolated tests**: Each test gets fresh container instance
- **Easy mocking**: Mock services via container.register()
- **No global state**: No need to reset singletons between tests

### 2. **Lifecycle Management**

- **Clear initialization**: Services wired during container bootstrap
- **Shutdown hooks**: Container can manage cleanup
- **Configuration injection**: Pass config via container.get_config()

### 3. **ONEX Compliance**

- **Consistent pattern**: Same DI pattern as nodes and services
- **Protocol-based**: Services resolved via protocol interfaces
- **Container-managed**: All services managed by ModelOnexContainer

### 4. **Flexibility**

- **Multiple containers**: Different configs for different environments
- **Service overrides**: Easy to replace services for testing
- **Lazy initialization**: Services created only when resolved

## Implementation Checklist

### Required Changes (PR for ONM-812)

1. ✅ Add `container_wiring.py` with `wire_infrastructure_services()`
2. ✅ Add container-aware convenience functions
3. ✅ Document container-based pattern in docstrings
4. ✅ Remove singleton pattern entirely (per no-backwards-compatibility policy)

### Follow-up Tasks

1. Update test fixtures to use container
2. Update examples and documentation
3. Add integration tests with real ModelOnexContainer
4. Update all consumers to container-based DI
5. Update `ProtocolBindingRegistry` and `EventBusBindingRegistry` to use container

## Risks and Mitigations

### Risk: Container Not Available

**Mitigation**: Provide `get_or_create_policy_registry(container)` helper that auto-registers if needed.

### Risk: Test Complexity

**Mitigation**: Provide mock container fixtures in conftest.py for easy testing.

### Risk: Documentation Drift

**Mitigation**: Update all documentation in same PR. Add examples to docstrings.

## Open Questions

1. **Container API surface**: Need to verify exact API of ModelOnexContainer (resolve, register, get_config methods)
2. **Container lifecycle**: Who creates and manages container instances? RuntimeHostProcess? Kernel?
3. **Configuration injection**: How should PolicyRegistry receive configuration (if needed)?
4. **Other registries**: Should ProtocolBindingRegistry and EventBusBindingRegistry also migrate in this PR?

## Conclusion

This design provides a clear path to migrate PolicyRegistry from singleton pattern to container-based DI. Per CLAUDE.md policy, all changes are breaking changes and no backwards compatibility is maintained. The singleton pattern should be removed entirely.

**Recommendation**: Implement container wiring in ONM-812 and remove the singleton pattern immediately.
