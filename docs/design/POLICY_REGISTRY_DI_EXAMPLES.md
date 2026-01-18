> **Navigation**: [Home](../index.md) > [Design](README.md) > Policy Registry DI Examples

# PolicyRegistry DI Integration - Code Examples

This document provides concrete code examples for integrating PolicyRegistry with ModelOnexContainer.

## Example 1: Basic Container Bootstrap

```python
# app_bootstrap.py - Application startup
"""Bootstrap ONEX infrastructure with container."""

from omnibase_core.models.container.model_onex_container import ModelOnexContainer
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
from omnibase_infra.enums import EnumPolicyType

# 1. Create container
container = ModelOnexContainer()

# 2. Wire infrastructure services (registers PolicyRegistry, handlers, event buses)
summary = wire_infrastructure_services(container)
print(f"Registered services: {summary['services']}")
# Output: Registered services: ['event_bus_registry', 'handler_registry', 'policy_registry']

# 3. Resolve and use PolicyRegistry
policy_registry = container.resolve("policy_registry")

# 4. Register policies
policy_registry.register_policy(
    policy_id="exponential_backoff",
    policy_class=ExponentialBackoffPolicy,
    policy_type=EnumPolicyType.ORCHESTRATOR,
    version="1.0.0",
)

# 5. Start application with container
app = Application(container)
app.run()
```

## Example 2: Node Using PolicyRegistry via DI

```python
# nodes/infrastructure_orchestrator/v1_0_0/node.py
"""Infrastructure orchestrator node with PolicyRegistry DI."""

from typing import TYPE_CHECKING

from omnibase_core.nodes import NodeEffect, ModelEffectInput, ModelEffectOutput

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.policy_registry import PolicyRegistry


class NodeInfrastructureOrchestratorEffect(NodeEffect):
    """Orchestrator node using PolicyRegistry via container DI.

    This node demonstrates proper DI pattern:
    1. Accept container in __init__
    2. Resolve PolicyRegistry from container
    3. Use registry to get policy instances
    """

    def __init__(self, container: ModelOnexContainer):
        """Initialize with container (DI pattern).

        Args:
            container: ONEX container with registered services.
        """
        super().__init__(container)

        # Resolve PolicyRegistry from container (DI)
        self.policy_registry: PolicyRegistry = container.resolve("policy_registry")

        # Also resolve other services as needed
        self.logger = container.get_tool("LOGGER")

    async def execute_effect(
        self,
        input_data: ModelEffectInput,
    ) -> ModelEffectOutput:
        """Execute orchestration with policy-driven decisions."""
        # 1. Get policy from registry
        retry_policy_cls = self.policy_registry.get(
            policy_id="exponential_backoff",
            policy_type="orchestrator",
        )

        # 2. Instantiate policy
        retry_policy = retry_policy_cls()

        # 3. Use policy to make decisions
        context = {"attempts": 3, "last_error": "connection_timeout"}
        decision = retry_policy.evaluate(context)

        # 4. Execute based on policy decision
        if decision["should_retry"]:
            delay = decision["retry_delay_seconds"]
            self.logger.info(f"Retrying after {delay}s based on policy decision")
            # ... retry logic ...

        return ModelEffectOutput(
            success=True,
            payload={"decision": decision},
        )
```

## Example 3: Service Class with PolicyRegistry

```python
# services/retry_service.py
"""Retry service using PolicyRegistry via DI."""

from typing import TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.policy_registry import PolicyRegistry


class RetryService:
    """Service that uses PolicyRegistry for retry strategies.

    Demonstrates:
    - Constructor injection of container
    - Resolving PolicyRegistry from container
    - Using multiple policies for different scenarios
    """

    def __init__(self, container: ModelOnexContainer):
        """Initialize with container dependency injection.

        Args:
            container: ONEX container with registered PolicyRegistry.
        """
        self.container = container
        self.policy_registry: PolicyRegistry = container.resolve("policy_registry")
        self.logger = container.get_tool("LOGGER")

    async def retry_with_policy(
        self,
        operation: callable,
        policy_id: str = "exponential_backoff",
        max_attempts: int = 3,
    ) -> dict[str, object]:
        """Retry operation using specified policy.

        Args:
            operation: Async callable to retry.
            policy_id: Policy to use for retry decisions.
            max_attempts: Maximum number of attempts.

        Returns:
            Result dict with success/failure info.
        """
        # Resolve policy from registry
        policy_cls = self.policy_registry.get(policy_id)
        policy = policy_cls()

        attempts = 0
        last_error = None

        while attempts < max_attempts:
            try:
                result = await operation()
                return {"success": True, "result": result, "attempts": attempts + 1}

            except Exception as e:
                attempts += 1
                last_error = e

                # Use policy to decide if/how to retry
                context = {
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                    "last_error": str(e),
                    "error_type": type(e).__name__,
                }

                decision = policy.evaluate(context)

                if not decision.get("should_retry", False):
                    self.logger.error(f"Policy decided not to retry: {decision}")
                    break

                delay = decision.get("retry_delay_seconds", 1.0)
                self.logger.info(
                    f"Retrying after {delay}s (attempt {attempts}/{max_attempts})"
                )
                await asyncio.sleep(delay)

        return {
            "success": False,
            "error": str(last_error),
            "attempts": attempts,
        }
```

## Example 4: Testing with Container

```python
# tests/unit/test_retry_service.py
"""Unit tests for RetryService with container-based DI."""

import pytest
from unittest.mock import Mock, AsyncMock

from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.enums import EnumPolicyType
from services.retry_service import RetryService


@pytest.fixture
def mock_container():
    """Mock ModelOnexContainer for testing."""
    container = Mock(spec=["resolve", "get_tool"])

    # Mock policy registry
    policy_registry = PolicyRegistry()
    container.resolve.return_value = policy_registry

    # Mock logger
    logger = Mock()
    container.get_tool.return_value = logger

    return container


@pytest.fixture
def retry_service(mock_container):
    """RetryService with mocked container."""
    return RetryService(mock_container)


@pytest.mark.asyncio
async def test_retry_with_policy_success(retry_service, mock_container):
    """Test successful retry with policy."""
    # Register test policy
    registry = mock_container.resolve.return_value

    class MockRetryPolicy:
        @property
        def policy_id(self) -> str:
            return "test_retry"

        @property
        def policy_type(self) -> EnumPolicyType:
            return EnumPolicyType.ORCHESTRATOR

        def evaluate(self, context):
            return {
                "should_retry": True,
                "retry_delay_seconds": 0.01,  # Fast for testing
            }

    registry.register_policy(
        policy_id="test_retry",
        policy_class=MockRetryPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )

    # Test operation that succeeds on 2nd attempt
    attempts = [0]

    async def flaky_operation():
        attempts[0] += 1
        if attempts[0] < 2:
            raise ConnectionError("Connection failed")
        return {"data": "success"}

    # Execute with retry
    result = await retry_service.retry_with_policy(
        operation=flaky_operation,
        policy_id="test_retry",
        max_attempts=3,
    )

    # Verify success
    assert result["success"] is True
    assert result["attempts"] == 2
    assert result["result"]["data"] == "success"


@pytest.mark.asyncio
async def test_retry_with_policy_failure(retry_service, mock_container):
    """Test retry exhaustion with policy."""
    # Register test policy that stops retrying
    registry = mock_container.resolve.return_value

    class MockNoRetryPolicy:
        @property
        def policy_id(self) -> str:
            return "no_retry"

        @property
        def policy_type(self) -> EnumPolicyType:
            return EnumPolicyType.ORCHESTRATOR

        def evaluate(self, context):
            # Stop retrying after 2 attempts
            return {"should_retry": context["attempts"] < 2}

    registry.register_policy(
        policy_id="no_retry",
        policy_class=MockNoRetryPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )

    # Test operation that always fails
    async def always_fail():
        raise ValueError("Always fails")

    # Execute with retry
    result = await retry_service.retry_with_policy(
        operation=always_fail,
        policy_id="no_retry",
        max_attempts=5,
    )

    # Verify failure
    assert result["success"] is False
    assert result["attempts"] == 2  # Policy stopped after 2
    assert "Always fails" in result["error"]
```

## Example 5: Integration Test with Real Container

```python
# tests/integration/test_policy_registry_integration.py
"""Integration tests with real ModelOnexContainer."""

import pytest


@pytest.mark.integration
def test_full_container_lifecycle():
    """Test PolicyRegistry lifecycle with real container."""
    # Skip if omnibase_core not available
    pytest.importorskip("omnibase_core")

    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
    from omnibase_infra.enums import EnumPolicyType

    # 1. Bootstrap container
    container = ModelOnexContainer()
    wire_infrastructure_services(container)

    # 2. Verify PolicyRegistry registered
    registry = container.resolve("policy_registry")
    assert registry is not None

    # 3. Register policy
    class TestPolicy:
        @property
        def policy_id(self) -> str:
            return "test_policy"

        @property
        def policy_type(self) -> EnumPolicyType:
            return EnumPolicyType.ORCHESTRATOR

        def evaluate(self, context):
            return {"result": "test"}

    registry.register_policy(
        policy_id="test_policy",
        policy_class=TestPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )

    # 4. Verify resolution
    policy_cls = registry.get("test_policy")
    assert policy_cls is TestPolicy

    # 5. Verify singleton per container
    registry2 = container.resolve("policy_registry")
    assert registry is registry2

    # 6. Create second container (isolated)
    container2 = ModelOnexContainer()
    wire_infrastructure_services(container2)
    registry3 = container2.resolve("policy_registry")

    # Verify different instances across containers
    assert registry3 is not registry

    # Verify policy not registered in second container
    assert not registry3.is_registered("test_policy")
```

## Example 6: Standalone Utilities with Container DI

```python
# utility_module.py - Utilities accepting container parameter
"""Standalone utilities using container-based DI.

Per CLAUDE.md policy, no backwards compatibility is maintained.
The singleton pattern has been removed entirely.
All code must use container-based DI.
"""

from omnibase_infra.runtime.container_wiring import get_or_create_policy_registry
from omnibase_infra.enums import EnumPolicyType


def register_default_policies(container: ModelOnexContainer) -> None:
    """Register default policies using container DI.

    Args:
        container: ONEX container instance.
    """
    registry = get_or_create_policy_registry(container)

    registry.register_policy(
        policy_id="exponential_backoff",
        policy_class=ExponentialBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )

    registry.register_policy(
        policy_id="linear_backoff",
        policy_class=LinearBackoffPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )
```

## Example 7: Migration Path

```python
# migration_example.py - Migrating from old to new pattern
"""Example showing migration from singleton to container-based DI."""

# =============================================================================
# BEFORE (Singleton Pattern - Deprecated)
# =============================================================================

from omnibase_infra.runtime.policy_registry import get_policy_registry


class OldRetryService:
    """Old pattern: Uses global singleton."""

    def __init__(self):
        # PROBLEM: Global singleton, hard to test, no isolation
        self.policy_registry = get_policy_registry()

    def retry_operation(self, policy_id):
        policy_cls = self.policy_registry.get(policy_id)
        return policy_cls()


# =============================================================================
# AFTER (Container-Based DI - Preferred)
# =============================================================================

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelOnexContainer
    from omnibase_infra.runtime.policy_registry import PolicyRegistry


class NewRetryService:
    """New pattern: Uses container-based DI."""

    def __init__(self, container: ModelOnexContainer):
        # SOLUTION: Container injection, easy to test, isolated
        self.policy_registry: PolicyRegistry = container.resolve("policy_registry")

    def retry_operation(self, policy_id):
        policy_cls = self.policy_registry.get(policy_id)
        return policy_cls()


# =============================================================================
# MIGRATION: Hybrid Approach (During Transition)
# =============================================================================

from typing import Optional


class HybridRetryService:
    """Hybrid pattern: Supports both old and new patterns during migration."""

    def __init__(
        self,
        container: Optional[ModelOnexContainer] = None,
        registry: Optional[PolicyRegistry] = None,
    ):
        """Initialize with container OR registry.

        Args:
            container: Optional container for DI (preferred).
            registry: Optional registry for direct injection (testing).

        Raises:
            ValueError: If neither container nor registry provided.
        """
        if container is not None:
            # Container-based DI (preferred)
            self.policy_registry = container.resolve("policy_registry")
        elif registry is not None:
            # Direct injection (for testing)
            self.policy_registry = registry
        else:
            # Per CLAUDE.md: no backwards compatibility - container required
            raise ValueError(
                "Either container or registry must be provided. "
                "Singleton pattern has been removed."
            )

    def retry_operation(self, policy_id):
        policy_cls = self.policy_registry.get(policy_id)
        return policy_cls()


# Usage examples:

# New way (preferred)
container = ModelOnexContainer()
wire_infrastructure_services(container)
service1 = HybridRetryService(container=container)

# Migration way (for testing)
test_registry = PolicyRegistry()
service2 = HybridRetryService(registry=test_registry)

# Old way (deprecated, backwards compatible)
service3 = HybridRetryService()  # Falls back to singleton
```

## Example 8: Container Configuration

```python
# config/container_config.py
"""Container configuration for different environments."""

from omnibase_core.models.container.model_onex_container import ModelOnexContainer
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services


def create_development_container() -> ModelOnexContainer:
    """Create container for development environment."""
    container = ModelOnexContainer()

    # Wire infrastructure services
    wire_infrastructure_services(container)

    # Add development-specific config
    container.register("environment", "development")
    container.register("debug_mode", True)

    return container


def create_production_container() -> ModelOnexContainer:
    """Create container for production environment."""
    container = ModelOnexContainer()

    # Wire infrastructure services
    wire_infrastructure_services(container)

    # Add production-specific config
    container.register("environment", "production")
    container.register("debug_mode", False)

    return container


def create_test_container() -> ModelOnexContainer:
    """Create container for testing with mocked services."""
    from unittest.mock import Mock

    container = ModelOnexContainer()

    # Wire infrastructure services
    wire_infrastructure_services(container)

    # Override with test doubles
    mock_logger = Mock()
    container.register("LOGGER", mock_logger)

    return container
```

## Summary

These examples demonstrate:

1. **Container bootstrap** - How to wire PolicyRegistry with container
2. **Node integration** - Using PolicyRegistry in ONEX nodes
3. **Service classes** - Using PolicyRegistry in service classes
4. **Testing** - Unit and integration tests with container
5. **Standalone utilities** - Container DI for utility functions
6. **Configuration** - Environment-specific container setup

**Key Takeaways**:

- ✅ **Always use container injection** for all code
- ✅ **Resolve PolicyRegistry from container** via `container.resolve("policy_registry")`
- ✅ **Test with mock containers** for isolated unit tests
- ✅ **No backwards compatibility** - Singleton pattern has been removed (per CLAUDE.md)
- ✅ **Wire services at startup** via `wire_infrastructure_services(container)`
