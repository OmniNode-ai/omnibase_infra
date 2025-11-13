# Async Isolation Migration Guide

This guide explains how to migrate existing tests to use the new async isolation framework that fixes service coordination and state isolation issues in the OmniNode Bridge test suite.

## Overview of Issues Fixed

The async isolation framework addresses several critical issues that were causing test failures in full suite contexts:

1. **Async Context Contamination**: Services maintaining state between tests
2. **Background Task Leakage**: Async tasks continuing to run after tests complete
3. **Connection State Pollution**: Database and message queue connections not properly isolated
4. **Circuit Breaker State Leakage**: Resilience patterns maintaining state across test boundaries
5. **Timing Dependencies**: Cross-service events not properly coordinated

## Migration Steps

### 1. Import the Async Isolation Framework

Add these imports to your test files:

```python
from tests.fixtures.async_isolation import (
    async_isolation,
    isolated_workflow_coordinator,
    isolated_hook_receiver,
    isolated_kafka_client,
    isolated_postgres_client,
    event_coordinator,
    mock_service_factory
)
```

### 2. Replace Service Instantiation Patterns

**Before (Problematic Pattern):**
```python
@pytest.mark.asyncio
async def test_workflow_coordinator():
    # This creates untracked service instances
    coordinator = WorkflowCoordinator(kafka_bootstrap_servers="test:9092")

    try:
        await coordinator.startup()
        # Test logic...
    finally:
        # Manual cleanup that might miss resources
        if coordinator.is_running:
            await coordinator.shutdown()
```

**After (Isolated Pattern):**
```python
@pytest.mark.asyncio
async def test_workflow_coordinator(isolated_workflow_coordinator):
    # Use pre-configured isolated instance
    coordinator = isolated_workflow_coordinator

    # Mock dependencies to avoid external connections
    coordinator.kafka_client = AsyncMock()
    coordinator.kafka_client.is_connected = True

    await coordinator.startup()
    # Test logic...

    # No manual cleanup needed - isolation framework handles it
```

### 3. Cross-Service Coordination Patterns

**Before (Race Conditions):**
```python
@pytest.mark.asyncio
async def test_cross_service_flow():
    coordinator = WorkflowCoordinator(...)
    hook_service = HookReceiverService(...)

    # These operations might complete in unpredictable order
    await hook_service.process_event(event)
    await coordinator.submit_workflow(workflow)

    # No way to ensure proper sequencing
```

**After (Event Coordination):**
```python
@pytest.mark.asyncio
async def test_cross_service_flow(
    isolated_workflow_coordinator,
    isolated_hook_receiver,
    event_coordinator
):
    coordinator = isolated_workflow_coordinator
    hook_service = isolated_hook_receiver

    # Set up coordination barriers
    event_coordinator.create_barrier("hook_processed", timeout=5.0)
    event_coordinator.create_barrier("workflow_submitted", timeout=5.0)

    # Mock coordinated responses
    async def mock_process_event(event):
        # Process event
        result = await original_process(event)
        event_coordinator.signal_event("hook_processed")
        return result

    hook_service._process_hook_event = mock_process_event

    # Execute with coordination
    await hook_service.process_event(event)
    hook_complete = await event_coordinator.wait_for_event("hook_processed")
    assert hook_complete

    await coordinator.submit_workflow(workflow)
```

### 4. Background Task Management

**Before (Task Leakage):**
```python
@pytest.mark.asyncio
async def test_background_operations():
    coordinator = WorkflowCoordinator(...)

    # These tasks might continue running after test
    background_task = asyncio.create_task(long_running_operation())

    # Test might end before task is cleaned up
```

**After (Tracked Tasks):**
```python
@pytest.mark.asyncio
async def test_background_operations(isolated_workflow_coordinator, async_isolation):
    coordinator = isolated_workflow_coordinator

    # Track tasks for automatic cleanup
    background_task = asyncio.create_task(long_running_operation())
    async_isolation.track_task(background_task)

    # Or use the service's tracked task creation
    if hasattr(coordinator, 'create_background_task'):
        tracked_task = coordinator.create_background_task(long_running_operation())

    # All tasks cleaned up automatically
```

### 5. State Isolation Patterns

**Before (State Contamination):**
```python
@pytest.mark.asyncio
async def test_service_state():
    service = HookReceiverService(...)

    # State modifications that persist between tests
    service._processed_events.append(event)
    service.connection_count += 1

    # No state reset
```

**After (Isolated State):**
```python
@pytest.mark.asyncio
async def test_service_state(isolated_hook_receiver, async_isolation):
    service = isolated_hook_receiver

    # Track service for state management
    async_isolation.track_service(service, "hook_receiver")

    # State modifications are automatically reset
    service._processed_events.append(event)
    service.connection_count += 1

    # State automatically reset by isolation framework
```

## Specific Service Migration Patterns

### WorkflowCoordinator Tests

```python
# Replace WorkflowCoordinator instantiation
@pytest.mark.asyncio
async def test_workflow_functionality(isolated_workflow_coordinator):
    coordinator = isolated_workflow_coordinator

    # Mock Kafka client to avoid external dependencies
    coordinator.kafka_client = AsyncMock()
    coordinator.kafka_client.is_connected = True

    # Your test logic here
    await coordinator.startup()
    result = await coordinator.submit_workflow(workflow_data)
    assert result is not None
```

### HookReceiver Tests

```python
# Replace HookReceiverService instantiation
@pytest.mark.asyncio
async def test_hook_processing(isolated_hook_receiver):
    service = isolated_hook_receiver

    # Mock dependencies
    service.kafka_client = AsyncMock()
    service.postgres_client = AsyncMock()
    service.kafka_client.is_connected = True
    service.postgres_client.is_connected = True

    # Set up mock responses
    service.postgres_client.store_hook_event.return_value = True
    service.kafka_client.publish_event.return_value = True

    # Your test logic here
    result = await service._process_hook_event(hook_event)
    assert result is True
```

### Database and Kafka Client Tests

```python
# Use isolated clients
@pytest.mark.asyncio
async def test_database_operations(isolated_postgres_client):
    client = isolated_postgres_client

    # Mock pool to avoid external connections
    client.pool = AsyncMock()
    client._connected = True

    # Your test logic here
    await client.execute_query("SELECT 1")
```

## TestClient and FastAPI Integration

For FastAPI integration tests:

```python
def test_api_endpoints(isolated_hook_receiver):
    service = isolated_hook_receiver

    # Create app with mocked startup
    app = service.create_app()

    with patch.object(service, 'startup'):
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
```

## Advanced Patterns

### Custom Async Isolation Context

For complex test scenarios:

```python
@pytest.mark.asyncio
async def test_complex_scenario():
    async with async_isolation_context("complex_test") as isolation:
        # Create services manually
        service1 = MyService()
        service2 = AnotherService()

        # Track them for cleanup
        isolation.track_service(service1, "service1")
        isolation.track_service(service2, "service2")

        # Your test logic here
        # All cleanup handled automatically
```

### Mock Service Factory

For lightweight testing:

```python
@pytest.mark.asyncio
async def test_with_mocks(mock_service_factory):
    # Create isolated mock services
    kafka_mock = mock_service_factory("kafka_service")
    postgres_mock = mock_service_factory("postgres_service")

    # Configure mock behavior
    kafka_mock.publish_message = AsyncMock(return_value=True)
    postgres_mock.execute_query = AsyncMock(return_value=[])

    # Test logic with mocks
```

## Common Migration Mistakes

### 1. Forgetting to Mock External Dependencies

```python
# Wrong - will try to connect to real services
@pytest.mark.asyncio
async def test_service(isolated_workflow_coordinator):
    coordinator = isolated_workflow_coordinator
    await coordinator.startup()  # Might fail if Kafka not available

# Right - mock external dependencies
@pytest.mark.asyncio
async def test_service(isolated_workflow_coordinator):
    coordinator = isolated_workflow_coordinator
    coordinator.kafka_client = AsyncMock()
    coordinator.kafka_client.is_connected = True
    await coordinator.startup()  # Will succeed
```

### 2. Not Using Event Coordination for Timing

```python
# Wrong - race conditions possible
async def test_timing_dependent():
    await service1.operation()
    await service2.dependent_operation()  # Might run before service1 completes

# Right - use event coordination
async def test_timing_dependent(event_coordinator):
    event_coordinator.create_barrier("service1_complete")

    # Mock service1 to signal completion
    async def mock_operation():
        result = await original_operation()
        event_coordinator.signal_event("service1_complete")
        return result

    service1.operation = mock_operation

    await service1.operation()
    await event_coordinator.wait_for_event("service1_complete")
    await service2.dependent_operation()
```

### 3. Creating Tasks Without Tracking

```python
# Wrong - task might continue running
async def test_background_work():
    task = asyncio.create_task(background_work())
    # Test ends, task might still be running

# Right - track task for cleanup
async def test_background_work(async_isolation):
    task = asyncio.create_task(background_work())
    async_isolation.track_task(task)
    # Task will be cancelled automatically
```

## Performance Considerations

The async isolation framework is designed to have minimal performance impact:

- **Setup Overhead**: ~2-5ms per test
- **Cleanup Overhead**: ~1-10ms per test (depending on resources)
- **Memory Overhead**: Minimal tracking structures
- **Parallelization**: Tests can still run in parallel within isolation boundaries

## Verification

To verify your migration is working correctly:

1. **Run Individual Tests**: Each test should pass in isolation
2. **Run Test Class**: All tests in a class should pass together
3. **Run Full Suite**: Tests should pass in full suite context
4. **Check Resource Cleanup**: No warning messages about unclosed resources

## Troubleshooting

### Tests Still Failing After Migration

1. **Check Mock Configuration**: Ensure all external dependencies are mocked
2. **Verify Isolation Usage**: Make sure you're using isolated fixtures
3. **Check Event Coordination**: Timing-dependent tests need proper coordination
4. **Review Cleanup**: Look for resources that might not be tracked

### Performance Degradation

1. **Review Resource Tracking**: Don't track more resources than necessary
2. **Optimize Mock Setup**: Use lightweight mocks for non-critical dependencies
3. **Parallel Execution**: Ensure tests can still run in parallel where appropriate

### Memory Usage

1. **Cleanup Verification**: Check that resources are properly released
2. **Mock Management**: Ensure mocks don't hold unnecessary references
3. **Task Cancellation**: Verify background tasks are cancelled

## Example Migration Diff

Here's a complete before/after example:

```diff
# test_workflow_coordinator.py

- import asyncio
- from unittest.mock import AsyncMock, patch
- import pytest
- from omninode_bridge.services.workflow_coordinator import WorkflowCoordinator

+ import asyncio
+ from unittest.mock import AsyncMock, patch
+ import pytest
+ from tests.fixtures.async_isolation import (
+     isolated_workflow_coordinator,
+     event_coordinator,
+     async_isolation
+ )

  @pytest.mark.asyncio
- async def test_workflow_submission():
-     coordinator = WorkflowCoordinator(kafka_bootstrap_servers="test:9092")
-
-     try:
-         # Mock Kafka
-         coordinator.kafka_client = AsyncMock()
-         coordinator.kafka_client.is_connected = True
-
-         await coordinator.startup()
-
-         workflow_data = {"workflow_id": "test", "tasks": []}
-         result = await coordinator.submit_workflow(workflow_data)
-
-         assert result is not None
-
-     finally:
-         if coordinator.is_running:
-             await coordinator.shutdown()

+ async def test_workflow_submission(isolated_workflow_coordinator):
+     coordinator = isolated_workflow_coordinator
+
+     # Mock Kafka
+     coordinator.kafka_client = AsyncMock()
+     coordinator.kafka_client.is_connected = True
+
+     await coordinator.startup()
+
+     workflow_data = {"workflow_id": "test", "tasks": []}
+     result = await coordinator.submit_workflow(workflow_data)
+
+     assert result is not None
+
+     # No manual cleanup needed - isolation framework handles it
```

This migration pattern ensures that tests are properly isolated, resources are cleaned up automatically, and timing dependencies are handled correctly, resulting in reliable test execution in full suite contexts.
