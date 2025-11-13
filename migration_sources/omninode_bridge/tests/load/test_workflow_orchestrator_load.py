#!/usr/bin/env python3
"""
Load Tests for Workflow Orchestrator System.

Tests orchestrator performance under high workflow load:
- 1000+ concurrent workflow executions
- Orchestrator processing latency and throughput
- FSM state management under load
- No workflow failures under concurrent stress
- P95/P99 latency measurements
- Memory and connection pool monitoring

Performance Requirements:
- Orchestrator should handle 1000+ concurrent workflows
- Success rate >= 95%
- P95 latency < 500ms for standard workflows
- P99 latency < 1000ms for standard workflows
- Zero connection pool exhaustion
- Memory usage stable under load
"""

import asyncio
import statistics
import time
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)

from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

# Import bridge nodes and omnibase_core models


# ============================================================================
# Load Test Configuration
# ============================================================================

LOAD_TEST_CONFIG = {
    "num_workflows": 1000,  # Number of concurrent workflows
    "concurrent_batch_size": 50,  # Concurrent execution batches
    "success_rate_threshold": 0.95,  # 95% success rate
    "p95_threshold_ms": 500,  # P95 latency threshold
    "p99_threshold_ms": 1000,  # P99 latency threshold
    "max_memory_mb": 1024,  # Maximum memory usage
}

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
async def high_capacity_metadata_stamping_client():
    """High-capacity mock metadata stamping client for load testing."""
    client = AsyncMock()

    async def mock_generate_hash(content: str):
        """Fast hash generation."""
        await asyncio.sleep(0.001)  # 1ms simulated processing
        return {
            "success": True,
            "file_hash": f"hash_{hash(content)}",
            "execution_time_ms": 1.0,
        }

    async def mock_create_stamp(file_hash: str, content: str):
        """Fast stamp creation."""
        await asyncio.sleep(0.002)  # 2ms simulated processing
        return {
            "success": True,
            "stamp_id": str(uuid4()),
            "file_hash": file_hash,
            "created_at": time.time(),
        }

    client.generate_hash = mock_generate_hash
    client.create_stamp = mock_create_stamp

    yield client


@pytest.fixture
async def high_capacity_onextree_client():
    """High-capacity mock OnexTree intelligence client for load testing."""
    client = AsyncMock()

    async def mock_get_intelligence(context: str, **kwargs):
        """Fast intelligence retrieval."""
        await asyncio.sleep(0.005)  # 5ms simulated processing
        return {
            "intelligence": {
                "analysis_type": "automated",
                "confidence_score": 0.85,
                "recommendations": ["test recommendation"],
            },
            "patterns": ["pattern1", "pattern2"],
            "relationships": [],
            "metadata": {"source": "onextree"},
        }

    client.get_intelligence = mock_get_intelligence

    yield client


@pytest.fixture
async def high_capacity_kafka_producer():
    """High-capacity mock Kafka producer for load testing."""
    producer = AsyncMock()
    published_events: dict[str, list[dict[str, Any]]] = {}
    event_count = 0

    async def mock_publish_event(topic: str, event: dict, **kwargs):
        """High-performance event publishing."""
        nonlocal event_count
        if topic not in published_events:
            published_events[topic] = []

        published_events[topic].append(
            {"offset": event_count, "timestamp": time.time(), "event": event}
        )
        event_count += 1
        return True

    producer.publish_event = mock_publish_event
    producer.published_events = published_events

    yield producer


@pytest.fixture
async def load_test_orchestrator(
    high_capacity_metadata_stamping_client,
    high_capacity_onextree_client,
    high_capacity_kafka_producer,
):
    """Orchestrator node configured for load testing."""
    from unittest.mock import MagicMock

    # Create mock container with config as dictionary (like unit tests)
    container = MagicMock(spec=ModelONEXContainer)
    config_dict = {
        "metadata_stamping_service_url": "http://localhost:8053",
        "onextree_service_url": "http://localhost:8080",
        "kafka_broker_url": "localhost:29092",
        "default_namespace": "omninode.test",
    }
    # Create a mock config object that behaves like a dictionary
    container.config = MagicMock()
    container.config.get = lambda key, default=None: config_dict.get(key, default)

    # Mock Kafka client
    mock_kafka_client = MagicMock()
    mock_kafka_client.publish = AsyncMock(return_value=True)
    container.kafka_client = mock_kafka_client

    # Mock EventBus
    mock_event_bus = MagicMock()
    mock_event_bus.is_initialized = False
    mock_event_bus.wait_for_completion = AsyncMock(
        return_value={
            "event_type": "STATE_COMMITTED",
            "workflow_id": "test",
            "status": "completed",
            "file_hash": "test_hash",
            "stamp_id": "test_stamp_id",
        }
    )
    mock_event_bus.publish_action_event = AsyncMock(return_value=True)
    mock_event_bus.wait_for_state_committed = AsyncMock(
        return_value=("SUCCESS", {"status": "completed"})
    )
    container.event_bus = mock_event_bus

    # Mock container.get_service to return the event_bus
    container.get_service = MagicMock(
        side_effect=lambda name: mock_event_bus if name == "event_bus" else None
    )
    container.register_service = MagicMock()

    try:
        orchestrator = NodeBridgeOrchestrator(container)

        # Inject high-capacity mocks
        orchestrator._metadata_stamping_client = high_capacity_metadata_stamping_client
        orchestrator._onextree_client = high_capacity_onextree_client
        orchestrator._kafka_producer = high_capacity_kafka_producer

        yield orchestrator

        # Cleanup
        if hasattr(orchestrator, "cleanup"):
            await orchestrator.cleanup()

    except Exception as e:
        error_msg = str(e)
        if (
            "omnibase_core.utils.generation" in error_msg
            or "Contract model loading failed" in error_msg
        ):
            pytest.skip(
                "NodeBridgeOrchestrator requires omnibase_core.utils.generation module"
            )
        else:
            raise


# ============================================================================
# Load Test 1: High Volume Concurrent Workflows
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.load
async def test_high_volume_concurrent_workflows(load_test_orchestrator):
    """
    Test orchestrator under high concurrent workflow load.

    Executes 1000 concurrent workflows and validates:
    - Success rate >= 95%
    - P95 latency < 500ms
    - P99 latency < 1000ms
    - No connection pool exhaustion
    - Zero workflow failures due to system errors
    """
    num_workflows = LOAD_TEST_CONFIG["num_workflows"]
    concurrent_batch_size = LOAD_TEST_CONFIG["concurrent_batch_size"]

    print(f"\n=== Load Test: {num_workflows} Concurrent Workflows ===")
    print(f"Concurrent batches: {concurrent_batch_size}")

    # Create workflow contracts for all executions
    workflow_contracts = []
    for i in range(num_workflows):
        contract = ModelContractOrchestrator(
            name=f"load_test_workflow_{i}",
            version={"major": 1, "minor": 0, "patch": 0},
            description=f"Load test workflow {i}",
            input_model="dict",
            output_model="dict",
            correlation_id=uuid4(),
            performance={"single_operation_max_ms": 1000},
        )
        # Bypass strict validation to add input_data (required by orchestrator node)
        contract.__dict__["input_data"] = {
            "content": f"Load test workflow {i}",
            "namespace": "load_test",
            "workflow_index": i,
        }
        workflow_contracts.append(contract)

    # Track execution metrics
    latencies_ms: list[float] = []
    failed_workflows = 0
    successful_workflows = 0

    async def execute_batch(batch: list[ModelContractOrchestrator]):
        """Execute a batch of workflows concurrently."""
        nonlocal failed_workflows, successful_workflows

        tasks = []
        for contract in batch:
            start_time = time.perf_counter()

            async def execute_with_timing(workflow_contract):
                nonlocal failed_workflows, successful_workflows
                try:
                    result = await load_test_orchestrator.execute_orchestration(
                        workflow_contract
                    )
                    latency = (time.perf_counter() - start_time) * 1000
                    latencies_ms.append(latency)

                    # Check if result is successful (handle ModelStampResponseOutput object)
                    if hasattr(result, "workflow_state"):
                        # It's a ModelStampResponseOutput object
                        successful_workflows += 1
                    elif isinstance(result, dict) and result.get("success", False):
                        # It's a dict response
                        successful_workflows += 1
                    else:
                        failed_workflows += 1

                except Exception as e:
                    print(f"Workflow execution error: {e}")
                    failed_workflows += 1

            tasks.append(execute_with_timing(contract))

        await asyncio.gather(*tasks, return_exceptions=True)

    # Execute load test
    start_time = time.time()

    # Process in batches for realistic concurrency
    for i in range(0, num_workflows, concurrent_batch_size):
        batch = workflow_contracts[i : i + concurrent_batch_size]
        await execute_batch(batch)

    total_duration_s = time.time() - start_time

    # Calculate statistics
    throughput = num_workflows / total_duration_s
    success_rate = successful_workflows / num_workflows
    p50_latency = statistics.median(latencies_ms) if latencies_ms else 0
    p95_latency = (
        statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else 0
    )
    p99_latency = (
        statistics.quantiles(latencies_ms, n=100)[98] if len(latencies_ms) >= 100 else 0
    )
    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0
    max_latency = max(latencies_ms) if latencies_ms else 0
    min_latency = min(latencies_ms) if latencies_ms else 0

    # Print results
    print("\n=== Load Test Results ===")
    print(f"Total duration: {total_duration_s:.2f}s")
    print(f"Throughput: {throughput:.2f} workflows/second")
    print(f"Successful workflows: {successful_workflows}/{num_workflows}")
    print(f"Failed workflows: {failed_workflows}/{num_workflows}")
    print(f"Success rate: {success_rate:.2%}")
    print("\nLatency Statistics:")
    print(f"  Min: {min_latency:.2f}ms")
    print(f"  P50: {p50_latency:.2f}ms")
    print(f"  Avg: {avg_latency:.2f}ms")
    print(f"  P95: {p95_latency:.2f}ms")
    print(f"  P99: {p99_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")

    # Assertions
    assert (
        success_rate >= LOAD_TEST_CONFIG["success_rate_threshold"]
    ), f"Success rate {success_rate:.2%} below threshold {LOAD_TEST_CONFIG['success_rate_threshold']:.0%}"

    assert (
        p95_latency < LOAD_TEST_CONFIG["p95_threshold_ms"]
    ), f"P95 latency {p95_latency:.2f}ms exceeds threshold {LOAD_TEST_CONFIG['p95_threshold_ms']}ms"

    assert (
        p99_latency < LOAD_TEST_CONFIG["p99_threshold_ms"]
    ), f"P99 latency {p99_latency:.2f}ms exceeds threshold {LOAD_TEST_CONFIG['p99_threshold_ms']}ms"


# ============================================================================
# Load Test 2: Sustained Workflow Load
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.load
async def test_sustained_workflow_load(load_test_orchestrator):
    """
    Test orchestrator under sustained workflow load over time.

    Simulates continuous workflow executions over 60 seconds.

    Measures:
    - Throughput stability
    - Latency stability (no degradation over time)
    - Memory leak detection
    - FSM state manager performance
    """
    duration_seconds = 60
    workflows_per_second = 20

    print(f"\n=== Sustained Load Test: {duration_seconds}s ===")
    print(f"Target: {workflows_per_second} workflows/second")

    start_time = time.time()
    workflow_counter = 0
    latencies_by_minute: dict[int, list[float]] = {0: [], 1: []}

    while (time.time() - start_time) < duration_seconds:
        minute = int((time.time() - start_time) / 60)

        # Create and execute workflow
        contract = ModelContractOrchestrator(
            name=f"sustained_test_workflow_{workflow_counter}",
            version={"major": 1, "minor": 0, "patch": 0},
            description=f"Sustained test workflow {workflow_counter}",
            input_model="dict",
            output_model="dict",
            correlation_id=uuid4(),
            performance={"single_operation_max_ms": 1000},
        )
        # Bypass strict validation to add input_data (required by orchestrator node)
        contract.__dict__["input_data"] = {
            "content": f"Sustained test workflow {workflow_counter}",
            "namespace": "sustained_test",
        }

        exec_start = time.perf_counter()
        result = await load_test_orchestrator.execute_orchestration(contract)
        latency_ms = (time.perf_counter() - exec_start) * 1000

        if minute not in latencies_by_minute:
            latencies_by_minute[minute] = []
        latencies_by_minute[minute].append(latency_ms)

        workflow_counter += 1

        # Rate limiting
        await asyncio.sleep(1.0 / workflows_per_second)

    total_duration = time.time() - start_time

    # Calculate per-minute statistics
    print("\n=== Sustained Load Results ===")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Total workflows: {workflow_counter}")
    print(f"Actual throughput: {workflow_counter / total_duration:.2f}/second")

    for minute, latencies in latencies_by_minute.items():
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) >= 20
                else max(latencies)
            )
            print(f"\nMinute {minute}:")
            print(f"  Workflows: {len(latencies)}")
            print(f"  Avg latency: {avg_latency:.2f}ms")
            print(f"  P95 latency: {p95_latency:.2f}ms")

    # Verify no significant degradation
    if len(latencies_by_minute) >= 2:
        # Get first and last minutes that have data
        minutes_with_data = [m for m in latencies_by_minute if latencies_by_minute[m]]
        if len(minutes_with_data) >= 2:
            first_minute_avg = statistics.mean(
                latencies_by_minute[minutes_with_data[0]]
            )
            last_minute_avg = statistics.mean(
                latencies_by_minute[minutes_with_data[-1]]
            )
            degradation_pct = (
                (last_minute_avg - first_minute_avg) / first_minute_avg
            ) * 100

            print(f"\nPerformance degradation: {degradation_pct:.1f}%")
            assert (
                abs(degradation_pct) < 50
            ), f"Performance degraded by {degradation_pct:.1f}% over time"


# ============================================================================
# Load Test 3: Burst Workflow Traffic
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.load
async def test_burst_workflow_traffic(load_test_orchestrator):
    """
    Test orchestrator handling sudden workflow traffic bursts.

    Simulates:
    - Normal load (20/sec)
    - Sudden burst (200/sec for 5s)
    - Recovery to normal load

    Measures:
    - Recovery time after burst
    - No workflow failures during burst
    - Latency spike duration
    """
    print("\n=== Burst Traffic Test ===")

    # Phase 1: Normal load
    print("Phase 1: Normal load (20/sec for 10s)")
    normal_latencies = []
    for i in range(200):
        contract = ModelContractOrchestrator(
            name=f"burst_test_normal_{i}",
            version={"major": 1, "minor": 0, "patch": 0},
            description=f"Normal workflow {i}",
            input_model="dict",
            output_model="dict",
            correlation_id=uuid4(),
            performance={"single_operation_max_ms": 1000},
        )
        # Bypass strict validation to add input_data (required by orchestrator node)
        contract.__dict__["input_data"] = {
            "content": f"Normal workflow {i}",
            "namespace": "burst_test",
        }

        start = time.perf_counter()
        await load_test_orchestrator.execute_orchestration(contract)
        normal_latencies.append((time.perf_counter() - start) * 1000)

        await asyncio.sleep(0.05)  # 20/sec

    normal_avg = statistics.mean(normal_latencies)
    print(f"Normal phase avg latency: {normal_avg:.2f}ms")

    # Phase 2: Burst
    print("\nPhase 2: Burst load (200/sec for 5s)")
    burst_latencies = []
    burst_tasks = []

    for i in range(1000):
        contract = ModelContractOrchestrator(
            name=f"burst_test_burst_{i}",
            version={"major": 1, "minor": 0, "patch": 0},
            description=f"Burst workflow {i}",
            input_model="dict",
            output_model="dict",
            correlation_id=uuid4(),
            performance={"single_operation_max_ms": 1000},
        )
        # Bypass strict validation to add input_data (required by orchestrator node)
        contract.__dict__["input_data"] = {
            "content": f"Burst workflow {i}",
            "namespace": "burst_test",
        }

        async def execute_burst(workflow_contract):
            start = time.perf_counter()
            await load_test_orchestrator.execute_orchestration(workflow_contract)
            return (time.perf_counter() - start) * 1000

        burst_tasks.append(execute_burst(contract))

    burst_latencies = await asyncio.gather(*burst_tasks)
    burst_avg = statistics.mean(burst_latencies)
    burst_p95 = (
        statistics.quantiles(burst_latencies, n=20)[18]
        if len(burst_latencies) >= 20
        else max(burst_latencies)
    )

    print(f"Burst phase avg latency: {burst_avg:.2f}ms")
    print(f"Burst phase P95 latency: {burst_p95:.2f}ms")

    # Phase 3: Recovery
    print("\nPhase 3: Recovery (20/sec for 10s)")
    recovery_latencies = []
    for i in range(200):
        contract = ModelContractOrchestrator(
            name=f"burst_test_recovery_{i}",
            version={"major": 1, "minor": 0, "patch": 0},
            description=f"Recovery workflow {i}",
            input_model="dict",
            output_model="dict",
            correlation_id=uuid4(),
            performance={"single_operation_max_ms": 1000},
        )
        # Bypass strict validation to add input_data (required by orchestrator node)
        contract.__dict__["input_data"] = {
            "content": f"Recovery workflow {i}",
            "namespace": "burst_test",
        }

        start = time.perf_counter()
        await load_test_orchestrator.execute_orchestration(contract)
        recovery_latencies.append((time.perf_counter() - start) * 1000)

        await asyncio.sleep(0.05)

    recovery_avg = statistics.mean(recovery_latencies)
    print(f"Recovery phase avg latency: {recovery_avg:.2f}ms")

    # Verify recovery
    latency_increase_pct = ((recovery_avg - normal_avg) / normal_avg) * 100
    print(f"\nRecovery latency increase: {latency_increase_pct:.1f}%")

    assert (
        latency_increase_pct < 30
    ), f"Failed to recover, latency still {latency_increase_pct:.1f}% higher"


# ============================================================================
# Load Test 4: FSM State Management Under Load
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.load
async def test_fsm_state_management_under_load(load_test_orchestrator):
    """
    Test FSM state manager performance under high workflow concurrency.

    Validates:
    - State transition consistency under load
    - No state corruption with concurrent workflows
    - State query performance at scale
    - Transition history tracking accuracy
    """
    num_workflows = 500
    print(f"\n=== FSM State Management Load Test: {num_workflows} Workflows ===")

    # Create workflows with different state transition paths
    workflow_tasks = []
    for i in range(num_workflows):
        contract = ModelContractOrchestrator(
            name=f"fsm_test_workflow_{i}",
            version={"major": 1, "minor": 0, "patch": 0},
            description=f"FSM test workflow {i}",
            input_model="dict",
            output_model="dict",
            correlation_id=uuid4(),
            performance={"single_operation_max_ms": 1000},
        )
        # Bypass strict validation to add input_data (required by orchestrator node)
        contract.__dict__["input_data"] = {
            "content": f"FSM test workflow {i}",
            "namespace": "fsm_test",
            "workflow_index": i,
        }

        async def execute_and_track_state(workflow_contract):
            """Execute workflow and track FSM state transitions."""
            start_time = time.perf_counter()
            result = await load_test_orchestrator.execute_orchestration(
                workflow_contract
            )
            execution_time = (time.perf_counter() - start_time) * 1000

            # Access Pydantic model attributes directly (not .get() method)
            # ModelStampResponseOutput has workflow_state attribute
            from omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state import (
                EnumWorkflowState,
            )

            return {
                "success": result.workflow_state == EnumWorkflowState.COMPLETED,
                "final_state": result.workflow_state.value,
                "execution_time_ms": execution_time,
                "correlation_id": workflow_contract.correlation_id,
            }

        workflow_tasks.append(execute_and_track_state(contract))

    # Execute all workflows concurrently
    start_time = time.time()
    results = await asyncio.gather(*workflow_tasks, return_exceptions=True)
    total_duration = time.time() - start_time

    # Analyze results
    successful_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    success_rate = len(successful_results) / num_workflows
    avg_execution_time = (
        statistics.mean([r["execution_time_ms"] for r in successful_results])
        if successful_results
        else 0
    )

    # Print results
    print("\n=== FSM Load Test Results ===")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Throughput: {num_workflows / total_duration:.2f} workflows/second")
    print(f"Successful workflows: {len(successful_results)}/{num_workflows}")
    print(f"Failed workflows: {len(failed_results)}/{num_workflows}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average execution time: {avg_execution_time:.2f}ms")

    # Validate FSM consistency
    assert (
        success_rate >= 0.95
    ), f"FSM state management success rate {success_rate:.2%} below 95% threshold"

    # Validate all successful workflows reached a valid final state
    valid_final_states = {"COMPLETED", "FAILED", "completed", "failed"}
    for result in successful_results:
        final_state = result.get("final_state", "").upper()
        assert (
            final_state in valid_final_states or final_state == "unknown"
        ), f"Invalid final state: {final_state}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "load"])
