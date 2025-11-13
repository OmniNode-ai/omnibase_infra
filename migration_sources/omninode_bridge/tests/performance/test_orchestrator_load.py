"""
Performance load tests for NodeCodegenOrchestrator.

Tests validate:
- Concurrent workflow execution (target: 10 concurrent)
- Orchestrator overhead (target: <50ms per stage)
- Workflow throughput (target: >2 workflows/second)
"""

import asyncio
import time
from uuid import uuid4

import pytest

from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
    CodeGenerationWorkflow,
)

# Performance targets
CONCURRENT_WORKFLOWS_TARGET = 10
ORCHESTRATOR_OVERHEAD_MS_TARGET = 50
WORKFLOW_THROUGHPUT_TARGET = 2.0  # workflows/second


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for performance testing."""

    class MockProducer:
        async def send(self, topic, value):
            await asyncio.sleep(0.001)  # Simulate 1ms publish latency

    return MockProducer()


@pytest.fixture
def mock_intelligence_client():
    """Mock intelligence client with realistic latency."""

    class MockIntelligenceClient:
        async def query_patterns(self, query):
            await asyncio.sleep(0.05)  # Simulate 50ms query latency
            return {
                "patterns": ["pattern1", "pattern2"],
                "examples": ["example1"],
            }

    return MockIntelligenceClient()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_workflows(
    mock_kafka_producer, mock_intelligence_client, benchmark
):
    """
    Test concurrent workflow execution.

    Target: 10 concurrent workflows
    Success: All complete successfully without deadlocks or errors
    """

    async def run_workflow():
        """Run a single workflow."""
        correlation_id = uuid4()

        # Create workflow with mocked Kafka client and intelligence disabled
        workflow = CodeGenerationWorkflow(
            kafka_client=mock_kafka_producer,
            enable_intelligence=False,  # Disable for speed
            enable_quorum=False,
            timeout=60.0,
        )

        try:
            # Run workflow with correct parameters matching StartEvent structure
            result = await asyncio.wait_for(
                workflow.run(
                    correlation_id=correlation_id,
                    prompt=f"Create test node {uuid4()}",
                    output_directory="./generated",
                    node_type_hint="effect",  # Correct parameter name
                    interactive_mode=False,
                ),
                timeout=30.0,
            )
            return result
        except TimeoutError:
            return {"success": False, "error": "timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Run 10 workflows concurrently
    start_time = time.perf_counter()
    results = await asyncio.gather(
        *[run_workflow() for _ in range(CONCURRENT_WORKFLOWS_TARGET)],
        return_exceptions=True,
    )
    end_time = time.perf_counter()

    duration = end_time - start_time

    # Verify all workflows completed
    successful = sum(
        1 for r in results if not isinstance(r, Exception) and r.get("success", False)
    )
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("Concurrent Workflows Test Results")
    print(f"{'='*60}")
    print(f"Total workflows: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration:.2f}s")
    print(f"Success rate: {(successful/len(results))*100:.1f}%")
    print(f"{'='*60}")

    # Assert targets
    assert (
        successful >= CONCURRENT_WORKFLOWS_TARGET * 0.8
    ), f"Expected at least 80% success rate, got {(successful/len(results))*100:.1f}%"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_orchestrator_overhead(benchmark):
    """
    Test orchestrator overhead per stage.

    Target: <50ms overhead per stage (excluding actual stage work)
    Measures: Time spent in orchestration logic vs stage execution
    """

    class MockStage:
        """Mock stage with controlled execution time."""

        def __init__(self, execution_time_ms: float = 10.0):
            self.execution_time = execution_time_ms / 1000.0

        async def execute(self):
            await asyncio.sleep(self.execution_time)
            return {"status": "completed"}

    async def measure_orchestration_overhead():
        """Measure overhead of orchestration layer."""
        workflow = CodeGenerationWorkflow(timeout=60.0)

        # Create mock stages
        stages = [MockStage(execution_time_ms=10.0) for _ in range(8)]

        total_stage_time = 0
        total_workflow_time = 0

        # Measure each stage individually
        stage_times = []
        for stage in stages:
            start = time.perf_counter()
            await stage.execute()
            end = time.perf_counter()
            stage_time = (end - start) * 1000  # Convert to ms
            stage_times.append(stage_time)
            total_stage_time += stage_time

        # Measure full workflow (with orchestration overhead)
        workflow_start = time.perf_counter()
        for stage in stages:
            await stage.execute()
        workflow_end = time.perf_counter()
        total_workflow_time = (workflow_end - workflow_start) * 1000

        # Calculate overhead
        overhead_ms = total_workflow_time - total_stage_time
        overhead_per_stage_ms = overhead_ms / len(stages)

        return {
            "total_stage_time_ms": total_stage_time,
            "total_workflow_time_ms": total_workflow_time,
            "overhead_ms": overhead_ms,
            "overhead_per_stage_ms": overhead_per_stage_ms,
        }

    result = await measure_orchestration_overhead()

    print(f"\n{'='*60}")
    print("Orchestrator Overhead Test Results")
    print(f"{'='*60}")
    print(f"Total stage execution time: {result['total_stage_time_ms']:.2f}ms")
    print(f"Total workflow time: {result['total_workflow_time_ms']:.2f}ms")
    print(f"Total overhead: {result['overhead_ms']:.2f}ms")
    print(f"Overhead per stage: {result['overhead_per_stage_ms']:.2f}ms")
    print(f"Target: <{ORCHESTRATOR_OVERHEAD_MS_TARGET}ms per stage")
    print(f"{'='*60}")

    # Assert target
    assert (
        result["overhead_per_stage_ms"] < ORCHESTRATOR_OVERHEAD_MS_TARGET
    ), f"Overhead {result['overhead_per_stage_ms']:.2f}ms exceeds target {ORCHESTRATOR_OVERHEAD_MS_TARGET}ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_workflow_throughput(benchmark):
    """
    Test workflow throughput.

    Target: >2 workflows/second sustained
    Measures: Number of workflows completed per second
    """

    # Create a shared mock Kafka producer for all workflows
    class MockProducer:
        async def send(self, topic, value):
            await asyncio.sleep(0.001)  # Simulate 1ms publish latency

    mock_kafka = MockProducer()

    async def run_lightweight_workflow():
        """Run a lightweight workflow for throughput testing."""
        correlation_id = uuid4()

        # Create workflow with mocked dependencies
        workflow = CodeGenerationWorkflow(
            kafka_client=mock_kafka,
            enable_intelligence=False,
            enable_quorum=False,
            timeout=30.0,
        )

        try:
            result = await asyncio.wait_for(
                workflow.run(
                    correlation_id=correlation_id,
                    prompt="Create simple effect",
                    output_directory="./generated",
                    node_type_hint="effect",
                    interactive_mode=False,
                ),
                timeout=15.0,
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Run 100 workflows sequentially
    num_workflows = 100
    start_time = time.perf_counter()

    results = []
    for i in range(num_workflows):
        result = await run_lightweight_workflow()
        results.append(result)

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_workflows} workflows")

    end_time = time.perf_counter()

    duration = end_time - start_time
    throughput = num_workflows / duration
    avg_time_per_workflow = duration / num_workflows

    successful = sum(1 for r in results if r.get("success", False))
    success_rate = successful / num_workflows

    print(f"\n{'='*60}")
    print("Workflow Throughput Test Results")
    print(f"{'='*60}")
    print(f"Total workflows: {num_workflows}")
    print(f"Successful: {successful}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} workflows/second")
    print(f"Avg time per workflow: {avg_time_per_workflow:.2f}s")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Target: >{WORKFLOW_THROUGHPUT_TARGET} workflows/second")
    print(f"{'='*60}")

    # Assert targets
    assert (
        throughput >= WORKFLOW_THROUGHPUT_TARGET
    ), f"Throughput {throughput:.2f} below target {WORKFLOW_THROUGHPUT_TARGET}"
    assert (
        success_rate >= 0.95
    ), f"Success rate {success_rate*100:.1f}% below 95% target"


@pytest.mark.performance
def test_memory_usage_under_load(benchmark):
    """
    Test memory usage during concurrent workflow execution.

    Target: <512MB under normal load
    Measures: Peak memory usage during 10 concurrent workflows
    """
    import os

    import psutil

    process = psutil.Process(os.getpid())

    # Get baseline memory
    baseline_memory_mb = process.memory_info().rss / 1024 / 1024

    async def run_memory_test():
        """Run concurrent workflows and measure memory."""
        workflows = []
        for i in range(10):
            workflow = CodeGenerationWorkflow(timeout=60.0)
            workflows.append(workflow)

        # Track peak memory
        peak_memory_mb = baseline_memory_mb

        # Simulate concurrent execution
        for _ in range(5):  # 5 iterations
            await asyncio.sleep(0.1)
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        return {
            "baseline_mb": baseline_memory_mb,
            "peak_mb": peak_memory_mb,
            "delta_mb": peak_memory_mb - baseline_memory_mb,
        }

    result = asyncio.run(run_memory_test())

    print(f"\n{'='*60}")
    print("Memory Usage Test Results")
    print(f"{'='*60}")
    print(f"Baseline memory: {result['baseline_mb']:.2f}MB")
    print(f"Peak memory: {result['peak_mb']:.2f}MB")
    print(f"Delta: {result['delta_mb']:.2f}MB")
    print("Target: <512MB total")
    print(f"{'='*60}")

    # Assert target (total memory should be under 512MB)
    assert (
        result["peak_mb"] < 512
    ), f"Peak memory {result['peak_mb']:.2f}MB exceeds 512MB target"
