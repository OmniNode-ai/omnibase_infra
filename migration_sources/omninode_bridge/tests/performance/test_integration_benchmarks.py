#!/usr/bin/env python3
"""
Integration Performance Benchmarks for Bridge System.

These benchmarks verify performance requirements from IMPLEMENTATION_ROADMAP.md:
1. Orchestrator throughput: 100 concurrent workflows in <10 seconds
2. Reducer aggregation: 1000 items in <100ms
3. Config load performance: <50ms

Correlation ID: c5c5ba1d-0642-4aa2-a7a0-086b9592ea67
Task: Integration Test Gaps - Task 3.4 (Performance Benchmark Tests)
"""

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omninode_bridge.config.config_loader import load_node_config, reload_config
from omninode_bridge.nodes.orchestrator.v1_0_0.models.model_stamp_request_input import (
    ModelStampRequestInput,
)
from omninode_bridge.nodes.reducer.v1_0_0.models.model_stamp_metadata_input import (
    ModelStampMetadataInput,
)

# Try importing ONEX
try:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_base import (
        EnumNodeType,
        ModelSemVer,
    )
    from omnibase_core.models.contracts.model_contract_reducer import (
        ModelContractReducer,
    )

    ONEX_AVAILABLE = True
except ImportError:
    ONEX_AVAILABLE = False

    class ModelONEXContainer:
        def __init__(self):
            self.config = {}
            self.value = {}

    class EnumNodeType:
        REDUCER = "reducer"

    class ModelSemVer:
        def __init__(self, major, minor, patch):
            pass

    class ModelContractReducer:
        def __init__(self, **kwargs):
            pass


@pytest.mark.performance
@pytest.mark.asyncio
async def test_orchestrator_throughput_100_concurrent():
    """
    Benchmark: 100 concurrent workflows should complete in <10 seconds.

    Performance Target: 10 workflows/second minimum throughput
    Success Criteria: Total execution time < 10 seconds
    """
    from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator

    # Create orchestrator
    container = ModelONEXContainer()
    container.config = {
        "metadata_stamping_service_url": "http://localhost:8053",
        "onextree_service_url": "http://localhost:8058",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.benchmark",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }

    with patch("consul.Consul"):
        orchestrator = NodeBridgeOrchestrator(container)

        # Mock Kafka
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        orchestrator.kafka_client = mock_kafka_client

        # Create workflow execution function
        async def execute_workflow(index: int):
            stamp_request = ModelStampRequestInput(
                file_path=f"/benchmark/workflow_{index}.pdf",
                file_content=f"Benchmark content {index}".encode(),
                namespace="omninode.bridge.benchmark",
                content_type="application/pdf",
            )

            mock_response = {
                "file_hash": f"blake3_benchmark_{index}",
                "stamp_id": str(uuid4()),
                "stamped_at": datetime.now(UTC).isoformat(),
            }

            with patch.object(
                orchestrator, "_call_stamping_service", return_value=mock_response
            ):
                return await orchestrator.execute_stamping_workflow(stamp_request)

        # Execute 100 concurrent workflows
        start_time = time.perf_counter()
        results = await asyncio.gather(*[execute_workflow(i) for i in range(100)])
        end_time = time.perf_counter()

        total_time_seconds = end_time - start_time

        # Verify results
        assert len(results) == 100
        successful_workflows = [
            r for r in results if r.workflow_state.value == "completed"
        ]
        assert len(successful_workflows) == 100

        # Verify performance target
        assert (
            total_time_seconds < 10.0
        ), f"100 workflows took {total_time_seconds:.2f}s (target: <10s)"

        # Calculate throughput
        throughput = 100 / total_time_seconds
        print(f"\n✅ Orchestrator Throughput: {throughput:.2f} workflows/second")
        print(f"   Total Time: {total_time_seconds:.3f}s")
        print(f"   Average per workflow: {(total_time_seconds * 1000) / 100:.2f}ms")

        await orchestrator.shutdown()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_reducer_aggregation_1000_items():
    """
    Benchmark: 1000 items aggregated in <100ms.

    Performance Target: >10,000 items/second aggregation throughput
    Success Criteria: 1000 items in <100ms
    """
    from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

    # Create reducer
    container = ModelONEXContainer()
    container.value = {
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.benchmark",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }

    with patch("consul.Consul"):
        reducer = NodeBridgeReducer(container)

        # Mock Kafka
        mock_kafka_client = AsyncMock()
        mock_kafka_client.is_connected = True
        mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
        reducer.kafka_client = mock_kafka_client

        # Create 1000 stamp metadata items
        items = []
        for i in range(1000):
            item = ModelStampMetadataInput(
                stamp_id=str(uuid4()),
                file_hash=f"blake3_item_{i}",
                file_path=f"/benchmark/item_{i}.pdf",
                file_size=1024 * (i % 100 + 1),  # Vary file sizes
                namespace=f"omninode.benchmark.ns_{i % 10}",  # 10 namespaces
                content_type="application/pdf",
                workflow_id=str(uuid4()),
                workflow_state="completed",
                processing_time_ms=10,
            )
            items.append(item.model_dump())

        # Create reducer contract
        contract = ModelContractReducer(
            name="benchmark_aggregation_1000",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Benchmark aggregation of 1000 items",
            node_type=EnumNodeType.REDUCER,
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
            input_state={"items": items},
        )

        # Execute aggregation with timing
        start_time = time.perf_counter()
        result = await reducer.execute_reduction(contract)
        end_time = time.perf_counter()

        aggregation_time_ms = (end_time - start_time) * 1000

        # Verify results
        assert result.total_items == 1000
        assert len(result.namespaces) == 10  # 10 unique namespaces

        # Verify performance target
        assert (
            aggregation_time_ms < 100.0
        ), f"1000 items took {aggregation_time_ms:.2f}ms (target: <100ms)"

        # Calculate throughput
        throughput = 1000 / (aggregation_time_ms / 1000)
        print("\n✅ Reducer Aggregation Performance:")
        print("   Items: 1000")
        print(f"   Time: {aggregation_time_ms:.2f}ms")
        print(f"   Throughput: {throughput:.0f} items/second")

        await reducer.shutdown()


@pytest.mark.performance
def test_config_load_performance():
    """
    Benchmark: Config load should complete in <50ms.

    Performance Target: Fast config loading for node startup
    Success Criteria: Config load <50ms (p95)
    """
    import tempfile

    import yaml

    # Create temporary config directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create realistic orchestrator config
        orchestrator_config = {
            "orchestrator": {
                "max_concurrent_workflows": 100,
                "timeout": 120,
                "health_check_interval": 30,
                "enable_intelligence": True,
            },
            "kafka": {
                "bootstrap_servers": "kafka-prod:9092",
                "topic_prefix": "omninode",
                "compression_type": "snappy",
            },
            "consul": {
                "host": "consul-prod",
                "port": 8500,
                "enable_registration": True,
            },
            "database": {
                "postgres": {
                    "host": "postgres-prod",
                    "port": 5432,
                    "database": "omninode_bridge",
                    "pool_size": 20,
                }
            },
        }

        with open(config_dir / "orchestrator.yaml", "w") as f:
            yaml.dump(orchestrator_config, f)

        # Create environment config
        env_config = {
            "orchestrator": {"max_concurrent_workflows": 200},
            "kafka": {"bootstrap_servers": "dev-kafka:9092"},
        }

        with open(config_dir / "development.yaml", "w") as f:
            yaml.dump(env_config, f)

        # Measure config load time over multiple iterations
        load_times = []
        for _ in range(100):
            reload_config()  # Clear cache
            start_time = time.perf_counter()
            config = load_node_config("orchestrator", "development", config_dir)
            end_time = time.perf_counter()
            load_time_ms = (end_time - start_time) * 1000
            load_times.append(load_time_ms)

        # Calculate statistics
        avg_time = sum(load_times) / len(load_times)
        p50_time = sorted(load_times)[int(len(load_times) * 0.50)]
        p95_time = sorted(load_times)[int(len(load_times) * 0.95)]
        p99_time = sorted(load_times)[int(len(load_times) * 0.99)]
        max_time = max(load_times)

        # Verify performance target
        assert p95_time < 50.0, f"Config load p95={p95_time:.2f}ms (target: <50ms)"

        print("\n✅ Config Load Performance:")
        print("   Iterations: 100")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   p50: {p50_time:.2f}ms")
        print(f"   p95: {p95_time:.2f}ms")
        print(f"   p99: {p99_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_end_to_end_workflow_latency():
    """
    Benchmark: Complete workflow latency from request to aggregation.

    Measures:
    - Orchestrator processing time
    - Reducer aggregation time
    - Total end-to-end latency

    Target: <200ms for complete workflow
    """
    from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
    from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer

    # Create nodes
    orchestrator_container = ModelONEXContainer()
    orchestrator_container.config = {
        "metadata_stamping_service_url": "http://localhost:8053",
        "onextree_service_url": "http://localhost:8058",
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.benchmark",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }

    reducer_container = ModelONEXContainer()
    reducer_container.value = {
        "kafka_broker_url": "localhost:9092",
        "default_namespace": "omninode.bridge.benchmark",
        "consul_enable_registration": False,
        "health_check_mode": False,
    }

    with patch("consul.Consul"):
        orchestrator = NodeBridgeOrchestrator(orchestrator_container)
        reducer = NodeBridgeReducer(reducer_container)

        # Mock Kafka
        for node in [orchestrator, reducer]:
            mock_kafka_client = AsyncMock()
            mock_kafka_client.is_connected = True
            mock_kafka_client.publish_with_envelope = AsyncMock(return_value=True)
            node.kafka_client = mock_kafka_client

        # Execute workflow
        stamp_request = ModelStampRequestInput(
            file_path="/benchmark/e2e.pdf",
            file_content=b"End-to-end benchmark content",
            namespace="omninode.bridge.benchmark",
            content_type="application/pdf",
        )

        mock_response = {
            "file_hash": "blake3_e2e_benchmark",
            "stamp_id": str(uuid4()),
            "stamped_at": datetime.now(UTC).isoformat(),
        }

        # Time orchestrator phase
        start_time = time.perf_counter()

        with patch.object(
            orchestrator, "_call_stamping_service", return_value=mock_response
        ):
            orchestrator_result = await orchestrator.execute_stamping_workflow(
                stamp_request
            )

        orchestrator_time = time.perf_counter() - start_time

        # Time reducer phase
        stamp_metadata = ModelStampMetadataInput(
            stamp_id=orchestrator_result.stamp_id,
            file_hash=orchestrator_result.file_hash,
            file_path="/benchmark/e2e.pdf",
            file_size=len(stamp_request.file_content),
            namespace="omninode.bridge.benchmark",
            content_type="application/pdf",
            workflow_id=orchestrator_result.workflow_id,
            workflow_state=orchestrator_result.workflow_state.value,
            processing_time_ms=orchestrator_result.processing_time_ms,
        )

        contract = ModelContractReducer(
            name="benchmark_e2e",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="E2E benchmark aggregation",
            node_type=EnumNodeType.REDUCER,
            input_model="ModelReducerInputState",
            output_model="ModelReducerOutputState",
            input_state={"items": [stamp_metadata.model_dump()]},
        )

        reducer_start = time.perf_counter()
        reducer_result = await reducer.execute_reduction(contract)
        reducer_time = time.perf_counter() - reducer_start

        total_time_ms = (orchestrator_time + reducer_time) * 1000

        # Verify performance
        assert (
            total_time_ms < 200.0
        ), f"E2E workflow took {total_time_ms:.2f}ms (target: <200ms)"

        print("\n✅ End-to-End Workflow Latency:")
        print(f"   Orchestrator: {orchestrator_time * 1000:.2f}ms")
        print(f"   Reducer: {reducer_time * 1000:.2f}ms")
        print(f"   Total: {total_time_ms:.2f}ms")

        await orchestrator.shutdown()
        await reducer.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
