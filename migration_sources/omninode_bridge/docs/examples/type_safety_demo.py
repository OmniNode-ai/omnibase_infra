"""
Type Safety Demonstration: Before vs After Dict[str, Any] Replacement.

This demo shows the concrete benefits of using TypedDict and Pydantic models
instead of Dict[str, Any] for type safety and IDE support.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.omninode_bridge.models.typed_models import (
    ConsulHealthCheckResult,
    DiscoveredServiceInstance,
    FSMTransitionMetadata,
    FSMTransitionRecord,
    PerformanceStats,
    PoolStatsResult,
)

# ==============================================================================
# BEFORE: Weak Typing with Dict[str, Any]
# ==============================================================================


def process_health_check_weak(result: dict[str, Any]) -> dict[str, Any]:
    """
    Process health check with weak typing.

    Problems:
    - No IDE autocomplete
    - Typos not caught until runtime
    - No structure documentation
    - Can't validate required fields
    """
    # Typo in field name - not caught by type checker!
    status = result.get("statsu", "unknown")  # Should be "status"

    # Wrong type - not caught until runtime!
    port = str(result.get("consul_port", 0))  # Should be int

    return {
        "healthy": status == "healthy",
        "port": port,
        # Missing required fields - not caught!
    }


def discover_services_weak(services: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Process discovered services with weak typing.

    Problems:
    - Can't validate service structure
    - No type hints for service fields
    - Easy to make mistakes with nested dicts
    """
    filtered = []
    for service in services:
        # Nested dict access - error-prone!
        if service.get("meta", {}).get("version") == "0.1.0":
            filtered.append(
                {
                    "id": service["id"],
                    "url": f"http://{service['address']}:{service['port']}",
                    # Typo in field name - not caught!
                    "tgs": service.get("tags", []),  # Should be "tags"
                }
            )
    return filtered


# ==============================================================================
# AFTER: Strong Typing with TypedDict
# ==============================================================================


def process_health_check_strong(
    result: ConsulHealthCheckResult,
) -> dict[str, bool | int]:
    """
    Process health check with strong typing.

    Benefits:
    - IDE autocomplete for all fields
    - Typos caught immediately
    - Structure documented in type
    - Required fields enforced
    """
    # Type checker catches typos immediately!
    # result["statsu"]  # ← Would be flagged by mypy
    status = result["status"]

    # Type checker enforces correct types!
    # port = str(result["consul_port"])  # ← Would warn about type mismatch
    port = result.get("consul_port", 8500)

    return {
        "healthy": status == "healthy",
        "port": port,
    }


def discover_services_strong(
    services: list[DiscoveredServiceInstance],
) -> list[dict[str, str | list[str]]]:
    """
    Process discovered services with strong typing.

    Benefits:
    - Validated service structure
    - Type-safe field access
    - IDE provides autocomplete
    - Clear API contract
    """
    filtered = []
    for service in services:
        # Type-safe access with IDE autocomplete!
        if service["meta"].get("version") == "0.1.0":
            filtered.append(
                {
                    "id": service["id"],
                    "url": f"http://{service['address']}:{service['port']}",
                    "tags": service["tags"],  # Type-safe access
                }
            )
    return filtered


# ==============================================================================
# Pydantic Validation Demo
# ==============================================================================


def validate_pool_stats(data: dict[str, Any]) -> PoolStatsResult | None:
    """
    Demonstrate Pydantic validation benefits.

    Pydantic models provide:
    - Automatic validation
    - Type coercion
    - Clear error messages
    - Immutable data (with frozen=True)
    """
    try:
        # Pydantic validates all fields automatically
        stats = PoolStatsResult(**data)
        return stats
    except Exception as e:
        print(f"Validation error: {e}")
        return None


# ==============================================================================
# FSM Transition Demo
# ==============================================================================


def record_fsm_transition_weak(
    workflow_id: UUID,
    from_state: str,
    to_state: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Record FSM transition with weak typing.

    Problems:
    - No validation of metadata structure
    - Unclear what fields are allowed
    - Easy to add wrong fields
    """
    return {
        "workflow_id": str(workflow_id),
        "from_state": from_state,
        "to_state": to_state,
        "metadata": metadata,
        # Missing required fields like "trigger" and "timestamp"
    }


def record_fsm_transition_strong(
    workflow_id: UUID,
    from_state: str,
    to_state: str,
    trigger: str,
    metadata: FSMTransitionMetadata,
) -> FSMTransitionRecord:
    """
    Record FSM transition with strong typing.

    Benefits:
    - Validated metadata structure
    - Clear required fields
    - Type-safe field access
    - Self-documenting
    """
    from datetime import UTC, datetime

    # Type checker ensures all required fields are present
    record: FSMTransitionRecord = {
        "from_state": from_state,
        "to_state": to_state,
        "trigger": trigger,
        "timestamp": datetime.now(UTC),
        "metadata": metadata,
    }
    return record


# ==============================================================================
# Performance Stats Demo
# ==============================================================================


def calculate_performance_weak(execution_times: list[float]) -> dict[str, Any]:
    """
    Calculate performance stats with weak typing.

    Problems:
    - No validation of numeric ranges
    - Can return negative times
    - No guarantee of required fields
    """
    if not execution_times:
        return {}

    sorted_times = sorted(execution_times)
    return {
        "avg": sum(sorted_times) / len(sorted_times),
        "p95": sorted_times[int(len(sorted_times) * 0.95)],
        # Missing fields like min, max, p99
        # No validation that values are >= 0
    }


def calculate_performance_strong(execution_times: list[float]) -> PerformanceStats:
    """
    Calculate performance stats with strong typing (Pydantic).

    Benefits:
    - Automatic validation (all values >= 0)
    - Guaranteed required fields
    - Clear return type
    - Immutable result
    """
    if not execution_times:
        # Pydantic ensures all fields have valid values
        return PerformanceStats(
            avg_execution_time_ms=0.0,
            min_execution_time_ms=0.0,
            max_execution_time_ms=0.0,
            p95_execution_time_ms=0.0,
            p99_execution_time_ms=0.0,
            sample_count=0,
        )

    sorted_times = sorted(execution_times)
    count = len(sorted_times)

    # Pydantic validates all values are >= 0
    return PerformanceStats(
        avg_execution_time_ms=sum(sorted_times) / count,
        min_execution_time_ms=sorted_times[0],
        max_execution_time_ms=sorted_times[-1],
        p95_execution_time_ms=sorted_times[int(count * 0.95)],
        p99_execution_time_ms=sorted_times[int(count * 0.99)],
        sample_count=count,
    )


# ==============================================================================
# Main Demo
# ==============================================================================


def main():
    """Run type safety demonstrations."""
    print("=" * 70)
    print("Type Safety Demonstration: Before vs After")
    print("=" * 70)
    print()

    # Demo 1: Health Check
    print("Demo 1: Health Check Processing")
    print("-" * 70)

    health_data: ConsulHealthCheckResult = {
        "status": "healthy",
        "consul_connected": True,
        "consul_host": "localhost",
        "consul_port": 8500,
    }

    result = process_health_check_strong(health_data)
    print(f"✅ Processed health check: {result}")
    print()

    # Demo 2: Service Discovery
    print("Demo 2: Service Discovery")
    print("-" * 70)

    services: list[DiscoveredServiceInstance] = [
        {
            "id": "service-1",
            "address": "192.168.1.10",
            "port": 8080,
            "tags": ["api", "v1"],
            "meta": {"version": "0.1.0", "region": "us-west"},
        },
        {
            "id": "service-2",
            "address": "192.168.1.11",
            "port": 8081,
            "tags": ["api", "v2"],
            "meta": {"version": "0.2.0", "region": "us-east"},
        },
    ]

    filtered = discover_services_strong(services)
    print(f"✅ Filtered {len(filtered)} services with version 0.1.0")
    print()

    # Demo 3: Pydantic Validation
    print("Demo 3: Pydantic Validation")
    print("-" * 70)

    # Valid data
    valid_stats = {
        "pool_size": 20,
        "available": 15,
        "in_use": 5,
        "utilization": 0.25,
    }
    stats = validate_pool_stats(valid_stats)
    print(f"✅ Valid stats: {stats}")

    # Invalid data (negative values)
    invalid_stats = {
        "pool_size": -20,  # Invalid: negative
        "available": 15,
        "in_use": 5,
        "utilization": 0.25,
    }
    invalid_result = validate_pool_stats(invalid_stats)
    print(f"❌ Invalid stats rejected: {invalid_result is None}")
    print()

    # Demo 4: FSM Transition
    print("Demo 4: FSM Transition Recording")
    print("-" * 70)

    metadata: FSMTransitionMetadata = {
        "namespace": "omninode.bridge",
        "reason": "workflow_completed",
        "execution_time_ms": 125.5,
    }

    transition = record_fsm_transition_strong(
        workflow_id=uuid4(),
        from_state="PROCESSING",
        to_state="COMPLETED",
        trigger="workflow_completed",
        metadata=metadata,
    )
    print(
        f"✅ Recorded transition: {transition['from_state']} → {transition['to_state']}"
    )
    print()

    # Demo 5: Performance Stats
    print("Demo 5: Performance Statistics")
    print("-" * 70)

    execution_times = [10.5, 15.2, 12.3, 18.7, 14.1, 16.8, 11.9, 20.4, 13.6, 17.2]
    perf_stats = calculate_performance_strong(execution_times)
    print("✅ Performance stats calculated:")
    print(f"   - Avg: {perf_stats.avg_execution_time_ms:.2f}ms")
    print(f"   - P95: {perf_stats.p95_execution_time_ms:.2f}ms")
    print(f"   - P99: {perf_stats.p99_execution_time_ms:.2f}ms")
    print(f"   - Samples: {perf_stats.sample_count}")
    print()

    print("=" * 70)
    print("Summary: Type Safety Benefits")
    print("=" * 70)
    print("✅ IDE autocomplete and type hints")
    print("✅ Compile-time error detection")
    print("✅ Self-documenting code")
    print("✅ Runtime validation (Pydantic)")
    print("✅ Safer refactoring")
    print("=" * 70)


if __name__ == "__main__":
    main()
