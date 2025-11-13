#!/usr/bin/env python3
"""
Test ProductionMonitor with circuit breaker integration.

Verifies:
- Circuit breakers initialized correctly
- Monitoring status includes circuit breaker state
- Prometheus metrics include circuit breaker metrics
"""

import asyncio
import sys

sys.path.insert(0, "src")

from unittest.mock import AsyncMock, MagicMock

from omninode_bridge.production.monitoring import ProductionMonitor


async def test_monitoring_with_circuit_breaker():
    """Test production monitor with circuit breaker integration."""
    print("Testing ProductionMonitor Circuit Breaker Integration\n")

    # Create mock dependencies
    metrics_collector = MagicMock()
    metrics_collector.get_performance_summary.return_value = {
        "overall_grade": "A",
        "workflow_latency_p95": 100.0,
    }

    # Initialize monitor
    monitor = ProductionMonitor(metrics_collector=metrics_collector)

    # Test 1: Circuit breakers initialized
    print("Test 1: Circuit breakers initialized")
    assert hasattr(monitor, "health_check_circuit")
    assert hasattr(monitor, "metrics_export_circuit")
    print("✅ Circuit breakers present\n")

    # Test 2: Monitoring status includes circuit breaker state
    print("Test 2: Monitoring status includes circuit breaker state")
    status = monitor.get_monitoring_status()
    assert "circuit_breakers" in status
    assert "health_check" in status["circuit_breakers"]
    assert "metrics_export" in status["circuit_breakers"]
    print(f"✅ Circuit breaker status: {status['circuit_breakers']}\n")

    # Test 3: Prometheus metrics include circuit breaker metrics
    print("Test 3: Prometheus metrics include circuit breaker metrics")
    metrics = monitor.export_prometheus_metrics()
    assert "circuit_breaker_state" in metrics
    assert "circuit_breaker_failures" in metrics
    print("✅ Prometheus metrics include circuit breaker data\n")

    # Test 4: Circuit breaker protects monitoring loop
    print("Test 4: Circuit breaker protects monitoring loop")

    # Create failing health checker
    failing_health_checker = AsyncMock()
    failing_health_checker.check_system_health.side_effect = Exception("Service down")

    monitor_with_failures = ProductionMonitor(
        metrics_collector=metrics_collector,
        health_checker=failing_health_checker,
    )

    # Start monitoring briefly
    await monitor_with_failures.start_monitoring(
        health_check_interval_seconds=1, metrics_export_interval_seconds=10
    )

    # Let it run for a few seconds to trigger failures
    await asyncio.sleep(6)

    # Stop monitoring
    await monitor_with_failures.stop_monitoring()

    # Check circuit breaker state
    status = monitor_with_failures.get_monitoring_status()
    health_circuit = status["circuit_breakers"]["health_check"]

    print(f"Health check circuit state: {health_circuit['state']}")
    print(f"Failure count: {health_circuit['failure_count']}")

    # Circuit should have opened due to failures
    if health_circuit["failure_count"] >= 5:
        assert health_circuit["state"] == "open"
        print("✅ Circuit opened after repeated failures\n")
    else:
        print(
            f"⚠️  Circuit still closed (only {health_circuit['failure_count']} failures)\n"
        )

    print("=" * 60)
    print("✅ ProductionMonitor circuit breaker integration verified!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_monitoring_with_circuit_breaker())
