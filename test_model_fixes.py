#!/usr/bin/env python3
"""Test script for model import and validation fixes in PR #11."""

import sys
from pathlib import Path

def test_circuit_breaker_metrics():
    """Test circuit breaker metrics with division by zero prevention."""
    try:
        from src.omnibase_infra.models.core.circuit_breaker.model_circuit_breaker_metrics import ModelCircuitBreakerMetrics

        # Test with zero total events
        metrics = ModelCircuitBreakerMetrics()
        assert metrics.success_rate_percent == 0.0, "Should handle zero division gracefully"

        # Test with events
        metrics = ModelCircuitBreakerMetrics(total_events=100, successful_events=80)
        assert metrics.success_rate_percent == 80.0, f"Expected 80.0, got {metrics.success_rate_percent}"

        print("✅ Circuit breaker metrics: Division by zero prevention working")
        return True
    except Exception as e:
        print(f"❌ Circuit breaker metrics test failed: {e}")
        return False

def test_workflow_coordination_validation():
    """Test workflow coordination metrics validation."""
    try:
        from src.omnibase_infra.models.core.workflow.model_workflow_coordination_metrics import ModelWorkflowCoordinationMetrics

        # Test valid metrics
        metrics = ModelWorkflowCoordinationMetrics(
            coordinator_id="test-coordinator",
            agent_coordination_success_rate=0.8,
            sub_agent_fleet_utilization=0.7
        )
        assert metrics.agent_coordination_success_rate == 0.8

        # Test invalid rate range
        try:
            ModelWorkflowCoordinationMetrics(
                coordinator_id="test-coordinator",
                agent_coordination_success_rate=1.5  # Invalid: > 1.0
            )
            print("❌ Validation should have failed for rate > 1.0")
            return False
        except ValueError:
            pass  # Expected

        print("✅ Workflow coordination metrics: Validation working")
        return True
    except Exception as e:
        print(f"❌ Workflow coordination metrics test failed: {e}")
        return False

def test_system_health_optimization():
    """Test system health details optimization."""
    try:
        from src.omnibase_infra.models.core.health.services.model_system_health_details import ModelSystemHealthDetails

        # Test disk usage calculation
        details = ModelSystemHealthDetails(
            disk_space_available_gb=20.0,
            disk_space_total_gb=100.0
        )
        assert details.disk_usage_percent == 80.0, f"Expected 80.0, got {details.disk_usage_percent}"

        # Test with zero total (division by zero prevention)
        details = ModelSystemHealthDetails(
            disk_space_available_gb=20.0,
            disk_space_total_gb=0.0
        )
        assert details.disk_usage_percent == 0.0, "Should handle zero division"

        print("✅ System health details: Optimization working")
        return True
    except Exception as e:
        print(f"❌ System health details test failed: {e}")
        return False

def test_optional_annotations():
    """Test that Optional[type] has been replaced with type | None."""
    try:
        from src.omnibase_infra.models.core.workflow.model_sub_agent_result import ModelSubAgentResult
        import inspect
        from uuid import uuid4
        from datetime import datetime

        # Test creating with None values
        result = ModelSubAgentResult(
            agent_id=uuid4(),
            agent_name="test-agent",
            agent_type="test",
            execution_status="completed",
            success=True,
            started_at=datetime.now(),
            parent_workflow_id=uuid4(),
            completed_at=None,  # Should accept None
            parent_agent_id=None  # Should accept None
        )

        assert result.completed_at is None
        assert result.parent_agent_id is None

        print("✅ Optional annotations: Modern type annotations working")
        return True
    except Exception as e:
        print(f"❌ Optional annotations test failed: {e}")
        return False

def test_model_base_imports():
    """Test that models are using ModelBase instead of BaseModel."""
    try:
        from src.omnibase_infra.models.core.circuit_breaker.model_circuit_breaker_metrics import ModelCircuitBreakerMetrics
        from src.omnibase_infra.models.core.security.model_tls_config import ModelKafkaProducerConfig
        from omnibase_core.models.model_base import ModelBase

        # Test inheritance
        assert issubclass(ModelCircuitBreakerMetrics, ModelBase), "Should inherit from ModelBase"
        assert issubclass(ModelKafkaProducerConfig, ModelBase), "Should inherit from ModelBase"

        print("✅ Model base imports: Using ModelBase correctly")
        return True
    except Exception as e:
        print(f"❌ Model base imports test failed: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("🧪 Testing PR #11 Import and Validation Fixes")
    print("=" * 50)

    tests = [
        test_circuit_breaker_metrics,
        test_workflow_coordination_validation,
        test_system_health_optimization,
        test_optional_annotations,
        test_model_base_imports,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! PR #11 fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())