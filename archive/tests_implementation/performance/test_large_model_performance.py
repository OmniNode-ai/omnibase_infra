"""Performance tests for large nested models.

Tests instantiation, validation, and serialization performance
for models with many fields (180+ fields like ModelAuditDetails).
"""

import time
from datetime import datetime
from uuid import uuid4

import pytest

try:
    from omnibase_infra.models.core.security.model_audit_details import ModelAuditDetails
    AUDIT_DETAILS_AVAILABLE = True
except ImportError:
    AUDIT_DETAILS_AVAILABLE = False

try:
    from omnibase_infra.models.core.health.services.model_system_health_details import ModelSystemHealthDetails
    SYSTEM_HEALTH_AVAILABLE = True
except ImportError:
    SYSTEM_HEALTH_AVAILABLE = False


class TestLargeModelPerformance:
    """Performance tests for large models with many fields."""

    @pytest.mark.skipif(not AUDIT_DETAILS_AVAILABLE, reason="ModelAuditDetails not available")
    def test_audit_details_instantiation_performance(self):
        """Test ModelAuditDetails instantiation performance (180+ fields)."""
        start_time = time.perf_counter()

        # Create multiple instances to test performance
        instances = []
        for i in range(100):
            audit_details = ModelAuditDetails(
                request_id=uuid4(),
                response_status=200,
                request_timestamp=datetime.now(),
                audit_message=f"Test audit message {i}",
                user_agent=f"TestAgent/{i}",
                remote_ip="192.168.1.1",
                session_id=uuid4(),
                data_classification="internal",
                threat_level="low",
                retention_period_days=365,
            )
            instances.append(audit_details)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_instance = total_time / 100

        print(f"\nAudit Details Performance:")
        print(f"Total time for 100 instances: {total_time:.4f}s")
        print(f"Average time per instance: {avg_time_per_instance:.6f}s")
        print(f"Instances per second: {100/total_time:.2f}")

        # Performance assertion: Should create instances reasonably fast
        assert avg_time_per_instance < 0.01, f"Instance creation too slow: {avg_time_per_instance:.6f}s per instance"
        assert len(instances) == 100

    @pytest.mark.skipif(not AUDIT_DETAILS_AVAILABLE, reason="ModelAuditDetails not available")
    def test_audit_details_serialization_performance(self):
        """Test ModelAuditDetails JSON serialization performance."""
        # Create a model instance with many fields populated
        audit_details = ModelAuditDetails(
            request_id=uuid4(),
            response_status=200,
            request_timestamp=datetime.now(),
            audit_message="Performance test audit message",
            user_agent="TestAgent/1.0",
            remote_ip="192.168.1.1",
            session_id=uuid4(),
            data_classification="internal",
            threat_level="low",
            retention_period_days=365,
        )

        start_time = time.perf_counter()

        # Serialize multiple times
        serialized_data = []
        for i in range(100):
            json_data = audit_details.model_dump()
            serialized_data.append(json_data)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_serialization = total_time / 100

        print(f"\nAudit Details Serialization Performance:")
        print(f"Total time for 100 serializations: {total_time:.4f}s")
        print(f"Average time per serialization: {avg_time_per_serialization:.6f}s")
        print(f"Serializations per second: {100/total_time:.2f}")

        # Performance assertion: Serialization should be fast
        assert avg_time_per_serialization < 0.001, f"Serialization too slow: {avg_time_per_serialization:.6f}s"
        assert len(serialized_data) == 100

        # Verify serialized data structure
        sample_data = serialized_data[0]
        assert "request_id" in sample_data
        assert "response_status" in sample_data
        assert sample_data["response_status"] == 200

    @pytest.mark.skipif(not SYSTEM_HEALTH_AVAILABLE, reason="ModelSystemHealthDetails not available")
    def test_system_health_details_performance(self):
        """Test ModelSystemHealthDetails performance (complex health calculations)."""
        start_time = time.perf_counter()

        # Create instances with health calculation logic
        instances = []
        for i in range(50):
            health_details = ModelSystemHealthDetails(
                cpu_usage_percent=50.0 + i,
                memory_usage_percent=60.0,
                disk_space_total_gb=1000.0,
                disk_space_available_gb=500.0 - i,
                connection_count=100 + i,
                error_count=i,
            )
            instances.append(health_details)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_instance = total_time / 50

        print(f"\nSystem Health Details Performance:")
        print(f"Total time for 50 instances: {total_time:.4f}s")
        print(f"Average time per instance: {avg_time_per_instance:.6f}s")
        print(f"Instances per second: {50/total_time:.2f}")

        # Performance assertion
        assert avg_time_per_instance < 0.01, f"Health model creation too slow: {avg_time_per_instance:.6f}s"
        assert len(instances) == 50

    def test_memory_usage_large_models(self):
        """Test memory usage for large model instantiation."""
        import sys

        # Get baseline memory
        if hasattr(sys, 'getsizeof'):
            # Simple memory test - create instances and measure
            instances = []

            # Create baseline
            baseline_size = sys.getsizeof(instances)

            # Add models if available
            if AUDIT_DETAILS_AVAILABLE:
                for i in range(10):
                    audit_details = ModelAuditDetails(
                        request_id=uuid4(),
                        response_status=200,
                        audit_message=f"Memory test {i}",
                    )
                    instances.append(audit_details)

            final_size = sys.getsizeof(instances)
            memory_per_instance = (final_size - baseline_size) / max(len(instances), 1)

            print(f"\nMemory Usage Test:")
            print(f"Baseline size: {baseline_size} bytes")
            print(f"Final size: {final_size} bytes")
            print(f"Estimated memory per instance: {memory_per_instance:.2f} bytes")

            # Basic assertion - memory usage should be reasonable
            assert memory_per_instance < 10000, f"Memory usage too high: {memory_per_instance:.2f} bytes per instance"

    def test_validation_performance_with_errors(self):
        """Test validation performance when encountering errors."""
        if not AUDIT_DETAILS_AVAILABLE:
            pytest.skip("ModelAuditDetails not available")

        start_time = time.perf_counter()

        # Test validation with various error conditions
        error_count = 0
        success_count = 0

        test_cases = [
            {"response_status": 200},  # Valid
            {"response_status": -1},   # Invalid (should cause validation error)
            {"response_status": 999},  # Invalid (should cause validation error)
            {"response_status": 404},  # Valid
            {"response_status": "invalid"},  # Invalid type
        ]

        for i in range(20):  # Test each case multiple times
            for case in test_cases:
                try:
                    ModelAuditDetails(**case)
                    success_count += 1
                except Exception:
                    error_count += 1

        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_attempts = len(test_cases) * 20
        avg_time_per_attempt = total_time / total_attempts

        print(f"\nValidation Performance (with errors):")
        print(f"Total time for {total_attempts} attempts: {total_time:.4f}s")
        print(f"Average time per validation: {avg_time_per_attempt:.6f}s")
        print(f"Successful validations: {success_count}")
        print(f"Validation errors: {error_count}")

        # Performance assertion - validation should be fast even with errors
        assert avg_time_per_attempt < 0.01, f"Validation too slow: {avg_time_per_attempt:.6f}s per attempt"
        assert error_count > 0, "Expected some validation errors in test cases"
        assert success_count > 0, "Expected some successful validations"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])