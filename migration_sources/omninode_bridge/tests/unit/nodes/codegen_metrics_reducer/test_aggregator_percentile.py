"""
Unit tests for MetricsAggregator percentile calculation.

Tests the _percentile method with focus on edge cases and correctness
for small sample sizes.
"""

import pytest

from omninode_bridge.nodes.codegen_metrics_reducer.v1_0_0.aggregator import (
    MetricsAggregator,
)


class TestPercentileCalculation:
    """Test suite for percentile calculation."""

    def test_percentile_empty_list(self):
        """Test percentile with empty list returns 0.0."""
        result = MetricsAggregator._percentile([], 0.95)
        assert result == 0.0

    def test_percentile_single_value(self):
        """Test percentile with single value returns that value."""
        data = [42.0]
        assert MetricsAggregator._percentile(data, 0.0) == 42.0
        assert MetricsAggregator._percentile(data, 0.5) == 42.0
        assert MetricsAggregator._percentile(data, 0.95) == 42.0
        assert MetricsAggregator._percentile(data, 1.0) == 42.0

    def test_percentile_two_values(self):
        """Test percentile with two values."""
        data = [1.0, 2.0]
        # p=0.5: ceil(0.5 * 2) = 1, index=0 -> 1.0
        assert MetricsAggregator._percentile(data, 0.5) == 1.0
        # p=0.95: ceil(0.95 * 2) = 2, index=1 -> 2.0
        assert MetricsAggregator._percentile(data, 0.95) == 2.0
        # p=0.99: ceil(0.99 * 2) = 2, index=1 -> 2.0
        assert MetricsAggregator._percentile(data, 0.99) == 2.0

    def test_percentile_ten_values(self):
        """Test percentile with ten values (common test size)."""
        data = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # p=0.5: ceil(0.5 * 10) = 5, index=4 -> 5
        assert MetricsAggregator._percentile(data, 0.5) == 5

        # p=0.95: ceil(0.95 * 10) = 10, index=9 -> 10
        assert MetricsAggregator._percentile(data, 0.95) == 10

        # p=0.99: ceil(0.99 * 10) = 10, index=9 -> 10
        assert MetricsAggregator._percentile(data, 0.99) == 10

    def test_percentile_boundary_values(self):
        """Test percentile with boundary percentile values."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # p=0 should return minimum
        assert MetricsAggregator._percentile(data, 0.0) == 1.0

        # p=1 should return maximum
        assert MetricsAggregator._percentile(data, 1.0) == 5.0

        # p<0 should return minimum
        assert MetricsAggregator._percentile(data, -0.1) == 1.0

        # p>1 should return maximum
        assert MetricsAggregator._percentile(data, 1.5) == 5.0

    def test_percentile_unsorted_data(self):
        """Test percentile with unsorted data."""
        data = [5.0, 1.0, 3.0, 2.0, 4.0]

        # Should still work correctly (function sorts internally)
        # p=0.5: ceil(0.5 * 5) = 3, index=2 -> 3.0 (median)
        assert MetricsAggregator._percentile(data, 0.5) == 3.0

        # p=0.8: ceil(0.8 * 5) = 4, index=3 -> 4.0
        assert MetricsAggregator._percentile(data, 0.8) == 4.0

    def test_percentile_duplicate_values(self):
        """Test percentile with duplicate values."""
        data = [1.0, 2.0, 2.0, 2.0, 5.0]

        # p=0.5: ceil(0.5 * 5) = 3, index=2 -> 2.0
        assert MetricsAggregator._percentile(data, 0.5) == 2.0

        # p=0.8: ceil(0.8 * 5) = 4, index=3 -> 2.0
        assert MetricsAggregator._percentile(data, 0.8) == 2.0

    def test_percentile_large_dataset(self):
        """Test percentile with large dataset."""
        data = list(range(1, 1001))  # 1 to 1000

        # p=0.5: ceil(0.5 * 1000) = 500, index=499 -> 500
        assert MetricsAggregator._percentile(data, 0.5) == 500

        # p=0.95: ceil(0.95 * 1000) = 950, index=949 -> 950
        assert MetricsAggregator._percentile(data, 0.95) == 950

        # p=0.99: ceil(0.99 * 1000) = 990, index=989 -> 990
        assert MetricsAggregator._percentile(data, 0.99) == 990

    def test_percentile_float_values(self):
        """Test percentile with float values."""
        data = [1.5, 2.7, 3.2, 4.8, 5.1, 6.9, 7.3, 8.6, 9.1, 10.5]

        # p=0.5: ceil(0.5 * 10) = 5, index=4 -> 5.1
        assert MetricsAggregator._percentile(data, 0.5) == 5.1

        # p=0.9: ceil(0.9 * 10) = 9, index=8 -> 9.1
        assert MetricsAggregator._percentile(data, 0.9) == 9.1

    def test_percentile_edge_case_small_p(self):
        """Test percentile with very small p values."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        # p=0.01: ceil(0.01 * 10) = 1, index=0 -> 1.0
        assert MetricsAggregator._percentile(data, 0.01) == 1.0

        # p=0.1: ceil(0.1 * 10) = 1, index=0 -> 1.0
        assert MetricsAggregator._percentile(data, 0.1) == 1.0

    def test_percentile_comparison_with_old_method(self):
        """
        Test that demonstrates the difference between old and new methods.

        This test shows why the old int(len*p) method was incorrect.
        """
        data = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Old method (incorrect):
        # p=0.95: int(10 * 0.95) = 9, data[9] = 10
        # p=0.99: int(10 * 0.99) = 9, data[9] = 10
        # Problem: 95th and 99th percentile give same result!

        # New method (correct):
        p95 = MetricsAggregator._percentile(data, 0.95)
        p99 = MetricsAggregator._percentile(data, 0.99)

        # Both should return 10 for this small dataset (nearest-rank method)
        # but the algorithm is now mathematically correct
        assert p95 == 10
        assert p99 == 10

        # For larger datasets, the methods would differ more significantly
        large_data = list(range(1, 101))  # 1 to 100

        # New method (correct):
        # p=0.95: ceil(0.95 * 100) = 95, index=94 -> 95
        # p=0.99: ceil(0.99 * 100) = 99, index=98 -> 99
        p95_large = MetricsAggregator._percentile(large_data, 0.95)
        p99_large = MetricsAggregator._percentile(large_data, 0.99)

        assert p95_large == 95
        assert p99_large == 99

        # Old method would have given:
        # p=0.95: int(100 * 0.95) = 95, data[95] = 96 (off by 1)
        # p=0.99: int(100 * 0.99) = 99, data[99] = 100 (off by 1)

    def test_percentile_three_values(self):
        """Test percentile with three values to verify median calculation."""
        data = [1.0, 2.0, 3.0]

        # p=0.5: ceil(0.5 * 3) = 2, index=1 -> 2.0
        assert MetricsAggregator._percentile(data, 0.5) == 2.0

        # p=0.33: ceil(0.33 * 3) = 1, index=0 -> 1.0
        assert MetricsAggregator._percentile(data, 0.33) == 1.0

        # p=0.67: ceil(0.67 * 3) = 3, index=2 -> 3.0
        assert MetricsAggregator._percentile(data, 0.67) == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
