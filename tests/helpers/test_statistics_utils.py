# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for statistics utilities.

This module tests the statistical utilities used in performance testing,
including PerformanceStats, MemoryTracker, binomial confidence intervals,
and the run_with_warmup functions.

Related:
    - OMN-955: Chaos and replay testing
    - statistics_utils.py: Implementation
"""

from __future__ import annotations

import pytest

from tests.helpers.statistics_utils import (
    BinomialConfidenceInterval,
    MemoryTracker,
    PerformanceStats,
    calculate_binomial_confidence_interval,
    minimum_sample_size_for_tolerance,
    run_with_warmup_sync,
)

# =============================================================================
# PerformanceStats Tests
# =============================================================================


class TestPerformanceStats:
    """Tests for PerformanceStats class."""

    def test_from_samples_basic(self) -> None:
        """Test basic statistics calculation."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = PerformanceStats.from_samples(samples)

        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0

    def test_from_samples_percentiles(self) -> None:
        """Test percentile calculations."""
        # Create a list of 100 values for clean percentile calculations
        samples = [float(i) for i in range(1, 101)]
        stats = PerformanceStats.from_samples(samples)

        assert stats.count == 100
        assert stats.p50 == pytest.approx(50.5, rel=0.01)
        assert stats.p90 == pytest.approx(90.1, rel=0.01)
        assert stats.p95 == pytest.approx(95.05, rel=0.01)
        assert stats.p99 == pytest.approx(99.01, rel=0.01)

    def test_from_samples_coefficient_of_variation(self) -> None:
        """Test coefficient of variation calculation."""
        # Low variance samples
        samples_low_variance = [1.0, 1.1, 1.0, 0.9, 1.0]
        stats_low = PerformanceStats.from_samples(samples_low_variance)
        assert stats_low.coefficient_of_variation < 0.1

        # High variance samples
        samples_high_variance = [1.0, 5.0, 2.0, 8.0, 3.0]
        stats_high = PerformanceStats.from_samples(samples_high_variance)
        assert stats_high.coefficient_of_variation > 0.5

    def test_from_samples_too_few_samples(self) -> None:
        """Test error handling for insufficient samples."""
        with pytest.raises(ValueError, match="At least 2 samples required"):
            PerformanceStats.from_samples([1.0])

        with pytest.raises(ValueError, match="At least 2 samples required"):
            PerformanceStats.from_samples([])

    def test_format_report(self) -> None:
        """Test report formatting."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = PerformanceStats.from_samples(samples)
        report = stats.format_report("Test Run")

        assert "Test Run" in report
        assert "Mean" in report
        assert "Median" in report
        assert "P95" in report


# =============================================================================
# MemoryTracker Tests
# =============================================================================


class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_start_and_stop(self) -> None:
        """Test basic start/stop lifecycle."""
        tracker = MemoryTracker()
        tracker.start()
        assert tracker._started is True

        tracker.stop()
        assert tracker._started is False

    def test_snapshot_requires_start(self) -> None:
        """Test that snapshot fails if not started."""
        tracker = MemoryTracker()

        with pytest.raises(RuntimeError, match="tracking not started"):
            tracker.snapshot("test")

    def test_snapshot_and_growth(self) -> None:
        """Test snapshot and memory growth calculation."""
        tracker = MemoryTracker()
        tracker.start()

        # Take baseline snapshot
        tracker.snapshot("baseline")

        # Allocate some memory
        data = list(range(100000))
        tracker.snapshot("after_allocation")

        growth = tracker.get_growth_mb("after_allocation")

        # Should have some growth (exact amount varies by Python version)
        assert growth >= 0  # At minimum, no negative growth

        # Clean up
        del data
        tracker.stop()

    def test_get_growth_without_start(self) -> None:
        """Test error when getting growth without start."""
        tracker = MemoryTracker()

        with pytest.raises(RuntimeError, match="tracking not started"):
            tracker.get_growth_mb()

    def test_get_growth_unknown_snapshot(self) -> None:
        """Test error for unknown snapshot name."""
        tracker = MemoryTracker()
        tracker.start()

        with pytest.raises(KeyError, match="not found"):
            tracker.get_growth_mb("nonexistent")

        tracker.stop()

    def test_format_report(self) -> None:
        """Test memory report formatting."""
        tracker = MemoryTracker()
        tracker.start()
        tracker.snapshot("test_snap")
        report = tracker.format_report()

        assert "Memory Report" in report
        assert "Baseline" in report
        assert "test_snap" in report

        tracker.stop()


# =============================================================================
# Binomial Confidence Interval Tests
# =============================================================================


class TestBinomialConfidenceInterval:
    """Tests for binomial confidence interval calculations."""

    def test_basic_interval(self) -> None:
        """Test basic confidence interval calculation."""
        # 30 successes out of 100 trials -> 30% observed rate
        ci = calculate_binomial_confidence_interval(30, 100, 0.95)

        assert isinstance(ci, BinomialConfidenceInterval)
        assert ci.observed_rate == 0.3
        assert ci.sample_size == 100
        assert ci.confidence_level == 0.95

        # 95% CI for p=0.3, n=100 should be approximately [0.21, 0.40]
        assert ci.lower < 0.3
        assert ci.upper > 0.3
        assert 0.20 <= ci.lower <= 0.25
        assert 0.35 <= ci.upper <= 0.42

    def test_zero_successes(self) -> None:
        """Test interval with zero successes."""
        ci = calculate_binomial_confidence_interval(0, 100, 0.95)

        assert ci.observed_rate == 0.0
        assert ci.lower == 0.0
        assert ci.upper > 0.0  # Upper bound should be positive

    def test_all_successes(self) -> None:
        """Test interval with all successes."""
        ci = calculate_binomial_confidence_interval(100, 100, 0.95)

        assert ci.observed_rate == 1.0
        assert ci.upper == 1.0
        assert ci.lower < 1.0  # Lower bound should be less than 1

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="must be positive"):
            calculate_binomial_confidence_interval(50, 0, 0.95)

        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_binomial_confidence_interval(-1, 100, 0.95)

        with pytest.raises(ValueError, match="cannot exceed"):
            calculate_binomial_confidence_interval(101, 100, 0.95)


class TestMinimumSampleSize:
    """Tests for minimum sample size calculation."""

    def test_basic_calculation(self) -> None:
        """Test basic sample size calculation."""
        # For 30% expected rate with 20% tolerance
        n = minimum_sample_size_for_tolerance(0.3, 0.2, 0.95)

        # Should need at least 50-100 samples
        assert n >= 50
        assert n <= 500  # Reasonable upper bound

    def test_tighter_tolerance_needs_more_samples(self) -> None:
        """Test that tighter tolerance requires more samples."""
        n_loose = minimum_sample_size_for_tolerance(0.3, 0.3, 0.95)
        n_tight = minimum_sample_size_for_tolerance(0.3, 0.1, 0.95)

        assert n_tight > n_loose

    def test_higher_confidence_needs_more_samples(self) -> None:
        """Test that higher confidence requires more samples."""
        n_low = minimum_sample_size_for_tolerance(0.3, 0.2, 0.90)
        n_high = minimum_sample_size_for_tolerance(0.3, 0.2, 0.99)

        assert n_high > n_low

    def test_minimum_returned(self) -> None:
        """Test that minimum sample size is at least 10."""
        # Edge case: p=0 or p=1 with variance=0
        n = minimum_sample_size_for_tolerance(0.0, 0.5, 0.95)
        assert n >= 10

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="expected_rate"):
            minimum_sample_size_for_tolerance(1.5, 0.2, 0.95)

        with pytest.raises(ValueError, match="tolerance"):
            minimum_sample_size_for_tolerance(0.3, 0.0, 0.95)

        with pytest.raises(ValueError, match="confidence_level"):
            minimum_sample_size_for_tolerance(0.3, 0.2, 1.5)


# =============================================================================
# Run with Warmup Tests
# =============================================================================


class TestRunWithWarmup:
    """Tests for run_with_warmup functions."""

    def test_run_with_warmup_sync_basic(self) -> None:
        """Test basic sync warmup execution."""
        call_count = 0

        def operation() -> None:
            nonlocal call_count
            call_count += 1

        timings = run_with_warmup_sync(
            operation=operation,
            iterations=5,
            warmup_iterations=2,
        )

        # Should have run 7 times total (2 warmup + 5 timed)
        assert call_count == 7
        # Should return 5 timing values
        assert len(timings) == 5
        # All timings should be positive
        assert all(t > 0 for t in timings)

    def test_run_with_warmup_sync_discards_warmup(self) -> None:
        """Test that warmup iterations are discarded."""
        results: list[int] = []

        def operation() -> None:
            results.append(len(results))

        timings = run_with_warmup_sync(
            operation=operation,
            iterations=3,
            warmup_iterations=2,
        )

        # Should have 5 results but only 3 timings
        assert len(results) == 5
        assert len(timings) == 3

    def test_run_with_warmup_sync_invalid_iterations(self) -> None:
        """Test error handling for invalid iterations."""
        with pytest.raises(ValueError, match="at least 1"):
            run_with_warmup_sync(lambda: None, iterations=0)


# =============================================================================
# Integration with assert_failure_rate_within_tolerance
# =============================================================================


class TestAssertFailureRateIntegration:
    """Integration tests for assert_failure_rate_within_tolerance."""

    def test_edge_case_zero_attempts(self) -> None:
        """Test that zero attempts raises assertion error."""
        from tests.chaos.conftest import assert_failure_rate_within_tolerance

        with pytest.raises(AssertionError, match="Cannot validate"):
            assert_failure_rate_within_tolerance(
                actual_failures=0,
                total_attempts=0,
                expected_rate=0.3,
            )

    def test_edge_case_zero_expected_rate(self) -> None:
        """Test behavior with expected rate of 0%."""
        from tests.chaos.conftest import assert_failure_rate_within_tolerance

        # Should pass with 0 failures
        assert_failure_rate_within_tolerance(
            actual_failures=0,
            total_attempts=100,
            expected_rate=0.0,
            warn_on_small_sample=False,
        )

        # Should pass with 1 failure (tolerance allows small number)
        assert_failure_rate_within_tolerance(
            actual_failures=1,
            total_attempts=100,
            expected_rate=0.0,
            tolerance=0.2,
            warn_on_small_sample=False,
        )

        # Should fail with many failures
        with pytest.raises(AssertionError, match="Expected 0 failures"):
            assert_failure_rate_within_tolerance(
                actual_failures=10,
                total_attempts=100,
                expected_rate=0.0,
                warn_on_small_sample=False,
            )

    def test_edge_case_100_percent_rate(self) -> None:
        """Test behavior with expected rate of 100%."""
        from tests.chaos.conftest import assert_failure_rate_within_tolerance

        # Should pass with all failures
        assert_failure_rate_within_tolerance(
            actual_failures=100,
            total_attempts=100,
            expected_rate=1.0,
            warn_on_small_sample=False,
        )

        # Should pass with 80% (within 20% tolerance)
        assert_failure_rate_within_tolerance(
            actual_failures=80,
            total_attempts=100,
            expected_rate=1.0,
            tolerance=0.2,
            warn_on_small_sample=False,
        )

        # Should fail with 50%
        with pytest.raises(AssertionError, match="Expected 100%"):
            assert_failure_rate_within_tolerance(
                actual_failures=50,
                total_attempts=100,
                expected_rate=1.0,
                tolerance=0.2,
                warn_on_small_sample=False,
            )

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        from tests.chaos.conftest import assert_failure_rate_within_tolerance

        # Negative failures
        with pytest.raises(AssertionError, match="must be >= 0"):
            assert_failure_rate_within_tolerance(
                actual_failures=-1,
                total_attempts=100,
                expected_rate=0.3,
            )

        # Failures > attempts
        with pytest.raises(AssertionError, match="total_attempts"):
            assert_failure_rate_within_tolerance(
                actual_failures=101,
                total_attempts=100,
                expected_rate=0.3,
            )

        # Invalid expected rate
        with pytest.raises(AssertionError, match="must be in"):
            assert_failure_rate_within_tolerance(
                actual_failures=30,
                total_attempts=100,
                expected_rate=1.5,
            )

        # Invalid tolerance
        with pytest.raises(AssertionError, match="must be > 0"):
            assert_failure_rate_within_tolerance(
                actual_failures=30,
                total_attempts=100,
                expected_rate=0.3,
                tolerance=0,
            )

    def test_sample_size_warning(self) -> None:
        """Test that small sample warning is included."""
        from tests.chaos.conftest import assert_failure_rate_within_tolerance

        # Pass but with small sample
        # Should include warning in error message if it would fail
        try:
            assert_failure_rate_within_tolerance(
                actual_failures=15,  # Way off from expected
                total_attempts=20,
                expected_rate=0.3,
                tolerance=0.2,
                warn_on_small_sample=True,
                minimum_sample_size=30,
            )
        except AssertionError as e:
            assert "WARNING" in str(e)
            assert "sample size" in str(e).lower()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TestPerformanceStats",
    "TestMemoryTracker",
    "TestBinomialConfidenceInterval",
    "TestMinimumSampleSize",
    "TestRunWithWarmup",
    "TestAssertFailureRateIntegration",
]
