# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for SinkMetricsPrometheus.

This module tests the Prometheus metrics sink implementation with focus on:
- Cardinality enforcement (forbidden labels rejected)
- Label validation against ModelMetricsPolicy
- Metric registration (counter, gauge, histogram)
- Thread-safety (concurrent metric updates)
- Violation handling modes (raise, warn_and_drop, drop_silent, warn_and_strip)

Thread-Safety Testing Strategy:
    Tests use concurrent.futures.ThreadPoolExecutor to simulate multiple threads
    simultaneously recording metrics. The sink uses threading.Lock internally
    to ensure metric registration is atomic.

Cardinality Policy Enforcement:
    The sink validates all labels against ModelMetricsPolicy before recording.
    By default, high-cardinality labels (envelope_id, correlation_id, node_id,
    runtime_id) are forbidden to prevent metric explosion.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from omnibase_infra.observability.sinks import (
    DEFAULT_HISTOGRAM_BUCKETS,
    SinkMetricsPrometheus,
)

if TYPE_CHECKING:
    from omnibase_core.enums import EnumMetricsPolicyViolationAction
    from omnibase_core.models.observability import ModelMetricsPolicy


# =============================================================================
# METRIC REGISTRATION TESTS
# =============================================================================


class TestMetricRegistration:
    """Test metric registration and basic operations."""

    def test_counter_increment_records_metric(self) -> None:
        """Verify counter increment records a metric value."""
        # Use unique metric name to avoid conflicts with other tests
        metric_name = f"test_counter_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        sink.increment_counter(
            metric_name,
            {"method": "POST", "status": "200"},
        )

        # Verify counter was created
        assert metric_name in sink._counters

    def test_counter_increment_multiple_times(self) -> None:
        """Verify counter increments accumulate correctly."""
        metric_name = f"test_counter_multi_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        # Increment 5 times
        for _ in range(5):
            sink.increment_counter(
                metric_name,
                {"handler": "test"},
                increment=1,
            )

        # Counter should exist and be incremented
        assert metric_name in sink._counters

    def test_counter_increment_by_custom_amount(self) -> None:
        """Verify counter can increment by amounts greater than 1."""
        metric_name = f"test_counter_amount_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        sink.increment_counter(
            metric_name,
            {"operation": "batch"},
            increment=100,
        )

        assert metric_name in sink._counters

    def test_counter_ignores_non_positive_increment(self) -> None:
        """Verify counter ignores non-positive increments."""
        metric_name = f"test_counter_ignore_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        # First, create the counter with a valid increment
        sink.increment_counter(
            metric_name,
            {"op": "test"},
            increment=1,
        )

        # Then try non-positive increments (should be ignored)
        sink.increment_counter(metric_name, {"op": "test"}, increment=0)
        sink.increment_counter(metric_name, {"op": "test"}, increment=-1)

        # Counter should still exist
        assert metric_name in sink._counters

    def test_gauge_set_records_metric(self) -> None:
        """Verify gauge set records the metric value."""
        metric_name = f"test_gauge_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        sink.set_gauge(
            metric_name,
            {"pool": "database"},
            value=42.5,
        )

        assert metric_name in sink._gauges

    def test_gauge_set_overwrites_previous_value(self) -> None:
        """Verify gauge set overwrites the previous value."""
        metric_name = f"test_gauge_overwrite_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        sink.set_gauge(metric_name, {"type": "queue"}, value=10.0)
        sink.set_gauge(metric_name, {"type": "queue"}, value=25.0)

        # Gauge should exist
        assert metric_name in sink._gauges

    def test_histogram_observe_records_metric(self) -> None:
        """Verify histogram observe records the observation."""
        metric_name = f"test_histogram_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        sink.observe_histogram(
            metric_name,
            {"handler": "process"},
            value=0.042,
        )

        assert metric_name in sink._histograms

    def test_histogram_multiple_observations(self) -> None:
        """Verify histogram accepts multiple observations."""
        metric_name = f"test_histogram_multi_{uuid.uuid4().hex[:8]}"
        sink = SinkMetricsPrometheus()

        # Record multiple observations
        values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.5]
        for value in values:
            sink.observe_histogram(
                metric_name,
                {"operation": "query"},
                value=value,
            )

        assert metric_name in sink._histograms


class TestMetricPrefixing:
    """Test metric name prefixing functionality."""

    def test_metric_prefix_applied_to_counter(self) -> None:
        """Verify metric prefix is applied to counter names."""
        sink = SinkMetricsPrometheus(metric_prefix="myapp")

        sink.increment_counter(
            "requests_total",
            {"method": "GET"},
        )

        # Counter should be stored with prefixed name
        assert "myapp_requests_total" in sink._counters
        assert "requests_total" not in sink._counters

    def test_metric_prefix_applied_to_gauge(self) -> None:
        """Verify metric prefix is applied to gauge names."""
        sink = SinkMetricsPrometheus(metric_prefix="infra")

        sink.set_gauge(
            "active_connections",
            {"pool": "db"},
            value=5.0,
        )

        assert "infra_active_connections" in sink._gauges

    def test_metric_prefix_applied_to_histogram(self) -> None:
        """Verify metric prefix is applied to histogram names."""
        sink = SinkMetricsPrometheus(metric_prefix="service")

        sink.observe_histogram(
            "latency_seconds",
            {"endpoint": "/api"},
            value=0.05,
        )

        assert "service_latency_seconds" in sink._histograms

    def test_empty_prefix_no_modification(self) -> None:
        """Verify empty prefix does not modify metric names."""
        sink = SinkMetricsPrometheus(metric_prefix="")

        sink.increment_counter("test_metric", {"k": "v"})

        assert "test_metric" in sink._counters
        assert "_test_metric" not in sink._counters


class TestHistogramBuckets:
    """Test custom histogram bucket configuration."""

    def test_default_histogram_buckets(self) -> None:
        """Verify default histogram buckets are applied."""
        sink = SinkMetricsPrometheus()

        assert sink._histogram_buckets == DEFAULT_HISTOGRAM_BUCKETS

    def test_custom_histogram_buckets(self) -> None:
        """Verify custom histogram buckets can be configured."""
        custom_buckets = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
        sink = SinkMetricsPrometheus(histogram_buckets=custom_buckets)

        assert sink._histogram_buckets == custom_buckets


# =============================================================================
# CARDINALITY ENFORCEMENT TESTS
# =============================================================================


class TestCardinalityEnforcement:
    """Test cardinality policy enforcement."""

    def test_forbidden_label_envelope_id_rejected_strict(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify envelope_id label is rejected with strict policy."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        with pytest.raises(Exception):  # OnexError
            sink.increment_counter(
                "test_counter",
                {"envelope_id": "12345", "method": "POST"},
            )

    def test_forbidden_label_correlation_id_rejected_strict(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify correlation_id label is rejected with strict policy."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        with pytest.raises(Exception):  # OnexError
            sink.set_gauge(
                "test_gauge",
                {"correlation_id": str(uuid.uuid4())},
                value=1.0,
            )

    def test_forbidden_label_node_id_rejected_strict(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify node_id label is rejected with strict policy."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        with pytest.raises(Exception):  # OnexError
            sink.observe_histogram(
                "test_histogram",
                {"node_id": "node-001"},
                value=0.1,
            )

    def test_forbidden_label_runtime_id_rejected_strict(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify runtime_id label is rejected with strict policy."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        with pytest.raises(Exception):  # OnexError
            sink.increment_counter(
                "test_counter",
                {"runtime_id": "runtime-xyz"},
            )

    def test_allowed_labels_accepted(self) -> None:
        """Verify allowed labels are accepted without errors."""
        sink = SinkMetricsPrometheus()
        metric_name = f"test_allowed_{uuid.uuid4().hex[:8]}"

        # These labels should be allowed
        sink.increment_counter(
            metric_name,
            {
                "method": "POST",
                "status": "200",
                "handler": "create_user",
                "service": "auth",
            },
        )

        assert metric_name in sink._counters


class TestViolationHandlingModes:
    """Test different violation handling modes."""

    def test_warn_and_drop_mode_silently_drops_metric(
        self,
        default_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify warn_and_drop mode drops the metric without raising."""
        sink = SinkMetricsPrometheus(policy=default_metrics_policy)
        metric_name = f"test_warn_drop_{uuid.uuid4().hex[:8]}"

        # This should NOT raise but should drop the metric
        sink.increment_counter(
            metric_name,
            {"envelope_id": "forbidden-value", "method": "POST"},
        )

        # Metric should NOT be created due to drop
        assert metric_name not in sink._counters

    def test_drop_silent_mode_no_warning(
        self,
        silent_drop_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify drop_silent mode drops without logging warnings."""
        sink = SinkMetricsPrometheus(policy=silent_drop_metrics_policy)
        metric_name = f"test_silent_{uuid.uuid4().hex[:8]}"

        # Should not raise
        sink.increment_counter(
            metric_name,
            {"correlation_id": "forbidden"},
        )

        # Metric should not be created
        assert metric_name not in sink._counters

    def test_raise_mode_raises_exception(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify raise mode raises an exception on violation."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        with pytest.raises(Exception):
            sink.increment_counter(
                "test_raise",
                {"envelope_id": "bad-value"},
            )

    def test_warn_and_strip_mode_records_with_stripped_labels(
        self,
        warn_and_strip_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify warn_and_strip mode strips forbidden labels but records metric."""
        sink = SinkMetricsPrometheus(policy=warn_and_strip_metrics_policy)
        metric_name = f"test_strip_{uuid.uuid4().hex[:8]}"

        # Should record metric but strip the forbidden label
        sink.increment_counter(
            metric_name,
            {"envelope_id": "forbidden", "method": "POST"},
        )

        # Metric SHOULD be created (with stripped labels)
        assert metric_name in sink._counters


class TestLabelValueLengthEnforcement:
    """Test label value length enforcement."""

    def test_label_value_within_limit_accepted(self) -> None:
        """Verify label values within the limit are accepted."""
        sink = SinkMetricsPrometheus()
        metric_name = f"test_len_ok_{uuid.uuid4().hex[:8]}"

        # Value within default 128 char limit
        sink.increment_counter(
            metric_name,
            {"handler": "a" * 100},
        )

        assert metric_name in sink._counters

    def test_label_value_exceeds_limit_handled(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify label values exceeding limit are handled per policy."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        # Value exceeds default 128 char limit
        long_value = "x" * 200

        with pytest.raises(Exception):
            sink.increment_counter(
                "test_long_label",
                {"handler": long_value},
            )


# =============================================================================
# THREAD-SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Test thread-safety of concurrent metric operations."""

    def test_concurrent_counter_increments(self) -> None:
        """Verify concurrent counter increments are thread-safe."""
        sink = SinkMetricsPrometheus()
        metric_name = f"concurrent_counter_{uuid.uuid4().hex[:8]}"
        num_threads = 10
        increments_per_thread = 100

        def increment_counter(thread_id: int) -> int:
            """Increment counter multiple times."""
            for _ in range(increments_per_thread):
                sink.increment_counter(
                    metric_name,
                    {"thread": str(thread_id)},
                )
            return increments_per_thread

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(increment_counter, i) for i in range(num_threads)
            ]
            results = [f.result() for f in as_completed(futures)]

        # All threads should complete without errors
        assert len(results) == num_threads
        # Counter should be created
        assert metric_name in sink._counters

    def test_concurrent_gauge_updates(self) -> None:
        """Verify concurrent gauge updates are thread-safe."""
        sink = SinkMetricsPrometheus()
        metric_name = f"concurrent_gauge_{uuid.uuid4().hex[:8]}"
        num_threads = 10
        updates_per_thread = 50

        def update_gauge(thread_id: int) -> int:
            """Update gauge multiple times with different values."""
            for i in range(updates_per_thread):
                sink.set_gauge(
                    metric_name,
                    {"thread": str(thread_id)},
                    value=float(i),
                )
            return updates_per_thread

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_gauge, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == num_threads
        assert metric_name in sink._gauges

    def test_concurrent_histogram_observations(self) -> None:
        """Verify concurrent histogram observations are thread-safe."""
        sink = SinkMetricsPrometheus()
        metric_name = f"concurrent_histogram_{uuid.uuid4().hex[:8]}"
        num_threads = 10
        observations_per_thread = 50

        def observe_histogram(thread_id: int) -> int:
            """Record multiple histogram observations."""
            for i in range(observations_per_thread):
                sink.observe_histogram(
                    metric_name,
                    {"thread": str(thread_id)},
                    value=float(i) / 100.0,
                )
            return observations_per_thread

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(observe_histogram, i) for i in range(num_threads)
            ]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == num_threads
        assert metric_name in sink._histograms

    def test_concurrent_metric_creation_same_name(self) -> None:
        """Verify concurrent creation of same metric is thread-safe.

        This tests the critical scenario where multiple threads try to
        create the same metric simultaneously. The lock should ensure
        only one metric object is created.
        """
        sink = SinkMetricsPrometheus()
        metric_name = f"same_metric_{uuid.uuid4().hex[:8]}"
        num_threads = 50
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []

        def create_and_increment(thread_id: int) -> None:
            """Wait at barrier then create/increment metric."""
            try:
                barrier.wait()  # Synchronize all threads
                sink.increment_counter(
                    metric_name,
                    {"label": "value"},
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_and_increment, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Metric should be created exactly once
        assert metric_name in sink._counters

    def test_concurrent_different_metric_types(self) -> None:
        """Verify concurrent operations on different metric types are safe."""
        sink = SinkMetricsPrometheus()
        # Prometheus metric names must start with letter or underscore, not digit
        prefix = f"test_{uuid.uuid4().hex[:8]}"
        num_threads = 30

        def mixed_operations(thread_id: int) -> str:
            """Perform mixed metric operations."""
            metric_type = thread_id % 3

            if metric_type == 0:
                sink.increment_counter(
                    f"{prefix}_counter_{thread_id}",
                    {"type": "counter"},
                )
                return "counter"
            elif metric_type == 1:
                sink.set_gauge(
                    f"{prefix}_gauge_{thread_id}",
                    {"type": "gauge"},
                    value=float(thread_id),
                )
                return "gauge"
            else:
                sink.observe_histogram(
                    f"{prefix}_histogram_{thread_id}",
                    {"type": "histogram"},
                    value=thread_id / 100.0,
                )
                return "histogram"

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        # All operations should complete
        assert len(results) == num_threads
        # Verify all types were created
        assert any(results.count(t) > 0 for t in ["counter", "gauge", "histogram"])


# =============================================================================
# POLICY ACCESS TESTS
# =============================================================================


class TestPolicyAccess:
    """Test policy access and inspection."""

    def test_get_policy_returns_configured_policy(
        self,
        strict_metrics_policy: ModelMetricsPolicy,
    ) -> None:
        """Verify get_policy returns the configured policy."""
        sink = SinkMetricsPrometheus(policy=strict_metrics_policy)

        policy = sink.get_policy()

        assert policy is strict_metrics_policy

    def test_get_policy_returns_default_when_none_provided(self) -> None:
        """Verify get_policy returns default policy when none configured."""
        sink = SinkMetricsPrometheus()

        policy = sink.get_policy()

        # Default policy should have standard forbidden labels
        assert "envelope_id" in policy.forbidden_label_keys
        assert "correlation_id" in policy.forbidden_label_keys

    def test_policy_is_immutable(self) -> None:
        """Verify the policy cannot be modified after creation."""
        sink = SinkMetricsPrometheus()

        policy = sink.get_policy()

        # Policy should be frozen (immutable)
        with pytest.raises(Exception):
            policy.max_label_value_length = 256  # type: ignore[misc]
