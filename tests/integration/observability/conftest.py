# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for observability integration tests.

This module provides shared fixtures for testing observability components:
- Isolated Prometheus registries to prevent metric conflicts
- Mock sinks for testing without side effects
- Test metrics policies with various violation modes
- Cleanup fixtures for proper test isolation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY, CollectorRegistry

if TYPE_CHECKING:
    from collections.abc import Generator

    from omnibase_core.enums import EnumMetricsPolicyViolationAction
    from omnibase_core.models.observability import ModelMetricsPolicy


# =============================================================================
# PROMETHEUS REGISTRY FIXTURES
# =============================================================================


@pytest.fixture
def isolated_registry() -> CollectorRegistry:
    """Create an isolated Prometheus registry for test isolation.

    Using isolated registries prevents metric conflicts between tests,
    especially when tests register metrics with the same name but different
    label sets.

    Returns:
        A fresh CollectorRegistry instance for the test.

    Note:
        The default REGISTRY is global and persists across tests. Using
        isolated registries ensures each test starts with a clean state.
    """
    return CollectorRegistry()


@pytest.fixture
def cleanup_default_registry() -> Generator[None, None, None]:
    """Cleanup fixture that unregisters test metrics from the default registry.

    This fixture yields immediately and performs cleanup after the test.
    Use this when tests must use the default REGISTRY (e.g., for
    generate_latest() compatibility).

    Yields:
        None. Cleanup runs on test teardown.

    Warning:
        This fixture attempts best-effort cleanup by removing collectors
        that may have been added during the test. Some metrics may persist
        if they were created before the test started.
    """
    # Track collectors before test
    collectors_before = set(REGISTRY._names_to_collectors.keys())

    yield

    # Remove collectors added during test
    collectors_after = set(REGISTRY._names_to_collectors.keys())
    new_collectors = collectors_after - collectors_before

    for name in new_collectors:
        try:
            collector = REGISTRY._names_to_collectors.get(name)
            if collector is not None:
                REGISTRY.unregister(collector)
        except Exception:
            # Silently ignore cleanup failures
            pass


# =============================================================================
# METRICS POLICY FIXTURES
# =============================================================================


@pytest.fixture
def default_metrics_policy() -> ModelMetricsPolicy:
    """Create a default ModelMetricsPolicy with standard settings.

    Returns:
        ModelMetricsPolicy with default forbidden labels and warn_and_drop behavior.
    """
    from omnibase_core.models.observability import ModelMetricsPolicy

    return ModelMetricsPolicy()


@pytest.fixture
def strict_metrics_policy() -> ModelMetricsPolicy:
    """Create a strict ModelMetricsPolicy that raises on violations.

    Use this policy when testing that violations are properly detected
    and raised as exceptions.

    Returns:
        ModelMetricsPolicy configured to raise errors on any violation.
    """
    from omnibase_core.enums import EnumMetricsPolicyViolationAction
    from omnibase_core.models.observability import ModelMetricsPolicy

    return ModelMetricsPolicy(
        on_violation=EnumMetricsPolicyViolationAction.RAISE,
    )


@pytest.fixture
def silent_drop_metrics_policy() -> ModelMetricsPolicy:
    """Create a ModelMetricsPolicy that silently drops violations.

    Use this policy when testing that metrics are silently dropped
    without warnings or exceptions.

    Returns:
        ModelMetricsPolicy configured to silently drop violating metrics.
    """
    from omnibase_core.enums import EnumMetricsPolicyViolationAction
    from omnibase_core.models.observability import ModelMetricsPolicy

    return ModelMetricsPolicy(
        on_violation=EnumMetricsPolicyViolationAction.DROP_SILENT,
    )


@pytest.fixture
def warn_and_strip_metrics_policy() -> ModelMetricsPolicy:
    """Create a ModelMetricsPolicy that warns and strips violating labels.

    Use this policy when testing that forbidden labels are removed
    while still recording the metric with remaining labels.

    Returns:
        ModelMetricsPolicy configured to warn and strip violating labels.
    """
    from omnibase_core.enums import EnumMetricsPolicyViolationAction
    from omnibase_core.models.observability import ModelMetricsPolicy

    return ModelMetricsPolicy(
        on_violation=EnumMetricsPolicyViolationAction.WARN_AND_STRIP,
    )


def create_metrics_policy(
    on_violation: EnumMetricsPolicyViolationAction,
    forbidden_label_keys: frozenset[str] | None = None,
    max_label_value_length: int = 128,
) -> ModelMetricsPolicy:
    """Factory function to create ModelMetricsPolicy with custom settings.

    Args:
        on_violation: How to handle policy violations.
        forbidden_label_keys: Set of label keys that are forbidden.
            Defaults to standard high-cardinality labels if None.
        max_label_value_length: Maximum allowed label value length.

    Returns:
        Configured ModelMetricsPolicy instance.
    """
    from omnibase_core.models.observability import ModelMetricsPolicy

    kwargs: dict[str, object] = {
        "on_violation": on_violation,
        "max_label_value_length": max_label_value_length,
    }

    if forbidden_label_keys is not None:
        kwargs["forbidden_label_keys"] = forbidden_label_keys

    return ModelMetricsPolicy(**kwargs)


# =============================================================================
# MOCK SINK FIXTURES
# =============================================================================


@pytest.fixture
def mock_metrics_sink() -> MagicMock:
    """Create a mock metrics sink for testing without Prometheus dependencies.

    Returns:
        MagicMock configured with the ProtocolHotPathMetricsSink interface.
    """
    sink = MagicMock()
    sink.increment_counter = MagicMock()
    sink.set_gauge = MagicMock()
    sink.observe_histogram = MagicMock()
    sink.get_policy = MagicMock()
    return sink


@pytest.fixture
def mock_logging_sink() -> MagicMock:
    """Create a mock logging sink for testing without structlog dependencies.

    Returns:
        MagicMock configured with the ProtocolHotPathLoggingSink interface.
    """
    sink = MagicMock()
    sink.emit = MagicMock()
    sink.flush = MagicMock()
    sink.buffer_size = 0
    sink.drop_count = 0
    sink.max_buffer_size = 1000
    return sink


# =============================================================================
# SINK INSTANCE FIXTURES
# =============================================================================


@pytest.fixture
def metrics_sink() -> Generator[
    SinkMetricsPrometheus,  # noqa: F821 - forward reference
    None,
    None,
]:
    """Create a real SinkMetricsPrometheus instance for integration testing.

    This fixture creates a fresh sink for each test. Note that metrics
    registered in the default Prometheus registry persist across tests.

    Yields:
        SinkMetricsPrometheus instance.
    """
    from omnibase_infra.observability.sinks import SinkMetricsPrometheus

    sink = SinkMetricsPrometheus()
    return sink
    # Cleanup is limited because Prometheus doesn't support metric removal


@pytest.fixture
def logging_sink() -> Generator[
    SinkLoggingStructured,  # noqa: F821 - forward reference
    None,
    None,
]:
    """Create a real SinkLoggingStructured instance for integration testing.

    This fixture creates a sink with a small buffer for testing buffer
    management behavior.

    Yields:
        SinkLoggingStructured instance.
    """
    from omnibase_infra.observability.sinks import SinkLoggingStructured

    sink = SinkLoggingStructured(max_buffer_size=100, output_format="json")
    yield sink
    # Flush any remaining entries
    try:
        sink.flush()
    except Exception:
        pass


# =============================================================================
# HOOK FIXTURES
# =============================================================================


@pytest.fixture
def hook_without_metrics() -> HookObservability:  # noqa: F821 - forward reference
    """Create a HookObservability instance without a metrics sink.

    This hook operates in timing-only mode where metrics emission is a no-op.

    Returns:
        HookObservability configured without metrics sink.
    """
    from omnibase_infra.observability.hooks import HookObservability

    return HookObservability(metrics_sink=None)


@pytest.fixture
def hook_with_mock_metrics(
    mock_metrics_sink: MagicMock,
) -> HookObservability:  # noqa: F821 - forward reference
    """Create a HookObservability instance with a mock metrics sink.

    This allows testing that the hook emits the correct metrics without
    actually recording them in Prometheus.

    Args:
        mock_metrics_sink: Mock sink fixture.

    Returns:
        HookObservability configured with mock metrics sink.
    """
    from omnibase_infra.observability.hooks import HookObservability

    return HookObservability(metrics_sink=mock_metrics_sink)


# =============================================================================
# FACTORY FIXTURES
# =============================================================================


@pytest.fixture
def factory() -> Generator[
    FactoryObservabilitySink,  # noqa: F821 - forward reference
    None,
    None,
]:
    """Create a FactoryObservabilitySink instance for testing.

    The factory is cleared of singletons after each test to ensure
    test isolation.

    Yields:
        FactoryObservabilitySink instance.
    """
    from omnibase_infra.observability import FactoryObservabilitySink

    factory_instance = FactoryObservabilitySink()
    yield factory_instance
    # Clear singletons to ensure test isolation
    factory_instance.clear_singletons()
