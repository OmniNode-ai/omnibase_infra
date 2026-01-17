# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pipeline hooks for cross-cutting observability concerns.

This module provides hooks that can be integrated into infrastructure pipelines
to enable observability instrumentation without modifying the core business logic.

Key Components:
    - HookObservability: Pipeline hook for timing, metrics, and context tracking

Design Philosophy:
    Hooks use contextvars for all per-operation state to ensure concurrency
    safety in async code. This is a CRITICAL design decision - using instance
    variables for timing state would cause race conditions when multiple
    concurrent operations use the same hook instance.

Usage Example:
    ```python
    from omnibase_infra.observability.hooks import HookObservability
    from omnibase_spi.protocols.observability import ProtocolHotPathMetricsSink

    # Create hook with optional metrics sink
    sink: ProtocolHotPathMetricsSink = get_metrics_sink()
    hook = HookObservability(metrics_sink=sink)

    # Use context manager for automatic timing
    with hook.operation_context("handler.process", correlation_id="abc-123"):
        result = await handler.execute(payload)

    # Or use manual timing for more control
    hook.before_operation("db.query", correlation_id="abc-123")
    try:
        result = await db.execute(query)
        hook.record_success()
    except Exception as e:
        hook.record_failure(type(e).__name__)
        raise
    finally:
        duration_ms = hook.after_operation()
    ```

See Also:
    - ProtocolHotPathMetricsSink: Metrics collection protocol
    - correlation.py: Correlation ID management (same contextvar pattern)
"""

from omnibase_infra.observability.hooks.hook_observability import HookObservability

__all__ = [
    "HookObservability",
]
