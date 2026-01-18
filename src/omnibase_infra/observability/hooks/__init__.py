# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pipeline hooks for cross-cutting observability concerns.

This module provides hooks that can be integrated into infrastructure pipelines
to enable observability instrumentation without modifying the core business logic.

Key Components:
    - HookObservability: Pipeline hook for timing, metrics, and context tracking
    - get_global_hook(): Thread-safe access to a global singleton hook
    - configure_global_hook(): Configure the global singleton on first use
    - clear_global_hook(): Reset the singleton for reconfiguration or testing
    - has_global_hook(): Check if a singleton exists

Design Philosophy:
    Hooks use contextvars for all per-operation state to ensure concurrency
    safety in async code. This is a CRITICAL design decision - using instance
    variables for timing state would cause race conditions when multiple
    concurrent operations use the same hook instance.

    The module also provides optional singleton support via get_global_hook()
    for applications that need a shared observability hook across components.
    The singleton is thread-safe and logs a warning when configuration is
    ignored for an existing instance.

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

    # Or use the global singleton
    from omnibase_infra.observability.hooks import get_global_hook
    hook = get_global_hook(metrics_sink=sink)  # Returns singleton
    ```

See Also:
    - ProtocolHotPathMetricsSink: Metrics collection protocol
    - correlation.py: Correlation ID management (same contextvar pattern)
"""

from omnibase_infra.observability.hooks.hook_observability import (
    HookObservability,
    clear_global_hook,
    configure_global_hook,
    get_global_hook,
    has_global_hook,
)

__all__ = [
    "HookObservability",
    "get_global_hook",
    "configure_global_hook",
    "clear_global_hook",
    "has_global_hook",
]
