"""
Dashboard components for monitoring and debugging omninode_bridge.

This package provides dashboard and monitoring tools for the omninode_bridge
infrastructure, including event tracing, metrics visualization, and debugging
utilities for autonomous code generation workflows.

Modules:
    codegen_event_tracer: Event tracing for code generation sessions

Components:
    CodegenEventTracer: Trace event flows for debugging sessions

Usage:
    ```python
    from omninode_bridge.dashboard import CodegenEventTracer
    from omninode_bridge.infrastructure.postgres_connection_manager import (
        PostgresConnectionManager,
        ModelPostgresConfig
    )

    # Initialize database connection
    config = ModelPostgresConfig.from_environment()
    db_manager = PostgresConnectionManager(config)
    await db_manager.initialize()

    # Create event tracer
    tracer = CodegenEventTracer(db_manager)

    # Trace session events
    # Note: Event tracing implementation has placeholder queries.
    # Full implementation coming in Phase 2.
    trace = await tracer.trace_session_events(session_id)
    ```
"""

from omninode_bridge.dashboard.codegen_event_tracer import CodegenEventTracer

__all__ = ["CodegenEventTracer"]
