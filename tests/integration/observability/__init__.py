# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for the ONEX observability layer.

This package contains integration tests for observability components:
- SinkMetricsPrometheus: Prometheus metrics sink with cardinality enforcement
- SinkLoggingStructured: Structured logging sink with buffer management
- HookObservability: Pipeline hook with contextvars for async safety
- Handler integration: Lifecycle and factory tests

Test Categories:
    - Thread-safety and concurrency testing
    - Cardinality policy enforcement
    - Buffer management policies (drop_oldest)
    - Contextvars isolation across async operations
    - Handler lifecycle management
"""
