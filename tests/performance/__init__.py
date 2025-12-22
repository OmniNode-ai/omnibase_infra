# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance tests for omnibase_infra.

This package contains performance and stress tests for infrastructure
components. These tests validate behavior under high load and measure
latency characteristics.

Test Categories:
    - High Volume: Sequential processing of many requests
    - Concurrent Load: Parallel request processing
    - Memory Bounds: Verify bounded memory usage
    - Latency Distribution: Measure p50, p95, p99 latencies

Usage:
    Run all performance tests:
        poetry run pytest tests/performance/ -v

    Run with performance marker only:
        poetry run pytest -m performance -v

    Skip performance tests in CI:
        poetry run pytest --ignore=tests/performance/

Related:
    - OMN-954: Effect node testing requirements
"""
