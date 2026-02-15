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
    - Query Performance: Database query plan analysis with EXPLAIN ANALYZE

Sub-packages:
    - database: PostgreSQL query performance tests with EXPLAIN ANALYZE
    - event_bus: Kafka/Redpanda event bus latency and throughput tests
    - registration: Effect node performance tests

Usage:
    Run all performance tests:
        uv run pytest tests/performance/ -v

    Run with performance marker only:
        uv run pytest -m performance -v

    Run database query performance tests:
        uv run pytest tests/performance/database/ -v

    Skip performance tests in CI:
        uv run pytest --ignore=tests/performance/

Related:
    - OMN-954: Effect node testing requirements
    - PR #101: Add updated_at index for audit queries
"""
