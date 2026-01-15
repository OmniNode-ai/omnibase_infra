# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance tests for Event Bus implementations.

This package contains performance and benchmark tests for:
- EventBusInmemory throughput and latency
- Concurrent publisher/subscriber performance
- Load testing and memory stability

Test Categories:
    - Throughput: Messages per second benchmarks
    - Latency: p50, p95, p99 latency measurements
    - Load: Sustained high-volume testing

Related:
    - OMN-57: Event bus performance testing (Phase 9)
    - EventBusInmemory: Primary implementation under test
    - EventBusKafka: Production implementation (tested with mocks)
"""
