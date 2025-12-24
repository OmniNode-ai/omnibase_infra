# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance tests for registration effect nodes.

This package contains performance tests for effect-level components
including idempotency stores, registration executors, and backend
integration.

Test Suites:
    - test_effect_performance.py: High volume, concurrent, memory tests
    - test_idempotency_store_performance.py: Store-specific stress tests

Related:
    - OMN-954: Effect node testing requirements
    - InMemoryEffectIdempotencyStore: Primary store under test
    - NodeRegistryEffect: Effect node implementation
"""
