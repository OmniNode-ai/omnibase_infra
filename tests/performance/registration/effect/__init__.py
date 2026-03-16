# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Performance tests for registration effect nodes.

This package contains performance tests for effect-level components
including idempotency stores, registration executors, and backend
integration.

Test Suites:
    - test_effect_performance.py: High volume, concurrent, memory tests
    - test_idempotency_store_performance.py: Store-specific stress tests

Related:
    - OMN-954: Effect node testing requirements
    - StoreEffectIdempotencyInmemory: Primary store under test
    - NodeRegistryEffect: Effect node implementation
"""
