# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database query performance tests for ONEX infrastructure.

This package contains performance tests that validate query efficiency
using PostgreSQL EXPLAIN ANALYZE to verify index usage and query plans.

Test Categories:
    - Index Verification: Confirm indexes are used for expected queries
    - Query Plan Analysis: Analyze sequential vs index scans
    - Audit Query Performance: Verify updated_at index efficiency
    - Time-Range Query Performance: Validate time-based filtering

Requirements:
    - PostgreSQL database (real or Docker-based)
    - Schema initialized with all migrations applied
    - POSTGRES_HOST, POSTGRES_PASSWORD environment variables

Usage:
    Run database performance tests:
        uv run pytest tests/performance/database/ -v

    Skip if PostgreSQL unavailable:
        Tests auto-skip with clear message if database is unreachable

Related:
    - PR #101: Add updated_at index for audit queries
    - OMN-944 (F1): Registration Projection Schema
"""
