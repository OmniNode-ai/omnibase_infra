# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for DLQ tracking service.

This module contains integration tests for the DLQ PostgreSQL tracking
service, validating behavior against real database infrastructure.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without database access:

Skip Conditions:
    - Skips if POSTGRES_HOST not set
    - Skips if POSTGRES_PASSWORD not set
    - Module-level ``pytestmark`` with ``pytest.mark.skipif`` used

Related Ticket: OMN-1032 - Complete DLQ Replay PostgreSQL Tracking Integration
"""
