# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for DLQ tracking service.  # ai-slop-ok: pre-existing

This module contains integration tests for the DLQ PostgreSQL tracking
service, validating behavior against real database infrastructure.

CI/CD Graceful Skip Behavior
============================  # ai-slop-ok: pre-existing

These tests skip gracefully in CI/CD environments without database access:

Skip Conditions:
    - Skips if OMNIBASE_INFRA_DB_URL (or POSTGRES_HOST/POSTGRES_PASSWORD fallback) not set
    - Module-level ``pytestmark`` with ``pytest.mark.skipif`` used

Related Ticket: OMN-1032 - Complete DLQ Replay PostgreSQL Tracking Integration
"""
