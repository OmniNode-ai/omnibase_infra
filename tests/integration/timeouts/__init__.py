# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for restart-safe durable timeout behavior.

This package contains integration tests verifying OMN-932 acceptance criteria:
- Deadlines stored in projections survive restarts
- Orchestrator queries for overdue entities periodically
- Timeout events emitted correctly
- Restart-safe behavior verified
- No in-memory-only deadlines
- Emission markers prevent duplicates

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
"""
