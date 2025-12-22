# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for registration orchestrator and handlers.

Tests validate:
- Orchestrator emits EVENTS only (no intents, no projections)
- Orchestrator uses injected `now` parameter (not system clock)
- Handler state decision logic and event emission
- Deduplication via projection emission markers (C2)
- Handler idempotency for duplicate/stale events

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - G2: Test orchestrator logic (EVENTS only, no I/O)
"""
