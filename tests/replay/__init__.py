# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Replay Verification Tests for OMN-955.

This package contains tests for verifying event replay correctness in the
ONEX infrastructure. The tests validate that:

1. Event sequences can be captured and replayed
2. Reducer replay is deterministic (same input -> same output)
3. Out-of-order events are detected and handled
4. Idempotent replay produces consistent results

Architecture:
    The tests follow the ONEX pure reducer pattern where:
    - Reducers are pure functions: reduce(state, event) -> output
    - Same inputs always produce same outputs (determinism)
    - Duplicate events are detected via last_processed_event_id (idempotency)
    - State is immutable (ModelRegistrationState frozen=True)

Related:
    - RegistrationReducer: Pure reducer under test
    - ModelRegistrationState: Immutable state model
    - OMN-955: Event Replay Verification ticket
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
"""

__all__: list[str] = []
