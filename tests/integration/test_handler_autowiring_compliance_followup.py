# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-8735 follow-up auto-wiring constructor compliance.

Verifies that HandlerRuntimeTick can be instantiated with no constructor
arguments, as required by the strict auto-wiring framework.

This handler was missed in the initial OMN-8735 pass and is fixed in
the follow-up PR (omnibase_infra#1325).

OMN-15959: the sibling ``test_handler_llm_cli_subprocess_no_args`` case that
used to live in this class was removed along with the deleted
HandlerLlmCliSubprocess handler.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestHandlerAutowiringComplianceFollowup:
    """Verify OMN-8735 follow-up: missed handlers instantiate with no constructor arguments."""

    def test_handler_runtime_tick_no_args(self) -> None:
        from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
            HandlerRuntimeTick,
        )

        handler = HandlerRuntimeTick()
        assert handler is not None
