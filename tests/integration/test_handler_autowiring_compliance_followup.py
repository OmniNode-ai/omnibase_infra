# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-8735 follow-up auto-wiring constructor compliance.

Verifies that HandlerLlmCliSubprocess and HandlerRuntimeTick can be instantiated
with no constructor arguments, as required by the strict auto-wiring framework.

These two handlers were missed in the initial OMN-8735 pass and are fixed in
the follow-up PR (omnibase_infra#1325).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestHandlerAutowiringComplianceFollowup:
    """Verify OMN-8735 follow-up: missed handlers instantiate with no constructor arguments."""

    def test_handler_llm_cli_subprocess_no_args(self) -> None:
        from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
            HandlerLlmCliSubprocess,
        )

        handler = HandlerLlmCliSubprocess()
        assert handler is not None

    def test_handler_runtime_tick_no_args(self) -> None:
        from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
            HandlerRuntimeTick,
        )

        handler = HandlerRuntimeTick()
        assert handler is not None
