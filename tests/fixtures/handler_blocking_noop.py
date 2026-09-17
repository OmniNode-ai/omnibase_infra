# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test-only fixture handler that never returns within the caller's bound (OMN-17516).

This module is NOT production code. It is the sibling of
``handler_correlated_noop`` with one difference that is the whole point: its
``handle`` blocks in a plain ``time.sleep`` for far longer than any bound a
test will declare, so the delegation terminal it would eventually publish never
reaches the CLI inside the caller's ``--timeout``.

**Why a real blocking handler rather than a stubbed ``run_receipt_mode``.**
``--timeout`` bounds only ``RuntimeLocal``'s ``asyncio.wait_for`` on the
terminal event, and that wait is reached only AFTER the entry handler returns.
On the in-process locus the delegate orchestrator does the whole delegation
inside that handler call, so the declared timeout covers none of the work that
actually takes the time, and the only bound left is the CLI's ``SIGALRM``
backstop. A stub that replaces ``run_receipt_mode`` cannot exhibit that,
because it removes the very layer whose bounding is in question
(``feedback_real_dispatch_path_tests``: a test that mocks the platform's own
resolution layer proves nothing about it). Driving the real CLI at this
contract exercises the real payload write, the real receipt mode, the real
``RuntimeLocal`` and the real in-memory bus.

``time.sleep`` is deliberate and is not a stand-in for something subtler: it is
a synchronous, non-cooperative block, which is exactly the shape
``asyncio.wait_for`` cannot preempt and ``SIGALRM`` can. Production handlers
MUST NOT import from or depend on this module.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

#: Long enough that no bound a test declares can be reached by the handler
#: returning on its own, so a test that passes proves the CLI bounded the call
#: rather than the handler finishing first. The test asserts elapsed time well
#: below this, so a regression that removes the bound fails on the clock
#: instead of hanging the suite for this long.
BLOCKING_SECONDS = 600


class ModelBlockingNoopRequest(BaseModel):
    """Test-only input model carrying the caller's correlation id."""

    correlation_id: str = ""
    prompt: str = ""
    task_type: str = ""


class ModelBlockingNoopResult(BaseModel):
    """Test-only output model that no caller ever actually receives."""

    status: str = "success"
    correlation_id: str


class HandlerBlockingNoop:
    """Block past any declared bound, so no terminal reaches the caller in time."""

    def handle(self, request: ModelBlockingNoopRequest) -> ModelBlockingNoopResult:
        time.sleep(BLOCKING_SECONDS)
        return ModelBlockingNoopResult(  # pragma: no cover - never reached
            status="success",
            correlation_id=request.correlation_id,
        )
