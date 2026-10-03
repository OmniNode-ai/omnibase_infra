# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the sync-revert watchdog tests."""

from __future__ import annotations

import pytest

from omnibase_core.models.ticket.model_done_write_decision import (
    ModelDoneWriteDecision,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.handlers import (
    handler_sync_revert_watchdog,
)


class _AllowingDoneWriteGuard:
    """Stands in for the bound-receipt gate in tests that are not about it.

    The tests that ARE about it (``test_done_write_receipt_gate.py``) pass their
    own guard to the handler, which wins over this default.
    """

    async def decide(
        self, *, ticket_id: str, description: str
    ) -> ModelDoneWriteDecision:
        return ModelDoneWriteDecision(allowed=True)


@pytest.fixture(autouse=True)
def _default_done_write_guard_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        handler_sync_revert_watchdog, "DoneWriteReceiptGuard", _AllowingDoneWriteGuard
    )
