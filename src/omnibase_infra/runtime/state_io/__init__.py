# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""State-io runtime dispatch seam (OMN-14208).

Opt-in, contract-declared boundary hook that gives an orchestrator handler a
durable, tenant-scoped, per-correlation_id FSM working row instead of
process-local in-memory state. See ``state_store_adapter`` for the storage
side and ``omnibase_infra.runtime.auto_wiring.handler_wiring`` for the
dispatch-time wiring that loads the row before ``handle()`` and CAS-persists
it after, gating publish on a successful persist.
"""

from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
    StateStoreAdapter,
)

__all__ = ["CONTEXTVAR_STATE_IO_ROWS", "StateStoreAdapter"]
