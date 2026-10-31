# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerWorkLedgerQuery (OMN-21683).

The ledger CLI reads the work ledger through ``HandlerWorkLedgerQuery``. This
module is the one place that builds it from the resolved bus transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.handlers.handler_work_ledger_query import HandlerWorkLedgerQuery

if TYPE_CHECKING:
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

__all__: list[str] = ["bind_ledger_query"]


def bind_ledger_query(
    transport: ModelKafkaEventBusConfig,
) -> HandlerWorkLedgerQuery:
    """Build the work-ledger query handler on the given bus transport."""
    return HandlerWorkLedgerQuery(transport)
