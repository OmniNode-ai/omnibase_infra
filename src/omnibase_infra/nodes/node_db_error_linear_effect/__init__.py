# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node: NodeDbErrorLinearEffect — Kafka consumer → Linear ticket reporter.

Consumes ``onex.evt.omnibase-infra.db-error.v1`` events and auto-creates
Linear tickets for unique PostgreSQL errors, with dedup and frequency
tracking via a ``db_error_tickets`` PostgreSQL table.

Related Tickets:
    - OMN-3408: Kafka Consumer → Linear Ticket Reporter (ONEX Node)
    - OMN-3407: PostgreSQL Error Emitter (hard prerequisite)
"""

from omnibase_infra.nodes.node_db_error_linear_effect.node import (
    NodeDbErrorLinearEffect,
)

__all__ = ["NodeDbErrorLinearEffect"]
