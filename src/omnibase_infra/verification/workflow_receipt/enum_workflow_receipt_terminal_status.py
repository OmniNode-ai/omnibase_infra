# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal status enum for the OMN-15095 replay-then-diff verifier.

Mirrors the two terminal values ``gateway_workflows_status_check`` allows
past ``published`` (``db/migrations/20260725_gateway_workflows_terminal.sql``,
OMN-15093, in ``omninode_infra``): ``'completed'`` and ``'failed'``.
"""

from __future__ import annotations

from enum import Enum


class EnumWorkflowReceiptTerminalStatus(str, Enum):
    """The only two terminal states a ``workflow_receipt.json`` can report."""

    COMPLETED = "completed"
    FAILED = "failed"


__all__ = ["EnumWorkflowReceiptTerminalStatus"]
