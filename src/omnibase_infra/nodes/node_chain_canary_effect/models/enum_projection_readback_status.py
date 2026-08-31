# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Outcome of the correlation-scoped projection readback (OMN-16963)."""

from __future__ import annotations

from enum import StrEnum


class EnumProjectionReadbackStatus(StrEnum):
    """Did the probe's routing decision reach a terminal state in the projection?

    This is the leg OMN-16963 adds. Before it, link 2 of OMN-16025 had no
    instrument at all: the canary asserted a terminal arrived on the bus and
    that quarantine was clean, and never read a projection. OMN-14843 is the
    standing proof that those are different layers — on stability-test, 26 of
    38 correlations sat non-terminal in ``delegation_workflow_state`` while
    the topic layer was healthy at the same moment (HW=100 against 102
    terminals). A lane in that condition reported the canary GREEN, because
    the layer the canary watched was the layer that was fine.

    ``STRANDED`` is the member that signature maps to, and it is deliberately
    distinct from ``ROW_ABSENT``: a row that exists and stopped mid-FSM is a
    projection defect, whereas no row at all may equally be a publish that
    never happened. Both are non-passing; conflating them would lose which
    layer to go look at.

    ``SKIPPED_NOT_CONFIGURED`` and ``ERROR`` are both NON-passing, and neither
    ever falls back to the bus terminal. Falling back is the defect — a leg
    that could not run makes no claim.
    """

    # A row for the probe's own correlation id reached a terminal FSM state.
    # This is what discharges OMN-16025 link 2.
    TERMINAL = "terminal"
    # A row exists for this correlation id but stopped short of terminal
    # (RECEIVED / ROUTED / INFERENCE_COMPLETED). The OMN-14843 signature.
    STRANDED = "stranded"
    # The projection was read for the budget window and carried no row at all
    # for this correlation id.
    ROW_ABSENT = "row_absent"
    # The readback was configured but could not be completed (store
    # unreachable, relation missing, query error). Fails closed.
    ERROR = "error"
    # No projection store was configured for the readback. No claim is made
    # about the routing decision, and therefore no green is available.
    SKIPPED_NOT_CONFIGURED = "skipped_not_configured"


__all__ = ["EnumProjectionReadbackStatus"]
