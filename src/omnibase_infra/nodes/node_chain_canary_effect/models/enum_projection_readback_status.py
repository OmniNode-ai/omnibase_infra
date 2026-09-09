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

    ``SKIPPED_NOT_CONFIGURED``, ``REFUSED`` and ``ERROR`` are all NON-passing,
    and none of them ever falls back to the bus terminal. Falling back is the
    defect — a leg that could not run makes no claim.

    ``REFUSED`` (OMN-18060) is deliberately distinct from ``ERROR``: an error
    is the store failing to answer, a refusal is this node declining to ask.
    The refusals are the ways a readback could have run and should not have —
    a DSN that reached the process through argv (where every other process on
    the host can read it), or a DSN whose role turns out to be ``SUPERUSER``
    or ``BYPASSRLS`` (a canary is a reader, and a reader with those attributes
    is exempt from the isolation the projection is supposed to enforce). Both
    would have produced a perfectly good green.
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
    # The readback COULD have run and this node declined to run it: the DSN
    # arrived on the command line, or the role it authenticates as carries
    # SUPERUSER / BYPASSRLS. Fails closed for the same reason as every other
    # non-passing member — a refusal is not a result (OMN-18060).
    REFUSED = "refused"


__all__ = ["EnumProjectionReadbackStatus"]
