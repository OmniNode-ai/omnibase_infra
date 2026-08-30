# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Outcome of the correlation-scoped terminal broker readback (OMN-16931)."""

from __future__ import annotations

from enum import StrEnum


class EnumTerminalReadbackStatus(StrEnum):
    """Did the probe's terminal actually land on the bus?

    This is the leg OMN-16931 added. Before it, ``terminal_landed`` was
    derived from the synchronous ingress HTTP response, which is a CLAIM
    about the chain rather than evidence from it. Run 33251822642 proved
    both directions of that error in one day: the ingress said ``ok=false``
    while the terminal was published to
    ``delegate-skill-completed.v1`` 2s later (a false RED), and OMN-15468
    is the standing proof that ``ok=true`` on this lane can accompany
    nothing durable at all (a false GREEN).

    ``SKIPPED_NOT_CONFIGURED`` and ``ERROR`` are both NON-passing, and
    neither ever falls back to the ingress response. Falling back is the
    defect; a leg that could not run makes no claim.
    """

    # The probe's own correlation id was read back off a declared terminal
    # topic. This is what discharges OMN-16025 link 4.
    FOUND = "found"
    # The declared terminal topics were read for the whole remaining budget
    # and the correlation id was not on any of them.
    NOT_FOUND = "not_found"
    # The readback was configured but could not be completed (broker
    # unreachable, topic unresolvable, scan error). Fails closed.
    ERROR = "error"
    # No broker was configured for the readback. No claim is made about the
    # terminal, and therefore no green is available.
    SKIPPED_NOT_CONFIGURED = "skipped_not_configured"


__all__ = ["EnumTerminalReadbackStatus"]
