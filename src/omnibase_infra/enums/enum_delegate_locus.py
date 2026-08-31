# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execution locus for ``onex delegate`` (OMN-17295 / OMN-17304).

WHERE the delegate orchestrator makes its accept/climb decision, as distinct
from WHICH TRANSPORT carries the events. Those were conflated: ``--bus kafka``
selected the transport and was read as relocating execution, so a "dev-lane
probe" ran the orchestrator from the caller's own venv and reported a result
about a machine it never touched.

.. versionadded:: OMN-17304
"""

from __future__ import annotations

from enum import Enum

__all__ = ["EnumDelegateLocus"]


class EnumDelegateLocus(str, Enum):
    """Requested or resolved execution locus for one delegation."""

    AUTO = "auto"
    """Resolve from the transport: a shared bus dispatches, in-memory hosts.

    Only ever a REQUEST — resolution always yields one of the two below, so a
    receipt never reports ``auto``.
    """

    IN_PROCESS = "in-process"
    """This CLI hosts the orchestrator and makes the decision itself.

    The offline/standalone path, and the only correct locus for an in-memory
    bus: nothing else can consume a bus that exists inside this process.
    """

    DEPLOYED_LANE = "deployed-lane"
    """The deployed runtime subscribed to the command topic makes the decision.

    This CLI publishes the typed command and awaits its own correlated
    terminal, hosting nothing. Requires a live consumer on that exact topic —
    with nobody bound there is no orchestrator to decide, and the run refuses
    rather than quietly deciding for itself.
    """
