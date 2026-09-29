# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How a boot-time Bifrost endpoint probe failed (OMN-19455)."""

from __future__ import annotations

from enum import StrEnum


class EnumBifrostEndpointProbeFailureKind(StrEnum):
    """Two failures with opposite boot consequences.

    ``UNREACHABLE``: nothing answered (refused, timed out, no route). The
    backend is marked dark and boot continues, because a dark rung is a known
    state the router already skips.

    ``REFUSED``: something answered and it is not what the binding declares (an
    HTTP error, an unreadable body, or a served id that differs). Boot refuses,
    because routing to it would attribute calls to the wrong model.
    """

    UNREACHABLE = "unreachable"
    REFUSED = "refused"
