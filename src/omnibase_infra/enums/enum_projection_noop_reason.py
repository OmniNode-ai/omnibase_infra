# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a projection handler wrote zero rows on purpose (OMN-26410).

A handler that returns ``rows_upserted: 0`` has two readings the runtime cannot
tell apart from the count alone: a write that was lost, and a delivery that owed
no write. This closed set is what a handler may attest for the second reading,
under the result key named in
``omnibase_infra.handlers.projection_noop_classification``.
"""

from __future__ import annotations

from enum import Enum


class EnumProjectionNoopReason(str, Enum):
    """The reasons a projection handler may attest for a designed zero-row return.

    - ``TICK``: the delivery is the runtime tick, a timer the handler sweeps or
      republishes on and that carries no source event. Honoured only for a
      delivery on the runtime tick topic, because a source event is owed a row
      whatever the handler claims.
    - ``SKIPPED``: the handler read the delivery and found nothing in it for this
      projection (for example an event that is not in the projection's scope).
    """

    TICK = "tick"
    SKIPPED = "skipped"


__all__ = ["EnumProjectionNoopReason"]
