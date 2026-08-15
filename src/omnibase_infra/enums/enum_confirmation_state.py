# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tri-state outcome of a durability confirmation attempt (OMN-15861).

Deliberately tri-state, not boolean. A confirmation that cannot resolve its
authoritative surface -- broker unreachable mid-readback, projection store
raising -- is NOT the same fact as "the record is definitively absent", and
collapsing the two into ``False`` is how an indeterminate result gets silently
treated as either success or a permanent failure. The platform's standing
doctrine is that ``UNKNOWN``/indeterminate **fails closed**: only ``CONFIRMED``
authorises a durable claim or an outbox truncation.
"""

from __future__ import annotations

from enum import Enum


class EnumConfirmationState(str, Enum):
    """Whether a produced record has been proven to have landed.

    Attributes:
        CONFIRMED: An authoritative surface (broker readback or projection)
            observed the record. This is the ONLY state that authorises acking
            a durable-outbox record or telling a caller "durable".
        UNCONFIRMED: The authoritative surface was reached and did NOT observe
            the record within the deadline. Retry is appropriate; the record
            must stay in the outbox.
        UNKNOWN: The confirmation attempt could not resolve -- the surface
            errored, or the receipt carried no coordinate to check. Fails
            closed: treated exactly like ``UNCONFIRMED`` for ack purposes, but
            reported distinctly so an operator can tell a broker that answered
            "not there" from a broker that never answered.
    """

    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    UNKNOWN = "unknown"


__all__: list[str] = ["EnumConfirmationState"]
