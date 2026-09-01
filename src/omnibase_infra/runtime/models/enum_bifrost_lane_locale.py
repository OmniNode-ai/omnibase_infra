# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execution locale of a Bifrost delegation lane (OMN-17502).

A lane's locale answers one question: **does this lane run where the shared lab
inference servers exist?** It is the axis that decides whether the lane's typed
overlay may — and must — declare local backend bindings at all.

Before OMN-17502 the overlay had no such axis, so "exactly the three authorized
lab backends" was the only schema-valid shape. That is correct for a lane whose
runtime sits on the lab network and wrong for a lane that does not: the onex-dev
cluster lane has never had local backends (its delegation has always resolved
cloud-only), and the .201 / .200 endpoints are refused from inside its
namespace. When the OMN-17150 fail-closed overlay requirement reached that lane
in a released image, the only ways to satisfy the schema were to mount lab
bindings the lane cannot reach — advertising three dead rungs, the OMN-17150
defect class done explicitly — or to crash-loop, which is what happened.
"""

from __future__ import annotations

from enum import Enum, unique

__all__ = ["EnumBifrostLaneLocale"]


@unique
class EnumBifrostLaneLocale(str, Enum):
    """Where a lane's runtime executes, and therefore what it may bind.

    Values:
        LAB: The lane runs on the lab network (the ``.201`` compose lanes —
            dev, stability-test, judge, collaborator lanes). Its overlay must
            declare EXACTLY the active local backend IDs, each pinned to its
            authorized host/port/model by ``_AUTHORIZED_BINDINGS``. This is the
            pre-OMN-17502 rule, unchanged — only now stated in the file rather
            than assumed by the schema.
        CLOUD: The lane runs where no lab backend is reachable (the onex-dev
            cluster lane; beta axiom 9 — cloud execution locale, BYOK). Its
            overlay declares ZERO local backends, and its delegation is served
            by the base contract's cloud backends alone.
    """

    LAB = "lab"
    CLOUD = "cloud"

    def __str__(self) -> str:
        """Return the plain value so log lines read ``cloud``, not the repr."""
        return self.value
