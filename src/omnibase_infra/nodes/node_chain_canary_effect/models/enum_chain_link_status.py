# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-link status for one canary run (OMN-16931)."""

from __future__ import annotations

from enum import StrEnum


class EnumChainLinkStatus(StrEnum):
    """Exactly one status per OMN-16025 link, per run.

    Only ``PASS`` counts toward ``links_proven``. Every other member is a
    different reason the link is not proven, and they are kept distinct on
    purpose: collapsing them is how "we never checked" becomes
    indistinguishable from "we checked and it was fine", which is the
    failure this whole ticket family is about.
    """

    # Real evidence, this run, for this correlation id.
    PASS = "pass"
    # Real evidence that the link did NOT hold.
    FAIL = "fail"
    # The leg exists and was configured, but could not be executed. Fails
    # closed — never counted as proven.
    ERROR = "error"
    # The leg exists but was not configured for this run. SKIP is not PASS.
    NOT_CONFIGURED = "not_configured"
    # The leg was not reached this run because an earlier link failed (e.g.
    # the ingress was unreachable, so nothing downstream was observed).
    # Distinct from FAIL: the canary is not entitled to call this link bad.
    NOT_EVALUATED = "not_evaluated"
    # No leg exists in the canary at all. The link's verdict carries the
    # ticket that owes the leg. This is an unpaid debt, not a result.
    NO_LEG = "no_leg"


__all__ = ["EnumChainLinkStatus"]
