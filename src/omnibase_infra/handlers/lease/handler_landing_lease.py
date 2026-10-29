# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The landing controller's lease kinds on the generic lease (OMN-21426, slice 3).

Plan OMN-21279 section 7 maps the controller's own leases onto three kinds of
the lease orchestrator:

- A1, fix-worker lease: ``landing-fix`` on ``pr:<repo>#<n>``, 2700 s.
- A2, cause lease: ``landing-cause`` on ``cause:<key>``, 5400 s.
- A3, lab-prover lease: ``lab-proof`` on ``repo:<repo>``, 10200 s.
- D5, CI-red owner claim: ``landing-cause`` on ``cause:<key>``, 5400 s.

A1's 2700 s is the controller's ``lease_seconds`` default and A2's 5400 s is
the plan's; A3's 10200 s is the lab prover's ``PROVER_DEADLINE_S`` (170 min).
``<repo>`` is the short repository name the ledger uses (``omnimarket#4151``):
a resource is one etcd key segment, so it never carries the owner's slash.
"""

from __future__ import annotations

import re
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.lease import (
    EnumLandingLeaseRow,
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
)

LANDING_FIX = "landing-fix"
LANDING_CAUSE = "landing-cause"
LAB_PROOF = "lab-proof"

LANDING_LEASE_KINDS: tuple[ModelLeaseKindPolicy, ...] = (
    ModelLeaseKindPolicy(kind=LANDING_FIX, capacity=1, ttl_s=2700),
    ModelLeaseKindPolicy(kind=LANDING_CAUSE, capacity=1, ttl_s=5400),
    ModelLeaseKindPolicy(kind=LAB_PROOF, capacity=1, ttl_s=170 * 60),
)

_REPO = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_SUBJECT = {
    EnumLandingLeaseRow.A1_FIX_WORKER: (LANDING_FIX, "pr", rf"^{_REPO}#[1-9][0-9]*$"),
    EnumLandingLeaseRow.A2_CAUSE: (LANDING_CAUSE, "cause", r"^[^/\s]+$"),
    EnumLandingLeaseRow.A3_LAB_PROVER: (LAB_PROOF, "repo", rf"^{_REPO}$"),
    EnumLandingLeaseRow.D5_CI_RED_OWNER: (LANDING_CAUSE, "cause", r"^[^/\s]+$"),
}


def landing_lease_request(
    row: EnumLandingLeaseRow, subject: str, *, holder: str, request_id: UUID
) -> ModelLeaseAcquireRequest:
    """The acquire the lease orchestrator grants for one landing lease row.

    ``subject`` is ``<repo>#<n>`` for A1, the cause key for A2 and D5, and
    ``<repo>`` for A3; anything else raises ``ValueError``.
    """
    kind, scheme, pattern = _SUBJECT[row]
    if not re.fullmatch(pattern, subject):
        raise ValueError(f"{row.value} subject {subject!r} does not match {pattern}")
    return ModelLeaseAcquireRequest(
        kind=kind,
        resource=f"{scheme}:{subject}",
        holder=holder,
        request_id=request_id,
    )


class HandlerLandingLease:
    """Maps a landing lease row and subject to its lease acquire; pure."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def request(
        self, row: EnumLandingLeaseRow, subject: str, *, holder: str, request_id: UUID
    ) -> ModelLeaseAcquireRequest:
        return landing_lease_request(row, subject, holder=holder, request_id=request_id)


__all__ = [
    "LAB_PROOF",
    "LANDING_CAUSE",
    "LANDING_FIX",
    "LANDING_LEASE_KINDS",
    "EnumLandingLeaseRow",
    "HandlerLandingLease",
    "landing_lease_request",
]
