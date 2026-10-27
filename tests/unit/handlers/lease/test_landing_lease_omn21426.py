# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21426 AC1: the landing controller's lease rows map onto three lease kinds."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.handlers.lease.handler_landing_lease import (
    LANDING_LEASE_KINDS,
    EnumLandingLeaseRow,
    landing_lease_request,
)

pytestmark = [pytest.mark.unit]


def test_three_kinds_with_the_plan_ttls() -> None:
    by_kind = {policy.kind: policy for policy in LANDING_LEASE_KINDS}
    assert set(by_kind) == {"landing-fix", "landing-cause", "lab-proof"}
    assert by_kind["landing-fix"].ttl_s == 2700
    assert by_kind["landing-cause"].ttl_s == 5400
    assert by_kind["lab-proof"].ttl_s == 170 * 60
    assert all(policy.capacity == 1 for policy in LANDING_LEASE_KINDS)


@pytest.mark.parametrize(
    ("row", "subject", "kind", "resource"),
    [
        (
            EnumLandingLeaseRow.A1_FIX_WORKER,
            "omnimarket#4151",
            "landing-fix",
            "pr:omnimarket#4151",
        ),
        (
            EnumLandingLeaseRow.A2_CAUSE,
            "ruff-red-1",
            "landing-cause",
            "cause:ruff-red-1",
        ),
        (
            EnumLandingLeaseRow.A3_LAB_PROVER,
            "omnibase_infra",
            "lab-proof",
            "repo:omnibase_infra",
        ),
        (
            EnumLandingLeaseRow.D5_CI_RED_OWNER,
            "ruff-red-1",
            "landing-cause",
            "cause:ruff-red-1",
        ),
    ],
)
def test_each_inventory_row_names_its_kind_and_resource(
    row: EnumLandingLeaseRow, subject: str, kind: str, resource: str
) -> None:
    request_id = uuid4()
    request = landing_lease_request(
        row, subject, holder="host-a/lane-a", request_id=request_id
    )
    assert (request.kind, request.resource) == (kind, resource)
    assert request.holder == "host-a/lane-a"
    assert request.request_id == request_id


def test_ci_red_owner_and_cause_share_one_resource() -> None:
    """D5 is the same kind as A2: one owner per cause, whichever row asks."""
    cause = landing_lease_request(
        EnumLandingLeaseRow.A2_CAUSE, "k", holder="a", request_id=uuid4()
    )
    owner = landing_lease_request(
        EnumLandingLeaseRow.D5_CI_RED_OWNER, "k", holder="b", request_id=uuid4()
    )
    assert (cause.kind, cause.resource) == (owner.kind, owner.resource)


@pytest.mark.parametrize(
    ("row", "subject"),
    [
        (EnumLandingLeaseRow.A1_FIX_WORKER, "OmniNode-ai/omnimarket#4151"),
        (EnumLandingLeaseRow.A1_FIX_WORKER, "omnimarket"),
        (EnumLandingLeaseRow.A1_FIX_WORKER, "omnimarket#x"),
        (EnumLandingLeaseRow.A3_LAB_PROVER, "OmniNode-ai/omnimarket"),
        (EnumLandingLeaseRow.A2_CAUSE, ""),
    ],
)
def test_a_subject_that_is_not_a_resource_key_is_refused(
    row: EnumLandingLeaseRow, subject: str
) -> None:
    with pytest.raises(ValueError):
        landing_lease_request(row, subject, holder="h", request_id=uuid4())
