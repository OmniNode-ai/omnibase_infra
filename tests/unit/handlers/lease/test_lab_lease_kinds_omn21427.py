# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21427 AC2 (declaration half): the lab lock rows are lease kinds with an expiry.

Every inventory row C1-C5 and C7-C9 of the OMN-21279 plan, and the B2 slot, is
declared once in ``lab_lease_kinds.yaml`` as a kind the generic lease store
accepts, each with a TTL, so a holder that stops renewing loses it at expiry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.lease.handler_lease_kinds import (
    LAB_LEASE_KINDS_PATH,
    HandlerLeaseKinds,
)
from omnibase_infra.models.lease import ModelLeaseKindPolicy

pytestmark = pytest.mark.unit

# inventory row -> kind, from the plan's section 7 table
INVENTORY_KINDS = {
    "B2": "slot",
    "C1": "prepr-project",
    "C2": "prepr-slot",
    "C3": "prepr-host",
    "C4": "compose-lane",
    "C5": "deploy-host",
    "C7": "host-reconcile",
    "C8": "lab-stack",
    "C9": "heavy-suite",
}


def test_every_lab_lock_row_is_a_declared_kind_with_an_expiry() -> None:
    declared = HandlerLeaseKinds().load_lease_kinds(LAB_LEASE_KINDS_PATH)
    by_kind = {entry.policy.kind: entry for entry in declared}
    assert set(by_kind) == set(INVENTORY_KINDS.values())
    for row, kind in INVENTORY_KINDS.items():
        entry = by_kind[kind]
        assert entry.inventory_row == row
        assert isinstance(entry.policy, ModelLeaseKindPolicy)
        assert entry.policy.ttl_s >= 2
        assert entry.replaces, f"{kind} names no mechanism it replaces"


def test_slot_capacity_is_per_host() -> None:
    slot = next(
        e.policy
        for e in HandlerLeaseKinds().load_lease_kinds(LAB_LEASE_KINDS_PATH)
        if e.policy.kind == "slot"
    )
    assert slot.capacity_of("slot:unlisted-host") == slot.capacity
    policy = ModelLeaseKindPolicy(
        kind="slot", capacity=2, resource_capacity={"slot:h200": 5}
    )
    assert policy.capacity_of("slot:h200") == 5
    assert policy.capacity_of("slot:h105") == 2


def test_a_duplicate_kind_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "kinds.yaml"
    path.write_text(
        "kinds:\n"
        "  - {kind: slot, inventory_row: B2, replaces: x, capacity: 1, ttl_s: 30}\n"
        "  - {kind: slot, inventory_row: B2, replaces: y, capacity: 1, ttl_s: 30}\n"
    )
    with pytest.raises(ValueError, match="declared twice"):
        HandlerLeaseKinds().load_lease_kinds(path)
