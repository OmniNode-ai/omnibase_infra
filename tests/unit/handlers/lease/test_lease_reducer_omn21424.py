# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424: lease reduction is a commutative join at and across fences."""

from __future__ import annotations

import itertools
from uuid import UUID

import pytest

from omnibase_infra.handlers.lease import reduce_lease_event
from omnibase_infra.models.lease import ModelLeaseProjectionRow, ModelLeaseReply
from omnibase_infra.models.lease.model_lease_reply import LeaseOutcome

pytestmark = pytest.mark.unit


def _event(
    outcome: LeaseOutcome, *, fence: int = 10, holder: str = ""
) -> ModelLeaseReply:
    return ModelLeaseReply(
        outcome=outcome,
        kind="k",
        resource="r",
        holder=holder,
        lease_id=UUID(int=fence),
        fence=fence,
    )


def test_same_fence_permutations_converge_and_replay_emits_no_intent() -> None:
    events = [
        _event("lease-granted", holder="a"),
        _event("lease-renewed"),
        _event("lease-released"),
        _event("lease-expired"),
    ]
    expected = {
        ("k", "r"): ModelLeaseProjectionRow(
            kind="k",
            resource="r",
            lease_id=UUID(int=10),
            fence=10,
            holder="a",
            state="expired",
        )
    }
    for order in itertools.permutations(events):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        for event in order:
            state, _ = reduce_lease_event(state, event)
        assert state == expected, order
        for event in events:
            after, intents = reduce_lease_event(state, event)
            assert after == expected
            assert intents == ()


@pytest.mark.parametrize("outcome", ["lease-released", "lease-expired"])
def test_terminal_before_grant_recovers_holder(outcome: LeaseOutcome) -> None:
    state, _ = reduce_lease_event({}, _event(outcome))
    terminal = state[("k", "r")]
    after, intents = reduce_lease_event(state, _event("lease-granted", holder="a"))
    expected = terminal.model_copy(update={"holder": "a"})
    assert after == {("k", "r"): expected}
    assert tuple(intent.row for intent in intents) == (expected,)
    assert state[("k", "r")] == terminal


@pytest.mark.parametrize("higher_holder", ["", "b"])
def test_two_fence_permutations_converge(higher_holder: str) -> None:
    events = [
        _event("lease-granted", holder="z"),
        _event("lease-expired"),
        _event("lease-granted", fence=17, holder=higher_holder),
        _event("lease-renewed", fence=17),
        _event("lease-released", fence=17),
    ]
    expected = {
        ("k", "r"): ModelLeaseProjectionRow(
            kind="k",
            resource="r",
            lease_id=UUID(int=17),
            fence=17,
            holder=higher_holder,
            state="released",
        )
    }
    for order in itertools.permutations(events):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        for event in order:
            state, _ = reduce_lease_event(state, event)
        assert state == expected, order


def test_lower_fence_after_higher_is_dropped() -> None:
    state, _ = reduce_lease_event({}, _event("lease-renewed", fence=17))
    original = dict(state)
    after, intents = reduce_lease_event(state, _event("lease-expired", holder="z"))
    assert after == original
    assert state == original
    assert intents == ()


@pytest.mark.parametrize(
    ("grant_holder", "terminal_holder"), [("a", ""), ("a", "z"), ("z", "a")]
)
def test_equal_fence_holder_join_is_commutative(
    grant_holder: str, terminal_holder: str
) -> None:
    events = [
        _event("lease-granted", holder=grant_holder),
        _event("lease-released", holder=terminal_holder),
    ]
    for order in itertools.permutations(events):
        state: dict[tuple[str, str], ModelLeaseProjectionRow] = {}
        for event in order:
            state, _ = reduce_lease_event(state, event)
        assert state[("k", "r")].holder == max(grant_holder, terminal_holder)
        assert state[("k", "r")].state == "released"
