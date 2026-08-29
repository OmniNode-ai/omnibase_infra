# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression for the subscriber-dispatcher-resolution ratchet (OMN-16939).

The defect class: a contract subscribes, joins the group, consumes, DLQ's every message
and COMMITS the offset — so the group reads ``Stable`` / ``MEMBERS 1`` / ``LAG 0`` forever
while 100% of the traffic is lost. Three prior ratchets all pass on that shape (see the
validator docstring), so this gate resolves every subscribe topic against the dispatcher
index the runtime will actually build.

The two RED cases below are the shapes observed bleeding on the .201 dev lane on
2026-08-29, reconstructed from the real contracts:

* ``node_pr_lifecycle_state_reducer`` — a sole ``operation_match`` entry owning one
  ``.cmd.`` topic and seven ``.evt.`` topics. 174 in / 174 DLQ over six hours.
* ``node_swarm_subtask_state_reducer`` — a full per-topic ``topic_match`` split WITH
  per-topic ``event_model``s but NO explicit ``message_category``. 128 DLQ per 40 minutes.
  This one matters most: it proves ``topic:`` alone does not fix the class, which is the
  trap that let the OMN-14605 fix pattern be applied and still leave the contract 100%
  broken.

Also asserts the derivation helpers the gate imports are the SAME ones the runtime uses
(``_prepare_handler_wiring`` calls them), so the gate cannot drift from live wiring — the
failure mode that let this class survive three gates.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    handler_wiring,
)
from omnibase_infra.validators.subscriber_dispatcher_resolution import (
    REASON_CATEGORY_MISMATCH,
    REASON_NO_ROUTE,
    load_baseline,
    scan,
    unresolved_subscriptions,
)

pytestmark = pytest.mark.unit

_HANDLER = ModelHandlerRef(name="HandlerX", module="omnimarket.fake")

_SWEEP_CMD = "onex.cmd.omnimarket.pr-lifecycle-sweep-start.v1"
_FIX_EVT = "onex.evt.omnimarket.pr-lifecycle-fix-completed.v1"
_MERGE_EVT = "onex.evt.omnimarket.pr-lifecycle-merge-completed.v1"

_EXEC_CMD = "onex.cmd.omnimarket.delegation-execute.v1"
_ESCALATION_EVT = "onex.evt.omnimarket.delegation-escalation-triggered.v1"


def _contract(
    *,
    name: str,
    subscribe_topics: tuple[str, ...],
    handlers: tuple[ModelHandlerRoutingEntry, ...],
    routing_strategy: str = "operation_match",
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/tmp/omn-16939/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics, publish_topics=()
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy=routing_strategy, handlers=handlers
        ),
    )


def test_red_pr_lifecycle_state_reducer_shape() -> None:
    """RED: the live 174-in / 174-DLQ shape — one operation_match entry, mixed topics.

    Every ``.evt.`` topic registers under ``command`` because the ``.cmd.`` sweep-start
    topic is ``subscribe_topics[0]``, so none of them can ever match.
    """
    findings = unresolved_subscriptions(
        [
            _contract(
                name="node_pr_lifecycle_state_reducer",
                subscribe_topics=(_SWEEP_CMD, _FIX_EVT, _MERGE_EVT),
                handlers=(
                    ModelHandlerRoutingEntry(
                        operation="reduce_pr_state", handler=_HANDLER
                    ),
                ),
            )
        ]
    )
    unresolved = {f.topic: f for f in findings}
    assert _SWEEP_CMD not in unresolved, "the command topic resolves; only events break"
    assert set(unresolved) == {_FIX_EVT, _MERGE_EVT}
    assert unresolved[_FIX_EVT].reason == REASON_CATEGORY_MISMATCH
    assert unresolved[_FIX_EVT].category == "event"


def test_red_topic_match_without_explicit_category_is_still_broken() -> None:
    """RED: the trap — a full per-topic split with event_models is NOT sufficient.

    ``node_swarm_subtask_state_reducer`` had exactly this shape and was 100%
    NO_DISPATCHER on all of its event topics. ``topic:`` picks WHICH topic an entry owns;
    the category still falls back to ``subscribe_topics[0]``.
    """
    findings = unresolved_subscriptions(
        [
            _contract(
                name="node_swarm_subtask_state_reducer",
                subscribe_topics=(_EXEC_CMD, _ESCALATION_EVT),
                routing_strategy="topic_match",
                handlers=(
                    ModelHandlerRoutingEntry(
                        operation="reduce_subtask_state",
                        topic=_EXEC_CMD,
                        handler=_HANDLER,
                    ),
                    ModelHandlerRoutingEntry(
                        operation="reduce_subtask_state",
                        topic=_ESCALATION_EVT,
                        handler=_HANDLER,
                    ),
                ),
            )
        ]
    )
    assert [f.topic for f in findings] == [_ESCALATION_EVT]
    assert findings[0].reason == REASON_CATEGORY_MISMATCH


def test_green_explicit_message_category_resolves() -> None:
    """GREEN: the applied fix — per-topic entries each declaring their own category."""
    findings = unresolved_subscriptions(
        [
            _contract(
                name="node_fixed",
                subscribe_topics=(_EXEC_CMD, _ESCALATION_EVT),
                routing_strategy="topic_match",
                handlers=(
                    ModelHandlerRoutingEntry(
                        operation="reduce_subtask_state",
                        topic=_EXEC_CMD,
                        message_category="command",
                        handler=_HANDLER,
                    ),
                    ModelHandlerRoutingEntry(
                        operation="reduce_subtask_state",
                        topic=_ESCALATION_EVT,
                        message_category="event",
                        handler=_HANDLER,
                    ),
                ),
            )
        ]
    )
    assert findings == []


def test_red_no_handler_routing_entry_at_all() -> None:
    """RED: a subscribe topic no entry is assigned registers zero routes."""
    findings = unresolved_subscriptions(
        [
            _contract(
                name="node_no_routes",
                subscribe_topics=(_FIX_EVT,),
                handlers=(),
            )
        ]
    )
    assert [f.reason for f in findings] == [REASON_NO_ROUTE]


def test_gate_uses_the_runtime_derivation_not_a_reimplementation() -> None:
    """The gate's category/message-type helpers ARE the ones _prepare_handler_wiring calls.

    A gate that re-derives this is free to drift from the runtime — which is precisely how
    this defect class survived three prior gates. Binding them here means a change to the
    runtime derivation that the gate does not follow turns this test red.
    """
    source = inspect.getsource(handler_wiring._prepare_handler_wiring)
    assert "derive_entry_message_category(contract, entry)" in source
    assert "derive_entry_message_types(contract, entry)" in source


def test_live_baseline_is_green_and_not_stale() -> None:
    """The seeded omnibase_infra baseline exactly matches the live finding set.

    Shrink-only in both directions: a new unresolved subscription fails as a violation,
    and a fixed one still listed fails as stale.
    """
    repo_root = Path(__file__).resolve().parents[2]
    findings, contract_count = scan(repo_root / "src" / "omnibase_infra")
    assert contract_count >= 60, (
        "contract scan collapsed; a green gate would be vacuous"
    )
    baseline = load_baseline(
        repo_root
        / "config"
        / "validation"
        / "subscriber_dispatcher_resolution_baseline.yaml"
    )
    live = {f.key for f in findings}
    assert live - baseline == set(), (
        "new unresolved subscription(s) — fix, do not baseline"
    )
    assert baseline - live == set(), "stale baseline entr(ies) — remove them"
