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


def test_the_live_174_dlq_shape_now_resolves_by_construction() -> None:
    """The OMN-16939 shape, and it is GREEN since OMN-18013.

    node_pr_lifecycle_state_reducer took 174 messages and DLQ'd 174 over six
    hours on exactly this contract: one unscoped ``operation_match`` entry over a
    ``.cmd.`` topic and two ``.evt.`` siblings. handler_wiring derived ONE
    category for the entry from ``subscribe_topics[0]`` — the ``.cmd.`` topic —
    and stamped ``command`` on all three routes, so both event topics were
    permanent NO_DISPATCHER while the consumer group read Stable / LAG 0.

    ``derive_route_message_category`` now takes each route's category from THAT
    route's own topic name, so this shape cannot mis-register any more. The gate
    finding is gone because the DEFECT is gone, not because it was baselined —
    there is no baseline. ``test_red_subscribe_topic_no_entry_is_assigned`` below is
    the live control proving the gate still fails on a real defect.
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
    assert [f.topic for f in findings] == [], (
        "the subscribe_topics[0] category mechanism is gone; every topic now "
        "registers under its own category"
    )


def test_per_topic_split_without_an_explicit_category_now_resolves() -> None:
    """The trap that a per-topic split alone used not to fix — now it does.

    ``node_swarm_subtask_state_reducer`` had a full ``topic_match`` split with
    per-entry event models and was still 100% NO_DISPATCHER on its event topic:
    ``topic:`` chose WHICH topic an entry owned, but the category still came from
    ``subscribe_topics[0]``. Since OMN-18013 the category comes from the topic,
    so the split is sufficient on its own and no ``message_category:`` is needed
    where the topic's name derives one.
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
    assert [f.topic for f in findings] == []


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


def test_red_subscribe_topic_no_entry_is_assigned() -> None:
    """RED: a subscribe topic no entry owns registers zero routes.

    This is the gate's live control — the proof it still fails on a real defect
    rather than having been quietly neutered by the OMN-18013 burn-down. The
    contract HAS a handler_routing entry, so the runtime does subscribe it; the
    entry is ``topic_match``-scoped to one topic, so the sibling topic is
    assigned to nothing and every message on it is consumed, matched against no
    route, DLQ'd and committed.
    """
    findings = unresolved_subscriptions(
        [
            _contract(
                name="node_no_routes",
                subscribe_topics=(_FIX_EVT, _MERGE_EVT),
                routing_strategy="topic_match",
                handlers=(
                    ModelHandlerRoutingEntry(
                        operation="reduce_pr_state",
                        topic=_FIX_EVT,
                        handler=_HANDLER,
                    ),
                ),
            )
        ]
    )
    assert [f.reason for f in findings] == [REASON_NO_ROUTE]
    assert [f.topic for f in findings] == [_MERGE_EVT]


def test_contract_with_no_handler_routing_entries_is_not_flagged() -> None:
    """OMN-18013: the runtime never auto-subscribes this shape, so nothing to resolve.

    A contract declaring subscribe topics and NO handler_routing entries is
    short-circuited to SKIPPED in ``_prepare_contract_wiring``; one whose routing
    block exists but parses to zero handlers is FAILED by the phantom-wiring
    check. Either way no Kafka subscription is created, so this gate's failure
    mode — consume, DLQ, commit, while the group reads Stable / LAG 0 — cannot
    occur, exactly as for ``plugin_managed``.

    This is not a convenience exemption; it removed two live FALSE POSITIVES.
    ``node_contract_registry_reducer`` and ``node_context_audit_dlq_effect`` have
    this shape, and the "fix" the finding invited — deleting the subscribe
    declaration — would have de-provisioned topics that the kernel's
    ``ContractRegistrationEventRouter`` and ``ContextAuditConsumer`` actually
    consume, and dropped enum members that live projection source imports.
    ``tests/unit/runtime/auto_wiring/test_omn17562_subscription_skip.py`` pins the
    runtime half of this invariant.
    """
    findings = unresolved_subscriptions(
        [
            _contract(
                name="node_kernel_consumed",
                subscribe_topics=(_FIX_EVT,),
                handlers=(),
            )
        ]
    )
    assert findings == [], (
        "a contract the runtime never subscribes has no unresolved subscription; "
        "flagging it invites deleting a declaration something else consumes"
    )


def test_gate_uses_the_runtime_derivation_not_a_reimplementation() -> None:
    """The gate's category/message-type helpers ARE the ones _prepare_handler_wiring calls.

    A gate that re-derives this is free to drift from the runtime — which is precisely how
    this defect class survived three prior gates. Binding them here means a change to the
    runtime derivation that the gate does not follow turns this test red.
    """
    source = inspect.getsource(handler_wiring._prepare_handler_wiring)
    assert "derive_entry_message_category(contract, entry)" in source
    assert "derive_entry_message_types(contract, entry)" in source


def test_live_scan_is_zero_with_no_baseline_at_all() -> None:
    """Every declared subscribe topic in this repo resolves to a dispatcher.

    This gate shipped with a 22-row shrink-only baseline. OMN-18013 burned it to
    ZERO and DELETED ``config/validation/subscriber_dispatcher_resolution_baseline.yaml``;
    ``load_baseline`` and the ``--baseline`` flag went with it, and
    ``no-baseline-refreeze`` refuses the file's return. A finding here is a live
    defect — a topic the runtime would subscribe to, consume, DLQ and COMMIT at
    LAG 0 — and the fix is the contract, because there is nowhere to record it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    findings, contract_count = scan(repo_root / "src" / "omnibase_infra")
    assert contract_count >= 60, (
        "contract scan collapsed; a green gate would be vacuous"
    )
    assert [f.key for f in findings] == [], (
        "unresolved subscription(s) — fix the contract, there is no baseline"
    )


def test_the_baseline_file_and_its_loader_are_gone() -> None:
    """The burned baseline cannot return through this gate's own module."""
    repo_root = Path(__file__).resolve().parents[2]
    assert not (
        repo_root
        / "config"
        / "validation"
        / "subscriber_dispatcher_resolution_baseline.yaml"
    ).exists()
    source = (
        repo_root
        / "src"
        / "omnibase_infra"
        / "validators"
        / "subscriber_dispatcher_resolution.py"
    ).read_text(encoding="utf-8")
    assert "--baseline" not in source
    assert "def load_baseline" not in source
