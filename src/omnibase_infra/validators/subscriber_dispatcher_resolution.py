# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every declared subscribe_topic must resolve to a REGISTERED DISPATCHER (OMN-16939).

THE DEFECT CLASS THIS GATE CLOSES
---------------------------------
A contract can subscribe to a topic, join the consumer group, consume every message and
commit the offset while having NO dispatcher registered for that message's (category,
message type). ``service_kernel`` logs ``failure_class=no_dispatcher`` and routes to the
DLQ — then commits anyway, so LAG can never rise. Group state reads ``Stable`` /
``MEMBERS 1`` / ``LAG 0`` forever while 100% of the traffic is lost.

Live-proven on the .201 dev lane 2026-08-29 (OMN-16939):
``node_pr_lifecycle_state_reducer`` took 174 messages in and DLQ'd 174 over six hours on
``onex.evt.omnimarket.pr-lifecycle-fix-completed.v1``; ``delegation-escalation-triggered``
was DLQ'ing at 128 per 40 minutes on the same mechanism.

WHY THE THREE EXISTING RATCHETS ALL PASS ON THAT SHAPE
-------------------------------------------------------
* ``scripts/check_subscribe_wiring_health.py`` (OMN-7385 / OMN-16795) asserts a declared
  subscribe topic has a contract PUBLISHER. The publisher was real and had produced 2,151
  messages — it passes.
* ``scripts/check_dispatcher_route_coverage.py`` (OMN-12858) asserts a contract
  subscribing to a ``onex.cmd.*`` topic declares SOME ``handler_routing`` /
  ``runtime_dispatch`` block. It never looks at ``.evt.`` topics and never checks which
  category the route actually registered under — it passes.
* ``omnibase_infra.validators.mixed_category_routing`` (OMN-14605) catches the derivation
  defect itself, but is scanned over ``src/omnibase_infra`` ONLY. Its own baseline header
  defers the other repos to "Lane B", which never landed — so omnimarket, where both live
  victims sit, had no gate at all.

This gate is the dispatcher-side sibling those three leave open: it resolves EVERY
subscribe topic of EVERY category against the dispatcher index the runtime will actually
build, and reports the reason resolution failed.

REAL HELPERS, NOT A REIMPLEMENTATION
------------------------------------
Topic assignment, category derivation and message-type indexing all import the REAL
production helpers (``_topics_for_handler_entry``, ``derive_entry_message_category``,
``derive_entry_message_types``) and the REAL discovery path
(``discover_contracts_from_paths``), so this gate observes exactly what
``_prepare_handler_wiring`` observes. ``derive_entry_message_category`` /
``derive_entry_message_types`` were extracted from ``_prepare_handler_wiring`` in this
same change precisely so the gate and the runtime cannot drift — a re-implementation is
how this class survived three prior gates.

FAILURE REASONS
---------------
``no_route``
    No handler_routing entry is assigned this topic by ``_topics_for_handler_entry``, so
    zero ``ModelDispatchRoute`` rows exist for it.
``category_mismatch``
    A route exists, but ``_prepare_handler_wiring`` stamped it with a category that
    differs from the topic's own real category. ``MessageDispatchEngine`` filters on the
    real category before any handler runs, so the route can never match. This is the
    OMN-14605 mechanism.
``message_type_unindexed``
    Category agrees, but the dispatcher is not indexed under the topic's message type.

RATCHET SEMANTICS (shrink-only; config/validation/subscriber_dispatcher_resolution_baseline.yaml)
--------------------------------------------------------------------------------------------------
  * a (contract, topic) NOT in the baseline that fails to resolve -> FAIL (no new
    instances, ever);
  * a (contract, topic) IN the baseline that now resolves -> FAIL until removed from the
    baseline (a fixed entry still listed is STALE, so the list cannot rot).
The baseline can only shrink. It is a burn-down list, NOT an amnesty list.

Usage (pre-commit / CI):
    PYTHONPATH=src uv run python -m omnibase_infra.validators.subscriber_dispatcher_resolution src/omnibase_infra
    uv run python -m omnibase_infra.validators.subscriber_dispatcher_resolution \\
    uv run python -m omnibase_infra.validators.subscriber_dispatcher_resolution \
        src/omnibase_infra
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_message_category,
    _topics_for_handler_entry,
    derive_entry_message_types,
    derive_route_message_category,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")

REASON_NO_ROUTE = "no_route"
REASON_CATEGORY_MISMATCH = "category_mismatch"
REASON_MESSAGE_TYPE_UNINDEXED = "message_type_unindexed"
REASON_UNDERIVABLE_CATEGORY = "underivable_category"

# A scan that discovers far fewer contracts than the tree actually has is a broken scan,
# not a clean tree. A gate over a collapsed set is vacuously green, so the validator fails
# closed below this floor rather than reporting success.
DEFAULT_MIN_EXPECTED_CONTRACTS = 40


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class UnresolvedSubscription:
    """A declared subscribe topic that no registered dispatcher can ever receive."""

    contract: str
    topic: str
    category: str
    reason: str
    detail: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.contract, self.topic)


def _resolve_topic(
    contract: ModelDiscoveredContract,
    topic: str,
) -> UnresolvedSubscription | None:
    """Return a finding when ``topic`` resolves to no live dispatcher, else ``None``."""
    real_category = _derive_message_category(topic)
    alias = derive_event_type_alias_for_topic(topic)

    if real_category is None:
        # OMN-18013: the topic's own name carries no category, so
        # ``EnumMessageCategory.from_topic`` returns None at dispatch and the
        # message is rejected as an invalid topic category BEFORE any route is
        # consulted. Before OMN-18013 the wiring stamped a silent "event"
        # default here and this was invisible.
        return UnresolvedSubscription(
            contract=contract.name,
            topic=topic,
            category="",
            reason=REASON_UNDERIVABLE_CATEGORY,
            detail=(
                "EnumMessageCategory.from_topic returns None for this topic, so "
                "MessageDispatchEngine rejects every message on it as an invalid "
                "topic category regardless of routing"
            ),
        )

    routing = contract.handler_routing
    entries = list(getattr(routing, "handlers", None) or []) if routing else []

    reason = REASON_NO_ROUTE
    detail = (
        "no handler_routing entry is assigned this topic by _topics_for_handler_entry"
    )

    for entry in entries:
        if topic not in _topics_for_handler_entry(contract, entry):
            continue
        try:
            route_category = derive_route_message_category(contract, entry, topic).value
        except ModelOnexError as exc:
            reason = REASON_CATEGORY_MISMATCH
            detail = f"route category is unresolvable for this topic: {exc}"
            continue
        if route_category != real_category:
            reason = REASON_CATEGORY_MISMATCH
            detail = (
                f"route registered under category={route_category!r} but messages arrive "
                f"as {real_category!r}"
            )
            continue
        message_types = derive_entry_message_types(contract, entry) or set()
        if (
            alias is not None
            and alias not in message_types
            and topic not in message_types
        ):
            reason = REASON_MESSAGE_TYPE_UNINDEXED
            detail = (
                f"category {real_category!r} agrees but the dispatcher is not indexed "
                f"under message type {alias!r}"
            )
            continue
        return None

    return UnresolvedSubscription(
        contract=contract.name,
        topic=topic,
        category=real_category,
        reason=reason,
        detail=detail,
    )


def unresolved_subscriptions(
    contracts: Iterable[ModelDiscoveredContract],
) -> list[UnresolvedSubscription]:
    """Every declared subscribe topic that cannot reach a registered dispatcher."""
    findings: list[UnresolvedSubscription] = []
    for contract in contracts:
        if contract.event_bus is None or not contract.event_bus.subscribe_topics:
            continue
        # plugin_managed: the DOMAIN PLUGIN owns the Kafka subscription, and the
        # runtime host explicitly skips creating one (OMN-10864,
        # handler_wiring.py "Auto-wiring (deferred): skipping Kafka subscription
        # for plugin-managed contract"). There is therefore no auto-wired
        # dispatcher to resolve to, and no traffic being consumed-and-DLQ'd:
        # this gate's whole failure mode cannot occur here. Scanning these
        # contracts anyway reported node_emit_daemon_runtime's three topics as
        # unresolved, and the burn-down nearly "fixed" that by deleting a live
        # plugin's declared consumption from the contract graph. This gate
        # observes exactly what the runtime observes — including what it skips.
        if contract.event_bus.plugin_managed:
            continue
        # OMN-18013, same principle one case wider. A contract declaring NO
        # handler_routing never reaches the subscribe decision at all:
        # _prepare_contract_wiring short-circuits `contract.handler_routing is
        # None` to a SKIPPED result with subscription_topics=[]. So, as with
        # plugin_managed, the runtime creates no auto-wired consumer and this
        # gate's failure mode (consume, DLQ, commit) cannot occur.
        #
        # Note the mechanism precisely, because the adjacent one does NOT apply:
        # handler_wiring's `no_live_dispatcher` branch is guarded by
        # `bool(prepared_wirings)`, which is False at zero entries, so it does
        # not cover this shape and no change was made there for it. The
        # present-but-EMPTY handler_routing shape DOES reach the subscribe
        # decision, and is caught fail-closed by the OMN-14141 phantom-wiring
        # guard (zero dispatchers registered with subscribe topics declared =>
        # FAILED, topic not subscribed). That guard is why skipping on an empty
        # entry list here is safe rather than a blind spot. Flagging these anyway is what produced the near-miss this
        # gate was written to avoid: node_contract_registry_reducer's three
        # contract-lifecycle topics ARE consumed, by the kernel's
        # ContractRegistrationEventRouter, and "fixing" the finding by deleting
        # the subscribe declaration would have erased a live, readiness-critical
        # consumption from the contract graph AND dropped the enum members that
        # models/projection/projection_contract_registry.py imports. The gate
        # observes exactly what the runtime observes, including what it skips.
        _routing = contract.handler_routing
        if not (list(getattr(_routing, "handlers", None) or []) if _routing else []):
            continue
        for topic in contract.event_bus.subscribe_topics:
            finding = _resolve_topic(contract, topic)
            if finding is not None:
                findings.append(finding)
    return findings


def scan(scan_root: Path) -> tuple[list[UnresolvedSubscription], int]:
    """Discover every contract under ``scan_root`` and return (findings, contract_count)."""
    contract_paths = sorted(
        p
        for p in scan_root.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    discovered = discover_contracts_from_paths(contract_paths)
    contracts = list(getattr(discovered, "contracts", discovered))
    return unresolved_subscriptions(contracts), len(contracts)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail any declared subscribe_topic that cannot resolve to a registered "
            "dispatcher for its real (category, message type) — the shape that consumes "
            "and commits while DLQ'ing 100% of traffic at LAG 0."
        )
    )
    parser.add_argument(
        "scan_root",
        nargs="?",
        default=str(DEFAULT_SCAN_ROOT),
        help="Root to scan for contract.yaml files.",
    )
    parser.add_argument(
        "--min-contracts",
        type=int,
        default=DEFAULT_MIN_EXPECTED_CONTRACTS,
        help="Vacuity floor: fail closed when fewer contracts than this are discovered.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    scan_root = Path(args.scan_root)

    findings, contract_count = scan(scan_root)
    if contract_count < args.min_contracts:
        sys.stderr.write(
            f"[subscriber-dispatcher-resolution] FAIL (vacuity guard): only "
            f"{contract_count} contracts discovered under {scan_root} (expected >= "
            f"{args.min_contracts}). The contract scan is broken; a gate over a collapsed "
            f"set proves nothing.\n"
        )
        return 1

    live: dict[tuple[str, str], UnresolvedSubscription] = {f.key: f for f in findings}
    violations = sorted(live)
    exit_code = 0

    if violations:
        exit_code = 1
        sys.stderr.write(
            "[subscriber-dispatcher-resolution] FAIL: declared subscribe topic(s) resolve "
            "to NO registered dispatcher. The runtime will consume, DLQ and COMMIT every "
            "message on these topics while the consumer group reads Stable / LAG 0 "
            "(OMN-16939):\n"
        )
        for key in violations:
            f = live[key]
            sys.stderr.write(
                f"  - {f.contract} :: {f.topic} [{f.category}] :: {f.reason}\n"
                f"      {f.detail}\n"
            )
        sys.stderr.write(
            "\n  Fix: give the topic a handler_routing entry that registers under its own "
            "real category — a per-topic `topic_match` entry carrying an explicit "
            "`message_category:` where the topic name derives no category, or DELETE "
            "the subscribe declaration if nothing consumes the topic. There is no "
            "baseline to add it to: OMN-18013 burned this ratchet to zero and deleted "
            "the file, and `no-baseline-refreeze` refuses its recreation.\n"
        )

    if exit_code == 0:
        sys.stderr.write(
            f"[subscriber-dispatcher-resolution] OK: {contract_count} contracts scanned, "
            f"0 unresolved subscription(s).\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
