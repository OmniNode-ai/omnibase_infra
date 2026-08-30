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
        src/omnimarket --baseline config/validation/subscriber_dispatcher_resolution_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_event_type_alias_from_topic,
    _derive_message_category,
    _topics_for_handler_entry,
    derive_entry_message_category,
    derive_entry_message_types,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")
DEFAULT_BASELINE = Path(
    "config/validation/subscriber_dispatcher_resolution_baseline.yaml"
)

REASON_NO_ROUTE = "no_route"
REASON_CATEGORY_MISMATCH = "category_mismatch"
REASON_MESSAGE_TYPE_UNINDEXED = "message_type_unindexed"

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
    alias = _derive_event_type_alias_from_topic(topic)

    routing = contract.handler_routing
    entries = list(getattr(routing, "handlers", None) or []) if routing else []

    reason = REASON_NO_ROUTE
    detail = (
        "no handler_routing entry is assigned this topic by _topics_for_handler_entry"
    )

    for entry in entries:
        if topic not in _topics_for_handler_entry(contract, entry):
            continue
        entry_category = derive_entry_message_category(contract, entry)
        if entry_category != real_category:
            reason = REASON_CATEGORY_MISMATCH
            detail = (
                f"route registered under category={entry_category!r} but messages arrive "
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


def load_baseline(baseline_path: Path) -> set[tuple[str, str]]:
    """Load the frozen shrink-only burn-down baseline of unresolved subscriptions."""
    if not baseline_path.is_file():
        return set()
    data = yaml.safe_load(baseline_path.read_text()) or {}
    return {
        (str(row["contract"]), str(row["topic"]))
        for row in (data.get("known_unresolved_subscriptions") or [])
    }


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
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Frozen shrink-only burn-down baseline.",
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
    scan_root, baseline_path = Path(args.scan_root), Path(args.baseline)

    findings, contract_count = scan(scan_root)
    if contract_count < args.min_contracts:
        sys.stderr.write(
            f"[subscriber-dispatcher-resolution] FAIL (vacuity guard): only "
            f"{contract_count} contracts discovered under {scan_root} (expected >= "
            f"{args.min_contracts}). The contract scan is broken; a gate over a collapsed "
            f"set proves nothing.\n"
        )
        return 1

    baseline = load_baseline(baseline_path)
    live: dict[tuple[str, str], UnresolvedSubscription] = {f.key: f for f in findings}

    violations = sorted(set(live) - baseline)
    stale = sorted(baseline - set(live))
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
            "`message_category:` (and an `event_model:` where several topics share a "
            "handler). Setting `topic:` alone is NOT sufficient: the category still falls "
            "back to subscribe_topics[0]. Do NOT add the topic to "
            f"{baseline_path} — that baseline is frozen and shrink-only.\n"
        )

    if stale:
        exit_code = 1
        sys.stderr.write(
            f"[subscriber-dispatcher-resolution] FAIL: subscription(s) now resolve to a "
            f"dispatcher but are still listed in {baseline_path}. Remove them; the "
            f"baseline is shrink-only and must never go stale:\n"
        )
        for contract, topic in stale:
            sys.stderr.write(f"  - {contract} :: {topic}\n")

    if exit_code == 0:
        sys.stderr.write(
            f"[subscriber-dispatcher-resolution] OK: {contract_count} contracts scanned, "
            f"{len(live)} unresolved subscription(s) (all in the frozen baseline), "
            f"0 new violations.\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
