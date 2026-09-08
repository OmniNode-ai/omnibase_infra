# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every route's message category comes from its own topic's name (OMN-18013).

OPERATOR RULING (2026-09-06), ITEM 1
------------------------------------
Topics are declared in every contract, so a topic/category mismatch must be
IMPOSSIBLE, not merely detectable. Two things are therefore true after
OMN-18013 and this gate proves both:

1. ``handler_wiring.derive_route_message_category`` stamps EACH
   ``ModelDispatchRoute`` with the category derived from THAT ROUTE'S OWN topic
   name. The pre-OMN-18013 code derived ONE category per handler entry — from
   ``contract.event_bus.subscribe_topics[0]``, i.e. from a LIST POSITION — and
   stamped it on every route the entry registered. ``MessageDispatchEngine``
   filters candidate routes by the real category of the topic a message
   arrived on, so every off-category sibling topic was permanently
   NO_DISPATCHER: consumed, DLQ'd and COMMITTED while the consumer group read
   Stable / LAG 0. Live-proven on ``node_swarm_fanout_orchestrator``
   (OMN-14606) and ``node_pr_lifecycle_state_reducer`` (OMN-16939: 174 in,
   174 DLQ'd over six hours).

2. THIS gate refuses, at authoring time, the two shapes that would still let a
   category be wrong:

   ``mixed_category_entry``
       one handler_routing entry is assigned topics spanning more than one real
       category. Routing is now per-topic so this no longer silently drops
       traffic, but the entry's single declared/derived category is then a
       statement that cannot be true of all its topics, and the operator ruling
       requires the contract to be unambiguous. Split it into per-topic
       ``topic:``-scoped entries.

   ``declared_category_contradicts_topic``
       an entry declares ``message_category:`` that disagrees with the category
       its topic's own name derives. That is a contract lie: the declaration
       loses to the topic at dispatch, so the route it produces is unreachable.
       An explicit declaration is permitted ONLY for a topic whose name derives
       nothing (``EnumMessageCategory.from_topic`` -> ``None``); anywhere else
       the name is the authority.

   ``underivable_category_undeclared``
       a subscribe topic whose name derives no category and whose owning entry
       declares none. The pre-OMN-18013 ``_derive_message_category`` returned an
       unconditional ``"event"`` here — a silent default of exactly the shape
       CLAUDE.md rule 8 forbids — while ``EnumMessageCategory.from_topic``
       returned ``None`` at dispatch and rejected the message as an invalid
       topic category. Registration and dispatch disagreed, and 32 live topics
       (``onex.dlq.*``, ``onex.snapshot.*``) sat in that gap.

NO BASELINE, BY CONSTRUCTION
----------------------------
``config/validation/mixed_category_routing_baseline.yaml`` (the OMN-14605
ratchet) was burned to zero and DELETED by OMN-18013. This module has no
``--baseline`` flag and no allowlist to add one to, and ``no-baseline-refreeze``
fails if the file reappears. A gate that can be re-frozen is a gate that will be.

REAL HELPERS, NOT A REIMPLEMENTATION
------------------------------------
Topic assignment and category derivation import the REAL production helpers
(``_topics_for_handler_entry``, ``_derive_message_category``,
``derive_route_message_category``) and the REAL discovery path
(``discover_contracts_from_paths``), so the gate observes exactly what
``_prepare_handler_wiring`` observes. If those change, the gate follows.

Usage (pre-commit / CI):
    uv run python -m omnibase_infra.validators.contract_topic_category \\
        --scope omnibase_infra
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_message_category,
    _topics_for_handler_entry,
    derive_route_message_category,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

# Every repo whose contracts this gate can be pointed at, and the vacuity floor
# for each. A scan that discovers far fewer contracts than the repo actually has
# is a broken scan, not a clean repo, and a gate over a collapsed set is
# vacuously green — so the validator fails closed below the floor. The floors are
# deliberately well under the live counts so an ordinary contract deletion does
# not trip them.
SCOPES: dict[str, tuple[str, int]] = {
    "omnibase_infra": ("src/omnibase_infra", 60),
    "omnimarket": ("src/omnimarket", 150),
    "omniintelligence": ("src/omniintelligence", 20),
    "omnimemory": ("src/omnimemory", 5),
    "omniclaude": ("src/omniclaude", 5),
    "omnibase_core": ("src/omnibase_core", 1),
}

REASON_MIXED = "mixed_category_entry"
REASON_CONTRADICTS = "declared_category_contradicts_topic"
REASON_UNDERIVABLE = "underivable_category_undeclared"


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class CategoryFinding:
    """One handler_routing entry whose route category cannot come from its topic."""

    contract: str
    handler: str
    operation: str
    reason: str
    detail: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.contract, self.handler, self.operation, self.reason)


def category_findings(
    contracts: Iterable[ModelDiscoveredContract],
) -> list[CategoryFinding]:
    """Every entry whose per-topic route category is ambiguous, contradicted or absent."""
    findings: list[CategoryFinding] = []
    for contract in contracts:
        routing = contract.handler_routing
        if routing is None or not getattr(routing, "handlers", None):
            continue
        for entry in routing.handlers:
            topics = _topics_for_handler_entry(contract, entry)
            if not topics:
                continue
            handler_name = getattr(getattr(entry, "handler", None), "name", "") or ""
            operation = (getattr(entry, "operation", None) or "").strip()

            categories = sorted(
                {
                    derived
                    for topic in topics
                    if (derived := _derive_message_category(topic))
                }
            )
            if len(categories) > 1:
                findings.append(
                    CategoryFinding(
                        contract=contract.name,
                        handler=handler_name,
                        operation=operation,
                        reason=REASON_MIXED,
                        detail=(
                            f"assigned topics span categories {'+'.join(categories)}: "
                            + ", ".join(sorted(topics))
                        ),
                    )
                )
                continue

            for topic in sorted(topics):
                try:
                    derive_route_message_category(contract, entry, topic)
                except ModelOnexError as exc:
                    derived = _derive_message_category(topic)
                    findings.append(
                        CategoryFinding(
                            contract=contract.name,
                            handler=handler_name,
                            operation=operation,
                            reason=(
                                REASON_CONTRADICTS
                                if derived is not None
                                else REASON_UNDERIVABLE
                            ),
                            detail=f"{topic}: {exc}",
                        )
                    )
    return findings


def scan(scan_root: Path) -> tuple[list[CategoryFinding], int]:
    """Discover every contract under ``scan_root`` and return (findings, count)."""
    contract_paths = sorted(
        p
        for p in scan_root.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    discovered = discover_contracts_from_paths(contract_paths)
    contracts = list(getattr(discovered, "contracts", discovered))
    return category_findings(contracts), len(contracts)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refuse any handler_routing entry whose route category cannot be taken "
            "from its own topic's name (OMN-18013 operator ruling item 1)."
        )
    )
    parser.add_argument(
        "--scope",
        choices=sorted(SCOPES),
        help=(
            "Repo whose contracts to scan. Selects the scan root and the vacuity "
            "floor. Per-repo scoping is what lets each repo be green independently."
        ),
    )
    parser.add_argument(
        "scan_root",
        nargs="?",
        default=None,
        help="Explicit root to scan (overrides --scope's default root).",
    )
    parser.add_argument(
        "--min-contracts",
        type=int,
        default=None,
        help="Vacuity floor override; defaults to the --scope floor.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.scope is None and args.scan_root is None:
        sys.stderr.write(
            "[contract-topic-category] FAIL: pass --scope <repo> (or an explicit "
            "scan_root). Defaulting the scope is how a gate silently scans nothing.\n"
        )
        return 1
    default_root, default_floor = (
        SCOPES[args.scope] if args.scope else (args.scan_root, 1)
    )
    scan_root = Path(args.scan_root or default_root)
    floor = args.min_contracts if args.min_contracts is not None else default_floor

    findings, contract_count = scan(scan_root)
    if contract_count < floor:
        sys.stderr.write(
            f"[contract-topic-category] FAIL (vacuity guard): only {contract_count} "
            f"contracts discovered under {scan_root} (expected >= {floor}). The "
            "contract scan is broken; a gate over a collapsed set proves nothing.\n"
        )
        return 1

    if findings:
        sys.stderr.write(
            "[contract-topic-category] FAIL: handler_routing entr(ies) whose route "
            "category does not come from the topic's own name (OMN-18013):\n"
        )
        for f in sorted(findings, key=lambda x: x.key):
            sys.stderr.write(
                f"  - {f.contract} :: {f.handler}/{f.operation or '(no-op)'} "
                f":: {f.reason}\n      {f.detail}\n"
            )
        sys.stderr.write(
            "\n  Fix: split a mixed entry into per-topic `topic:`-scoped entries; "
            "remove a `message_category:` that contradicts its topic's name; declare "
            "`message_category:` only for a topic whose name derives none, or delete "
            "that subscription. There is no baseline: OMN-18013 burned the OMN-14605 "
            "ratchet to zero and deleted the file, and `no-baseline-refreeze` refuses "
            "its recreation.\n"
        )
        return 1

    sys.stderr.write(
        f"[contract-topic-category] OK: {contract_count} contracts scanned under "
        f"{scan_root}, every route category derived from its own topic.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
