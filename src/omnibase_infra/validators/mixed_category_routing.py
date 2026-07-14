# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Forbid a handler_routing entry whose assigned topics span >1 real category (OMN-14605).

THE DEFECT CLASS THIS GATE CLOSES
---------------------------------
``_prepare_handler_wiring`` (``omnibase_infra.runtime.auto_wiring.handler_wiring``)
derives a handler_routing entry's message category EXACTLY ONCE — from an explicit
``entry.message_category`` override, else from ``contract.event_bus.subscribe_topics[0]``
(the contract's FIRST declared subscribe topic). That single category is then stamped
onto EVERY ``ModelDispatchRoute`` the entry registers, even when
``_topics_for_handler_entry`` assigns that entry MULTIPLE topics spanning different real
categories (``.evt.``=event, ``.cmd.``=command, ``.intent.``=intent).

``MessageDispatchEngine._find_matching_dispatchers`` filters candidate routes by the
REAL category of the topic a message arrived on (derived from the topic string) BEFORE
any handler runs. So any topic in a mixed-category entry whose real category differs from
the one category the entry was registered under is PERMANENTLY unroutable — every message
on it hits ``NO_DISPATCHER``, is logged as a WARNING and routed to DLQ (if declared) or
silently dropped. No error, no crash: the container boots green with ``wired=N failed=0``.

Live-verified on ``node_swarm_fanout_orchestrator`` (omnimarket): its 3 ``.evt.``
delegation topics were registered under ``command`` (because ``.cmd.``…swarm-fanout.v1 is
``subscribe_topics[0]``), so every delegation completion/escalation/failure event was
NO_DISPATCHER and its FSM could never leave COLLECTING (OMN-14606). Same "quiet death"
family as OMN-14517 (dispatcher-ID collision) and OMN-14593/OMN-14594 (missing
payload-type-scoping).

AGREEMENT WITH THE CORE RUNTIME INVARIANT (OMN-12549)
-----------------------------------------------------
``omnibase_core.runtime.mixin_node_dispatch.MixinNodeDispatch`` — the canonical future
dispatch engine — HARD-REJECTS a route whose ``message_category`` disagrees with its
dispatcher's registered category (``_validate_route_dispatcher_category``). A single
dispatcher spanning topics of >1 real category is exactly that shape. This gate flags at
authoring/CI time precisely what MixinNodeDispatch rejects at registration time, so the
CI gate and the runtime invariant cannot silently diverge (OMN-14208 seam discipline).
``tests/ci/test_mixed_category_routing_gate.py`` proves the agreement with a shared
synthetic contract driven through BOTH surfaces.

REAL HELPERS, NOT A REIMPLEMENTATION
------------------------------------
Topic assignment and category derivation import the REAL production helpers
(``_topics_for_handler_entry`` + ``_derive_message_category``) and the REAL discovery
path (``discover_contracts_from_paths``), so this gate observes exactly what
``_prepare_handler_wiring`` observes. If those helpers change, the gate follows.

RATCHET SEMANTICS (shrink-only; config/validation/mixed_category_routing_baseline.yaml)
---------------------------------------------------------------------------------------
  * an entry NOT in the baseline that is mixed-category  -> FAIL (no new instances, ever);
  * an entry IN the baseline that is no longer mixed-category -> FAIL until removed from
    the baseline (a fixed entry still listed is STALE, so the list cannot rot).
The baseline can only shrink. Baseline entries that remain mixed-category are green
(WARN-on-baseline) so this lands green day-1 and hard-fails only on new/growth.

Usage (pre-commit / CI):
    PYTHONPATH=src uv run python -m omnibase_infra.validators.mixed_category_routing src/omnibase_infra
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
    _derive_message_category,
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")
DEFAULT_BASELINE = Path("config/validation/mixed_category_routing_baseline.yaml")

# A scan that discovers far fewer contracts than the repo actually has is a broken scan,
# not a clean repo. A gate over a collapsed set is vacuously green, so the validator fails
# closed below this floor rather than reporting success. omnibase_infra has ~150 contracts.
MIN_EXPECTED_CONTRACTS = 60


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class MixedCategoryEntry:
    """A handler_routing entry whose assigned topics span more than one real category."""

    contract: str
    handler: str
    operation: str
    categories: tuple[
        str, ...
    ]  # sorted distinct real categories, e.g. ("command", "event")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.contract, self.handler, self.operation)


def mixed_category_findings(
    contracts: Iterable[ModelDiscoveredContract],
) -> list[MixedCategoryEntry]:
    """Every handler_routing entry whose ``_topics_for_handler_entry`` topics span >1 real
    category, computed with the REAL production helpers so the result matches
    ``_prepare_handler_wiring`` exactly.
    """
    findings: list[MixedCategoryEntry] = []
    for contract in contracts:
        routing = contract.handler_routing
        if routing is None or not getattr(routing, "handlers", None):
            continue
        for entry in routing.handlers:
            topics = _topics_for_handler_entry(contract, entry)
            if len(topics) < 2:
                continue
            categories = sorted({_derive_message_category(topic) for topic in topics})
            if len(categories) < 2:
                continue
            handler_name = getattr(getattr(entry, "handler", None), "name", "") or ""
            operation = (getattr(entry, "operation", None) or "").strip()
            findings.append(
                MixedCategoryEntry(
                    contract=contract.name,
                    handler=handler_name,
                    operation=operation,
                    categories=tuple(categories),
                )
            )
    return findings


def scan(scan_root: Path) -> tuple[list[MixedCategoryEntry], int]:
    """Discover every contract under ``scan_root`` and return (findings, contract_count)."""
    contract_paths = sorted(
        p
        for p in scan_root.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    discovered = discover_contracts_from_paths(contract_paths)
    contracts = list(getattr(discovered, "contracts", discovered))
    return mixed_category_findings(contracts), len(contracts)


def load_baseline(baseline_path: Path) -> set[tuple[str, str, str]]:
    """Load the frozen shrink-only burn-down baseline of known mixed-category entries."""
    if not baseline_path.is_file():
        return set()
    data = yaml.safe_load(baseline_path.read_text()) or {}
    return {
        (str(row["contract"]), str(row["handler"]), str(row.get("operation") or ""))
        for row in (data.get("known_mixed_category_entries") or [])
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail any handler_routing entry whose _topics_for_handler_entry-assigned "
            "topics span more than one real message category (permanent NO_DISPATCHER "
            "for the off-category topics)."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    scan_root, baseline_path = Path(args.scan_root), Path(args.baseline)

    findings, contract_count = scan(scan_root)
    if contract_count < MIN_EXPECTED_CONTRACTS:
        sys.stderr.write(
            f"[mixed-category-routing] FAIL (vacuity guard): only {contract_count} "
            f"contracts discovered under {scan_root} (expected >= {MIN_EXPECTED_CONTRACTS}). "
            f"The contract scan is broken; a gate over a collapsed set proves nothing.\n"
        )
        return 1

    baseline = load_baseline(baseline_path)
    live: dict[tuple[str, str, str], MixedCategoryEntry] = {f.key: f for f in findings}

    violations = sorted(set(live) - baseline)
    stale = sorted(baseline - set(live))
    exit_code = 0

    if violations:
        exit_code = 1
        sys.stderr.write(
            "[mixed-category-routing] FAIL: handler_routing entry(ies) assign topics that "
            "span MORE THAN ONE real category. handler_wiring stamps ONE category per "
            "entry, so the off-category topics register as permanent NO_DISPATCHER "
            "(silent drop / DLQ) and MixinNodeDispatch (OMN-12549) hard-rejects the same "
            "shape:\n"
        )
        for key in violations:
            f = live[key]
            cats = "+".join(f.categories)
            sys.stderr.write(
                f"  - {f.contract} :: {f.handler}/{f.operation or '(no-op)'} :: {cats}\n"
            )
        sys.stderr.write(
            "\n  Split the entry into per-topic `topic_match` entries, each with its own "
            "explicit `message_category:` (see OMN-14594). Do NOT add the entry to "
            f"{baseline_path} — that baseline is frozen and shrink-only. If the handler "
            "cannot process a dispatched event of that topic's shape, the fix is a "
            "rearchitecture, not a mechanical split (see OMN-14606/OMN-14607).\n"
        )

    if stale:
        exit_code = 1
        sys.stderr.write(
            f"[mixed-category-routing] FAIL: entry(ies) are no longer mixed-category but "
            f"are still listed in {baseline_path}. Remove them; the baseline is shrink-only "
            f"and must never go stale:\n"
        )
        for contract, handler, operation in stale:
            sys.stderr.write(f"  - {contract} :: {handler}/{operation or '(no-op)'}\n")

    if exit_code == 0:
        sys.stderr.write(
            f"[mixed-category-routing] OK: {contract_count} contracts scanned, "
            f"{len(live)} mixed-category entr{'y' if len(live) == 1 else 'ies'} "
            f"(all in the frozen baseline), 0 new violations.\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
