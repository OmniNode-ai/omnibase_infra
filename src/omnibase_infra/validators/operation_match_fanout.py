# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Forbid an ``operation_match`` handler_routing entry that falls through to full
topic fan-out, for ANY contract (OMN-16088).

THE DEFECT CLASS THIS GATE CLOSES
----------------------------------
``_topics_for_handler_entry`` (``omnibase_infra.runtime.auto_wiring.handler_wiring``)
resolves the subscribe topics a ``handler_routing`` entry is assigned in three steps:
an explicit ``entry.topic`` (topic_match strategy) wins outright, else an explicit
``entry.event_type`` alias wins, else — when ``entry.event_model is None``, which is
always true for an ``operation_match`` entry (that strategy routes on envelope
``operation``, not on a payload model) — the entry falls through to
``return topics``: it is assigned **every** subscribe topic the contract declares,
regardless of how many other handler entries exist or what each entry's declared
``operation:`` is.

An ``operation_match`` entry that carries ``operation:`` reads, at a glance, as
"this handler only receives messages for this one operation." It does not: unless the
entry ALSO pins itself to a topic (via ``topic:`` or ``event_type:``), the wiring
resolver hands it every topic on the contract's event bus — full fan-out — and nothing
downstream re-filters by ``operation`` at dispatch-route registration time. Every other
entry on the same contract independently computes the identical "all topics" answer, so
N operation-scoped handlers each register for all N topics: a message on any one topic
reaches every handler, and the ones whose ``operation`` does not match crash or misfire.

Live-verified precedent: ``node_gateway_attach_effect`` (OMN-15978) had exactly this
shape — 3 operation_match entries (``gateway.attach`` / ``gateway.heartbeat`` /
``gateway.detach``), none pinned to a topic, 3 subscribe topics — so all 3 handlers
registered on all 3 topics, producing permanent ``HANDLER_ERROR`` on 2 of every 3
deliveries. The fix (omnibase_infra#2729) added an explicit ``entry.topic`` to each of
the 3 entries. That fix was node-scoped; this gate is the generic AC4 that was split out
to prove NO other/new contract can reintroduce the same shape (origin: OMN-15978 AC4,
split at close 2026-08-16).

REAL HELPER, NOT A REIMPLEMENTATION
------------------------------------
Topic assignment imports the REAL production helper (``_topics_for_handler_entry``) and
the REAL discovery path (``discover_contracts_from_paths``), so this gate observes
exactly what ``_prepare_handler_wiring`` observes. If that helper changes, the gate
follows — it can never silently drift out of agreement with the runtime it is guarding.

RATCHET SEMANTICS (shrink-only; config/validation/operation_match_fanout_baseline.yaml)
----------------------------------------------------------------------------------------
Live-scanning the repo at authoring time surfaced 7 pre-existing instances of this shape
beyond the already-fixed ``node_gateway_attach_effect`` (e.g.
``node_artifact_change_detector_effect``'s 3 handlers, which are ALSO already tracked in
``mixed_category_routing_baseline.yaml`` as MISSING_HANDLE — a real defect, not a false
positive). Mirroring the sibling OMN-14605 gate exactly:
  * an entry NOT in the baseline that fans out -> FAIL (no new instances, ever);
  * an entry IN the baseline that no longer fans out -> FAIL until removed from the
    baseline (a fixed entry still listed is STALE, so the list cannot rot).
The baseline can only shrink. Baseline entries stay green (WARN-on-baseline) so this
gate lands green day-1 and hard-fails only on new/growth. ``node_gateway_attach_effect``
(fixed by OMN-15978) is intentionally NOT in the baseline — it must pass the gate
unchanged, proving no regression on the fixed node.

Usage (pre-commit / CI):
    PYTHONPATH=src uv run python -m omnibase_infra.validators.operation_match_fanout src/omnibase_infra
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import _topics_for_handler_entry
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")
DEFAULT_BASELINE = Path("config/validation/operation_match_fanout_baseline.yaml")

# A scan that discovers far fewer contracts than the repo actually has is a broken scan,
# not a clean repo. A gate over a collapsed set is vacuously green, so the validator fails
# closed below this floor rather than reporting success. omnibase_infra has ~150 contracts.
MIN_EXPECTED_CONTRACTS = 60


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class FanoutEntry:
    """An ``operation_match`` handler_routing entry assigned every subscribe topic."""

    contract: str
    handler: str
    operation: str
    topic_count: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.contract, self.handler, self.operation)


def operation_match_fanout_findings(
    contracts: Iterable[ModelDiscoveredContract],
) -> list[FanoutEntry]:
    """Every handler_routing entry that declares ``operation:`` (operation_match
    strategy) but no ``topic:``/``event_type:`` scoping, whose
    ``_topics_for_handler_entry``-assigned topics span the contract's ENTIRE
    subscribe-topic set with more than one topic — the fall-through-to-fan-out shape.

    Computed with the REAL production helper so the result matches
    ``_prepare_handler_wiring`` exactly.
    """
    findings: list[FanoutEntry] = []
    for contract in contracts:
        routing = contract.handler_routing
        if routing is None or not getattr(routing, "handlers", None):
            continue
        for entry in routing.handlers:
            operation = (entry.operation or "").strip()
            if not operation:
                continue  # not an operation_match entry
            if (entry.topic or "").strip():
                continue  # pinned to a topic (topic_match override) — scoped, safe
            if (entry.event_type or "").strip():
                continue  # pinned to an event_type alias — scoped, safe
            topics = _topics_for_handler_entry(contract, entry)
            if len(topics) <= 1:
                continue  # nothing to fan out to, or already scoped to one topic
            handler_name = getattr(getattr(entry, "handler", None), "name", "") or ""
            findings.append(
                FanoutEntry(
                    contract=contract.name,
                    handler=handler_name,
                    operation=operation,
                    topic_count=len(topics),
                )
            )
    return findings


def scan(scan_root: Path) -> tuple[list[FanoutEntry], int]:
    """Discover every contract under ``scan_root`` and return (findings, contract_count)."""
    contract_paths = sorted(
        p
        for p in scan_root.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    discovered = discover_contracts_from_paths(contract_paths)
    contracts = list(getattr(discovered, "contracts", discovered))
    return operation_match_fanout_findings(contracts), len(contracts)


def load_baseline(baseline_path: Path) -> set[tuple[str, str, str]]:
    """Load the frozen shrink-only burn-down baseline of known fan-through entries."""
    if not baseline_path.is_file():
        return set()
    data = yaml.safe_load(baseline_path.read_text()) or {}
    return {
        (str(row["contract"]), str(row["handler"]), str(row.get("operation") or ""))
        for row in (data.get("known_fanout_entries") or [])
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail any operation_match handler_routing entry (operation: set, no "
            "topic:/event_type: scoping) whose _topics_for_handler_entry-assigned "
            "topics span the contract's full subscribe-topic set (permanent "
            "cross-operation fan-out / mis-dispatch)."
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
            f"[operation-match-fanout] FAIL (vacuity guard): only {contract_count} "
            f"contracts discovered under {scan_root} (expected >= {MIN_EXPECTED_CONTRACTS}). "
            f"The contract scan is broken; a gate over a collapsed set proves nothing.\n"
        )
        return 1

    baseline = load_baseline(baseline_path)
    live: dict[tuple[str, str, str], FanoutEntry] = {f.key: f for f in findings}

    violations = sorted(set(live) - baseline)
    stale = sorted(baseline - set(live))
    exit_code = 0

    if violations:
        exit_code = 1
        sys.stderr.write(
            "[operation-match-fanout] FAIL: operation_match handler_routing entry(ies) "
            "carry `operation:` with no `topic:`/`event_type:` scoping, so "
            "_topics_for_handler_entry falls through to `return topics` and assigns the "
            "entry EVERY subscribe topic on the contract. Every other operation_match "
            "entry on the same contract independently resolves the same full topic set, "
            "so N operation-scoped handlers each register for all N topics — a message "
            "on ANY topic reaches every handler, and the ones whose `operation` does not "
            "match crash or misfire (the OMN-15978 node_gateway_attach_effect shape):\n"
        )
        for key in violations:
            f = live[key]
            sys.stderr.write(
                f"  - {f.contract} :: {f.handler}/{f.operation} :: "
                f"fanned out to {f.topic_count} topics\n"
            )
        sys.stderr.write(
            "\n  Fix: add an explicit `topic:` (or `event_type:`) to the entry so it "
            "owns exactly the one topic its operation applies to (see "
            f"node_gateway_attach_effect/contract.yaml, OMN-15978). Do NOT add the entry "
            f"to {baseline_path} — that baseline is frozen and shrink-only.\n"
        )

    if stale:
        exit_code = 1
        sys.stderr.write(
            f"[operation-match-fanout] FAIL: entry(ies) no longer fan out but are still "
            f"listed in {baseline_path}. Remove them; the baseline is shrink-only and "
            f"must never go stale:\n"
        )
        for contract, handler, operation in stale:
            sys.stderr.write(f"  - {contract} :: {handler}/{operation or '(no-op)'}\n")

    if exit_code == 0:
        sys.stderr.write(
            f"[operation-match-fanout] OK: {contract_count} contracts scanned, "
            f"{len(live)} fan-through entr{'y' if len(live) == 1 else 'ies'} "
            f"(all in the frozen baseline), 0 new violations.\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
