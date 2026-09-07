# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The event type a handler matches is read from a contract, never hand-typed (OMN-18013).

OPERATOR RULING (2026-09-06), ITEM 2
------------------------------------
Topics are declared in every contract. The string a handler compares an incoming
``event_type`` against must therefore be DERIVED from the contract that declares
the topic — never typed into the handler by hand — so a mismatch between what
the bus carries and what the handler expects is impossible rather than merely
detectable.

WHAT THE BUS ACTUALLY CARRIES
-----------------------------
``derive_event_type_alias_for_topic`` is the single source for the wire alias on
both sides (OMN-17296): ``onex.evt.<producer>.<name>.v1`` carries
``<producer>.<name>``. It is NOT the topic string. Before OMN-18013 the
auto-wired consume boundary also honoured an ``event_type`` field lifted out of
the raw payload — an untyped, uncontracted string no contract declares and no
gate checks — which let a publisher re-key a message onto any dispatcher and was
the only path by which a hand-typed alias was ever reachable. That override is
removed; this gate removes the shapes that depended on it.

THE TWO REFUSED SHAPES
----------------------
``alias_matches_no_subscribed_topic``
    a ``handler_routing`` entry declares ``event_type:`` that is neither one of
    the contract's own subscribe topics nor
    ``derive_event_type_alias_for_topic`` of one. The dispatcher is then indexed
    under a key the consume boundary never produces, so the entry owns zero
    topics and the subscription is dead. Twelve such entries existed across the
    corpus at the time of the ruling.

``hand_typed_event_type_literal``
    a handler module compares an event type (or a topic) against a literal
    ``onex.*`` string or a ``.v1`` suffix. Even where such a comparison happens
    to be right today it is a second, uncontracted spelling of a name the
    contract already owns, and it silently rots when the topic is renamed. The
    fix is to read the topic from the contract and derive the alias from it —
    see ``node_coding_agent_orchestrator``'s
    ``EVENT_TYPE_WORKSPACE_VALIDATED`` for the shape.

Usage (pre-commit / CI):
    uv run python -m omnibase_infra.validators.handler_event_type_source \\
        --scope omnibase_infra
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
from omnibase_infra.validators.contract_topic_category import SCOPES

REASON_ALIAS_UNMATCHED = "alias_matches_no_subscribed_topic"
REASON_HAND_TYPED = "hand_typed_event_type_literal"

# A comparison of some value against a literal ONEX topic, or against a bare
# ``…v1`` suffix. Both spell a name the contract already owns.
_LITERAL_TOPIC = re.compile(r"onex\.(?:evt|cmd|intent|dlq|snapshot)\..+")
_SUFFIX_MATCH = re.compile(r"\.v\d+$")

# The existing, deliberate escape hatch for a topic constant that has no contract
# to read it from yet. It must name a reason; a bare marker is not enough.
_ALLOW = re.compile(r"#\s*onex-topic-allow:\s*\S")


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class EventTypeFinding:
    """One place an event type is spelled by hand rather than read from a contract."""

    location: str
    reason: str
    detail: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.location, self.reason, self.detail)


def contract_alias_findings(
    contracts: Iterable[ModelDiscoveredContract],
) -> list[EventTypeFinding]:
    """Entries whose declared ``event_type:`` matches none of their own topics."""
    findings: list[EventTypeFinding] = []
    for contract in contracts:
        routing = contract.handler_routing
        if routing is None or not getattr(routing, "handlers", None):
            continue
        topics = tuple(
            contract.event_bus.subscribe_topics if contract.event_bus else ()
        )
        acceptable = set(topics) | {
            alias
            for topic in topics
            if (alias := derive_event_type_alias_for_topic(topic)) is not None
        }
        for entry in routing.handlers:
            declared = (getattr(entry, "event_type", None) or "").strip()
            if not declared or declared in acceptable:
                continue
            handler = getattr(getattr(entry, "handler", None), "name", "") or "?"
            findings.append(
                EventTypeFinding(
                    location=f"{contract.name} :: {handler}",
                    reason=REASON_ALIAS_UNMATCHED,
                    detail=(
                        f"event_type: {declared!r} is neither a subscribe topic of this "
                        f"contract nor the alias derived from one. Accepted aliases: "
                        + (
                            ", ".join(sorted(acceptable))
                            or "(contract subscribes to nothing)"
                        )
                    ),
                )
            )
    return findings


def _allowed_lines(text: str) -> set[int]:
    """Line numbers carrying the sanctioned ``# onex-topic-allow: <reason>`` marker."""
    return {
        n for n, line in enumerate(text.splitlines(), start=1) if _ALLOW.search(line)
    }


def _literal_strings_in_matching_positions(
    tree: ast.AST,
) -> list[tuple[int, str]]:
    """Every string literal used as a COMPARISON operand or an endswith/startswith arg.

    Walking the AST rather than the raw text is what keeps the gate off comments
    and docstrings: a comment is not in the tree at all, and a docstring is an
    ``ast.Expr`` this never descends into for constants. Only a literal the code
    actually MATCHES ON is a finding — quoting a topic in prose is not a defect.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for operand in operands:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                    hits.append((operand.lineno, operand.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in ("endswith", "startswith"):
                continue
            for arg in node.args:
                targets = arg.elts if isinstance(arg, ast.Tuple) else [arg]
                for target in targets:
                    if isinstance(target, ast.Constant) and isinstance(
                        target.value, str
                    ):
                        hits.append((target.lineno, target.value))
    return hits


def source_literal_findings(scan_root: Path) -> list[EventTypeFinding]:
    """Handler modules that compare an event type against a hand-typed literal."""
    findings: list[EventTypeFinding] = []
    for path in sorted(scan_root.rglob("handlers/*.py")):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        allowed = _allowed_lines(text)
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:  # a file that does not parse is a real failure
            findings.append(
                EventTypeFinding(
                    location=f"{path}:{exc.lineno or 0}",
                    reason=REASON_HAND_TYPED,
                    detail=f"module does not parse, so it cannot be checked: {exc.msg}",
                )
            )
            continue
        for lineno, value in _literal_strings_in_matching_positions(tree):
            if lineno in allowed:
                continue
            if not (_LITERAL_TOPIC.fullmatch(value) or _SUFFIX_MATCH.search(value)):
                continue
            findings.append(
                EventTypeFinding(
                    location=f"{path}:{lineno}",
                    reason=REASON_HAND_TYPED,
                    detail=(
                        f"matches on the hand-typed literal {value!r}; read the topic "
                        "from the contract and derive its alias instead"
                    ),
                )
            )
    return findings


def scan(scan_root: Path) -> tuple[list[EventTypeFinding], int]:
    """Return (findings, contract_count) for ``scan_root``."""
    contract_paths = sorted(
        p
        for p in scan_root.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    discovered = discover_contracts_from_paths(contract_paths)
    contracts = list(getattr(discovered, "contracts", discovered))
    findings = contract_alias_findings(contracts) + source_literal_findings(scan_root)
    return findings, len(contracts)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refuse a handler event_type that is hand-typed rather than derived "
            "from the contract that declares the topic (OMN-18013 item 2)."
        )
    )
    parser.add_argument("--scope", choices=sorted(SCOPES))
    parser.add_argument("scan_root", nargs="?", default=None)
    parser.add_argument("--min-contracts", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.scope is None and args.scan_root is None:
        sys.stderr.write(
            "[handler-event-type-source] FAIL: pass --scope <repo> (or an explicit "
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
            f"[handler-event-type-source] FAIL (vacuity guard): only {contract_count} "
            f"contracts discovered under {scan_root} (expected >= {floor}).\n"
        )
        return 1

    if findings:
        sys.stderr.write(
            "[handler-event-type-source] FAIL: event type(s) spelled by hand instead "
            "of read from the contract that declares the topic (OMN-18013):\n"
        )
        for f in sorted(findings, key=lambda x: x.key):
            sys.stderr.write(f"  - {f.location} :: {f.reason}\n      {f.detail}\n")
        sys.stderr.write(
            "\n  Fix: read the topic from the contract (event_bus.subscribe_topics) and "
            "derive the wire alias with derive_event_type_alias_for_topic — the single "
            "source for that alias on both sides of the wire (OMN-17296). See "
            "node_coding_agent_orchestrator's EVENT_TYPE_WORKSPACE_VALIDATED. There is "
            "no baseline for this gate and none may be added.\n"
        )
        return 1

    sys.stderr.write(
        f"[handler-event-type-source] OK: {contract_count} contracts and every "
        f"handler module under {scan_root} read their event types from a contract.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
