#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatcher route-coverage gate (OMN-12858 / OMN-12879 / OMN-12880).

Static analysis: for every contract that subscribes to a command topic
(``onex.cmd.*``), assert that the contract declares a dispatcher route via
``handler_routing`` or ``runtime_dispatch``.  Missing wiring causes the
command to go to DLQ silently — this gate catches that regression before
it reaches the runtime.

Coverage included (OMN-12880):
- Primary ``event_bus.subscribe_topics`` entries.
- ``compatibility_publish_topics`` entries are treated as sender-side only
  (a publisher does not require a handler route for topics it emits to
  compatibility surfaces — routes are required only on the subscriber side).

Changed-command-topic coverage (OMN-12879):
- When ``--changed-contracts`` paths are provided (space-separated list from
  CI diff), only those contracts are checked for missing routes.  This makes
  failures actionable to the PR author without blocking on pre-existing
  ratchet violations.

Usage::

    # Full scan (both repos checked out as siblings)
    uv run python scripts/check_dispatcher_route_coverage.py \\
        --contracts-dir src/omnibase_infra/nodes \\
        --extra-contracts-dir ../omnimarket/src/omnimarket/nodes

    # CI changed-contract mode (OMN-12879)
    uv run python scripts/check_dispatcher_route_coverage.py \\
        --contracts-dir src/omnibase_infra/nodes \\
        --extra-contracts-dir ../omnimarket/src/omnimarket/nodes \\
        --changed-contracts "src/.../contract.yaml src/.../contract.yaml"

    # Verbose output
    uv run python scripts/check_dispatcher_route_coverage.py \\
        --contracts-dir src/omnibase_infra/nodes \\
        --extra-contracts-dir ../omnimarket/src/omnimarket/nodes \\
        --verbose

Exit codes:
    0 = all command-topic subscriptions have a dispatcher route (or are allowlisted)
    1 = one or more command-topic subscriptions are missing dispatcher wiring

[OMN-12858, OMN-12879, OMN-12880]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Allow running as a standalone script when omnibase_infra is not installed.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CMD_TOPIC_PREFIX = "onex.cmd."

# ---------------------------------------------------------------------------
# Allowlist: command-topic subscriptions that are expected to have no
# handler_routing / runtime_dispatch because they are consumed by an
# internal broker outside the standard dispatch surface.
# Format: "topic": "reason | owner | expiry"
# ---------------------------------------------------------------------------
_ALLOWLISTED_CMD_TOPICS: dict[str, str] = {
    # Pattern-B dispatch broker owns its own consumption model; the contract
    # declares the subscription but RuntimePatternBBroker processes the
    # messages directly, bypassing handler_routing.
    "onex.cmd.omnibase-infra.pattern-b-dispatch.v1": (
        "Consumed by RuntimePatternBBroker directly, not via handler_routing "
        "| owner: jonah | expiry: 2027-01-01"
    ),
    # Transitional HTTP bridge (OMN-2756): subscribes for observability/future
    # Kafka-native switch but is currently an HTTP-dispatch node.  Dispatcher
    # route will be added when the node is migrated to native Kafka dispatch.
    # Ref: node_contract_resolver_bridge/contract.yaml — metadata.transitional=true
    "onex.cmd.platform.contract-resolve-requested.v1": (
        "Transitional HTTP bridge node_contract_resolver_bridge (OMN-2756); "
        "processes via FastAPI not Kafka handler_routing | owner: jonah | expiry: 2027-01-01"
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContractInfo:
    """Parsed subset of a contract.yaml needed for route coverage checking."""

    name: str
    contract_path: Path
    subscribe_topics: tuple[str, ...]
    has_handler_routing: bool
    has_runtime_dispatch: bool

    @property
    def has_dispatcher_route(self) -> bool:
        return self.has_handler_routing or self.has_runtime_dispatch


@dataclass
class RouteCoverageFailure:
    """One missing-route finding."""

    topic: str
    contract_name: str
    contract_path: Path


@dataclass
class RouteCoverageReport:
    """Result of the full scan."""

    scanned: list[ContractInfo] = field(default_factory=list)
    failures: list[RouteCoverageFailure] = field(default_factory=list)
    allowlisted: list[tuple[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------
def _parse_topic_list(raw: Any) -> tuple[str, ...]:
    """Parse a subscribe_topics or publish_topics value from YAML.

    Handles both flat-string lists and schema-B structured dicts::

        subscribe_topics:
          - "onex.cmd.foo.bar.v1"
          - topic: "onex.cmd.foo.baz.v1"
            operation: "ingest"
    """
    if not raw:
        return ()
    topics: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            topics.append(entry)
        elif isinstance(entry, dict):
            t = entry.get("topic", "")
            if isinstance(t, str) and t:
                topics.append(t)
    return tuple(topics)


def _parse_contract(path: Path) -> ContractInfo | None:
    """Parse a contract.yaml and return a ContractInfo, or None on error."""
    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None

    name: str = raw.get("name") or path.parent.name

    # event_bus.subscribe_topics
    event_bus: dict[str, Any] = raw.get("event_bus") or {}
    subscribe_topics = _parse_topic_list(event_bus.get("subscribe_topics"))

    # Dispatcher route declarations
    has_handler_routing = bool(raw.get("handler_routing"))
    has_runtime_dispatch = bool(raw.get("runtime_dispatch"))

    return ContractInfo(
        name=name,
        contract_path=path,
        subscribe_topics=subscribe_topics,
        has_handler_routing=has_handler_routing,
        has_runtime_dispatch=has_runtime_dispatch,
    )


def _scan_contracts_dir(contracts_dir: Path) -> list[ContractInfo]:
    """Recursively find and parse all contract.yaml files under a directory."""
    contracts: list[ContractInfo] = []
    for p in sorted(contracts_dir.rglob("contract.yaml")):
        info = _parse_contract(p)
        if info is not None:
            contracts.append(info)
    return contracts


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------
def check_route_coverage(
    contracts_dirs: list[Path],
    *,
    changed_contract_paths: frozenset[Path] | None = None,
    verbose: bool = False,
) -> RouteCoverageReport:
    """Run the dispatcher route-coverage check.

    Args:
        contracts_dirs: List of directories to scan for contract.yaml files.
        changed_contract_paths: When provided (OMN-12879 mode), only report
            failures for contracts whose path is in this set.  The full scan
            populates ``scanned``; the filter only affects ``failures``.
            Pass ``None`` to check all contracts (full-scan mode).
        verbose: Print per-contract debug lines to stdout.

    Returns:
        A :class:`RouteCoverageReport` with the scan results.
    """
    report = RouteCoverageReport()

    for contracts_dir in contracts_dirs:
        if not contracts_dir.is_dir():
            if verbose:
                print(f"  [SKIP missing dir] {contracts_dir}", flush=True)
            continue
        for info in _scan_contracts_dir(contracts_dir):
            report.scanned.append(info)

            for topic in info.subscribe_topics:
                if not topic.startswith(_CMD_TOPIC_PREFIX):
                    continue  # only command topics require dispatch wiring

                if topic in _ALLOWLISTED_CMD_TOPICS:
                    report.allowlisted.append((topic, _ALLOWLISTED_CMD_TOPICS[topic]))
                    if verbose:
                        print(
                            f"  [ALLOWLISTED] {info.name}: {topic}",
                            flush=True,
                        )
                    continue

                # OMN-12879: changed-contract mode — skip unchanged contracts.
                if changed_contract_paths is not None:
                    if info.contract_path not in changed_contract_paths:
                        if verbose:
                            print(
                                f"  [SKIP unchanged] {info.name}: {topic}",
                                flush=True,
                            )
                        continue

                if info.has_dispatcher_route:
                    if verbose:
                        route_kind = (
                            "handler_routing"
                            if info.has_handler_routing
                            else "runtime_dispatch"
                        )
                        print(
                            f"  [OK  {route_kind}] {info.name}: {topic}",
                            flush=True,
                        )
                    continue

                # Failure: subscribed command topic with no dispatcher route.
                report.failures.append(
                    RouteCoverageFailure(
                        topic=topic,
                        contract_name=info.name,
                        contract_path=info.contract_path,
                    )
                )
                if verbose:
                    print(
                        f"  [FAIL] {info.name}: {topic}  -- no dispatcher route",
                        flush=True,
                    )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that every command-topic subscription has a dispatcher route "
            "(OMN-12858 / OMN-12879 / OMN-12880)."
        ),
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=_REPO_ROOT / "src" / "omnibase_infra" / "nodes",
        help="Primary contracts directory to scan (default: src/omnibase_infra/nodes)",
    )
    parser.add_argument(
        "--extra-contracts-dir",
        dest="extra_contracts_dirs",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional contract directories (repeatable; "
            "e.g. ../omnimarket/src/omnimarket/nodes)"
        ),
    )
    parser.add_argument(
        "--changed-contracts",
        type=str,
        default="",
        help=(
            "OMN-12879: space-separated list of contract.yaml paths that changed in "
            "this PR diff.  When non-empty, only those contracts are checked."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit per-contract status lines",
    )
    args = parser.parse_args()

    contracts_dirs: list[Path] = [args.contracts_dir, *args.extra_contracts_dirs]

    # OMN-12879: parse changed-contract paths
    changed_contract_paths: frozenset[Path] | None = None
    if args.changed_contracts.strip():
        changed_contract_paths = frozenset(
            Path(p).resolve() for p in args.changed_contracts.split() if p.strip()
        )

    print(
        "================================================================",
        flush=True,
    )
    print(
        "Dispatcher Route-Coverage Gate (OMN-12858 / OMN-12879 / OMN-12880)",
        flush=True,
    )
    print(
        "================================================================",
        flush=True,
    )
    for d in contracts_dirs:
        print(f"  scanning: {d}", flush=True)
    if changed_contract_paths is not None:
        print(
            f"  changed-contract mode (OMN-12879): {len(changed_contract_paths)} paths",
            flush=True,
        )

    report = check_route_coverage(
        contracts_dirs,
        changed_contract_paths=changed_contract_paths,
        verbose=args.verbose,
    )

    print(
        f"\nContracts scanned: {len(report.scanned)}"
        f"  |  allowlisted: {len(report.allowlisted)}"
        f"  |  failures: {len(report.failures)}",
        flush=True,
    )

    if report.failures:
        print(
            "\nFAILURES — command topics subscribed without a dispatcher route:\n",
            flush=True,
        )
        for f in report.failures:
            print(f"  topic:    {f.topic}")
            print(f"  contract: {f.contract_name}")
            print(f"  path:     {f.contract_path}")
            print("  fix:      add handler_routing or runtime_dispatch to the contract")
            print()
        print(
            f"ERROR: {len(report.failures)} command topic(s) lack dispatcher wiring. "
            "Messages on these topics will go to DLQ at runtime.",
            flush=True,
        )
        return 1

    if changed_contract_paths is not None:
        print(
            "\nAll changed command-topic subscriptions have dispatcher routes. PASS.",
            flush=True,
        )
    else:
        print(
            "\nAll command-topic subscriptions have dispatcher routes. PASS.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
