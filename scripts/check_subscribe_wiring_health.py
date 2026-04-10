#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Subscribe-topic wiring health check.

Static analysis that verifies every contract-declared subscribe_topic has
at least one matching publish_topic from another contract. Detects "dead
letter" subscriptions where a node declares it consumes from a topic but
no node in the system publishes to it.

Also checks the reverse: every publish_topic should have at least one
subscriber (warning only, not blocking).

This catches the exact class of bug where:
- A contract.yaml declares subscribe_topics
- But no consumer runtime wiring exists (no publisher feeds the topic)
- Messages are silently lost or the subscription is purely aspirational

Uses the existing ContractTopicExtractor for YAML parsing.

Usage::

    uv run python scripts/check_subscribe_wiring_health.py
    uv run python scripts/check_subscribe_wiring_health.py --verbose
    uv run python scripts/check_subscribe_wiring_health.py --contracts-dir src/omnibase_infra/nodes
    uv run python scripts/check_subscribe_wiring_health.py --extra-contracts-dir ../omniclaude/src/omniclaude/nodes

Exit codes:
    0 = all subscribe topics have at least one publisher (or are allowlisted)
    1 = one or more dead-letter subscribe topics found

[OMN-7385]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as a standalone script
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from omnibase_infra.topics.contract_topic_extractor import ContractTopicExtractor

# ---------------------------------------------------------------------------
# Allowlist: subscribe topics that are intentionally consumed from external
# sources (webhooks, CLI triggers, cross-repo publishers not in this scan).
# Format: "topic": "reason | owner | expiry"
# ---------------------------------------------------------------------------

_EXTERNAL_PUBLISHER_ALLOWLIST: dict[str, str] = {
    # omniclaude publishes these via hook scripts, not contract.yaml
    "onex.evt.omniclaude.session-started.v1": "Published by omniclaude hooks, not contract-declared | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omniclaude.session-ended.v1": "Published by omniclaude SessionEnd hook, not contract-declared | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omniclaude.prompt-submitted.v1": "Published by omniclaude hooks, not contract-declared | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omniclaude.tool-executed.v1": "Published by omniclaude hooks, not contract-declared | owner: jonah | expiry: 2026-12-01",
    "onex.cmd.omniintelligence.claude-hook-event.v1": "Published by omniclaude hooks | owner: jonah | expiry: 2026-12-01",
    "onex.cmd.omniintelligence.tool-content.v1": "Published by omniclaude hooks | owner: jonah | expiry: 2026-12-01",
    # GitHub webhooks are external triggers
    "onex.evt.github.pr-webhook.v1": "Published by GitHub webhook relay, not a node | owner: jonah | expiry: 2026-12-01",
    "onex.evt.github.push-webhook.v1": "Published by GitHub webhook relay | owner: jonah | expiry: 2026-12-01",
}

# ---------------------------------------------------------------------------
# Baseline allowlist: pre-existing dead-letter subscriptions (OMN-7385).
# These topics have subscribe_topics declared in contracts but no matching
# publish_topics in any contract. Each entry represents a known gap.
# New entries are tech debt. Removing entries (by adding publisher contracts)
# is the goal.
#
# Format: "topic": "reason | owner | expiry"
# Current baseline: 2026-04-02 (44 entries)
# Target: 0 entries
# ---------------------------------------------------------------------------
# fmt: off
_BASELINE_DEAD_LETTER_ALLOWLIST: dict[str, str] = {
    # Build loop cmd topics — triggered by CLI (claude -p), not Kafka publisher
    "onex.cmd.omnibase-infra.build-loop-start.v1": "Triggered by cron-buildloop.sh via claude -p, not Kafka | owner: jonah | expiry: 2026-12-01",
    # Chain learning — publisher nodes not yet implemented
    "onex.cmd.omnibase-infra.chain-learn.v1": "Chain learning publisher not yet wired | owner: jonah | expiry: 2026-09-01",
    # Delegation — request comes from omniclaude hooks, not contract-declared
    "onex.cmd.omnibase-infra.delegation-request.v1": "Delegation request from omniclaude hooks | owner: jonah | expiry: 2026-09-01",
    # LLM infrastructure — requests come from orchestrators via intents, not Kafka publish
    "onex.cmd.omnibase-infra.llm-completion-request.v1": "LLM request via intent routing, not direct publish | owner: jonah | expiry: 2026-09-01",
    "onex.cmd.omnibase-infra.llm-embedding-request.v1": "LLM embedding request via intent routing | owner: jonah | expiry: 2026-09-01",
    "onex.cmd.omnibase-infra.llm-inference-request.v1": "LLM request via intent routing | owner: jonah | expiry: 2026-09-01",
    "onex.cmd.omnibase-infra.vector-store-request.v1": "Vector store request via intent routing | owner: jonah | expiry: 2026-09-01",
    # Artifact reconciliation — triggered externally
    "onex.cmd.artifact.reconcile.v1": "Triggered by CI/webhook, not Kafka publisher | owner: jonah | expiry: 2026-09-01",
    # Contract resolution — triggered by runtime, not Kafka publisher
    "onex.cmd.platform.contract-resolve-requested.v1": "Contract resolution triggered by runtime | owner: jonah | expiry: 2026-09-01",
    # Intent storage queries — internal runtime queries, not event-sourced
    "onex.cmd.platform.intent-query-distribution.v1": "Internal runtime query pattern | owner: jonah | expiry: 2026-09-01",
    "onex.cmd.platform.intent-query-session.v1": "Internal runtime query pattern | owner: jonah | expiry: 2026-09-01",
    # Ledger operations — internal runtime
    "onex.cmd.platform.ledger-append.v1": "Internal runtime ledger operation | owner: jonah | expiry: 2026-09-01",
    "onex.cmd.platform.ledger-query.v1": "Internal runtime ledger query | owner: jonah | expiry: 2026-09-01",
    # Router — request comes from omniclaude hooks
    "onex.cmd.router.route-request.v1": "Route request from omniclaude hooks | owner: jonah | expiry: 2026-09-01",
    # RSD scoring — triggered externally
    "onex.cmd.rsd.score.v1": "RSD scoring triggered externally | owner: jonah | expiry: 2026-09-01",
    # Skill commands — triggered by Claude skill invocations, not Kafka
    "onex.cmd.skill.merge-sweep.v1": "Triggered by /merge-sweep skill, not Kafka | owner: jonah | expiry: 2026-09-01",
    "onex.cmd.skill.scope-check.v1": "Triggered by /scope-check skill | owner: jonah | expiry: 2026-09-01",
    # Build loop events — classify and fill phases not yet publishing
    "onex.evt.omnibase-infra.build-loop-classify-completed.v1": "Classify phase effect node not yet implemented | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.build-loop-fill-completed.v1": "Fill phase effect node not yet implemented | owner: jonah | expiry: 2026-09-01",
    # Chain events — replay/verify effect nodes pending
    "onex.evt.omnibase-infra.chain-replay-result.v1": "Chain replay effect pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.chain-verified.v1": "Chain verify effect pending | owner: jonah | expiry: 2026-09-01",
    # Infrastructure monitoring — published by runtime internals
    "onex.evt.omnibase-infra.consumer-health.v1": "Published by runtime health monitor, not contract | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.db-error.v1": "Published by DB error handler, not contract | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.quality-gate-result.v1": "Published by CI pipeline, not contract | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.routing-decision.v1": "Published by routing runtime, not contract | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.runtime-error.v1": "Published by runtime error handler | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.service-lifecycle.v1": "Published by service lifecycle manager | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.system-alert.v1": "Published by alert system | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.tool-update.v1": "Published by tool updater | owner: jonah | expiry: 2026-09-01",
    # Context audit DLQ — published by omniclaude hooks
    "onex.evt.omniclaude.context-audit-dlq.v1": "Published by omniclaude context audit | owner: jonah | expiry: 2026-09-01",
    # Contract lifecycle — published by contract management runtime, not contract-declared
    "onex.evt.platform.contract-deregistered.v1": "Published by contract management runtime | owner: jonah | expiry: 2026-09-01",
    "onex.evt.platform.contract-registered.v1": "Published by contract management runtime | owner: jonah | expiry: 2026-09-01",
    # Intent classification — published by omniintelligence, not in this scan
    "onex.evt.platform.intent-classified.v1": "Published by omniintelligence, cross-repo | owner: jonah | expiry: 2026-09-01",
    # Merge gate — decision published by CI integration, not contract
    "onex.evt.platform.merge-gate-decision.v1": "Published by CI merge gate integration | owner: jonah | expiry: 2026-09-01",
    # Router events — published by routing runtime
    "onex.evt.router.health-snapshot.v1": "Published by routing runtime | owner: jonah | expiry: 2026-09-01",
    "onex.evt.router.routing-outcome.v1": "Published by routing runtime | owner: jonah | expiry: 2026-09-01",
    "onex.evt.router.scoring-decision.v1": "Published by routing runtime | owner: jonah | expiry: 2026-09-01",
    # RSD events — effect nodes pending
    "onex.evt.rsd.data-fetched.v1": "RSD data fetch effect pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.rsd.scores-calculated.v1": "RSD scores compute pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.rsd.scores-stored.v1": "RSD scores store effect pending | owner: jonah | expiry: 2026-09-01",
    # Runtime tick — published by runtime scheduler, not contract
    "onex.evt.runtime.tick.v1": "Published by runtime scheduler | owner: jonah | expiry: 2026-09-01",
    # Merge sweep workflow events — effect nodes pending
    "onex.evt.skill.merge-sweep-auto-merged.v1": "Merge sweep effect pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.skill.merge-sweep-classified.v1": "Merge sweep classify effect pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.skill.merge-sweep-pr-list.v1": "Merge sweep PR list effect pending | owner: jonah | expiry: 2026-09-01",
    # Scope workflow events — effect nodes pending
    "onex.evt.skill.scope-extracted.v1": "Scope extract effect pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.skill.scope-file-read.v1": "Scope file read effect pending | owner: jonah | expiry: 2026-09-01",
    "onex.evt.skill.scope-manifest-written.v1": "Scope manifest write effect pending | owner: jonah | expiry: 2026-09-01",
    # Gmail archive cleanup — runtime tick published by scheduler
    "onex.int.platform.runtime-tick.v1": "Published by runtime scheduler | owner: jonah | expiry: 2026-09-01",
    # Onboarding — triggered by omniclaude /onboarding skill, not Kafka publisher
    "onex.cmd.omnibase-infra.onboarding-start.v1": "Triggered by /onboarding skill via claude -p, not Kafka | owner: jonah | expiry: 2026-12-01",
}
# fmt: on

# DLQ and broadcast topics are infrastructure-scoped, skip them
_INFRASTRUCTURE_PREFIXES = (".dlq.", ".broadcast.")


def _is_infrastructure_topic(topic: str) -> bool:
    """Check if a topic is infrastructure-scoped (DLQ, broadcast)."""
    return any(prefix in topic for prefix in _INFRASTRUCTURE_PREFIXES)


def check_wiring_health(
    contracts_dirs: list[Path],
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Check subscribe/publish topic wiring across all contracts.

    Args:
        contracts_dirs: Directories to scan for contract.yaml files.
        verbose: Print detailed output.

    Returns:
        Tuple of (errors, warnings).
        errors: Dead-letter subscribe topics (no publisher exists).
        warnings: Orphan publish topics (no subscriber exists).
    """
    extractor = ContractTopicExtractor()

    # Collect all topics across all directories
    all_subscribe: dict[str, list[str]] = defaultdict(list)  # topic -> [node_names]
    all_publish: dict[str, list[str]] = defaultdict(list)  # topic -> [node_names]

    for contracts_dir in contracts_dirs:
        if not contracts_dir.exists():
            if verbose:
                print(f"SKIP: Directory not found: {contracts_dir}")
            continue

        manifest = extractor.scan(contracts_dir)

        for node_name, node_topics in manifest.nodes.items():
            for topic in node_topics.subscribe_topics:
                if not _is_infrastructure_topic(topic):
                    all_subscribe[topic].append(node_name)
            for topic in node_topics.publish_topics:
                if not _is_infrastructure_topic(topic):
                    all_publish[topic].append(node_name)

    if verbose:
        print(f"Scanned: {sum(1 for d in contracts_dirs if d.exists())} directories")
        print(f"Subscribe topics: {len(all_subscribe)}")
        print(f"Publish topics: {len(all_publish)}")
        print()

    errors: list[str] = []
    warnings: list[str] = []

    # Check: every subscribe topic should have a publisher
    for topic, subscribers in sorted(all_subscribe.items()):
        if topic in _EXTERNAL_PUBLISHER_ALLOWLIST:
            if verbose:
                print(
                    f"  ALLOWLISTED (external): {topic} (subscribed by {', '.join(subscribers)})"
                )
            continue

        if topic in _BASELINE_DEAD_LETTER_ALLOWLIST:
            if verbose:
                print(
                    f"  ALLOWLISTED (baseline): {topic} (subscribed by {', '.join(subscribers)})"
                )
            continue

        if topic not in all_publish:
            errors.append(
                f"DEAD_LETTER: {topic} subscribed by [{', '.join(subscribers)}] "
                f"but no contract publishes to it"
            )
        elif verbose:
            publishers = all_publish[topic]
            print(
                f"  OK: {topic} "
                f"(pub: {', '.join(publishers)} -> sub: {', '.join(subscribers)})"
            )

    # Check: every publish topic should have a subscriber (warning only)
    for topic, publishers in sorted(all_publish.items()):
        if _is_infrastructure_topic(topic):
            continue
        if topic not in all_subscribe and topic not in _EXTERNAL_PUBLISHER_ALLOWLIST:
            warnings.append(
                f"NO_SUBSCRIBER: {topic} published by [{', '.join(publishers)}] "
                f"but no contract subscribes to it"
            )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check subscribe-topic wiring health across contracts"
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=_REPO_ROOT / "src" / "omnibase_infra" / "nodes",
        help="Primary contracts directory to scan",
    )
    parser.add_argument(
        "--extra-contracts-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional contract directories (e.g., cross-repo nodes)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed wiring status",
    )
    args = parser.parse_args()

    dirs = [args.contracts_dir] + args.extra_contracts_dir
    errors, warnings = check_wiring_health(dirs, verbose=args.verbose)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)} orphan publish topics):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n{'=' * 60}")
        print(f"WIRING HEALTH: FAIL ({len(errors)} dead-letter subscriptions)")
        print(f"{'=' * 60}")
        for e in errors:
            print(f"  - {e}")
        print("\nEach dead-letter topic means a contract declares a subscription")
        print("but no contract in the system publishes to that topic.")
        print("Fix: add the topic to a publisher's publish_topics, or add to allowlist")
        print("if the publisher is external (webhook, CLI, cross-repo).")
        return 1

    print("WIRING HEALTH: PASS (no dead-letter subscriptions found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
