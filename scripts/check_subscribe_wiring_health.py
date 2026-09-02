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
import re
import sys
from collections import defaultdict
from datetime import UTC, date, datetime
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
    "onex.evt.omniclaude.phase-metrics.v1": "Published by omniclaude emit-daemon hooks, not contract-declared | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omniclaude.notification-blocked.v1": "Published by omniclaude emit-daemon hooks, not contract-declared | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omniclaude.notification-completed.v1": "Published by omniclaude emit-daemon hooks, not contract-declared | owner: jonah | expiry: 2026-12-01",
    # GitHub webhooks are external triggers
    "onex.evt.github.pr-webhook.v1": "Published by GitHub webhook relay, not a node | owner: jonah | expiry: 2026-12-01",
    # Runner usage events are produced by self-hosted runner telemetry outside
    # contract-declared node publishers.
    "onex.evt.omninode.runner-usage-recorded.v1": "Published by runner telemetry outside node contracts | owner: jonah | expiry: 2026-12-01",
    # omnimarket build-loop orchestrator publishes the workflow terminal event;
    # node_build_loop_projection_compute (this repo) consumes it. Cross-repo
    # publisher lives in omnimarket and is not visible to this scan.
    "onex.evt.omnimarket.build-loop-orchestrator-completed.v1": "Published by omnimarket node_build_loop_orchestrator (cross-repo) | owner: jonah | expiry: 2026-12-01",
    # Pattern B dispatch commands enter through local runtime transport / skill clients;
    # RuntimePatternBBroker consumes them but no contract-declared node publishes them.
    "onex.cmd.omnibase-infra.pattern-b-dispatch.v1": "Published by local runtime transport / runtime-backed skill clients | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omnibase-infra.runtime-manifest-published.v1": "Published by runtime startup self-report, not a contract-declared node | owner: jonah | expiry: 2026-12-01",
    # Baselines batch compute — triggered by scripts/run_baselines_batch_compute.py CLI publisher,
    # not a contract-declared node publisher. The script publishes to this topic to trigger the
    # node_baselines_batch_compute effect node. (OMN-11177)
    "onex.cmd.omnibase-infra.baselines-batch-compute.v1": "Published by scripts/run_baselines_batch_compute.py CLI trigger, not a contract-declared node | owner: jonah | expiry: 2026-12-01",
    # Savings correlation batch compute — same self-only-command-topic shape as
    # baselines-batch-compute above, but triggered directly by an in-process
    # asyncio loop in service_kernel.py (HandlerSavingsCorrelation.
    # run_correlation_batch), never published to over Kafka at all. Declared
    # purely so operation_match auto-wiring resolves a real handler for the
    # node's savings.correlation_batch_compute capability. (OMN-16293)
    "onex.cmd.omnibase-infra.savings-correlation-batch-compute.v1": "Triggered directly by an in-process periodic asyncio loop in service_kernel.py (HandlerSavingsCorrelation), never published over Kafka | owner: jonah | expiry: 2026-12-01",
    # Gateway attach control-plane command topics (OMN-15750) — published by the
    # edge-side dialer (customer-premise / .201 test-lane connector,
    # docker/docker-compose.gateway-attach-test-lane.yml +
    # scripts/proof/gateway_attach_e2e_proof.py), never by another
    # contract-declared node. Same external-CLI-publisher shape as the
    # baselines-batch-compute entry above.
    "onex.cmd.omnibase-infra.gateway-attach-request.v1": "Published by the edge-side attach dialer (customer-premise/.201 test-lane connector), not a contract-declared node | owner: jonah | expiry: 2026-12-01",
    "onex.cmd.omnibase-infra.gateway-heartbeat-request.v1": "Published by the edge-side attach dialer (customer-premise/.201 test-lane connector), not a contract-declared node | owner: jonah | expiry: 2026-12-01",
    "onex.cmd.omnibase-infra.gateway-detach-request.v1": "Published by the edge-side attach dialer (customer-premise/.201 test-lane connector), not a contract-declared node | owner: jonah | expiry: 2026-12-01",
    # Runner-fleet-maintain tick — triggered by the reused OMN-13915
    # runner-fleet-canary 15-min GitHub-hosted schedule (OMN-13942 Increment 1),
    # not a contract-declared Kafka publisher.
    "onex.cmd.omnibase-infra.runner-fleet-maintain-start.v1": "Triggered by the reused OMN-13915 runner-fleet-canary GHA schedule, not Kafka | owner: jonah | expiry: 2026-12-01",
    # OMN-15006: node_ledger_projection_compute widened subscribe_topics to
    # OCC governance + omnimarket-redeploy topics whose canonical publisher
    # contracts live in onex_change_control / omnimarket (cross-repo, not
    # visible to this scan). Topic strings are onex_change_control's own
    # GovernanceTopic registry (OMN-8635).
    "onex.evt.occ.nightly-promotion.v1": "Published by onex_change_control nightly dev-to-main promotion pipeline (GovernanceTopic.NIGHTLY_PROMOTION), cross-repo | owner: jonah | expiry: 2026-12-01",
    "onex.evt.onex-change-control.governance-check-completed.v1": "Published by onex_change_control governance check pipeline (GovernanceTopic.GOVERNANCE_CHECK_COMPLETED), cross-repo | owner: jonah | expiry: 2026-12-01",
    "onex.evt.onex-change-control.contract-drift-detected.v1": "Published by onex_change_control drift detection (GovernanceTopic.CONTRACT_DRIFT_DETECTED), cross-repo | owner: jonah | expiry: 2026-12-01",
    "onex.evt.onex-change-control.cosmetic-compliance-scored.v1": "Published by onex_change_control cosmetic lint tooling (GovernanceTopic.COSMETIC_COMPLIANCE_SCORED), cross-repo | owner: jonah | expiry: 2026-12-01",
    "onex.cmd.omnimarket.redeploy-start.v1": "Published by onex_change_control promotion tooling, consumed by omnimarket node_redeploy (GovernanceTopic.RUNTIME_DEPLOYMENT_REQUEST, OMN-12576), cross-repo | owner: jonah | expiry: 2026-12-01",
    "onex.evt.omnimarket.runtime-deployment-proof.v1": "Published by omnimarket node_redeploy per-lane probe (GovernanceTopic.RUNTIME_DEPLOYMENT_PROOF, OMN-12576), cross-repo | owner: jonah | expiry: 2026-12-01",
    # OMN-15168 (epic OMN-15154): node_ledger_projection_compute widened
    # subscribe_topics to steel_onslaught's forwarded terminal-match topic
    # whose canonical publisher lives in the private steel_onslaught repo
    # (cross-repo, not visible to this scan; infra ↛ steel forbids importing
    # it). Topic string is steel_onslaught's own STEEL_MATCH_TERMINAL_TOPIC
    # (kafka_forwarder.py, OMN-15167, merged f378cd48).
    "onex.evt.steel-onslaught.match-terminal.v1": "Published by steel_onslaught's KafkaTerminalEventForwarder (STEEL_MATCH_TERMINAL_TOPIC, OMN-15167), cross-repo (private personal repo) | owner: jonah | expiry: 2026-12-01",
    # OMN-16265: node_fault_inject_fixture_compute's command topic is
    # deliberately published only by an external fault-injection caller
    # (a manual run per knowledge-base:runbooks/fault-inject-fixture-dlq-offset-withholding.md,
    # or a future automated boundary-regression script), never by another
    # contract-declared node — same external-CLI-trigger shape as the
    # baselines-batch-compute entry above.
    "onex.cmd.omnibase-infra.fault-inject-fixture.v1": "Published by an external fault-injection caller (manual run or future regression script per the fixture's runbook), not a contract-declared node | owner: jonah | expiry: 2026-12-01",
}

# ---------------------------------------------------------------------------
# Baseline allowlist: pre-existing dead-letter subscriptions (OMN-7385).
# These topics have subscribe_topics declared in contracts but no matching
# publish_topics in any contract. Each entry represents a known gap.
# New entries are tech debt. Removing entries (by adding publisher contracts)
# is the goal.
#
# Format: "topic": "reason | owner | expiry"
# Current baseline: 2026-04-10 (45 entries)
# Target: 0 entries
# ---------------------------------------------------------------------------
# fmt: off
_BASELINE_DEAD_LETTER_ALLOWLIST: dict[str, str] = {
    # Coding-agent workflow external entrypoint — published by workflow clients
    # (the coding-agent CLI thin-publisher), not a contract-declared node (OMN-13247).
    "onex.cmd.omnibase-infra.coding-agent-invoke.v1": "Published by coding-agent workflow clients as the external entrypoint | owner: jonah | expiry: 2026-12-01",
    # Build loop cmd topics — triggered by CLI (claude -p), not Kafka publisher
    "onex.cmd.omnibase-infra.build-loop-append.v1": "Routed via intent from node_build_loop_projection_compute, not Kafka publish | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    "onex.cmd.omnibase-infra.pr-state-upsert.v1": "Routed via intent from node_pr_state_projection_compute, not Kafka publish | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1": "Routed via intent from node_gateway_link_health_projection_compute, not Kafka publish (OMN-15570) | owner: jonah | expiry: 2026-12-01",
    # Gateway heartbeat — published imperatively by ServiceGatewayForwarder.publish_heartbeat
    # (node_bus_forwarder_effect/services/service_gateway_forwarder.py), not via a
    # contract-declared event_bus.publish_topics entry -- node_bus_forwarder_effect's
    # own contract only declares mirror_topics (this checker reads publish_topics
    # only), which the 2026-08-08 gateway lift architecture assessment already flags
    # as a gap in that node's own contract, not something OMN-15570 introduces.
    "onex.evt.omnibase-infra.gateway-heartbeat.v1": "Published imperatively by ServiceGatewayForwarder.publish_heartbeat, not via contract publish_topics (OMN-15570 adds the first contract-declared consumer) | owner: jonah | expiry: 2026-12-01",
    "onex.cmd.omnibase-infra.validation-ledger-append.v1": "Routed via intent from node_validation_ledger_projection_compute, not Kafka publish | owner: jonah | expiry: 2026-12-01",
    # Chain learning — publisher nodes not yet implemented
    "onex.cmd.omnibase-infra.chain-learn.v1": "Chain learning publisher not yet wired | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Topic migration — command issued by operator/runtime, not a contract-declared publisher (OMN-12623)
    "onex.cmd.omnibase-infra.topic-migration-execute.v1": "Migration command issued by operator/runtime (node_topic_migration_executor_effect), not Kafka publisher | owner: jonah | expiry: 2026-12-01",
    # Delegation — request comes from omniclaude hooks, not contract-declared
    # LLM infrastructure — requests come from orchestrators via intents, not Kafka publish
    "onex.cmd.omnibase-infra.llm-completion-request.v1": "LLM request via intent routing, not direct publish | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.cmd.omnibase-infra.llm-embedding-request.v1": "LLM embedding request via intent routing | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.cmd.omnibase-infra.llm-inference-request.v1": "LLM request via intent routing | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.cmd.omnibase-infra.vector-store-request.v1": "Vector store request via intent routing | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Artifact reconciliation — triggered externally
    "onex.cmd.artifact.reconcile.v1": "Triggered by CI/webhook, not Kafka publisher | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    # Contract resolution — triggered by runtime, not Kafka publisher
    "onex.cmd.platform.contract-resolve-requested.v1": "Contract resolution triggered by runtime | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Intent storage queries — internal runtime queries, not event-sourced
    # Ledger operations — internal runtime
    "onex.cmd.platform.ledger-append.v1": "Internal runtime ledger operation | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    "onex.cmd.platform.ledger-query.v1": "Internal runtime ledger query | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    # Router — request comes from omniclaude hooks
    "onex.cmd.router.route-request.v1": "Route request from omniclaude hooks | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    # RSD scoring — triggered externally
    "onex.cmd.rsd.score.v1": "RSD scoring triggered externally | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    # Skill commands — triggered by Claude skill invocations, not Kafka
    "onex.cmd.skill.merge-sweep.v1": "Triggered by /merge-sweep skill, not Kafka | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    "onex.cmd.skill.scope-check.v1": "Triggered by /scope-check skill | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    # Build loop events — classify and fill phases not yet publishing
    # Chain events — replay/verify effect nodes pending
    "onex.evt.omnibase-infra.chain-replay-result.v1": "Chain replay effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.omnibase-infra.chain-verified.v1": "Chain verify effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Infrastructure monitoring — published by runtime internals
    "onex.evt.omnibase-infra.consumer-health.v1": "Published by runtime health monitor, not contract | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    "onex.evt.omnibase-infra.db-error.v1": "Published by DB error handler, not contract | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    "onex.evt.omnibase-infra.runtime-error.v1": "Published by runtime error handler | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    "onex.evt.omnibase-infra.service-lifecycle.v1": "Published by service lifecycle manager | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.omnibase-infra.system-alert.v1": "Published by alert system | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.omnibase-infra.tool-update.v1": "Published by tool updater | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Context audit DLQ — published by omniclaude hooks
    "onex.evt.omniclaude.context-audit-dlq.v1": "Published by omniclaude context audit | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    # Contract lifecycle — published by contract management runtime, not contract-declared
    "onex.evt.platform.contract-deregistered.v1": "Published by contract management runtime | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    "onex.evt.platform.contract-registered.v1": "Published by contract management runtime | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: stated in-repo publisher confirmed by non-enum source reference]",
    # Intent classification — published by omniintelligence, not in this scan
    # Merge gate — decision published by CI integration, not contract
    "onex.evt.platform.merge-gate-decision.v1": "Published by CI merge gate integration | owner: jonah | expiry: 2026-12-01 [OMN-16795 re-verified 2026-08-27: publisher is outside omnibase_infra, so this claim is NOT falsifiable from this repo; exemption kept on that basis]",
    # Router events — published by routing runtime
    "onex.evt.router.health-snapshot.v1": "Published by routing runtime | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.router.routing-outcome.v1": "Published by routing runtime | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.router.scoring-decision.v1": "Published by routing runtime | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # RSD events — effect nodes pending
    "onex.evt.rsd.data-fetched.v1": "RSD data fetch effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.rsd.scores-calculated.v1": "RSD scores compute pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.rsd.scores-stored.v1": "RSD scores store effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Runtime tick — published by runtime scheduler, not contract
    # Merge sweep workflow events — effect nodes pending
    "onex.evt.skill.merge-sweep-auto-merged.v1": "Merge sweep effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.skill.merge-sweep-classified.v1": "Merge sweep classify effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.skill.merge-sweep-pr-list.v1": "Merge sweep PR list effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Scope workflow events — effect nodes pending
    "onex.evt.skill.scope-extracted.v1": "Scope extract effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.skill.scope-file-read.v1": "Scope file read effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    "onex.evt.skill.scope-manifest-written.v1": "Scope manifest write effect pending | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
    # Gmail archive cleanup — runtime tick published by scheduler
    # Onboarding — triggered by omniclaude /onboarding skill, not Kafka publisher
    "onex.cmd.omnibase-infra.onboarding-start.v1": "Triggered by /onboarding skill via claude -p, not Kafka | owner: jonah | expiry: 2026-12-01",
    # Remote-agent invoke — emitted by node_delegation_orchestrator (OMN-9620 epic);
    # its contract addition is tracked separately and pending in another wave.
    "onex.cmd.omnibase-infra.remote-agent-invoke.v1": "Publisher pending in delegation_orchestrator (OMN-9620 epic) | owner: jonah | expiry: 2026-10-01 [OMN-16795 2026-08-27: NOT verified — the topic appears only in its own topic enum, so the stated publisher could not be confirmed in this repo. Short leash: prove the publisher or delete the subscribe declaration]",
}
# fmt: on

# DLQ and broadcast topics are infrastructure-scoped, skip them
_INFRASTRUCTURE_PREFIXES = (".dlq.", ".broadcast.")


def _is_infrastructure_topic(topic: str) -> bool:
    """Check if a topic is infrastructure-scoped (DLQ, broadcast)."""
    return any(prefix in topic for prefix in _INFRASTRUCTURE_PREFIXES)


# ---------------------------------------------------------------------------
# Allowlist hygiene (OMN-16795)
#
# Until now the ``expiry:`` in every allowlist reason was decoration: nothing
# read it. That is what let a 45-entry baseline accumulate with dates already
# lapsing — an exemption list nobody can be forced off is not a baseline, it is
# a permanent amnesty with a date-shaped comment attached.
#
# Three failure modes, all previously silent:
#   EXPIRED   the owner's own stated deadline has passed (inclusive: on the
#             stated day the exemption is over).
#   MALFORMED no parseable ``expiry:``, so the entry could never expire.
#   STALE     no contract subscribes to the topic any more, so the entry is
#             dead weight inflating the list and hiding its real size.
#
# STALE matters as much as EXPIRED: an allowlist that keeps entries for topics
# nobody consumes reports a debt number that is mostly fiction, and the fiction
# is what makes the real entries easy to ignore.
# ---------------------------------------------------------------------------

_EXPIRY_RE = re.compile(r"expiry:\s*(\d{4})-(\d{2})-(\d{2})")
_OWNER_RE = re.compile(r"owner:\s*([^|]+)")


def _parse_allowlist_expiry(reason: str) -> date | None:
    """Extract the ``expiry: YYYY-MM-DD`` date from an allowlist reason.

    Returns ``None`` when the field is absent or not a real calendar date —
    both of which the hygiene check treats as MALFORMED rather than as
    "no expiry, therefore fine".
    """
    match = _EXPIRY_RE.search(reason)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _parse_allowlist_owner(reason: str) -> str:
    match = _OWNER_RE.search(reason)
    return match.group(1).strip() if match else "unowned"


def collect_subscribed_topics(contracts_dirs: list[Path]) -> set[str]:
    """Every non-infrastructure topic some contract in these dirs subscribes to."""
    extractor = ContractTopicExtractor()
    subscribed: set[str] = set()
    for contracts_dir in contracts_dirs:
        if not contracts_dir.exists():
            continue
        manifest = extractor.scan(contracts_dir)
        for node_topics in manifest.nodes.values():
            for topic in node_topics.subscribe_topics:
                if not _is_infrastructure_topic(topic):
                    subscribed.add(topic)
    return subscribed


def check_allowlist_hygiene(
    allowlists: dict[str, str],
    subscribed_topics: set[str],
    today: date,
) -> list[str]:
    """Return one error per expired, malformed, or stale allowlist entry.

    Args:
        allowlists: merged ``topic -> "reason | owner: X | expiry: YYYY-MM-DD"``.
        subscribed_topics: topics some contract actually subscribes to.
        today: the clock, injected so the enforcement itself is testable.
    """
    errors: list[str] = []
    for topic, reason in sorted(allowlists.items()):
        owner = _parse_allowlist_owner(reason)
        expiry = _parse_allowlist_expiry(reason)

        if topic not in subscribed_topics:
            errors.append(
                f"STALE: {topic} is allowlisted but NO contract subscribes to it "
                f"(owner: {owner}). The exemption is dead weight — delete the entry."
            )
            continue

        if expiry is None:
            errors.append(
                f"MALFORMED: {topic} has no parseable 'expiry: YYYY-MM-DD' "
                f"(owner: {owner}). An entry that cannot expire is a permanent "
                f"amnesty; give it a real date."
            )
            continue

        if expiry <= today:
            errors.append(
                f"EXPIRED: {topic} exemption lapsed on {expiry.isoformat()} "
                f"(owner: {owner}). Either fix the gap (add a contract publisher) "
                f"or renew with a FRESH reason and expiry that says what changed."
            )
    return errors


def check_wiring_health(
    contracts_dirs: list[Path],
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Check subscribe/publish topic wiring across all contracts.

    Allowlist hygiene (OMN-16795) is deliberately NOT checked here — see
    :func:`check_allowlist_hygiene`, which ``main()`` runs against the real
    contract tree. Folding it in here would make every caller that scans a
    partial or synthetic directory report the entire allowlist as stale.

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

    # OMN-16795: the allowlists themselves must stay honest. Run this only here,
    # against the REAL contract tree — a caller scanning a partial/synthetic
    # directory has no business being told the whole allowlist is stale.
    errors.extend(
        check_allowlist_hygiene(
            allowlists={
                **_EXTERNAL_PUBLISHER_ALLOWLIST,
                **_BASELINE_DEAD_LETTER_ALLOWLIST,
            },
            subscribed_topics=collect_subscribed_topics(dirs),
            today=datetime.now(UTC).date(),
        )
    )

    if warnings:
        print(f"\nWARNINGS ({len(warnings)} orphan publish topics):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n{'=' * 60}")
        print(f"WIRING HEALTH: FAIL ({len(errors)} problems)")
        print(f"{'=' * 60}")
        for e in errors:
            print(f"  - {e}")
        print("\nDEAD_LETTER: a contract declares a subscription but no contract in")
        print("the system publishes to that topic. Fix: add the topic to a")
        print("publisher's publish_topics, or allowlist it if the publisher is")
        print("external (webhook, CLI, cross-repo).")
        print("\nEXPIRED / MALFORMED / STALE: an allowlist entry is no longer honest")
        print("(OMN-16795). Fix the underlying gap, renew with a FRESH reason and")
        print("expiry saying what changed, or delete the entry. Do NOT bulk-extend:")
        print("a date nobody re-verified is what made this list unbounded.")
        return 1

    print("WIRING HEALTH: PASS (no dead-letter subscriptions, allowlists clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
