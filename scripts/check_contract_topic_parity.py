#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI parity gate: detect Python-only topic registry entries not covered by contracts (OMN-4600).

Topics in ALL_PROVISIONED_SUFFIXES that appear in NO contract.yaml AND are NOT in
the legacy allowlist below will cause this script to exit 1 with an actionable diff.

This creates migration pressure: every NEW topic must be added to a contract.yaml,
not directly to platform_topic_suffixes.py.

LEGACY ALLOWLIST GOVERNANCE
---------------------------
Each entry MUST include: reason | owner | expiry

Format: "suffix": "reason | owner: <who> | expiry: <date or sprint>"

Entries without complete comments will be rejected in PR review.
New entries are tech debt. Removing entries (via contract.yaml coverage) is the goal.

Current baseline (OMN-4600, 2026-03-11): 252 pre-migration entries.
Target: 0 entries (all topics declared in contract.yaml or topics.yaml).

Usage::

    # Run as CI check (exit 0 = pass, exit 1 = fail)
    uv run python scripts/check_contract_topic_parity.py

    # Verbose: show all contract-covered topics too
    uv run python scripts/check_contract_topic_parity.py --verbose

    # Show allowlist entries separately
    uv run python scripts/check_contract_topic_parity.py --show-allowlist

Exit codes:
    0 -- All Python-only topics are covered by contracts or allowlisted
    1 -- One or more Python-only topics are neither contract-covered nor allowlisted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Legacy allowlist — pre-migration entries from OMN-4600 baseline (2026-03-11).
# Format: "suffix": "reason | owner: <who> | expiry: <date>"
# ---------------------------------------------------------------------------
# fmt: off
_LEGACY_ALLOWLIST: dict[str, str] = {
    # --- omniclaude skill cmd topics (migrating via topics.yaml, OMN-4592/OMN-4594) ---
    "onex.cmd.omniclaude.action-logging.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.agent-observability.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.auto-merge.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.brainstorming.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.checkpoint.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ci-failures.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ci-fix-pipeline.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ci-watch.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.condition-based-waiting.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.crash-recovery.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.create-followup-tickets.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.create-ticket.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.decompose-epic.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.deep-dive.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.defense-in-depth.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.deploy-local-plugin.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.dispatching-parallel-agents.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.epic-team.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.executing-plans.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.finishing-a-development-branch.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.fix-prs.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.gap-analysis.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.gap-fix.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.generate-node.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.linear-insights.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.linear-ticket-management.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.local-review.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.log-execution.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.merge-sweep.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.onex-status.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.parallel-solve.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pipeline-audit.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pipeline-metrics.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.plan-ticket.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.plan-to-tickets.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pr-polish.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pr-queue-pipeline.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pr-release-ready.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pr-review-comprehensive.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pr-review-dev.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.pr-watch.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.project-status.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.receiving-code-review.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.release.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.requesting-code-review.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.review-all-prs.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.review-cycle.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.root-cause-tracing.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.rrh.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.setup-statusline.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.sharing-skills.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.slack-gate.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.subagent-driven-development.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.suggest-work.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.systematic-debugging.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.test-driven-development.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.testing-anti-patterns.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.testing-skills-with-subagents.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ticket-pipeline.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ticket-plan.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ticket-work.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.ultimate-validate.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.using-git-worktrees.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.using-superpowers.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.velocity-estimate.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.verification-before-completion.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.writing-plans.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniclaude.writing-skills.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    # --- omniclaude evt skill topics (completed/failed pairs, same migration path) ---
    "onex.evt.omniclaude.action-logging-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.action-logging-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.agent-observability-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.agent-observability-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.auto-merge-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.auto-merge-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.brainstorming-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.brainstorming-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.checkpoint-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.checkpoint-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ci-failures-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ci-failures-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ci-fix-pipeline-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ci-fix-pipeline-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ci-watch-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ci-watch-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.condition-based-waiting-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.condition-based-waiting-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.crash-recovery-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.crash-recovery-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.create-followup-tickets-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.create-followup-tickets-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.create-ticket-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.create-ticket-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.decompose-epic-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.decompose-epic-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.deep-dive-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.deep-dive-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.defense-in-depth-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.defense-in-depth-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.deploy-local-plugin-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.deploy-local-plugin-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.dispatching-parallel-agents-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.dispatching-parallel-agents-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.epic-team-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.epic-team-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.executing-plans-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.executing-plans-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.finishing-a-development-branch-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.finishing-a-development-branch-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.fix-prs-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.fix-prs-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.gap-analysis-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.gap-analysis-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.gap-fix-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.gap-fix-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.generate-node-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.generate-node-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.linear-insights-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.linear-insights-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.linear-ticket-management-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.linear-ticket-management-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.local-review-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.local-review-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.log-execution-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.log-execution-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.merge-sweep-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.merge-sweep-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.onex-status-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.onex-status-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.parallel-solve-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.parallel-solve-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pipeline-audit-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pipeline-audit-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pipeline-metrics-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pipeline-metrics-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.plan-ticket-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.plan-ticket-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.plan-to-tickets-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.plan-to-tickets-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-polish-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-polish-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-queue-pipeline-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-queue-pipeline-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-release-ready-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-release-ready-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-review-comprehensive-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-review-comprehensive-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-review-dev-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-review-dev-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-watch-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.pr-watch-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.project-status-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.project-status-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.receiving-code-review-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.receiving-code-review-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.release-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.release-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.requesting-code-review-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.requesting-code-review-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.review-all-prs-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.review-all-prs-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.review-cycle-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.review-cycle-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.root-cause-tracing-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.root-cause-tracing-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.rrh-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.rrh-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.setup-statusline-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.setup-statusline-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.sharing-skills-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.sharing-skills-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.slack-gate-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.slack-gate-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.subagent-driven-development-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.subagent-driven-development-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.suggest-work-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.suggest-work-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.systematic-debugging-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.systematic-debugging-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.test-driven-development-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.test-driven-development-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.testing-anti-patterns-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.testing-anti-patterns-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.testing-skills-with-subagents-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.testing-skills-with-subagents-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ticket-pipeline-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ticket-pipeline-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ticket-plan-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ticket-plan-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ticket-work-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ticket-work-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ultimate-validate-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.ultimate-validate-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.using-git-worktrees-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.using-git-worktrees-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.using-superpowers-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.using-superpowers-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.velocity-estimate-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.velocity-estimate-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.verification-before-completion-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.verification-before-completion-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.writing-plans-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.writing-plans-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.writing-skills-completed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.writing-skills-failed.v1": "pre-migration skill topic; topics.yaml covers this once OMN-4594 wired | owner: jonah | expiry: 2026-06-01",
    # --- omniclaude special/non-skill topics (need contract.yaml in omniclaude) ---
    "onex.evt.omniclaude.agent-actions-dlq.v1": "DLQ topic, no contract.yaml yet; needs omniclaude node contract | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.agent-observability-dlq.v1": "DLQ topic, no contract.yaml yet; needs omniclaude node contract | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.skill-lifecycle-dlq.v1": "DLQ topic for skill-lifecycle consumer (OMN-5445); provisioned in platform_topic_suffixes; needs omniclaude node contract | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.audit-compression-triggered.v1": "context audit topic [OMN-5240]; produced by omniclaude, needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.audit-context-budget-exceeded.v1": "context audit topic [OMN-5240]; produced by omniclaude, needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.audit-dispatch-validated.v1": "context audit topic [OMN-5240]; produced by omniclaude, needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.audit-return-bounded.v1": "context audit topic [OMN-5240]; produced by omniclaude, needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.audit-scope-violation.v1": "context audit topic [OMN-5240]; produced by omniclaude, needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    # OMN-7114: contract.yaml added in node_context_audit_dlq_effect
    "onex.evt.omniclaude.fix-transition.v1": "lifecycle transition topic added in OMN-4572; needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.skill-completed.v1": "global skill lifecycle topic [OMN-2934]; needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniclaude.skill-started.v1": "global skill lifecycle topic [OMN-2934]; needs contract.yaml | owner: jonah | expiry: 2026-06-01",
    # --- omniclaude agent observability topics (OMN-6066..OMN-6072) ---
    # Produced by omniclaude agent hooks, consumed by ServiceAgentActionsConsumer.
    # Added to platform_topic_suffixes in this PR to replace raw string literals.
    # Contract.yaml coverage tracked in omniclaude repo.
    "onex.evt.omniclaude.agent-actions.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniclaude.routing-decision.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniclaude.agent-transformation.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniclaude.performance-metrics.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniclaude.detection-failure.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniclaude.agent-execution-logs.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniclaude.agent-status.v1": "observability topic produced by omniclaude [OMN-6066]; needs contract.yaml in omniclaude | owner: jonah | expiry: 2026-09-01",
    # --- omniintelligence topics (need contract.yaml in omniintelligence) ---
    "onex.cmd.omniintelligence.claude-hook-event.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniintelligence.decision-recorded.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniintelligence.pattern-lifecycle-transition.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniintelligence.routing-decision.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omniintelligence.session-outcome.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.decision-recorded.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.intent-classified.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.llm-call-completed.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.pattern-learned.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.pattern-lifecycle-transitioned.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.pattern-promoted.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omniintelligence.pattern-stored.v1": "pre-migration; needs contract.yaml in omniintelligence repo | owner: jonah | expiry: 2026-06-01",
    # --- omnimemory topics (need contract.yaml in omnimemory) ---
    "onex.cmd.omnimemory.archive-memory.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.crawl-requested.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.crawl-tick.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.expire-memory.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.intent-query-requested.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.memory-retrieval-requested.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.restore-memory.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnimemory.runtime-tick.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.document-changed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.document-discovered.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.document-indexed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.document-parse-failed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.document-removed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.intent-query-response.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.intent-store-failed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.intent-stored.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.lifecycle-transition-failed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-accessed.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-archive-initiated.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-archived.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-deleted.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-expired.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-restored.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-retrieval-response.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-retrieved.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-stored.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    "onex.evt.omnimemory.memory-updated.v1": "pre-migration; needs contract.yaml in omnimemory repo | owner: jonah | expiry: 2026-06-01",
    # --- omnibase-infra service-level topics (emitted by services, not nodes) ---
    "onex.evt.omnibase-infra.wiring-health-snapshot.v1": "emitted by WiringHealthChecker service; no contract.yaml node needed — service-level emission | owner: jonah | expiry: 2026-09-01",
    # --- platform/cross-cutting topics ---
    "onex.evt.omnibase-infra.circuit-breaker.v1": "new topic (OMN-5293); publisher-only, no node contract.yaml yet | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.runtime-error.v1": "new topic (OMN-5649); emitted by monitor_logs.py RuntimeErrorEmitter; contract.yaml added in OMN-5650 | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.error-triaged.v1": "new topic (OMN-5650); emitted by NodeRuntimeErrorTriageEffect; contract.yaml added in OMN-5650 | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omnibase-infra.gmail-archive-purged.v1": "pre-migration; needs contract.yaml in omnibase_infra | owner: jonah | expiry: 2026-06-01",
    "onex.evt.pattern.discovered.v1": "pre-migration; needs contract.yaml in omniintelligence | owner: jonah | expiry: 2026-06-01",
    "onex.evt.platform.resolution-decided.v1": "pre-migration; needs contract.yaml in omnibase_infra | owner: jonah | expiry: 2026-06-01",
    "onex.evt.platform.feature-flag-changed.v1": "new topic added in OMN-5580; contract.yaml needed once consuming node is wired | owner: jonah | expiry: 2026-06-01",
    "onex.evt.platform.service-heartbeat.v1": "new topic added in OMN-5184; contract.yaml needed once consuming node is wired | owner: jonah | expiry: 2026-06-01",
    "onex.snapshot.platform.registration-snapshots.v1": "non-standard kind 'snapshot'; topic validated separately via ValidateTopicSuffix skip-list | owner: jonah | expiry: 2026-06-01",
    # --- consumer health pipeline topics (OMN-5529) ---
    "onex.evt.omnibase-infra.consumer-health.v1": "OMN-5515; contract.yaml will be added with NodeConsumerHealthTriageEffect in OMN-5520 | owner: jonah | expiry: 2026-06-01",
    "onex.cmd.omnibase-infra.consumer-restart.v1": "OMN-5515; contract.yaml will be added with NodeConsumerHealthTriageEffect in OMN-5520 | owner: jonah | expiry: 2026-06-01",
    # --- DLQ aggregation topic (OMN-6136) ---
    "onex.evt.platform.dlq-message.v1": "OMN-6136; cross-published by MixinKafkaDlq for omnidash /dlq dashboard | owner: jonah | expiry: 2026-06-01",
    # --- runner health pipeline topics (OMN-6082) ---
    # OMN-7114: contract.yaml added in node_runner_health_snapshot_effect
    "onex.evt.omnibase-infra.eval-completed.v1": "OMN-6798; emitted by ServiceAutoEvalRunner; contract.yaml deferred until eval node created | owner: jonah | expiry: 2026-09-01",
    # --- row count diagnostic probe (OMN-5653) ---
    # OMN-7114: contract.yaml added in node_row_count_diagnostic_effect
    # --- AST code extraction pipeline topics (OMN-5669) ---
    "onex.cmd.omniintelligence.code-crawl-requested.v1": "OMN-5669; contract.yaml in omniintelligence repo (cross-repo provisioning) | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniintelligence.code-file-discovered.v1": "OMN-5669; contract.yaml in omniintelligence repo (cross-repo provisioning) | owner: jonah | expiry: 2026-09-01",
    "onex.evt.omniintelligence.code-entities-extracted.v1": "OMN-5669; contract.yaml in omniintelligence repo (cross-repo provisioning) | owner: jonah | expiry: 2026-09-01",
    # --- GitHub PR merged event (OMN-6726) ---
    "onex.evt.github.pr-merged.v1": "OMN-6726; produced by GHA workflow, no node contract — external event source | owner: jonah | expiry: 2026-09-01",
    # --- Post-merge check chain result (OMN-6727) ---
    "onex.evt.github.post-merge-result.v1": "OMN-6727; produced by PostMergeConsumer service, no node contract — service-level topic | owner: jonah | expiry: 2026-09-01",
}
# fmt: on

_ALLOWLISTED_SUFFIXES: frozenset[str] = frozenset(_LEGACY_ALLOWLIST.keys())


def _repo_root() -> Path:
    """Resolve repo root relative to this script."""
    return Path(__file__).parent.parent.resolve()


def _load_provisioned_suffixes() -> tuple[str, ...]:
    """Import ALL_PROVISIONED_SUFFIXES from the installed package."""
    repo_root = _repo_root()
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from omnibase_infra.topics.platform_topic_suffixes import ALL_PROVISIONED_SUFFIXES

    return ALL_PROVISIONED_SUFFIXES


def _load_contract_suffixes(contracts_root: Path) -> frozenset[str]:
    """Scan all contract.yaml files under contracts_root and collect declared topics."""
    repo_root = _repo_root()
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

    extractor = ContractTopicExtractor()
    entries = extractor.extract(contracts_root)
    return frozenset(e.topic for e in entries)


def _find_contracts_root(repo_root: Path) -> Path:
    """Locate the contracts/nodes root directory."""
    candidates = [
        repo_root / "src" / "omnibase_infra" / "nodes",
        repo_root / "nodes",
        repo_root / "contracts",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find contracts root. Tried: {[str(c) for c in candidates]}"
    )


def run_parity_check(verbose: bool = False, show_allowlist: bool = False) -> int:
    """Run the parity check. Returns exit code (0 = pass, 1 = fail)."""
    repo_root = _repo_root()

    # Load provisioned suffixes from Python registry
    try:
        provisioned = _load_provisioned_suffixes()
    except ImportError as exc:
        print(
            f"ERROR: Could not import ALL_PROVISIONED_SUFFIXES: {exc}", file=sys.stderr
        )
        return 1

    # Load contract-derived topics
    try:
        contracts_root = _find_contracts_root(repo_root)
        contract_suffixes = _load_contract_suffixes(contracts_root)
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: Could not load contract topics: {exc}", file=sys.stderr)
        return 1

    # Classify each provisioned suffix
    contract_covered: list[str] = []
    allowlisted: list[str] = []
    python_only: list[str] = []

    for suffix in sorted(provisioned):
        if suffix in contract_suffixes:
            contract_covered.append(suffix)
        elif suffix in _ALLOWLISTED_SUFFIXES:
            allowlisted.append(suffix)
        else:
            python_only.append(suffix)

    # Print allowlist section (intentionally visible)
    if show_allowlist:
        print(f"\n{'=' * 70}")
        print(
            f"LEGACY ALLOWLIST  ({len(allowlisted)} entries — tech debt, migrate these)"
        )
        print(f"{'=' * 70}")
        for suffix, note in sorted(_LEGACY_ALLOWLIST.items()):
            print(f"  {suffix}")
            print(f"    {note}")
        print()

    # Metrics summary
    total = len(provisioned)
    coverage_pct = len(contract_covered) * 100 // total if total else 0
    print(f"\nTopic parity report ({contracts_root.relative_to(repo_root)}):")
    print(f"  provisioned total:  {total}")
    print(f"  contract-covered:   {len(contract_covered)} ({coverage_pct}%)")
    print(f"  allowlisted:        {len(allowlisted)} (tech debt — target: 0)")
    print(f"  python-only gaps:   {len(python_only)}")

    if verbose and contract_covered:
        print(f"\nContract-covered topics ({len(contract_covered)}):")
        for suffix in contract_covered:
            print(f"  ✓  {suffix}")

    if python_only:
        print(f"\n{'=' * 70}")
        print("PARITY GATE FAILURE")
        print(f"{'=' * 70}")
        print(
            f"\n{len(python_only)} topic(s) in ALL_PROVISIONED_SUFFIXES have no contract.yaml"
            " declaration and are not in the legacy allowlist.\n"
        )
        print(
            "To fix: add a contract.yaml entry for each topic, OR add to _LEGACY_ALLOWLIST"
        )
        print(
            "(with required 'reason | owner | expiry' comment) as a temporary exemption.\n"
        )
        print("Python-only gaps:")
        for suffix in python_only:
            print(f"  ✗  {suffix}")
        print()
        return 1

    print(
        f"\n✓ Parity gate passed — {len(contract_covered)} contract-covered,"
        f" {len(allowlisted)} allowlisted (tech debt)."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI parity gate: detect Python-only topic registry entries (OMN-4600)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all contract-covered topics"
    )
    parser.add_argument(
        "--show-allowlist",
        action="store_true",
        help="Print the legacy allowlist section",
    )
    args = parser.parse_args()

    exit_code = run_parity_check(
        verbose=args.verbose, show_allowlist=args.show_allowlist
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
