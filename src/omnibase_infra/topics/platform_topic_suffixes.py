"""Platform and domain topic suffixes for ONEX infrastructure.

This module defines topic suffixes for:
  1. Platform-reserved topics (producer: ``platform``) -- infrastructure internals
  2. Domain plugin topics (producer: ``omniintelligence``, ``pattern``, etc.) --
     provisioned by the runtime so domain plugins find their topics ready.

Domain services should NOT import individual suffix constants from this module.
They should subscribe to topics by name from their own contracts. The combined
``ALL_PROVISIONED_TOPIC_SPECS`` registry is consumed by ``TopicProvisioner`` at
startup to create all required topics.

Topic Suffix Format:
    onex.<kind>.<producer>.<event-name>.v<version>

    Structure:
        - onex: Required prefix for all ONEX topics
        - kind: Message category (evt, cmd, intent, snapshot, dlq)
        - producer: Routing domain -- ``platform`` for infrastructure,
          domain name for plugins (e.g., ``omniintelligence``)
        - event-name: Descriptive name using kebab-case
        - version: Semantic version (v1, v2, etc.)

    Kinds:
        evt - Event topics (state changes, notifications)
        cmd - Command topics (requests for action)
        intent - Intent topics (internal workflow coordination)
        snapshot - Snapshot topics (periodic state snapshots)
        dlq - Dead letter queue topics

    Examples:
        onex.evt.platform.node-registration.v1
        onex.cmd.platform.request-introspection.v1
        onex.intent.platform.runtime-tick.v1
        onex.cmd.omniintelligence.claude-hook-event.v1

Usage:
    from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION

    # Compose full topic with tenant/namespace prefix
    full_topic = f"{tenant}.{namespace}.{SUFFIX_NODE_REGISTRATION}"

See Also:
    omnibase_core.validation.validate_topic_suffix - Validation function
    omnibase_core.validation.compose_full_topic - Topic composition utility
    model_topic_spec.ModelTopicSpec - Per-topic creation spec
"""

from __future__ import annotations

from omnibase_core.errors import OnexError
from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

# =============================================================================
# PLATFORM-RESERVED TOPIC SUFFIXES
# =============================================================================

# Node lifecycle events
SUFFIX_NODE_REGISTRATION: str = "onex.evt.platform.node-registration.v1"
"""Topic suffix for node registration events.

Published when a node registers with the runtime. Contains node metadata,
capabilities, and health check configuration.
"""

SUFFIX_NODE_INTROSPECTION: str = "onex.evt.platform.node-introspection.v1"
"""Topic suffix for node introspection events.

Published when a node responds to an introspection request. Contains node
capabilities, supported operations, and current state.
"""

SUFFIX_REGISTRY_REQUEST_INTROSPECTION: str = (
    "onex.evt.platform.registry-request-introspection.v1"
)
"""Topic suffix for registry-initiated introspection request events.

Published when the registry requests introspection from nodes during the
registration workflow. The registration orchestrator subscribes to this topic
to trigger node registration processing.
"""

SUFFIX_NODE_HEARTBEAT: str = "onex.evt.platform.node-heartbeat.v1"
"""Topic suffix for node heartbeat events.

Published periodically by nodes to indicate liveness. Contains timestamp,
resource usage metrics, and health status.
"""

# Command topics
SUFFIX_REQUEST_INTROSPECTION: str = "onex.cmd.platform.request-introspection.v1"
"""Topic suffix for introspection request commands.

Published to request introspection from a specific node or all nodes.
Nodes respond on the SUFFIX_NODE_INTROSPECTION topic.
"""

# FSM and state management
SUFFIX_FSM_STATE_TRANSITIONS: str = "onex.evt.platform.fsm-state-transitions.v1"
"""Topic suffix for FSM state transition events.

Published when a node's finite state machine transitions between states.
Contains previous state, new state, trigger event, and transition metadata.
"""

# Runtime coordination
SUFFIX_RUNTIME_TICK: str = "onex.intent.platform.runtime-tick.v1"
"""Topic suffix for runtime tick intents.

Internal topic for runtime orchestration. Triggers periodic tasks like
heartbeat collection, health checks, and scheduled workflows.
"""

# Registration snapshots
SUFFIX_REGISTRATION_SNAPSHOTS: str = "onex.snapshot.platform.registration-snapshots.v1"
"""Topic suffix for registration snapshot events.

Published periodically with aggregated registration state. Used for
dashboard displays and monitoring systems.
"""

# Contract lifecycle events (used by ContractRegistrationEventRouter in kernel)
SUFFIX_CONTRACT_REGISTERED: str = "onex.evt.platform.contract-registered.v1"
"""Topic suffix for contract registration events.

Published when a node contract is registered with the runtime.
"""

SUFFIX_CONTRACT_DEREGISTERED: str = "onex.evt.platform.contract-deregistered.v1"
"""Topic suffix for contract deregistration events.

Published when a node contract is deregistered from the runtime.
"""

# Registration ACK commands
SUFFIX_NODE_REGISTRATION_ACKED: str = "onex.cmd.platform.node-registration-acked.v1"
"""Topic suffix for node registration ACK commands.

Published by a node after it receives a registration-accepted event,
confirming that the node acknowledges successful registration.
"""

# Resolution event ledger (OMN-2895 / Phase 6 of OMN-2897)
SUFFIX_RESOLUTION_DECIDED: str = "onex.evt.platform.resolution-decided.v1"
"""Topic suffix for resolution decision audit events.

Published after every tiered dependency resolution decision. Records the
full tier progression, proofs attempted, and final outcome for audit,
replay, and intelligence.

Producer: ServiceResolutionEventPublisher
Consumer: Audit log, intelligence pipeline, replay infrastructure
"""

# =============================================================================
# INTELLIGENCE DOMAIN TOPIC SUFFIXES (omniintelligence plugin)
# =============================================================================
# These topics are consumed/produced by PluginIntelligence. They are provisioned
# alongside platform topics so the plugin finds them ready at startup.

# Command topics (inbound to intelligence pipeline)
SUFFIX_INTELLIGENCE_CLAUDE_HOOK_EVENT: str = (
    "onex.cmd.omniintelligence.claude-hook-event.v1"
)
"""Topic for Claude hook events dispatched to the intelligence pipeline."""

SUFFIX_INTELLIGENCE_SESSION_OUTCOME: str = (
    "onex.cmd.omniintelligence.session-outcome.v1"
)
"""Topic for session outcome commands (success/failure attribution)."""

SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITION: str = (
    "onex.cmd.omniintelligence.pattern-lifecycle-transition.v1"
)
"""Topic for pattern lifecycle transition commands."""

# Event topics (outbound from intelligence pipeline)
SUFFIX_INTELLIGENCE_INTENT_CLASSIFIED: str = (
    "onex.evt.omniintelligence.intent-classified.v1"
)
"""Topic for intent classification events."""

SUFFIX_INTELLIGENCE_PATTERN_LEARNED: str = (
    "onex.evt.omniintelligence.pattern-learned.v1"
)
"""Topic for pattern learning events (new pattern discovered)."""

SUFFIX_INTELLIGENCE_PATTERN_STORED: str = "onex.evt.omniintelligence.pattern-stored.v1"
"""Topic for pattern storage events (pattern persisted to DB)."""

SUFFIX_INTELLIGENCE_PATTERN_PROMOTED: str = (
    "onex.evt.omniintelligence.pattern-promoted.v1"
)
"""Topic for pattern promotion events (candidate -> validated)."""

SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITIONED: str = (
    "onex.evt.omniintelligence.pattern-lifecycle-transitioned.v1"
)
"""Topic for pattern lifecycle transition completion events."""

SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED: str = (
    "onex.evt.omniintelligence.llm-call-completed.v1"
)
"""Topic for LLM call completed metrics events.

Published by LLM inference handlers after each call. Contains per-call
token counts, cost, and latency for the cost aggregation pipeline.
"""

SUFFIX_INTELLIGENCE_PATTERN_DISCOVERED: str = "onex.evt.pattern.discovered.v1"
"""Topic for generic pattern discovery events."""

# =============================================================================
# OMNIMEMORY DOMAIN TOPIC SUFFIXES (omnimemory plugin)
# =============================================================================
# These topics are consumed/produced by PluginOmnimemory. They are provisioned
# alongside platform topics so the plugin finds them ready at startup.

# Event topics (outbound from omnimemory pipeline)
SUFFIX_OMNIMEMORY_DOCUMENT_DISCOVERED: str = (
    "onex.evt.omnimemory.document-discovered.v1"
)
"""Topic for document discovery events (new document found during crawl)."""

SUFFIX_OMNIMEMORY_DOCUMENT_CHANGED: str = "onex.evt.omnimemory.document-changed.v1"
"""Topic for document change events (existing document content changed)."""

SUFFIX_OMNIMEMORY_DOCUMENT_REMOVED: str = "onex.evt.omnimemory.document-removed.v1"
"""Topic for document removal events (document deleted or no longer accessible)."""

SUFFIX_OMNIMEMORY_DOCUMENT_INDEXED: str = "onex.evt.omnimemory.document-indexed.v1"
"""Topic for document indexed events (document successfully indexed into vector store)."""

# Command topics (inbound to omnimemory pipeline)
SUFFIX_OMNIMEMORY_CRAWL_TICK: str = "onex.cmd.omnimemory.crawl-tick.v1"
"""Topic for crawl tick commands (periodic scheduler trigger for crawl cycle)."""

SUFFIX_OMNIMEMORY_CRAWL_REQUESTED: str = "onex.cmd.omnimemory.crawl-requested.v1"
"""Topic for crawl requested commands (explicit crawl request for a document source)."""

# =============================================================================
# OMNICLAUDE SKILL TOPIC SUFFIXES (omniclaude plugin)
# =============================================================================
# These topics are consumed/produced by OmniClaude skill orchestrator nodes.
# Each skill has 3 topics: a command topic (subscribe) and two event topics
# (success + failure). Topics are extracted from contract.yaml files in
# omniclaude/src/omniclaude/nodes/node_skill_*/contract.yaml.
#
# Unlike platform and intelligence topics, individual suffix constants are NOT
# defined here because domain services subscribe via their own contracts.
# The suffixes are listed as strings and converted to ModelTopicSpec entries
# in the ALL_OMNICLAUDE_TOPIC_SPECS tuple below.

_OMNICLAUDE_SKILL_TOPIC_SUFFIXES: tuple[str, ...] = (
    # ----- Skill lifecycle observability topics (OMN-2934) -----
    # Emitted by handle_skill_requested() in omniclaude on every skill invocation.
    # run_id is the join key: started and completed for the same invocation
    # share the same run_id and land on the same Kafka partition.
    "onex.evt.omniclaude.skill-started.v1",
    "onex.evt.omniclaude.skill-completed.v1",
    # ----- Command topics (skill invocation requests) -----
    "onex.cmd.omniclaude.action-logging.v1",
    "onex.cmd.omniclaude.agent-observability.v1",
    "onex.cmd.omniclaude.auto-merge.v1",
    "onex.cmd.omniclaude.brainstorming.v1",
    "onex.cmd.omniclaude.checkpoint.v1",
    "onex.cmd.omniclaude.ci-failures.v1",
    "onex.cmd.omniclaude.ci-fix-pipeline.v1",
    "onex.cmd.omniclaude.ci-watch.v1",
    "onex.cmd.omniclaude.condition-based-waiting.v1",
    "onex.cmd.omniclaude.crash-recovery.v1",
    "onex.cmd.omniclaude.create-followup-tickets.v1",
    "onex.cmd.omniclaude.create-ticket.v1",
    "onex.cmd.omniclaude.decompose-epic.v1",
    "onex.cmd.omniclaude.deep-dive.v1",
    "onex.cmd.omniclaude.defense-in-depth.v1",
    "onex.cmd.omniclaude.deploy-local-plugin.v1",
    "onex.cmd.omniclaude.dispatching-parallel-agents.v1",
    "onex.cmd.omniclaude.epic-team.v1",
    "onex.cmd.omniclaude.executing-plans.v1",
    "onex.cmd.omniclaude.finishing-a-development-branch.v1",
    "onex.cmd.omniclaude.fix-prs.v1",
    "onex.cmd.omniclaude.gap-analysis.v1",
    "onex.cmd.omniclaude.gap-fix.v1",
    "onex.cmd.omniclaude.generate-node.v1",
    "onex.cmd.omniclaude.linear-insights.v1",
    "onex.cmd.omniclaude.linear-ticket-management.v1",
    "onex.cmd.omniclaude.local-review.v1",
    "onex.cmd.omniclaude.log-execution.v1",
    "onex.cmd.omniclaude.merge-sweep.v1",
    "onex.cmd.omniclaude.onex-status.v1",
    "onex.cmd.omniclaude.parallel-solve.v1",
    "onex.cmd.omniclaude.pipeline-audit.v1",
    "onex.cmd.omniclaude.pipeline-metrics.v1",
    "onex.cmd.omniclaude.plan-ticket.v1",
    "onex.cmd.omniclaude.plan-to-tickets.v1",
    "onex.cmd.omniclaude.pr-polish.v1",
    "onex.cmd.omniclaude.pr-queue-pipeline.v1",
    "onex.cmd.omniclaude.pr-release-ready.v1",
    "onex.cmd.omniclaude.pr-review-comprehensive.v1",
    "onex.cmd.omniclaude.pr-review-dev.v1",
    "onex.cmd.omniclaude.pr-watch.v1",
    "onex.cmd.omniclaude.project-status.v1",
    "onex.cmd.omniclaude.receiving-code-review.v1",
    "onex.cmd.omniclaude.release.v1",
    "onex.cmd.omniclaude.requesting-code-review.v1",
    "onex.cmd.omniclaude.review-all-prs.v1",
    "onex.cmd.omniclaude.review-cycle.v1",
    "onex.cmd.omniclaude.root-cause-tracing.v1",
    "onex.cmd.omniclaude.rrh.v1",
    "onex.cmd.omniclaude.setup-statusline.v1",
    "onex.cmd.omniclaude.sharing-skills.v1",
    "onex.cmd.omniclaude.slack-gate.v1",
    "onex.cmd.omniclaude.subagent-driven-development.v1",
    "onex.cmd.omniclaude.suggest-work.v1",
    "onex.cmd.omniclaude.systematic-debugging.v1",
    "onex.cmd.omniclaude.test-driven-development.v1",
    "onex.cmd.omniclaude.testing-anti-patterns.v1",
    "onex.cmd.omniclaude.testing-skills-with-subagents.v1",
    "onex.cmd.omniclaude.ticket-pipeline.v1",
    "onex.cmd.omniclaude.ticket-plan.v1",
    "onex.cmd.omniclaude.ticket-work.v1",
    "onex.cmd.omniclaude.ultimate-validate.v1",
    "onex.cmd.omniclaude.using-git-worktrees.v1",
    "onex.cmd.omniclaude.using-superpowers.v1",
    "onex.cmd.omniclaude.velocity-estimate.v1",
    "onex.cmd.omniclaude.verification-before-completion.v1",
    "onex.cmd.omniclaude.writing-plans.v1",
    "onex.cmd.omniclaude.writing-skills.v1",
    # ----- Event topics (skill completion) -----
    "onex.evt.omniclaude.action-logging-completed.v1",
    "onex.evt.omniclaude.action-logging-failed.v1",
    "onex.evt.omniclaude.agent-observability-completed.v1",
    "onex.evt.omniclaude.agent-observability-failed.v1",
    "onex.evt.omniclaude.auto-merge-completed.v1",
    "onex.evt.omniclaude.auto-merge-failed.v1",
    "onex.evt.omniclaude.brainstorming-completed.v1",
    "onex.evt.omniclaude.brainstorming-failed.v1",
    "onex.evt.omniclaude.checkpoint-completed.v1",
    "onex.evt.omniclaude.checkpoint-failed.v1",
    "onex.evt.omniclaude.ci-failures-completed.v1",
    "onex.evt.omniclaude.ci-failures-failed.v1",
    "onex.evt.omniclaude.ci-fix-pipeline-completed.v1",
    "onex.evt.omniclaude.ci-fix-pipeline-failed.v1",
    "onex.evt.omniclaude.ci-watch-completed.v1",
    "onex.evt.omniclaude.ci-watch-failed.v1",
    "onex.evt.omniclaude.condition-based-waiting-completed.v1",
    "onex.evt.omniclaude.condition-based-waiting-failed.v1",
    "onex.evt.omniclaude.crash-recovery-completed.v1",
    "onex.evt.omniclaude.crash-recovery-failed.v1",
    "onex.evt.omniclaude.create-followup-tickets-completed.v1",
    "onex.evt.omniclaude.create-followup-tickets-failed.v1",
    "onex.evt.omniclaude.create-ticket-completed.v1",
    "onex.evt.omniclaude.create-ticket-failed.v1",
    "onex.evt.omniclaude.decompose-epic-completed.v1",
    "onex.evt.omniclaude.decompose-epic-failed.v1",
    "onex.evt.omniclaude.deep-dive-completed.v1",
    "onex.evt.omniclaude.deep-dive-failed.v1",
    "onex.evt.omniclaude.defense-in-depth-completed.v1",
    "onex.evt.omniclaude.defense-in-depth-failed.v1",
    "onex.evt.omniclaude.deploy-local-plugin-completed.v1",
    "onex.evt.omniclaude.deploy-local-plugin-failed.v1",
    "onex.evt.omniclaude.dispatching-parallel-agents-completed.v1",
    "onex.evt.omniclaude.dispatching-parallel-agents-failed.v1",
    "onex.evt.omniclaude.epic-team-completed.v1",
    "onex.evt.omniclaude.epic-team-failed.v1",
    "onex.evt.omniclaude.executing-plans-completed.v1",
    "onex.evt.omniclaude.executing-plans-failed.v1",
    "onex.evt.omniclaude.finishing-a-development-branch-completed.v1",
    "onex.evt.omniclaude.finishing-a-development-branch-failed.v1",
    "onex.evt.omniclaude.fix-prs-completed.v1",
    "onex.evt.omniclaude.fix-prs-failed.v1",
    "onex.evt.omniclaude.gap-analysis-completed.v1",
    "onex.evt.omniclaude.gap-analysis-failed.v1",
    "onex.evt.omniclaude.gap-fix-completed.v1",
    "onex.evt.omniclaude.gap-fix-failed.v1",
    "onex.evt.omniclaude.generate-node-completed.v1",
    "onex.evt.omniclaude.generate-node-failed.v1",
    "onex.evt.omniclaude.linear-insights-completed.v1",
    "onex.evt.omniclaude.linear-insights-failed.v1",
    "onex.evt.omniclaude.linear-ticket-management-completed.v1",
    "onex.evt.omniclaude.linear-ticket-management-failed.v1",
    "onex.evt.omniclaude.local-review-completed.v1",
    "onex.evt.omniclaude.local-review-failed.v1",
    "onex.evt.omniclaude.log-execution-completed.v1",
    "onex.evt.omniclaude.log-execution-failed.v1",
    "onex.evt.omniclaude.merge-sweep-completed.v1",
    "onex.evt.omniclaude.merge-sweep-failed.v1",
    "onex.evt.omniclaude.onex-status-completed.v1",
    "onex.evt.omniclaude.onex-status-failed.v1",
    "onex.evt.omniclaude.parallel-solve-completed.v1",
    "onex.evt.omniclaude.parallel-solve-failed.v1",
    "onex.evt.omniclaude.pipeline-audit-completed.v1",
    "onex.evt.omniclaude.pipeline-audit-failed.v1",
    "onex.evt.omniclaude.pipeline-metrics-completed.v1",
    "onex.evt.omniclaude.pipeline-metrics-failed.v1",
    "onex.evt.omniclaude.plan-ticket-completed.v1",
    "onex.evt.omniclaude.plan-ticket-failed.v1",
    "onex.evt.omniclaude.plan-to-tickets-completed.v1",
    "onex.evt.omniclaude.plan-to-tickets-failed.v1",
    "onex.evt.omniclaude.pr-polish-completed.v1",
    "onex.evt.omniclaude.pr-polish-failed.v1",
    "onex.evt.omniclaude.pr-queue-pipeline-completed.v1",
    "onex.evt.omniclaude.pr-queue-pipeline-failed.v1",
    "onex.evt.omniclaude.pr-release-ready-completed.v1",
    "onex.evt.omniclaude.pr-release-ready-failed.v1",
    "onex.evt.omniclaude.pr-review-comprehensive-completed.v1",
    "onex.evt.omniclaude.pr-review-comprehensive-failed.v1",
    "onex.evt.omniclaude.pr-review-dev-completed.v1",
    "onex.evt.omniclaude.pr-review-dev-failed.v1",
    "onex.evt.omniclaude.pr-watch-completed.v1",
    "onex.evt.omniclaude.pr-watch-failed.v1",
    "onex.evt.omniclaude.project-status-completed.v1",
    "onex.evt.omniclaude.project-status-failed.v1",
    "onex.evt.omniclaude.receiving-code-review-completed.v1",
    "onex.evt.omniclaude.receiving-code-review-failed.v1",
    "onex.evt.omniclaude.release-completed.v1",
    "onex.evt.omniclaude.release-failed.v1",
    "onex.evt.omniclaude.requesting-code-review-completed.v1",
    "onex.evt.omniclaude.requesting-code-review-failed.v1",
    "onex.evt.omniclaude.review-all-prs-completed.v1",
    "onex.evt.omniclaude.review-all-prs-failed.v1",
    "onex.evt.omniclaude.review-cycle-completed.v1",
    "onex.evt.omniclaude.review-cycle-failed.v1",
    "onex.evt.omniclaude.root-cause-tracing-completed.v1",
    "onex.evt.omniclaude.root-cause-tracing-failed.v1",
    "onex.evt.omniclaude.rrh-completed.v1",
    "onex.evt.omniclaude.rrh-failed.v1",
    "onex.evt.omniclaude.setup-statusline-completed.v1",
    "onex.evt.omniclaude.setup-statusline-failed.v1",
    "onex.evt.omniclaude.sharing-skills-completed.v1",
    "onex.evt.omniclaude.sharing-skills-failed.v1",
    "onex.evt.omniclaude.slack-gate-completed.v1",
    "onex.evt.omniclaude.slack-gate-failed.v1",
    "onex.evt.omniclaude.subagent-driven-development-completed.v1",
    "onex.evt.omniclaude.subagent-driven-development-failed.v1",
    "onex.evt.omniclaude.suggest-work-completed.v1",
    "onex.evt.omniclaude.suggest-work-failed.v1",
    "onex.evt.omniclaude.systematic-debugging-completed.v1",
    "onex.evt.omniclaude.systematic-debugging-failed.v1",
    "onex.evt.omniclaude.test-driven-development-completed.v1",
    "onex.evt.omniclaude.test-driven-development-failed.v1",
    "onex.evt.omniclaude.testing-anti-patterns-completed.v1",
    "onex.evt.omniclaude.testing-anti-patterns-failed.v1",
    "onex.evt.omniclaude.testing-skills-with-subagents-completed.v1",
    "onex.evt.omniclaude.testing-skills-with-subagents-failed.v1",
    "onex.evt.omniclaude.ticket-pipeline-completed.v1",
    "onex.evt.omniclaude.ticket-pipeline-failed.v1",
    "onex.evt.omniclaude.ticket-plan-completed.v1",
    "onex.evt.omniclaude.ticket-plan-failed.v1",
    "onex.evt.omniclaude.ticket-work-completed.v1",
    "onex.evt.omniclaude.ticket-work-failed.v1",
    "onex.evt.omniclaude.ultimate-validate-completed.v1",
    "onex.evt.omniclaude.ultimate-validate-failed.v1",
    "onex.evt.omniclaude.using-git-worktrees-completed.v1",
    "onex.evt.omniclaude.using-git-worktrees-failed.v1",
    "onex.evt.omniclaude.using-superpowers-completed.v1",
    "onex.evt.omniclaude.using-superpowers-failed.v1",
    "onex.evt.omniclaude.velocity-estimate-completed.v1",
    "onex.evt.omniclaude.velocity-estimate-failed.v1",
    "onex.evt.omniclaude.verification-before-completion-completed.v1",
    "onex.evt.omniclaude.verification-before-completion-failed.v1",
    "onex.evt.omniclaude.writing-plans-completed.v1",
    "onex.evt.omniclaude.writing-plans-failed.v1",
    "onex.evt.omniclaude.writing-skills-completed.v1",
    "onex.evt.omniclaude.writing-skills-failed.v1",
)
"""All OmniClaude skill topic suffixes extracted from contract.yaml files.

Includes skill lifecycle observability topics (OMN-2934):
  - ``onex.evt.omniclaude.skill-started.v1``   (emitted before dispatch)
  - ``onex.evt.omniclaude.skill-completed.v1`` (emitted after dispatch)

Each skill node also defines 3 topics:
  - Command: ``onex.cmd.omniclaude.<skill-name>.v1`` (subscribe)
  - Success: ``onex.evt.omniclaude.<skill-name>-completed.v1`` (publish)
  - Failure: ``onex.evt.omniclaude.<skill-name>-failed.v1`` (publish)

Source: ``omniclaude/src/omniclaude/nodes/node_skill_*/contract.yaml``
"""

# =============================================================================
# TOPIC CATALOG TOPIC SUFFIXES
# =============================================================================

SUFFIX_TOPIC_CATALOG_QUERY: str = "onex.cmd.platform.topic-catalog-query.v1"
"""Topic suffix for topic catalog query commands.

Published when a client requests the current topic catalog. Contains optional
filters (topic_pattern, include_inactive) and a correlation_id for
request-response matching.
"""

SUFFIX_TOPIC_CATALOG_RESPONSE: str = "onex.evt.platform.topic-catalog-response.v1"
"""Topic suffix for topic catalog response events.

Published in response to a catalog query. Contains the full list of topic
entries with publisher/subscriber counts, plus catalog metadata.
"""

SUFFIX_TOPIC_CATALOG_CHANGED: str = "onex.evt.platform.topic-catalog-changed.v1"
"""Topic suffix for topic catalog change notification events.

Published when topics are added or removed from the catalog. Contains
delta tuples (topics_added, topics_removed) sorted alphabetically for
deterministic ordering.
"""

# =============================================================================
# PLATFORM TOPIC SPEC REGISTRY
# =============================================================================


# Build snapshot topic kafka_config from ModelSnapshotTopicConfig.default().
# Deferred import to avoid circular dependency; lazy initialization is safe
# because this module is only imported at startup.
def _snapshot_kafka_config() -> dict[str, str]:
    """Build Kafka config for the snapshot topic from ModelSnapshotTopicConfig."""
    from omnibase_infra.models.projection.model_snapshot_topic_config import (
        ModelSnapshotTopicConfig,
    )

    return ModelSnapshotTopicConfig.default().to_kafka_config()


ALL_PLATFORM_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    ModelTopicSpec(suffix=SUFFIX_NODE_REGISTRATION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_INTROSPECTION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_REGISTRY_REQUEST_INTROSPECTION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_HEARTBEAT, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_REQUEST_INTROSPECTION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_FSM_STATE_TRANSITIONS, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_RUNTIME_TICK, partitions=1),
    ModelTopicSpec(
        suffix=SUFFIX_REGISTRATION_SNAPSHOTS,
        partitions=1,
        kafka_config=_snapshot_kafka_config(),
    ),
    ModelTopicSpec(suffix=SUFFIX_CONTRACT_REGISTERED, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_CONTRACT_DEREGISTERED, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_REGISTRATION_ACKED, partitions=6),
    # Resolution event ledger (OMN-2895)
    ModelTopicSpec(suffix=SUFFIX_RESOLUTION_DECIDED, partitions=3),
    # Topic catalog topics (low-throughput coordination, 1 partition each)
    ModelTopicSpec(
        suffix=SUFFIX_TOPIC_CATALOG_QUERY,
        partitions=1,
        kafka_config={"retention.ms": "3600000", "cleanup.policy": "delete"},
    ),
    ModelTopicSpec(
        suffix=SUFFIX_TOPIC_CATALOG_RESPONSE,
        partitions=1,
        kafka_config={"retention.ms": "3600000", "cleanup.policy": "delete"},
    ),
    ModelTopicSpec(
        suffix=SUFFIX_TOPIC_CATALOG_CHANGED,
        partitions=1,
        kafka_config={"retention.ms": "604800000", "cleanup.policy": "delete"},
    ),
)
"""Complete tuple of all platform topic specs with per-topic configuration.

Each spec defines the topic suffix, partition count, replication factor, and
optional Kafka config overrides. TopicProvisioner iterates this registry
to create topics on startup.
"""

# =============================================================================
# INTELLIGENCE DOMAIN TOPIC SPEC REGISTRY
# =============================================================================

ALL_INTELLIGENCE_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # Command topics (3 partitions each — matches e2e compose)
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_CLAUDE_HOOK_EVENT, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_SESSION_OUTCOME, partitions=3),
    ModelTopicSpec(
        suffix=SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITION, partitions=3
    ),
    # Event topics (3 partitions each)
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_INTENT_CLASSIFIED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_LEARNED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_STORED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_PROMOTED, partitions=3),
    ModelTopicSpec(
        suffix=SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITIONED, partitions=3
    ),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_DISCOVERED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED, partitions=3),
)
"""Intelligence domain topic specs provisioned for PluginIntelligence."""

# =============================================================================
# OMNIMEMORY DOMAIN TOPIC SPEC REGISTRY
# =============================================================================

ALL_OMNIMEMORY_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # Event topics (3 partitions each)
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_DISCOVERED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_CHANGED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_REMOVED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_INDEXED, partitions=3),
    # Command topics (3 partitions each)
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_CRAWL_TICK, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_CRAWL_REQUESTED, partitions=3),
)
"""Omnimemory domain topic specs provisioned for PluginOmnimemory."""

# =============================================================================
# OMNICLAUDE SKILL TOPIC SPEC REGISTRY
# =============================================================================

ALL_OMNICLAUDE_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = tuple(
    ModelTopicSpec(suffix=suffix, partitions=1)
    for suffix in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES
)
"""OmniClaude skill topic specs provisioned for skill orchestrator nodes.

Each skill topic gets 1 partition (low-throughput skill dispatch -- each skill
invocation is a single message). 206 topics total (68 skills x 3 topics each + 2 lifecycle topics [OMN-2934]).

Source: ``omniclaude/src/omniclaude/nodes/node_skill_*/contract.yaml``
"""

# =============================================================================
# COMBINED PROVISIONED TOPIC SPECS
# =============================================================================

ALL_PROVISIONED_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    ALL_PLATFORM_TOPIC_SPECS
    + ALL_INTELLIGENCE_TOPIC_SPECS
    + ALL_OMNIMEMORY_TOPIC_SPECS
    + ALL_OMNICLAUDE_TOPIC_SPECS
)
"""All topic specs to be provisioned by TopicProvisioner at startup.

Combines platform-reserved, domain plugin, and OmniClaude skill topic specs
into a single registry consumed by service_topic_manager.py. This is the single
source of truth for topic creation.
"""

# =============================================================================
# AGGREGATE SUFFIX TUPLES
# =============================================================================

ALL_PLATFORM_SUFFIXES: tuple[str, ...] = tuple(
    spec.suffix for spec in ALL_PLATFORM_TOPIC_SPECS
)
"""Complete tuple of all platform-reserved topic suffixes.

Derived from ALL_PLATFORM_TOPIC_SPECS for backwards compatibility with
validation code that iterates suffix strings.
"""

ALL_PROVISIONED_SUFFIXES: tuple[str, ...] = tuple(
    spec.suffix for spec in ALL_PROVISIONED_TOPIC_SPECS
)
"""Complete tuple of all provisioned topic suffixes (platform + domain).

Derived from ALL_PROVISIONED_TOPIC_SPECS. Includes both platform-reserved
and domain plugin topics.
"""

# =============================================================================
# IMPORT-TIME VALIDATION
# =============================================================================


def _validate_all_suffixes() -> None:
    """Validate all suffixes at import time to fail fast on invalid format.

    Raises:
        OnexError: If any suffix fails validation with details about which
            suffix failed and why.
    """
    for suffix in ALL_PROVISIONED_SUFFIXES:
        result = validate_topic_suffix(suffix)
        if not result.is_valid:
            raise OnexError(f"Invalid topic suffix '{suffix}': {result.error}")


# Run validation at import time
_validate_all_suffixes()
