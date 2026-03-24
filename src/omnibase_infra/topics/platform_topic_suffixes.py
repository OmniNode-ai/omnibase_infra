# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

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

import os

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

SUFFIX_SERVICE_HEARTBEAT: str = "onex.evt.platform.service-heartbeat.v1"
"""Topic suffix for service-level heartbeat events.

Published periodically by runtime services (e.g. omninode-runtime, effects-runtime)
to report service health, uptime, resource usage, and restart count. Consumed by
monitoring infrastructure for service failure detection and health dashboards.
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

# Registration acceptance events
SUFFIX_NODE_REGISTRATION_ACCEPTED: str = (
    "onex.evt.platform.node-registration-accepted.v1"
)
"""Topic suffix for node registration acceptance events.

Published by the registration orchestrator when a node's registration is
accepted. Nodes subscribe to this topic to confirm their registration
and emit ACK commands in response.
"""

# Registration ACK commands
SUFFIX_NODE_REGISTRATION_ACKED: str = "onex.cmd.platform.node-registration-acked.v1"
"""Topic suffix for node registration ACK commands.

Published by a node after it receives a registration-accepted event,
confirming that the node acknowledges successful registration.
"""

# Resolution event ledger (OMN-2895 / Phase 6 of OMN-2897)
SUFFIX_FEATURE_FLAG_CHANGED: str = "onex.evt.platform.feature-flag-changed.v1"
"""Topic suffix for feature flag state change events.

Published when a feature flag is toggled via the control-plane API.
Contains flag name, new value, previous value, and env_var reference.

Producer: Registry API (update_feature_flag)
Consumer: Runtime services for dynamic flag reload
"""

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

SUFFIX_INTELLIGENCE_DECISION_RECORDED_EVT: str = (
    "onex.evt.omniintelligence.decision-recorded.v1"
)
"""Topic for decision recorded events (outbound from intelligence pipeline).

Published by omniintelligence decision_emitter on every model routing decision.
Records the final routing decision for audit, replay, and downstream consumers.

Producer: omniintelligence decision_emitter
Consumer: Audit log, intelligence pipeline, omnidash routing analytics
"""

SUFFIX_INTELLIGENCE_DECISION_RECORDED_CMD: str = (
    "onex.cmd.omniintelligence.decision-recorded.v1"
)
"""Topic for decision recorded commands (coordination channel).

Published by omniintelligence decision_emitter alongside the evt topic on every
model routing decision. Used for downstream command acknowledgement and replay.

Producer: omniintelligence decision_emitter
Consumer: Intelligence coordination, replay infrastructure
"""

SUFFIX_OMNIINTELLIGENCE_ROUTING_DECISION_CMD: str = (
    "onex.cmd.omniintelligence.routing-decision.v1"
)
"""Internal control plane routing decision command topic.

Producer: omniclaude (cross-domain CMD — see hooks/topics.py ROUTING_DECISION_CMD)
Consumer: omniintelligence routing decision handler
Lifecycle: internal_control (OMN-3294)
Governance: topic_allowlist.yaml (not a contract topic_base)
"""

# =============================================================================
# OMNIMEMORY DOMAIN TOPIC SUFFIXES (omnimemory plugin)
# =============================================================================
# These topics are consumed/produced by PluginOmnimemory. They are provisioned
# alongside platform topics so the plugin finds them ready at startup.

# Document crawl event topics (outbound from omnimemory crawl pipeline)
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

SUFFIX_OMNIMEMORY_DOCUMENT_PARSE_FAILED: str = (
    "onex.evt.omnimemory.document-parse-failed.v1"
)
"""Topic for document parse failure events (document could not be parsed during indexing)."""

# Crawl command topics (inbound to omnimemory crawl pipeline)
SUFFIX_OMNIMEMORY_CRAWL_TICK: str = "onex.cmd.omnimemory.crawl-tick.v1"
"""Topic for crawl tick commands (periodic scheduler trigger for crawl cycle)."""

SUFFIX_OMNIMEMORY_CRAWL_REQUESTED: str = "onex.cmd.omnimemory.crawl-requested.v1"
"""Topic for crawl requested commands (explicit crawl request for a document source).

Note: No subscriber is currently declared in any omnimemory contract.yaml. The topic is
provisioned to ensure the correct broker topic exists if a subscriber is added in the
future. See OMN-2941.
"""

SUFFIX_OMNIMEMORY_RUNTIME_TICK: str = "onex.cmd.omnimemory.runtime-tick.v1"
"""Topic for runtime tick commands (internal periodic tick for omnimemory orchestrator)."""

# Intent pipeline topics
SUFFIX_OMNIMEMORY_INTENT_STORED: str = "onex.evt.omnimemory.intent-stored.v1"
"""Topic for intent stored events (intent classification successfully persisted)."""

SUFFIX_OMNIMEMORY_INTENT_STORE_FAILED: str = (
    "onex.evt.omnimemory.intent-store-failed.v1"
)
"""Topic for intent store failed events (intent classification persistence failure)."""

SUFFIX_OMNIMEMORY_INTENT_QUERY_REQUESTED: str = (
    "onex.cmd.omnimemory.intent-query-requested.v1"
)
"""Topic for intent query request commands (request to query stored intents)."""

SUFFIX_OMNIMEMORY_INTENT_QUERY_RESPONSE: str = (
    "onex.evt.omnimemory.intent-query-response.v1"
)
"""Topic for intent query response events (query results for stored intents)."""

# Memory lifecycle event topics (outbound from memory lifecycle pipeline)
SUFFIX_OMNIMEMORY_MEMORY_STORED: str = "onex.evt.omnimemory.memory-stored.v1"
"""Topic for memory stored events (memory entry successfully persisted)."""

SUFFIX_OMNIMEMORY_MEMORY_RETRIEVED: str = "onex.evt.omnimemory.memory-retrieved.v1"
"""Topic for memory retrieved events (memory entry successfully fetched)."""

SUFFIX_OMNIMEMORY_MEMORY_UPDATED: str = "onex.evt.omnimemory.memory-updated.v1"
"""Topic for memory updated events (memory entry content or metadata changed)."""

SUFFIX_OMNIMEMORY_MEMORY_DELETED: str = "onex.evt.omnimemory.memory-deleted.v1"
"""Topic for memory deleted events (memory entry permanently removed)."""

SUFFIX_OMNIMEMORY_MEMORY_ACCESSED: str = "onex.evt.omnimemory.memory-accessed.v1"
"""Topic for memory accessed events (memory entry read access recorded)."""

SUFFIX_OMNIMEMORY_MEMORY_EXPIRED: str = "onex.evt.omnimemory.memory-expired.v1"
"""Topic for memory expired events (memory entry TTL elapsed and marked expired)."""

SUFFIX_OMNIMEMORY_MEMORY_ARCHIVED: str = "onex.evt.omnimemory.memory-archived.v1"
"""Topic for memory archived events (memory entry moved to long-term archive)."""

SUFFIX_OMNIMEMORY_MEMORY_ARCHIVE_INITIATED: str = (
    "onex.evt.omnimemory.memory-archive-initiated.v1"
)
"""Topic for memory archive initiated events (archive workflow started for a memory entry)."""

SUFFIX_OMNIMEMORY_MEMORY_RESTORED: str = "onex.evt.omnimemory.memory-restored.v1"
"""Topic for memory restored events (archived memory entry restored to active state)."""

SUFFIX_OMNIMEMORY_LIFECYCLE_TRANSITION_FAILED: str = (
    "onex.evt.omnimemory.lifecycle-transition-failed.v1"
)
"""Topic for lifecycle transition failure events (FSM transition could not be completed)."""

# Memory lifecycle command topics (inbound to memory lifecycle pipeline)
SUFFIX_OMNIMEMORY_MEMORY_RETRIEVAL_REQUESTED: str = (
    "onex.cmd.omnimemory.memory-retrieval-requested.v1"
)
"""Topic for memory retrieval request commands (request to retrieve a memory entry)."""

SUFFIX_OMNIMEMORY_MEMORY_RETRIEVAL_RESPONSE: str = (
    "onex.evt.omnimemory.memory-retrieval-response.v1"
)
"""Topic for memory retrieval response events (retrieval results for a memory query)."""

SUFFIX_OMNIMEMORY_EXPIRE_MEMORY: str = "onex.cmd.omnimemory.expire-memory.v1"
"""Topic for expire memory commands (command to expire a memory entry by ID)."""

SUFFIX_OMNIMEMORY_ARCHIVE_MEMORY: str = "onex.cmd.omnimemory.archive-memory.v1"
"""Topic for archive memory commands (command to archive a memory entry by ID)."""

SUFFIX_OMNIMEMORY_RESTORE_MEMORY: str = "onex.cmd.omnimemory.restore-memory.v1"
"""Topic for restore memory commands (command to restore an archived memory entry)."""

# =============================================================================
# OMNIBASE_INFRA DOMAIN TOPIC SUFFIXES (omnibase-infra gmail nodes)
# =============================================================================
# These topics are produced by omnibase_infra gmail effect nodes. They are
# provisioned alongside platform topics so consumers find them ready at startup.

SUFFIX_GMAIL_ARCHIVE_PURGED: str = "onex.evt.omnibase-infra.gmail-archive-purged.v1"
"""Topic suffix for Gmail archive purge summary events.

Published by NodeGmailArchiveCleanupEffect after each cleanup run when any
messages were deleted or errors occurred. This is a fire-and-forget summary
event — there are no downstream consumers in the current implementation.

No consumer rationale: Gmail archive cleanup events are audit-trail events
only. Downstream systems (dashboards, alerting) may subscribe in the future,
but no service currently depends on this event for correctness. The topic is
provisioned to ensure the correct name exists on the broker and to prevent
consumer misconfiguration if a subscriber is added later.

Producer: NodeGmailArchiveCleanupEffect / HandlerGmailArchiveCleanup
Consumer: None (intentionally fire-and-forget; see OMN-2937)
"""

SUFFIX_BASELINES_COMPUTED: str = "onex.evt.omnibase-infra.baselines-computed.v1"
"""Emitted after baseline ROI computation completes for a pattern cohort.

Published by the omnibase_infra baselines compute node after each baseline
ROI computation cycle completes. The omnidash /baselines route subscribes to
this topic to display baseline metrics in real time.

Producer: omnibase_infra baselines compute node (TODO: implement — OMN-4296)
Consumer: omnidash /baselines dashboard
"""

SUFFIX_WIRING_HEALTH_SNAPSHOT: str = "onex.evt.omnibase-infra.wiring-health-snapshot.v1"
"""Topic suffix for wiring health snapshot events.

Published by WiringHealthChecker after each health evaluation cycle.
Each event carries per-topic emission/consumption counts and overall health status.

Producer: WiringHealthChecker (OMN-5292)
Consumer: omnidash /wiring-health dashboard
"""

SUFFIX_CIRCUIT_BREAKER_STATE: str = "onex.evt.omnibase-infra.circuit-breaker.v1"
"""Topic suffix for circuit breaker state transition events.

Published by CircuitBreakerEventPublisher whenever a circuit breaker transitions
between states: CLOSED → OPEN, OPEN → HALF_OPEN, or HALF_OPEN → CLOSED.

Each event carries service_name, state, failure_count, threshold, and timestamp.

Producer: CircuitBreakerEventPublisher (MixinAsyncCircuitBreaker integrations)
Consumer: omnidash /circuit-breaker dashboard (OMN-5293)
"""

# =============================================================================
# PLATFORM DLQ AGGREGATION TOPIC SUFFIX (OMN-6136)
# =============================================================================

SUFFIX_PLATFORM_DLQ_MESSAGE: str = "onex.evt.platform.dlq-message.v1"
"""Topic suffix for DLQ aggregation events consumed by omnidash.

Published by MixinKafkaDlq as a cross-publish alongside the category-specific
DLQ topic (onex.dlq.{category}.v1).  Omnidash subscribes to this single
aggregation topic to project DLQ messages into the dlq_messages read-model table.

Producer: MixinKafkaDlq (cross-publish on each DLQ publish)
Consumer: omnidash ReadModelConsumer (platform-projections.ts)
Ticket: OMN-6136
"""

# =============================================================================
# OMNIBASE_INFRA CONSUMER HEALTH TOPIC SUFFIXES (OMN-5515 / OMN-5529)
# =============================================================================

SUFFIX_CONSUMER_HEALTH: str = "onex.evt.omnibase-infra.consumer-health.v1"
"""Topic suffix for consumer health events.

Published by ConsumerHealthEmitter when consumer lifecycle events occur
(heartbeat failures, session timeouts, rebalances, etc.).

Producer: ConsumerHealthEmitter (EventBusKafka, standalone consumers via MixinConsumerHealth)
Consumer: NodeConsumerHealthTriageEffect, omnidash /consumer-health dashboard
"""

SUFFIX_CONSUMER_RESTART_CMD: str = "onex.cmd.omnibase-infra.consumer-restart.v1"
"""Topic suffix for consumer restart commands.

Published by NodeConsumerHealthTriageEffect when graduated response
escalates to automated restart.

Producer: NodeConsumerHealthTriageEffect
Consumer: MixinConsumerHealth (standalone consumers)
"""

# =============================================================================
# OMNIBASE_INFRA RUNTIME ERROR TOPIC SUFFIXES (OMN-5517 / OMN-5529)
# =============================================================================

SUFFIX_RUNTIME_ERROR: str = "onex.evt.omnibase-infra.runtime-error.v1"
"""Topic suffix for runtime error events.

Published by RuntimeLogEventBridge (logging.Handler) when ERROR/WARNING
log records are captured from allowlisted Python loggers.

Producer: RuntimeLogEventBridge
Consumer: NodeRuntimeErrorTriageEffect, omnidash /runtime-errors dashboard
"""

# Full topic name (not a suffix) — named as such to be unambiguous.
# Used by monitor_logs.py postgres error emitter and downstream consumers.
TOPIC_DB_ERROR_V1: str = "onex.evt.omnibase-infra.db-error.v1"
"""Full topic name for PostgreSQL error events (OMN-3407).

Published by the postgres error emitter in ``scripts/monitor_logs.py`` when
a PostgreSQL ERROR log block is detected and deduplicated. Each event carries
a ``ModelDbErrorEvent`` payload (JSON-serialized).

Producer: monitor_logs.py PostgresErrorEmitter
Consumer: OMN-3406 CI UUID cast misuse validator (omniintelligence)

Note: Named ``TOPIC_DB_ERROR_V1`` (full name, not a suffix) to make it
unambiguous that callers use this string directly rather than composing it
with a tenant/namespace prefix.
"""

TOPIC_ERROR_TRIAGED_V1: str = "onex.evt.omnibase-infra.error-triaged.v1"
"""Full topic name for runtime error triage result events (OMN-5650).

Published by NodeRuntimeErrorTriageEffect after processing a runtime error
event. Each event carries a ``ModelRuntimeErrorTriageResult`` payload.

Producer: NodeRuntimeErrorTriageEffect
Consumer: omnidash /runtime-errors dashboard (OMN-5654)
"""

SUFFIX_RUNNER_HEALTH_SNAPSHOT: str = "onex.evt.omnibase-infra.runner-health-snapshot.v1"
"""Topic suffix for runner health snapshot events (OMN-6082).

Published by the runner health CLI (cli_runner_health.py) every collection
cycle. Each event carries a ``ModelRunnerHealthSnapshot`` payload.

Producer: cli_runner_health.py (cron-scheduled)
Consumer: omnidash (future)
"""

# =============================================================================
# OMNIBASE_INFRA DOMAIN TOPIC SPEC REGISTRY
# =============================================================================

ALL_OMNIBASE_INFRA_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # Gmail cleanup events (3 partitions — low-throughput, one event per run)
    ModelTopicSpec(suffix=SUFFIX_GMAIL_ARCHIVE_PURGED, partitions=3),
    # PostgreSQL error events (3 partitions — low-throughput, error-driven)
    ModelTopicSpec(suffix=TOPIC_DB_ERROR_V1, partitions=3),
    # Runtime error triage results (6 partitions — matches runtime-error partitions)
    ModelTopicSpec(suffix=TOPIC_ERROR_TRIAGED_V1, partitions=6),
    # Baselines ROI computation results (1 partition — low-throughput, per-cohort)
    ModelTopicSpec(
        suffix=SUFFIX_BASELINES_COMPUTED,
        partitions=1,
        kafka_config={
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
        },  # 7 days
    ),
    # Circuit breaker state transitions (3 partitions — low-throughput, state-change-driven)
    ModelTopicSpec(
        suffix=SUFFIX_CIRCUIT_BREAKER_STATE,
        partitions=3,
        kafka_config={
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
        },  # 7 days
    ),
    # Wiring health snapshots (3 partitions — low-throughput, per-evaluation-cycle)
    ModelTopicSpec(
        suffix=SUFFIX_WIRING_HEALTH_SNAPSHOT,
        partitions=3,
        kafka_config={
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
        },  # 7 days
    ),
    # Consumer health events (3 partitions — event-driven, OMN-5515)
    ModelTopicSpec(
        suffix=SUFFIX_CONSUMER_HEALTH,
        partitions=3,
        kafka_config={
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
        },  # 7 days
    ),
    # Consumer restart commands (1 partition — low-throughput commands, OMN-5515)
    ModelTopicSpec(
        suffix=SUFFIX_CONSUMER_RESTART_CMD,
        partitions=1,
        kafka_config={
            "retention.ms": "86400000",
            "cleanup.policy": "delete",
        },  # 1 day — commands are short-lived
    ),
    # Runtime error events (3 partitions — event-driven, OMN-5517)
    ModelTopicSpec(
        suffix=SUFFIX_RUNTIME_ERROR,
        partitions=3,
        kafka_config={
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
        },  # 7 days
    ),
    # Runner health snapshot events (1 partition — low-throughput, OMN-6082)
    ModelTopicSpec(
        suffix=SUFFIX_RUNNER_HEALTH_SNAPSHOT,
        partitions=1,
        kafka_config={},
    ),
)
"""Omnibase_infra domain topic specs for internal effect nodes.

Covers gmail cleanup events, PostgreSQL error events, and runner health
snapshots. Topics are provisioned so the correct broker topic name exists
even if no consumer is registered yet.
"""

# =============================================================================
# VALIDATION DOMAIN TOPIC SUFFIXES (cross-repo validation pipeline)
# =============================================================================
# These topics are produced by the cross-repo validation pipeline. They are
# provisioned so omnidash EventBusDataSource auto-subscribes via the onex.*
# pattern match. Previously the pipeline bypassed topic provisioning by seeding
# the DB directly (OMN-5042); registering here closes that gap.

SUFFIX_VALIDATION_CROSS_REPO_RUN_STARTED: str = (
    "onex.evt.validation.cross-repo-run-started.v1"
)
"""Topic suffix for cross-repo validation run started events.

Published at the beginning of each cross-repo validation run to signal
that a new validation cycle has started. omnidash subscribes to this topic
via the EventBusDataSource onex.* subscription pattern.

Producer: cross-repo validation pipeline (OMN-5050)
Consumer: omnidash EventBusDataSource
"""

SUFFIX_VALIDATION_CROSS_REPO_VIOLATIONS_BATCH: str = (
    "onex.evt.validation.cross-repo-violations-batch.v1"
)
"""Topic suffix for cross-repo validation violations batch events.

Published during a cross-repo validation run for each batch of violations
detected. Each message contains a set of violation records for processing
and display by downstream consumers (e.g., omnidash).

Producer: cross-repo validation pipeline (OMN-5050)
Consumer: omnidash EventBusDataSource
"""

SUFFIX_VALIDATION_CROSS_REPO_RUN_COMPLETED: str = (
    "onex.evt.validation.cross-repo-run-completed.v1"
)
"""Topic suffix for cross-repo validation run completed events.

Published at the end of each cross-repo validation run to signal that the
validation cycle has finished and all violation batches have been emitted.
omnidash subscribes to this topic via the EventBusDataSource onex.* pattern.

Producer: cross-repo validation pipeline (OMN-5050)
Consumer: omnidash EventBusDataSource
"""

# =============================================================================
# VALIDATION DOMAIN TOPIC SPEC REGISTRY
# =============================================================================

ALL_VALIDATION_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # Cross-repo run lifecycle events (3 partitions — low-throughput, one per run)
    ModelTopicSpec(suffix=SUFFIX_VALIDATION_CROSS_REPO_RUN_STARTED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_VALIDATION_CROSS_REPO_VIOLATIONS_BATCH, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_VALIDATION_CROSS_REPO_RUN_COMPLETED, partitions=3),
)
"""Validation domain topic specs for the cross-repo validation pipeline (OMN-5050).

Provisioned so omnidash EventBusDataSource auto-subscribes via the onex.*
topic catalog pattern. Previously bypassed via DB seed in OMN-5042; this
registration closes that gap and makes topic creation deterministic at
TopicProvisioner startup.
"""

# =============================================================================
# OMNICLAUDE AGENT TRACE TOPIC SUFFIXES
# =============================================================================
# Non-skill domain topics produced by omniclaude agent trace system.
# These are NOT skill lifecycle topics -- they are infrastructure-level trace
# events emitted by the agent trace pipeline (OMN-3264 et al.).
#
# Provisioned here so the broker topic exists at startup. No consumer is
# currently registered (see OMN-4572 for consumer implementation tracking).

SUFFIX_OMNICLAUDE_AGENT_TRACE_FIX_TRANSITION: str = (
    "onex.evt.omniclaude.fix-transition.v1"
)
"""Topic for agent trace fix transition events.

Published by omniclaude fix_transition.py (TopicBase.AGENT_TRACE_FIX_TRANSITION)
when a new passing ChangeFrame resolves a previously open failure within the same
trace session. Records which failure was fixed, which frames were involved, and
the diff that produced the fix.

Producer: omniclaude trace/fix_transition.py (emit_fix_transition_event)
Consumer: None currently registered (OMN-4572 — consumer implementation pending)

Note: Provisioned to guarantee topic exists on broker at startup. Data loss is
acceptable by design (non-blocking emit pattern in producer). See fix_transition.py.
"""

_OMNICLAUDE_AGENT_TRACE_TOPIC_SUFFIXES: tuple[str, ...] = (
    SUFFIX_OMNICLAUDE_AGENT_TRACE_FIX_TRANSITION,
)
"""Agent trace topic suffixes for the omniclaude trace pipeline.

Provisioned to guarantee broker topic existence for trace events emitted
by omniclaude. No consumer is currently registered for these topics.
"""

# =============================================================================
# OMNICLAUDE AGENT OBSERVABILITY TOPIC SUFFIXES
# =============================================================================
# Canonical topic names for omniclaude agent observability events consumed by
# omnibase_infra observability services. These are the source (non-DLQ) topics.
# Reference via these constants instead of raw string literals (OMN-3343).

SUFFIX_OMNICLAUDE_AGENT_ACTIONS: str = "onex.evt.omniclaude.agent-actions.v1"
"""Agent actions observability topic. Emitted by omniclaude agent hooks.

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_ROUTING_DECISION: str = "onex.evt.omniclaude.routing-decision.v1"
"""Routing decision observability topic. Emitted by omniclaude routing hooks.

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_AGENT_TRANSFORMATION: str = (
    "onex.evt.omniclaude.agent-transformation.v1"
)
"""Agent transformation observability topic. Emitted by omniclaude polymorphic agent.

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_PERFORMANCE_METRICS: str = (
    "onex.evt.omniclaude.performance-metrics.v1"
)
"""Performance metrics observability topic. Emitted by omniclaude hooks.

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_DETECTION_FAILURE: str = "onex.evt.omniclaude.detection-failure.v1"
"""Detection failure observability topic. Emitted by omniclaude hooks.

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_AGENT_EXECUTION_LOGS: str = (
    "onex.evt.omniclaude.agent-execution-logs.v1"
)
"""Agent execution logs observability topic. Emitted by omniclaude TopicBase.EXECUTION_LOGS (OMN-2902).

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_AGENT_STATUS: str = "onex.evt.omniclaude.agent-status.v1"
"""Agent status observability topic. Emitted by omniclaude TopicBase.AGENT_STATUS (OMN-2846, OMN-2903).

Consumed by omnibase_infra ServiceAgentActionsConsumer.
"""

SUFFIX_OMNICLAUDE_SKILL_STARTED: str = "onex.evt.omniclaude.skill-started.v1"
"""Skill lifecycle start topic. Emitted by omniclaude skill dispatch hooks.

Consumed by omnibase_infra ServiceSkillLifecycleConsumer.
"""

SUFFIX_OMNICLAUDE_SKILL_COMPLETED: str = "onex.evt.omniclaude.skill-completed.v1"
"""Skill lifecycle completion topic. Emitted by omniclaude skill dispatch hooks.

Consumed by omnibase_infra ServiceSkillLifecycleConsumer.
"""

# =============================================================================
# OMNICLAUDE OBSERVABILITY DLQ TOPIC SUFFIXES
# =============================================================================
# Dead letter queue topics for OmniClaude observability consumers. These are
# NOT skill topics -- they are infrastructure-level DLQ destinations for
# messages that permanently fail processing in the agent-actions consumer.
#
# Provisioned here so broker auto-creation is guaranteed at startup. Without
# provisioning, DLQ writes fail when broker auto-creation is disabled (OMN-2945).

SUFFIX_OMNICLAUDE_AGENT_ACTIONS_DLQ: str = "onex.evt.omniclaude.agent-actions-dlq.v1"
"""Dead letter queue topic for the agent-actions observability consumer.

Messages that fail validation or exceed max retry count in
ConfigAgentActionsConsumer are forwarded to this topic. Matches the
hardcoded default in
``omnibase_infra.services.observability.agent_actions.config.ConfigAgentActionsConsumer.dlq_topic``.

Producer: agent-actions consumer (ServiceAgentActionsConsumer)
Consumer: observability alerting, incident recovery tooling
"""

SUFFIX_OMNICLAUDE_AGENT_OBSERVABILITY_DLQ: str = (
    "onex.evt.omniclaude.agent-observability-dlq.v1"
)
"""Dead letter queue topic for the legacy agent-observability consumer.

Messages that fail deserialization or exceed max retry count in
the legacy ``consumers/agent_actions_consumer.py`` are forwarded to this topic.
Matches ``TopicBase.AGENT_OBSERVABILITY_DLQ`` in omniclaude (OMN-2959).

Producer: legacy agent-observability consumer (consumers/agent_actions_consumer.py)
Consumer: observability alerting, incident recovery tooling
"""

SUFFIX_OMNICLAUDE_SKILL_LIFECYCLE_DLQ: str = (
    "onex.evt.omniclaude.skill-lifecycle-dlq.v1"
)
"""Dead letter queue topic for the skill-lifecycle observability consumer.

Messages that fail validation or exceed max retry count in
ConfigSkillLifecycleConsumer are forwarded to this topic. Matches the
hardcoded default in
``omnibase_infra.services.observability.skill_lifecycle.config.ConfigSkillLifecycleConsumer.dlq_topic``.

Producer: skill-lifecycle consumer (ServiceSkillLifecycleConsumer)
Consumer: observability alerting, incident recovery tooling
"""

_OMNICLAUDE_OBSERVABILITY_DLQ_TOPIC_SUFFIXES: tuple[str, ...] = (
    SUFFIX_OMNICLAUDE_AGENT_ACTIONS_DLQ,
    SUFFIX_OMNICLAUDE_AGENT_OBSERVABILITY_DLQ,
    SUFFIX_OMNICLAUDE_SKILL_LIFECYCLE_DLQ,
)
"""DLQ topic suffixes for OmniClaude observability consumers.

These topics are provisioned separately from skill topics to make the
DLQ contract explicit and auditable via the provisioning registry.
"""

# =============================================================================
# OMNICLAUDE CONTEXT AUDIT TOPIC SUFFIXES (OMN-5240)
# =============================================================================
# Context integrity audit event topics produced by the omniclaude context
# audit pipeline (OMN-5234). Consumed by the ContextAuditConsumer
# (omnibase_infra.services.observability.context_audit) for persistence
# to PostgreSQL and enforcement tracking.

SUFFIX_OMNICLAUDE_AUDIT_DISPATCH_VALIDATED: str = (
    "onex.evt.omniclaude.audit-dispatch-validated.v1"
)
"""Topic for validated dispatch audit events.

Producer: omniclaude context audit pipeline
Consumer: ContextAuditConsumer (omnibase_infra)
"""

SUFFIX_OMNICLAUDE_AUDIT_SCOPE_VIOLATION: str = (
    "onex.evt.omniclaude.audit-scope-violation.v1"
)
"""Topic for scope violation audit events.

Producer: omniclaude context audit pipeline
Consumer: ContextAuditConsumer (omnibase_infra)
"""

SUFFIX_OMNICLAUDE_AUDIT_CONTEXT_BUDGET_EXCEEDED: str = (
    "onex.evt.omniclaude.audit-context-budget-exceeded.v1"
)
"""Topic for context budget exceeded audit events.

Producer: omniclaude context audit pipeline
Consumer: ContextAuditConsumer (omnibase_infra)
"""

SUFFIX_OMNICLAUDE_AUDIT_RETURN_BOUNDED: str = (
    "onex.evt.omniclaude.audit-return-bounded.v1"
)
"""Topic for return-bounded audit events.

Producer: omniclaude context audit pipeline
Consumer: ContextAuditConsumer (omnibase_infra)
"""

SUFFIX_OMNICLAUDE_AUDIT_COMPRESSION_TRIGGERED: str = (
    "onex.evt.omniclaude.audit-compression-triggered.v1"
)
"""Topic for compression-triggered audit events.

Producer: omniclaude context audit pipeline
Consumer: ContextAuditConsumer (omnibase_infra)
"""

SUFFIX_OMNICLAUDE_CONTEXT_AUDIT_DLQ: str = "onex.evt.omniclaude.context-audit-dlq.v1"
"""Dead letter queue topic for the context audit consumer.

Messages that fail validation or exceed max retry count in
ContextAuditConsumer are forwarded to this topic.

Producer: ContextAuditConsumer (omnibase_infra)
Consumer: observability alerting, incident recovery tooling
"""

_OMNICLAUDE_CONTEXT_AUDIT_TOPIC_SUFFIXES: tuple[str, ...] = (
    SUFFIX_OMNICLAUDE_AUDIT_DISPATCH_VALIDATED,
    SUFFIX_OMNICLAUDE_AUDIT_SCOPE_VIOLATION,
    SUFFIX_OMNICLAUDE_AUDIT_CONTEXT_BUDGET_EXCEEDED,
    SUFFIX_OMNICLAUDE_AUDIT_RETURN_BOUNDED,
    SUFFIX_OMNICLAUDE_AUDIT_COMPRESSION_TRIGGERED,
    SUFFIX_OMNICLAUDE_CONTEXT_AUDIT_DLQ,
)
"""Context audit topic suffixes for the omniclaude audit pipeline (OMN-5240).

Provisioned to guarantee broker topic existence for context integrity
audit events. Consumed by ContextAuditConsumer for PostgreSQL persistence.
"""

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
    ModelTopicSpec(suffix=SUFFIX_SERVICE_HEARTBEAT, partitions=3),
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
    ModelTopicSpec(suffix=SUFFIX_NODE_REGISTRATION_ACCEPTED, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_REGISTRATION_ACKED, partitions=6),
    # Feature flag changes (OMN-5580)
    ModelTopicSpec(
        suffix=SUFFIX_FEATURE_FLAG_CHANGED,
        partitions=1,
        kafka_config={"retention.ms": "604800000", "cleanup.policy": "delete"},
    ),
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
    # DLQ aggregation topic (OMN-6136) — consumed by omnidash /dlq dashboard
    ModelTopicSpec(suffix=SUFFIX_PLATFORM_DLQ_MESSAGE, partitions=3),
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
    # Decision recording topics (OMN-2943 — previously unprovisioned gap)
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_DECISION_RECORDED_EVT, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_DECISION_RECORDED_CMD, partitions=3),
    # Routing decision CMD topic (OMN-4299 — cross-domain cmd from omniclaude)
    ModelTopicSpec(
        suffix=SUFFIX_OMNIINTELLIGENCE_ROUTING_DECISION_CMD,
        partitions=1,
        kafka_config={
            "retention.ms": "86400000"
        },  # 1 day — command topics are short-lived
    ),
)
"""Intelligence domain topic specs provisioned for PluginIntelligence."""

# =============================================================================
# OMNIMEMORY DOMAIN TOPIC SPEC REGISTRY
# =============================================================================

ALL_OMNIMEMORY_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # --- Document crawl event topics ---
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_DISCOVERED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_CHANGED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_REMOVED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_INDEXED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_DOCUMENT_PARSE_FAILED, partitions=3),
    # --- Crawl command topics ---
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_CRAWL_TICK, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_CRAWL_REQUESTED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_RUNTIME_TICK, partitions=3),
    # --- Intent pipeline topics ---
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_INTENT_STORED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_INTENT_STORE_FAILED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_INTENT_QUERY_REQUESTED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_INTENT_QUERY_RESPONSE, partitions=3),
    # --- Memory lifecycle event topics ---
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_STORED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_RETRIEVED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_UPDATED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_DELETED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_ACCESSED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_EXPIRED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_ARCHIVED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_ARCHIVE_INITIATED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_RESTORED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_LIFECYCLE_TRANSITION_FAILED, partitions=3),
    # --- Memory lifecycle command topics ---
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_RETRIEVAL_REQUESTED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_MEMORY_RETRIEVAL_RESPONSE, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_EXPIRE_MEMORY, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_ARCHIVE_MEMORY, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_OMNIMEMORY_RESTORE_MEMORY, partitions=3),
)
"""Omnimemory domain topic specs provisioned for PluginOmnimemory.

Covers all topics declared in omnimemory contract.yaml files (OMN-2941):
  - Document crawl pipeline: discovered, changed, removed, indexed, parse-failed
  - Crawl commands: crawl-tick, crawl-requested (no current subscriber), runtime-tick
  - Intent pipeline: stored, store-failed, query-requested, query-response
  - Memory lifecycle events: stored, retrieved, updated, deleted, accessed, expired,
    archived, archive-initiated, restored, lifecycle-transition-failed
  - Memory lifecycle commands: retrieval-requested, retrieval-response, expire,
    archive, restore
"""

# =============================================================================
# OMNICLAUDE SKILL TOPIC SPEC REGISTRY
# =============================================================================

ALL_OMNICLAUDE_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # Skill orchestrator topics (1 partition -- low-throughput skill dispatch)
    *tuple(
        ModelTopicSpec(suffix=suffix, partitions=1)
        for suffix in _OMNICLAUDE_SKILL_TOPIC_SUFFIXES
    ),
    # Observability DLQ topics (3 partitions -- matches agent-actions consumer throughput)
    *tuple(
        ModelTopicSpec(suffix=suffix, partitions=3)
        for suffix in _OMNICLAUDE_OBSERVABILITY_DLQ_TOPIC_SUFFIXES
    ),
    # Agent trace topics (3 partitions -- low-throughput trace events per session)
    *tuple(
        ModelTopicSpec(suffix=suffix, partitions=3)
        for suffix in _OMNICLAUDE_AGENT_TRACE_TOPIC_SUFFIXES
    ),
    # Context audit topics (3 partitions -- OMN-5240 context integrity audit events)
    *tuple(
        ModelTopicSpec(suffix=suffix, partitions=3)
        for suffix in _OMNICLAUDE_CONTEXT_AUDIT_TOPIC_SUFFIXES
    ),
)
"""OmniClaude topic specs provisioned for skill orchestrator nodes and observability.

Skill topics: 1 partition each (low-throughput skill dispatch -- each skill
invocation is a single message). 207 topics total (68 skills x 3 topics each + 2 lifecycle topics [OMN-2934] + 1 DLQ topic [OMN-2945]).

Observability DLQ topics: 3 partitions each (matches agent-actions consumer
throughput). Provisioned to guarantee broker topic existence when auto-creation
is disabled (OMN-2945).

Agent trace topics: 3 partitions each (low-throughput trace events). Provisioned
to guarantee broker topic existence. No consumer currently registered (OMN-4572).

Source: ``omniclaude/src/omniclaude/nodes/node_skill_*/contract.yaml``
"""

# =============================================================================
# COMBINED PROVISIONED TOPIC SPECS
# =============================================================================


def _omnimemory_enabled() -> bool:
    """Return True when omnimemory connection config is present.

    Infers from OMNIMEMORY_MEMGRAPH_HOST being set (connection config presence),
    rather than requiring a separate OMNIMEMORY_ENABLED feature flag.
    Also accepts legacy OMNIMEMORY_ENABLED for backwards compatibility during
    migration.

    This function is called once at module import time to compute
    ALL_PROVISIONED_TOPIC_SPECS. It reads directly from os.environ so
    that tests can override via monkeypatch or environment manipulation.
    """
    # Primary: infer from connection config [OMN-5358]
    if os.environ.get("OMNIMEMORY_MEMGRAPH_HOST", "").strip():  # ONEX_EXCLUDE: env
        return True
    # Legacy fallback: explicit flag (to be removed after full rollout)
    return os.environ.get(
        "OMNIMEMORY_ENABLED", ""
    ).strip().lower() in {  # ONEX_EXCLUDE: env
        "1",
        "true",
        "yes",
        "on",
    }


ALL_PROVISIONED_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    ALL_PLATFORM_TOPIC_SPECS
    + ALL_INTELLIGENCE_TOPIC_SPECS
    + (ALL_OMNIMEMORY_TOPIC_SPECS if _omnimemory_enabled() else ())
    + ALL_OMNIBASE_INFRA_TOPIC_SPECS
    + ALL_VALIDATION_TOPIC_SPECS
    + ALL_OMNICLAUDE_TOPIC_SPECS
)
"""All topic specs to be provisioned by TopicProvisioner at startup.

Combines platform-reserved, domain plugin, and OmniClaude skill topic specs
into a single registry consumed by service_topic_manager.py. This is the single
source of truth for topic creation.

OmniMemory topics (ALL_OMNIMEMORY_TOPIC_SPECS) are included only when
OMNIMEMORY_ENABLED is set to a truthy value ("1", "true", "yes", "on").
When OMNIMEMORY_ENABLED is unset or falsy, omnimemory topics are not
provisioned — avoiding orphan topics on brokers where omnimemory is inactive.
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
