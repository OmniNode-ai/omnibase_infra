# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consumer that projects decision events into Qdrant.

Subscribes to: onex.evt.omniintelligence.decision-recorded.v1
Consumer group: omnibase_infra.session_registry.decision_project.v1

For each event:
  1. Extract ModelDecisionRecord from payload
  2. Build embedding text via build_embedding_text()
  3. Call EmbeddingClient to get 1024-dim vector
  4. Upsert into Qdrant "session_decisions" collection

Also indexes decisions from coordination signals:
  - PR_MERGED with rationale -> "Decided to merge PR #N: {rationale}"
  - TICKET_COMPLETED with summary -> "Completed OMN-XXXX: {summary}"

Part of the Multi-Session Coordination Layer (OMN-6850, Task 13).

Architecture:
    Kafka (onex.evt.omniintelligence.decision-recorded.v1)
           |
           v
    DecisionProjector.extract_decision()
           |
           v
    EmbeddingClient.embed() -> 1024-dim vector
           |
           v
    DecisionEmbedderWriter.upsert() -> Qdrant

Design decisions:
    - Pure function extraction: extract_decision() and
      extract_decision_from_coordination() are pure functions that
      return ModelDecisionRecord from raw event dicts. This enables
      unit testing without Kafka, Qdrant, or embedding dependencies.
    - Events without task_id are silently skipped.
    - Coordination signals (PR_MERGED, TICKET_COMPLETED) are converted
      into synthetic decision records for semantic recall.
"""

from __future__ import annotations

import logging

from omnibase_infra.services.session_registry.decision_embedder import (
    DecisionEmbedderWriter,
    EmbeddingClient,
    ModelDecisionRecord,
    build_embedding_text,
)
from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_INTELLIGENCE_DECISION_RECORDED_EVT,
)

logger = logging.getLogger(__name__)

# Consumer group for the decision projector.
CONSUMER_GROUP = "omnibase_infra.session_registry.decision_project.v1"

# Kafka topic for decision recorded events (from canonical topic registry).
DECISION_RECORDED_TOPIC = SUFFIX_INTELLIGENCE_DECISION_RECORDED_EVT  # onex-topic-allow: pending contract auto-wiring

# Coordination signal types that produce synthetic decisions.
_COORDINATION_SIGNAL_TYPES = frozenset({"PR_MERGED", "TICKET_COMPLETED"})


class DecisionProjector:
    """Projects decision events from Kafka into Qdrant via embeddings.

    The projector has two extraction paths:

    1. ``extract_decision()`` -- extracts from decision.recorded events
    2. ``extract_decision_from_coordination()`` -- extracts from
       coordination signals (PR_MERGED, TICKET_COMPLETED)

    Both return ModelDecisionRecord or None (skip).

    The ``project()`` method orchestrates the full pipeline:
    extract -> embed -> upsert.
    """

    def __init__(
        self,
        embedder: EmbeddingClient | None,
        qdrant: DecisionEmbedderWriter | None,
    ) -> None:
        self._embedder = embedder
        self._qdrant = qdrant

    def extract_decision(
        self,
        event: dict[str, object],
    ) -> ModelDecisionRecord | None:
        """Extract a decision record from a decision.recorded event.

        Returns None if the event has no task_id or is malformed.
        Never raises -- all errors are logged and swallowed to preserve
        projector liveness.

        Args:
            event: Raw deserialized Kafka event dict.

        Returns:
            A ModelDecisionRecord, or None if the event should be skipped.
        """
        try:
            payload = event.get("payload")
            if isinstance(payload, dict):
                source = payload
            else:
                source = event

            task_id = source.get("task_id")
            if not task_id or not isinstance(task_id, str):
                return None

            decision_text = source.get("decision_text", "")
            if not isinstance(decision_text, str) or not decision_text:
                return None

            session_id = str(source.get("session_id", "unknown"))
            rationale = source.get("rationale", "")
            context = str(rationale) if rationale else ""
            emitted_at = str(source.get("emitted_at", ""))

            return ModelDecisionRecord(
                task_id=task_id,
                session_id=session_id,
                decision_text=decision_text,
                context=context,
                timestamp=emitted_at,
            )

        except Exception:
            logger.exception("Failed to extract decision from event")
            return None

    def extract_decision_from_coordination(
        self,
        signal: dict[str, object],
    ) -> ModelDecisionRecord | None:
        """Extract a synthetic decision from a coordination signal.

        Coordination signals like PR_MERGED and TICKET_COMPLETED carry
        implicit decisions that are useful for semantic recall.

        Formats:
            PR_MERGED: "Decided to merge PR #{pr_number}: {rationale}"
            TICKET_COMPLETED: "Completed {task_id}: {summary}"

        Args:
            signal: Raw coordination signal dict with signal_type, task_id, etc.

        Returns:
            A ModelDecisionRecord, or None if the signal should be skipped.
        """
        try:
            signal_type = signal.get("signal_type")
            if not isinstance(signal_type, str):
                return None
            if signal_type not in _COORDINATION_SIGNAL_TYPES:
                return None

            task_id = signal.get("task_id")
            if not task_id or not isinstance(task_id, str):
                return None

            session_id = str(signal.get("session_id", "unknown"))
            emitted_at = str(signal.get("emitted_at", ""))

            if signal_type == "PR_MERGED":
                pr_number = signal.get("pr_number", "?")
                rationale = signal.get("rationale", "")
                decision_text = f"Decided to merge PR #{pr_number}: {rationale}"
                context = "coordination:pr_merged"
            elif signal_type == "TICKET_COMPLETED":
                summary = signal.get("summary", "")
                decision_text = f"Completed {task_id}: {summary}"
                context = "coordination:ticket_completed"
            else:
                return None

            return ModelDecisionRecord(
                task_id=task_id,
                session_id=session_id,
                decision_text=decision_text,
                context=context,
                timestamp=emitted_at,
            )

        except Exception:
            logger.exception("Failed to extract decision from coordination signal")
            return None

    async def project(self, event: dict[str, object]) -> bool:
        """Full projection pipeline: extract -> embed -> upsert.

        Args:
            event: Raw deserialized Kafka event dict.

        Returns:
            True if the event was successfully projected, False if skipped.
        """
        if self._embedder is None or self._qdrant is None:
            logger.warning("DecisionProjector not fully initialized (embedder/qdrant)")
            return False

        # Try decision.recorded extraction first.
        record = self.extract_decision(event)

        # Fall back to coordination signal extraction.
        if record is None:
            record = self.extract_decision_from_coordination(event)

        if record is None:
            return False

        try:
            text = build_embedding_text(record)
            vector = await self._embedder.embed(text)
            self._qdrant.upsert(record, vector)
            logger.info(
                "Projected decision for task=%s text=%s",
                record.task_id,
                record.decision_text[:50],
            )
            return True
        except Exception:
            logger.exception(
                "Failed to project decision for task=%s",
                record.task_id,
            )
            return False


__all__ = [
    "CONSUMER_GROUP",
    "DECISION_RECORDED_TOPIC",
    "DecisionProjector",
]
