# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for DecisionProjector.

Tests extraction of decision records from decision.recorded events
and coordination signals, plus the full projection pipeline with mocks.

Part of OMN-6850, Task 13.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.services.session_registry.decision_projector import (
    CONSUMER_GROUP,
    DECISION_RECORDED_TOPIC,
    DecisionProjector,
)


@pytest.mark.unit
class TestDecisionProjectorExtraction:
    """Tests for DecisionProjector.extract_decision()."""

    def test_extracts_decision_from_decision_recorded_event(self) -> None:
        event = {
            "event_type": "decision.recorded",
            "payload": {
                "session_id": "s1",
                "task_id": "OMN-1234",
                "decision_text": "Use approach B for retry logic",
                "rationale": "Approach A requires new dependency",
                "emitted_at": "2026-03-27T14:00:00Z",
            },
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision(event)
        assert record is not None
        assert record.decision_text == "Use approach B for retry logic"
        assert record.task_id == "OMN-1234"
        assert record.session_id == "s1"
        assert record.context == "Approach A requires new dependency"

    def test_skips_events_without_task_id(self) -> None:
        event = {
            "event_type": "decision.recorded",
            "payload": {
                "session_id": "s1",
                "task_id": None,
                "decision_text": "something",
            },
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision(event)
        assert record is None

    def test_skips_events_without_decision_text(self) -> None:
        event = {
            "event_type": "decision.recorded",
            "payload": {
                "session_id": "s1",
                "task_id": "OMN-1234",
                "decision_text": "",
            },
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision(event)
        assert record is None

    def test_extracts_from_flat_event(self) -> None:
        """Events without a nested payload dict should still work."""
        event = {
            "event_type": "decision.recorded",
            "task_id": "OMN-5678",
            "session_id": "s2",
            "decision_text": "Use Kafka for events",
            "rationale": "",
            "emitted_at": "2026-03-27T15:00:00Z",
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision(event)
        assert record is not None
        assert record.task_id == "OMN-5678"

    def test_handles_malformed_event(self) -> None:
        """Malformed events should return None, not raise."""
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision({})
        assert record is None


@pytest.mark.unit
class TestDecisionProjectorCoordination:
    """Tests for DecisionProjector.extract_decision_from_coordination()."""

    def test_pr_merged_signal(self) -> None:
        signal = {
            "signal_type": "PR_MERGED",
            "task_id": "OMN-1234",
            "session_id": "s1",
            "pr_number": 47,
            "rationale": "All tests pass, approved by reviewer",
            "emitted_at": "2026-03-27T16:00:00Z",
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision_from_coordination(signal)
        assert record is not None
        assert "merge PR #47" in record.decision_text
        assert "All tests pass" in record.decision_text
        assert record.context == "coordination:pr_merged"

    def test_ticket_completed_signal(self) -> None:
        signal = {
            "signal_type": "TICKET_COMPLETED",
            "task_id": "OMN-5678",
            "session_id": "s1",
            "summary": "Implemented retry logic with exponential backoff",
            "emitted_at": "2026-03-27T17:00:00Z",
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision_from_coordination(signal)
        assert record is not None
        assert "Completed OMN-5678" in record.decision_text
        assert "retry logic" in record.decision_text
        assert record.context == "coordination:ticket_completed"

    def test_skips_unknown_signal_type(self) -> None:
        signal = {
            "signal_type": "UNKNOWN",
            "task_id": "OMN-1234",
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision_from_coordination(signal)
        assert record is None

    def test_skips_signal_without_task_id(self) -> None:
        signal = {
            "signal_type": "PR_MERGED",
            "task_id": None,
            "pr_number": 47,
        }
        projector = DecisionProjector(embedder=None, qdrant=None)
        record = projector.extract_decision_from_coordination(signal)
        assert record is None


@pytest.mark.unit
class TestDecisionProjectorProject:
    """Tests for the full projection pipeline."""

    @pytest.mark.asyncio
    async def test_project_success(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_writer = MagicMock()

        projector = DecisionProjector(embedder=mock_embedder, qdrant=mock_writer)
        event = {
            "event_type": "decision.recorded",
            "payload": {
                "session_id": "s1",
                "task_id": "OMN-1234",
                "decision_text": "Use Postgres",
                "rationale": "Best fit for relational data",
                "emitted_at": "2026-03-27T14:00:00Z",
            },
        }
        result = await projector.project(event)
        assert result is True
        mock_embedder.embed.assert_awaited_once()
        mock_writer.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_skips_empty_event(self) -> None:
        mock_embedder = MagicMock()
        mock_writer = MagicMock()

        projector = DecisionProjector(embedder=mock_embedder, qdrant=mock_writer)
        result = await projector.project({})
        assert result is False
        mock_embedder.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_project_falls_back_to_coordination(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1024)
        mock_writer = MagicMock()

        projector = DecisionProjector(embedder=mock_embedder, qdrant=mock_writer)
        signal = {
            "signal_type": "PR_MERGED",
            "task_id": "OMN-1234",
            "session_id": "s1",
            "pr_number": 47,
            "rationale": "Tests pass",
        }
        result = await projector.project(signal)
        assert result is True
        mock_writer.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_returns_false_when_uninitialized(self) -> None:
        projector = DecisionProjector(embedder=None, qdrant=None)
        event = {
            "payload": {
                "task_id": "OMN-1234",
                "session_id": "s1",
                "decision_text": "Use Postgres",
            },
        }
        result = await projector.project(event)
        assert result is False


@pytest.mark.unit
class TestDecisionProjectorConstants:
    """Tests for module-level constants."""

    def test_consumer_group(self) -> None:
        assert CONSUMER_GROUP == "omnibase_infra.session_registry.decision_project.v1"

    def test_topic(self) -> None:
        assert (
            DECISION_RECORDED_TOPIC == "onex.evt.omniintelligence.decision-recorded.v1"
        )
