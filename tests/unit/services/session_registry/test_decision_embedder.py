# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for decision embedding models and utilities.

Tests ModelDecisionRecord creation, embedding text generation,
and deterministic point ID generation.

Part of OMN-6850, Task 12/13.
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.session_registry.decision_embedder import (
    ModelDecisionRecord,
    build_embedding_text,
    decision_point_id,
)


@pytest.mark.unit
class TestModelDecisionRecord:
    """Tests for ModelDecisionRecord Pydantic model."""

    def test_decision_record_creation(self) -> None:
        record = ModelDecisionRecord(
            task_id="OMN-1234",
            session_id="session-abc",
            decision_text="Chose approach B: extend ModelFoo rather than creating ModelBar",
            context="Working on auth middleware refactor",
        )
        assert record.task_id == "OMN-1234"
        assert record.session_id == "session-abc"
        assert "approach B" in record.decision_text

    def test_decision_record_minimal(self) -> None:
        record = ModelDecisionRecord(
            task_id="OMN-5678",
            session_id="s1",
            decision_text="Use Postgres",
        )
        assert record.context == ""
        assert record.timestamp == ""

    def test_decision_record_frozen(self) -> None:
        record = ModelDecisionRecord(
            task_id="OMN-1234",
            session_id="s1",
            decision_text="Use Postgres",
        )
        with pytest.raises(Exception):
            record.task_id = "OMN-9999"  # type: ignore[misc]


@pytest.mark.unit
class TestBuildEmbeddingText:
    """Tests for build_embedding_text utility."""

    def test_embedding_text_includes_context(self) -> None:
        record = ModelDecisionRecord(
            task_id="OMN-1234",
            session_id="s1",
            decision_text="Use Postgres for session state",
            context="Multi-session coordination design",
        )
        text = build_embedding_text(record)
        assert "Use Postgres" in text
        assert "Multi-session" in text
        assert "OMN-1234" in text

    def test_embedding_text_without_context(self) -> None:
        record = ModelDecisionRecord(
            task_id="OMN-5678",
            session_id="s1",
            decision_text="Use Kafka for events",
        )
        text = build_embedding_text(record)
        assert "Use Kafka" in text
        assert "OMN-5678" in text
        assert "Context" not in text

    def test_embedding_text_format(self) -> None:
        record = ModelDecisionRecord(
            task_id="OMN-1",
            session_id="s1",
            decision_text="decision A",
            context="ctx B",
        )
        text = build_embedding_text(record)
        assert text == "Task OMN-1: decision A. Context: ctx B"


@pytest.mark.unit
class TestDecisionPointId:
    """Tests for deterministic point ID generation."""

    def test_deterministic(self) -> None:
        id1 = decision_point_id("OMN-1234", "Use Postgres")
        id2 = decision_point_id("OMN-1234", "Use Postgres")
        assert id1 == id2

    def test_different_text_different_id(self) -> None:
        id1 = decision_point_id("OMN-1234", "Use Postgres")
        id2 = decision_point_id("OMN-1234", "Use Redis")
        assert id1 != id2

    def test_different_task_different_id(self) -> None:
        id1 = decision_point_id("OMN-1234", "Use Postgres")
        id2 = decision_point_id("OMN-5678", "Use Postgres")
        assert id1 != id2
