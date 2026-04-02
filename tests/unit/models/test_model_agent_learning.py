# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for agent learning record models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.models.agent_learning import (
    EnumLearningMatchType,
    EnumLearningTaskType,
    ModelAgentLearning,
    ModelAgentLearningQuery,
)


@pytest.mark.unit
class TestModelAgentLearning:
    def test_create_minimal(self) -> None:
        learning = ModelAgentLearning(
            session_id=uuid4(),
            repo="omnibase_infra",
            resolution_summary="Fixed CI by excluding generated files from ruff.",
            created_at=datetime.now(tz=UTC),
        )
        assert learning.repo == "omnibase_infra"
        assert learning.task_type == EnumLearningTaskType.UNKNOWN
        assert learning.confidence == 0.8
        assert learning.access_count == 0

    def test_create_full(self) -> None:
        learning = ModelAgentLearning(
            session_id=uuid4(),
            repo="omnidash",
            file_paths_touched=["src/app/api/route.ts", "src/lib/db.ts"],
            error_signatures=["TypeError: Cannot read properties of undefined"],
            resolution_summary="The API route was missing the async keyword.",
            ticket_id="OMN-7100",
            task_type=EnumLearningTaskType.CI_FIX,
            confidence=0.95,
            created_at=datetime.now(tz=UTC),
        )
        assert learning.file_paths_touched == ("src/app/api/route.ts", "src/lib/db.ts")
        assert learning.error_signatures == (
            "TypeError: Cannot read properties of undefined",
        )
        assert learning.ticket_id == "OMN-7100"

    def test_frozen(self) -> None:
        learning = ModelAgentLearning(
            session_id=uuid4(),
            repo="omnibase_core",
            resolution_summary="Test.",
            created_at=datetime.now(tz=UTC),
        )
        with pytest.raises(Exception):
            learning.repo = "other"  # type: ignore[misc]

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelAgentLearning(
                session_id=uuid4(),
                repo="x",
                resolution_summary="x",
                confidence=1.5,
                created_at=datetime.now(tz=UTC),
            )


@pytest.mark.unit
class TestModelAgentLearningQuery:
    def test_error_query(self) -> None:
        query = ModelAgentLearningQuery(
            match_type=EnumLearningMatchType.ERROR_SIGNATURE,
            error_text="ModuleNotFoundError: No module named 'omnibase_core.models.plan'",
            repo="omniclaude",
        )
        assert query.match_type == EnumLearningMatchType.ERROR_SIGNATURE
        assert query.error_text is not None

    def test_context_query(self) -> None:
        query = ModelAgentLearningQuery(
            match_type=EnumLearningMatchType.TASK_CONTEXT,
            repo="omnibase_infra",
            file_paths=["docker/migrations/forward/"],
            task_type=EnumLearningTaskType.MIGRATION,
        )
        assert query.match_type == EnumLearningMatchType.TASK_CONTEXT

    def test_auto_query(self) -> None:
        query = ModelAgentLearningQuery(
            match_type=EnumLearningMatchType.AUTO,
            error_text="ImportError: cannot import name 'Foo'",
            repo="omnibase_core",
        )
        assert query.match_type == EnumLearningMatchType.AUTO
