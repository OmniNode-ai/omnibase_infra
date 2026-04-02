# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for agent learning record construction.

Tests that learning records can be built from realistic session data
with correct repo extraction, ticket parsing, and task classification.
Full write -> store -> retrieve cycle requires Postgres + Qdrant (deferred to runtime tests).
"""

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from omnibase_infra.models.agent_learning.enum_learning_task_type import (
    EnumLearningTaskType,
)
from omnibase_infra.models.agent_learning.model_agent_learning import (
    ModelAgentLearning,
)
from omnibase_infra.services.agent_learning_extraction.consumer import (
    build_learning_record,
)


@pytest.mark.integration
class TestAgentLearningE2E:
    def test_build_and_validate_record(self) -> None:
        """Test that a learning record can be built from session data."""
        record = build_learning_record(
            session_id=uuid4(),
            working_dir="/Volumes/PRO-G40/Code/omni_home/omnibase_infra",
            branch="jonah/omn-7200-fix-migration-backfill",
            resolution_summary=(
                "The registration_projections table required a backfill step after "
                "adding the node_family column. Added a migration that updates "
                "existing rows with the default value."
            ),
            file_paths=[
                "docker/migrations/forward/055_add_node_family.sql",
                "docker/migrations/forward/056_backfill_node_family.sql",
            ],
            error_signatures=[
                'psycopg2.errors.NotNullViolation: null value in column "node_family"',
            ],
            created_at=datetime.now(tz=UTC),
        )

        assert record.repo == "omnibase_infra"
        assert record.ticket_id == "OMN-7200"
        assert record.task_type == EnumLearningTaskType.MIGRATION
        assert len(record.error_signatures) == 1
        assert len(record.file_paths_touched) == 2
        assert "backfill" in record.resolution_summary
