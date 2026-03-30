# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for session graph schema enums and Cypher DDL.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 8).
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.session_registry.graph_schema import (
    SESSION_GRAPH_SCHEMA,
    EnumNodeLabel,
    EnumRelationshipType,
)


@pytest.mark.unit
class TestEnumNodeLabel:
    """EnumNodeLabel must contain exactly the expected graph node labels."""

    def test_member_count(self) -> None:
        assert len(EnumNodeLabel) == 5

    def test_expected_members(self) -> None:
        expected = {"SESSION", "TASK", "FILE", "PULL_REQUEST", "REPOSITORY"}
        assert set(EnumNodeLabel.__members__.keys()) == expected

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (EnumNodeLabel.SESSION, "Session"),
            (EnumNodeLabel.TASK, "Task"),
            (EnumNodeLabel.FILE, "File"),
            (EnumNodeLabel.PULL_REQUEST, "PullRequest"),
            (EnumNodeLabel.REPOSITORY, "Repository"),
        ],
    )
    def test_values(self, member: EnumNodeLabel, value: str) -> None:
        assert member.value == value

    def test_is_str_enum(self) -> None:
        for member in EnumNodeLabel:
            assert isinstance(member, str)


@pytest.mark.unit
class TestEnumRelationshipType:
    """EnumRelationshipType must contain exactly the expected relationship types."""

    def test_member_count(self) -> None:
        assert len(EnumRelationshipType) == 5

    def test_expected_members(self) -> None:
        expected = {"WORKS_ON", "TOUCHES", "DEPENDS_ON", "PRODUCED", "BELONGS_TO"}
        assert set(EnumRelationshipType.__members__.keys()) == expected

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (EnumRelationshipType.WORKS_ON, "WORKS_ON"),
            (EnumRelationshipType.TOUCHES, "TOUCHES"),
            (EnumRelationshipType.DEPENDS_ON, "DEPENDS_ON"),
            (EnumRelationshipType.PRODUCED, "PRODUCED"),
            (EnumRelationshipType.BELONGS_TO, "BELONGS_TO"),
        ],
    )
    def test_values(self, member: EnumRelationshipType, value: str) -> None:
        assert member.value == value

    def test_is_str_enum(self) -> None:
        for member in EnumRelationshipType:
            assert isinstance(member, str)


@pytest.mark.unit
class TestSessionGraphSchema:
    """SESSION_GRAPH_SCHEMA DDL string must contain all constraints and indexes."""

    def test_is_nonempty_string(self) -> None:
        assert isinstance(SESSION_GRAPH_SCHEMA, str)
        assert len(SESSION_GRAPH_SCHEMA) > 0

    @pytest.mark.parametrize(
        "label",
        ["Session", "Task", "File", "PullRequest", "Repository"],
    )
    def test_constraint_for_each_node_label(self, label: str) -> None:
        assert f":{label})" in SESSION_GRAPH_SCHEMA
        assert "CREATE CONSTRAINT" in SESSION_GRAPH_SCHEMA

    def test_uniqueness_constraint_count(self) -> None:
        count = SESSION_GRAPH_SCHEMA.count("CREATE CONSTRAINT")
        assert count == 5

    def test_index_on_task_status(self) -> None:
        assert "CREATE INDEX ON :Task(status)" in SESSION_GRAPH_SCHEMA

    def test_index_on_session_last_activity(self) -> None:
        assert "CREATE INDEX ON :Session(last_activity)" in SESSION_GRAPH_SCHEMA

    def test_schema_matches_migration(self) -> None:
        """Schema string must match the Cypher migration file content (DDL lines only)."""
        from pathlib import Path

        migration_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "omnibase_infra"
            / "migrations"
            / "memgraph"
            / "session_graph_001.cypher"
        )
        if not migration_path.exists():
            pytest.skip("Migration file not found at expected path")

        migration_text = migration_path.read_text()
        # Extract DDL lines (non-comment, non-empty) from migration
        migration_ddl_lines = [
            line.strip()
            for line in migration_text.splitlines()
            if line.strip() and not line.strip().startswith("//")
        ]
        schema_ddl_lines = [
            line.strip() for line in SESSION_GRAPH_SCHEMA.splitlines() if line.strip()
        ]
        assert migration_ddl_lines == schema_ddl_lines
