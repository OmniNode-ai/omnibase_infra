# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for Memgraph conflict detection and dependency chain query builders.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 10).
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.session_registry.graph_queries import (
    build_dependency_chain_query,
    build_file_conflict_query,
    build_lineage_tree_query,
)


@pytest.mark.unit
class TestBuildFileConflictQuery:
    """Tests for build_file_conflict_query."""

    def test_contains_touches_relationship(self) -> None:
        query = build_file_conflict_query("OMN-1234")
        assert "TOUCHES" in query

    def test_contains_task_id(self) -> None:
        query = build_file_conflict_query("OMN-1234")
        assert "OMN-1234" in query

    def test_filters_other_tasks(self) -> None:
        query = build_file_conflict_query("OMN-1234")
        assert "t1 <> t2" in query

    def test_filters_active_status(self) -> None:
        query = build_file_conflict_query("OMN-1234")
        assert "t2.status = 'active'" in query

    def test_returns_shared_files(self) -> None:
        query = build_file_conflict_query("OMN-1234")
        assert "shared_files" in query

    def test_returns_conflicting_task_id(self) -> None:
        query = build_file_conflict_query("OMN-1234")
        assert "t2.task_id" in query

    def test_different_task_id_produces_different_query(self) -> None:
        q1 = build_file_conflict_query("OMN-1234")
        q2 = build_file_conflict_query("OMN-5678")
        assert "OMN-1234" in q1
        assert "OMN-5678" in q2
        assert q1 != q2


@pytest.mark.unit
class TestBuildDependencyChainQuery:
    """Tests for build_dependency_chain_query."""

    def test_contains_depends_on_relationship(self) -> None:
        query = build_dependency_chain_query("OMN-1234")
        assert "DEPENDS_ON" in query

    def test_contains_variable_length_path(self) -> None:
        query = build_dependency_chain_query("OMN-1234")
        assert "*1..5" in query

    def test_contains_task_id(self) -> None:
        query = build_dependency_chain_query("OMN-1234")
        assert "OMN-1234" in query

    def test_returns_chain(self) -> None:
        query = build_dependency_chain_query("OMN-1234")
        assert "chain" in query

    def test_extracts_task_ids_from_nodes(self) -> None:
        query = build_dependency_chain_query("OMN-1234")
        assert "nodes(path)" in query
        assert "task_id" in query


@pytest.mark.unit
class TestBuildLineageTreeQuery:
    """Tests for build_lineage_tree_query."""

    def test_contains_task_id(self) -> None:
        query = build_lineage_tree_query("OMN-1234")
        assert "OMN-1234" in query

    def test_optional_match_sessions(self) -> None:
        query = build_lineage_tree_query("OMN-1234")
        assert "OPTIONAL MATCH" in query
        assert "WORKS_ON" in query

    def test_optional_match_files(self) -> None:
        query = build_lineage_tree_query("OMN-1234")
        assert "TOUCHES" in query

    def test_optional_match_pull_requests(self) -> None:
        query = build_lineage_tree_query("OMN-1234")
        assert "PRODUCED" in query
        assert "PullRequest" in query

    def test_returns_distinct_collections(self) -> None:
        query = build_lineage_tree_query("OMN-1234")
        assert "DISTINCT" in query
        assert "sessions" in query
        assert "files" in query
        assert "prs" in query
