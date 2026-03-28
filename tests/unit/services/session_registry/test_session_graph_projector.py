# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for session graph projector (Kafka -> Memgraph).

Validates the pure build_graph_mutations() function and ModelConfigGraphProjector.
No Kafka or Memgraph connections required — all tests are unit-level.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 9).
"""

from __future__ import annotations

import pytest

from omnibase_infra.services.session_registry.enum_node_label import EnumNodeLabel
from omnibase_infra.services.session_registry.enum_relationship_type import (
    EnumRelationshipType,
)
from omnibase_infra.services.session_registry.graph_projector import (
    _extract_repo,
    build_graph_mutations,
)
from omnibase_infra.services.session_registry.model_config_graph_projector import (
    ModelConfigGraphProjector,
)
from omnibase_infra.services.session_registry.model_graph_mutation import (
    ModelGraphMutation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_event() -> dict[str, object]:
    """Minimal event with task_id and session_id."""
    return {
        "event_type": "prompt.submitted",
        "task_id": "OMN-1234",
        "session_id": "sess-abc",
    }


@pytest.fixture
def tool_executed_event() -> dict[str, object]:
    """tool.executed event with file_path."""
    return {
        "event_type": "tool.executed",
        "task_id": "OMN-1234",
        "session_id": "sess-abc",
        "file_path": "/Volumes/PRO-G40/Code/omni_home/omnibase_infra/src/foo.py",
    }


@pytest.fixture
def pr_merged_event() -> dict[str, object]:
    """coordination.signal pr_merged event."""
    return {
        "event_type": "coordination.signal",
        "signal_type": "pr_merged",
        "task_id": "OMN-1234",
        "session_id": "sess-abc",
        "pr_number": 42,
        "repo": "omnibase_infra",
    }


@pytest.fixture
def rebase_needed_event() -> dict[str, object]:
    """coordination.signal rebase_needed event."""
    return {
        "event_type": "coordination.signal",
        "signal_type": "rebase_needed",
        "task_id": "OMN-1234",
        "session_id": "sess-abc",
        "related_task_id": "OMN-5678",
    }


# ===================================================================
# Tests: build_graph_mutations — skip conditions
# ===================================================================


class TestBuildGraphMutationsSkip:
    """Events that should produce no mutations."""

    @pytest.mark.unit
    def test_event_without_task_id_returns_empty(self) -> None:
        event = {"event_type": "prompt.submitted", "session_id": "sess-abc"}
        assert build_graph_mutations(event) == []

    @pytest.mark.unit
    def test_event_without_session_id_returns_empty(self) -> None:
        event = {"event_type": "prompt.submitted", "task_id": "OMN-1234"}
        assert build_graph_mutations(event) == []

    @pytest.mark.unit
    def test_empty_event_returns_empty(self) -> None:
        assert build_graph_mutations({}) == []

    @pytest.mark.unit
    def test_none_task_id_returns_empty(self) -> None:
        event = {
            "event_type": "tool.executed",
            "task_id": None,
            "session_id": "sess-abc",
        }
        assert build_graph_mutations(event) == []


# ===================================================================
# Tests: build_graph_mutations — base Session/Task/WORKS_ON
# ===================================================================


class TestBuildGraphMutationsBase:
    """All events with task_id + session_id produce base mutations."""

    @pytest.mark.unit
    def test_base_mutation_always_present(self, base_event: dict[str, object]) -> None:
        mutations = build_graph_mutations(base_event)
        assert len(mutations) >= 1
        base = mutations[0]
        assert EnumNodeLabel.SESSION in base.cypher
        assert EnumNodeLabel.TASK in base.cypher
        assert EnumRelationshipType.WORKS_ON in base.cypher
        assert base.params["session_id"] == "sess-abc"
        assert base.params["task_id"] == "OMN-1234"

    @pytest.mark.unit
    def test_base_mutation_is_merge(self, base_event: dict[str, object]) -> None:
        mutations = build_graph_mutations(base_event)
        assert all("MERGE" in m.cypher for m in mutations)

    @pytest.mark.unit
    def test_unknown_event_type_produces_only_base(self) -> None:
        event = {
            "event_type": "unknown.event",
            "task_id": "OMN-1234",
            "session_id": "sess-abc",
        }
        mutations = build_graph_mutations(event)
        assert len(mutations) == 1


# ===================================================================
# Tests: build_graph_mutations — tool.executed
# ===================================================================


class TestBuildGraphMutationsToolExecuted:
    """tool.executed events produce File and TOUCHES mutations."""

    @pytest.mark.unit
    def test_file_node_and_touches_edge(
        self, tool_executed_event: dict[str, object]
    ) -> None:
        mutations = build_graph_mutations(tool_executed_event)
        # base + TOUCHES + BELONGS_TO (file->repo)
        assert len(mutations) == 3

        touches = mutations[1]
        assert EnumNodeLabel.FILE in touches.cypher
        assert EnumRelationshipType.TOUCHES in touches.cypher
        assert touches.params["path"] == tool_executed_event["file_path"]

    @pytest.mark.unit
    def test_repo_extracted_and_belongs_to(
        self, tool_executed_event: dict[str, object]
    ) -> None:
        mutations = build_graph_mutations(tool_executed_event)
        belongs = mutations[2]
        assert EnumNodeLabel.REPOSITORY in belongs.cypher
        assert EnumRelationshipType.BELONGS_TO in belongs.cypher
        assert belongs.params["repo"] == "omnibase_infra"

    @pytest.mark.unit
    def test_tool_executed_without_file_path(self) -> None:
        event = {
            "event_type": "tool.executed",
            "task_id": "OMN-1234",
            "session_id": "sess-abc",
        }
        mutations = build_graph_mutations(event)
        # Only base mutation
        assert len(mutations) == 1

    @pytest.mark.unit
    def test_tool_executed_unrecognized_path_no_repo(self) -> None:
        event = {
            "event_type": "tool.executed",
            "task_id": "OMN-1234",
            "session_id": "sess-abc",
            "file_path": "/var/data/scratch/foo.py",
        }
        mutations = build_graph_mutations(event)
        # base + TOUCHES (no BELONGS_TO since repo can't be extracted)
        assert len(mutations) == 2


# ===================================================================
# Tests: build_graph_mutations — coordination.signal pr_merged
# ===================================================================


class TestBuildGraphMutationsPrMerged:
    """coordination.signal pr_merged produces PR node and PRODUCED edge."""

    @pytest.mark.unit
    def test_pr_node_and_produced_edge(
        self, pr_merged_event: dict[str, object]
    ) -> None:
        mutations = build_graph_mutations(pr_merged_event)
        # base + PRODUCED + PR BELONGS_TO repo
        assert len(mutations) == 3

        produced = mutations[1]
        assert EnumNodeLabel.PULL_REQUEST in produced.cypher
        assert EnumRelationshipType.PRODUCED in produced.cypher
        assert produced.params["pr_id"] == "omnibase_infra#42"

    @pytest.mark.unit
    def test_pr_belongs_to_repo(self, pr_merged_event: dict[str, object]) -> None:
        mutations = build_graph_mutations(pr_merged_event)
        belongs = mutations[2]
        assert EnumNodeLabel.REPOSITORY in belongs.cypher
        assert EnumRelationshipType.BELONGS_TO in belongs.cypher
        assert belongs.params["repo"] == "omnibase_infra"

    @pytest.mark.unit
    def test_pr_merged_without_repo(self) -> None:
        event = {
            "event_type": "coordination.signal",
            "signal_type": "pr_merged",
            "task_id": "OMN-1234",
            "session_id": "sess-abc",
            "pr_number": 99,
        }
        mutations = build_graph_mutations(event)
        # base + PRODUCED (no BELONGS_TO without repo)
        assert len(mutations) == 2
        assert mutations[1].params["pr_id"] == "99"

    @pytest.mark.unit
    def test_pr_merged_without_pr_number(self) -> None:
        event = {
            "event_type": "coordination.signal",
            "signal_type": "pr_merged",
            "task_id": "OMN-1234",
            "session_id": "sess-abc",
        }
        mutations = build_graph_mutations(event)
        # Only base
        assert len(mutations) == 1


# ===================================================================
# Tests: build_graph_mutations — coordination.signal rebase_needed
# ===================================================================


class TestBuildGraphMutationsRebaseNeeded:
    """coordination.signal rebase_needed produces DEPENDS_ON edge."""

    @pytest.mark.unit
    def test_depends_on_edge(self, rebase_needed_event: dict[str, object]) -> None:
        mutations = build_graph_mutations(rebase_needed_event)
        # base + DEPENDS_ON
        assert len(mutations) == 2

        dep = mutations[1]
        assert EnumRelationshipType.DEPENDS_ON in dep.cypher
        assert dep.params["task_id"] == "OMN-1234"
        assert dep.params["related_task_id"] == "OMN-5678"

    @pytest.mark.unit
    def test_rebase_needed_without_related_task_id(self) -> None:
        event = {
            "event_type": "coordination.signal",
            "signal_type": "rebase_needed",
            "task_id": "OMN-1234",
            "session_id": "sess-abc",
        }
        mutations = build_graph_mutations(event)
        # Only base
        assert len(mutations) == 1


# ===================================================================
# Tests: envelope wrapping
# ===================================================================


class TestBuildGraphMutationsEnvelope:
    """Events wrapped in an envelope with payload key."""

    @pytest.mark.unit
    def test_envelope_wrapped_event(self) -> None:
        envelope = {
            "event_type": "hook.event",
            "payload": {
                "event_type": "tool.executed",
                "task_id": "OMN-1234",
                "session_id": "sess-abc",
                "file_path": "/Volumes/PRO-G40/Code/omni_worktrees/OMN-1234/omniclaude/src/main.py",
            },
        }
        mutations = build_graph_mutations(envelope)
        # base + TOUCHES + BELONGS_TO
        assert len(mutations) == 3
        assert mutations[2].params["repo"] == "omniclaude"


# ===================================================================
# Tests: _extract_repo
# ===================================================================


class TestExtractRepo:
    """Repository name extraction from file paths."""

    @pytest.mark.unit
    def test_omni_home_path(self) -> None:
        path = "/Volumes/PRO-G40/Code/omni_home/omnibase_infra/src/foo.py"
        assert _extract_repo(path) == "omnibase_infra"

    @pytest.mark.unit
    def test_omni_worktrees_path(self) -> None:
        path = "/Volumes/PRO-G40/Code/omni_worktrees/OMN-1234/omniclaude/src/bar.py"
        assert _extract_repo(path) == "omniclaude"

    @pytest.mark.unit
    def test_unrecognized_path(self) -> None:
        assert _extract_repo("/var/data/scratch/foo.py") is None

    @pytest.mark.unit
    def test_empty_string(self) -> None:
        assert _extract_repo("") is None


# ===================================================================
# Tests: ModelConfigGraphProjector
# ===================================================================


class TestModelConfigGraphProjector:
    """Configuration model for the graph projector."""

    @pytest.mark.unit
    def test_defaults(self) -> None:
        config = ModelConfigGraphProjector()
        assert config.bootstrap_servers == "localhost:19092"
        assert (
            config.consumer_group == "omnibase_infra.session_registry.graph_project.v1"
        )
        assert config.topic_pattern == r"onex\.evt\.omniclaude\..*"

    @pytest.mark.unit
    def test_bolt_uri(self) -> None:
        config = ModelConfigGraphProjector(memgraph_host="memgraph", memgraph_port=7688)
        assert config.bolt_uri == "bolt://memgraph:7688"

    @pytest.mark.unit
    def test_frozen(self) -> None:
        config = ModelConfigGraphProjector()
        with pytest.raises(Exception):
            config.bootstrap_servers = "other:9092"  # type: ignore[misc]

    @pytest.mark.unit
    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelConfigGraphProjector(unknown_field="value")  # type: ignore[call-arg]


# ===================================================================
# Tests: ModelGraphMutation
# ===================================================================


class TestModelGraphMutation:
    """ModelGraphMutation dataclass."""

    @pytest.mark.unit
    def test_frozen(self) -> None:
        m = ModelGraphMutation(cypher="MERGE (n:Foo {id: $id})", params={"id": "1"})
        with pytest.raises(AttributeError):
            m.cypher = "other"  # type: ignore[misc]

    @pytest.mark.unit
    def test_default_params(self) -> None:
        m = ModelGraphMutation(cypher="RETURN 1")
        assert m.params == {}


# ===================================================================
# Tests: idempotency (Doctrine D3)
# ===================================================================


class TestIdempotency:
    """Verify all mutations use MERGE (never CREATE) for replay safety."""

    @pytest.mark.unit
    def test_all_mutations_are_merge(
        self,
        base_event: dict[str, object],
        tool_executed_event: dict[str, object],
        pr_merged_event: dict[str, object],
        rebase_needed_event: dict[str, object],
    ) -> None:
        for event in [
            base_event,
            tool_executed_event,
            pr_merged_event,
            rebase_needed_event,
        ]:
            mutations = build_graph_mutations(event)
            for m in mutations:
                assert "MERGE" in m.cypher, f"Non-MERGE mutation: {m.cypher}"
                assert "CREATE" not in m.cypher, f"CREATE found: {m.cypher}"

    @pytest.mark.unit
    def test_duplicate_events_produce_same_mutations(
        self, tool_executed_event: dict[str, object]
    ) -> None:
        """Replaying the same event produces identical mutations."""
        first = build_graph_mutations(tool_executed_event)
        second = build_graph_mutations(tool_executed_event)
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=False):
            assert a.cypher == b.cypher
            assert a.params == b.params
