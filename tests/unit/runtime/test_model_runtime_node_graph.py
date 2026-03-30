# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ModelRuntimeNodeGraph declarative graph definition (OMN-6306)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_runtime_node_graph import (
    ModelNodeConfig,
    ModelNodeEdge,
    ModelRuntimeNodeGraph,
)


@pytest.mark.unit
class TestModelNodeConfig:
    """Tests for ModelNodeConfig."""

    def test_minimal_node(self) -> None:
        """Node with only required fields."""
        node = ModelNodeConfig(
            name="error-triage",
            handler_class="omnibase_infra.nodes.HandlerRuntimeErrorTriage",
        )
        assert node.name == "error-triage"
        assert node.subscribe_topics == ()
        assert node.publish_topics == ()
        assert node.enabled is True

    def test_full_node(self) -> None:
        """Node with all fields populated."""
        node = ModelNodeConfig(
            name="error-triage",
            handler_class="omnibase_infra.nodes.HandlerRuntimeErrorTriage",
            subscribe_topics=("onex.evt.omnibase-infra.runtime-error.v1",),
            publish_topics=("onex.evt.omnibase-infra.runtime-error-triage-result.v1",),
            enabled=False,
        )
        assert len(node.subscribe_topics) == 1
        assert len(node.publish_topics) == 1
        assert node.enabled is False

    def test_frozen(self) -> None:
        """ModelNodeConfig is immutable."""
        node = ModelNodeConfig(name="test", handler_class="some.Handler")
        with pytest.raises(ValidationError):
            node.name = "changed"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelNodeConfig(
                name="test",
                handler_class="some.Handler",
                unexpected="value",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestModelNodeEdge:
    """Tests for ModelNodeEdge."""

    def test_edge_creation(self) -> None:
        """Edge with source and target."""
        edge = ModelNodeEdge(source="postgres", target="error-triage")
        assert edge.source == "postgres"
        assert edge.target == "error-triage"

    def test_frozen(self) -> None:
        """ModelNodeEdge is immutable."""
        edge = ModelNodeEdge(source="a", target="b")
        with pytest.raises(ValidationError):
            edge.source = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestModelRuntimeNodeGraph:
    """Tests for ModelRuntimeNodeGraph."""

    def test_minimal_graph(self) -> None:
        """Graph with a single node, no edges."""
        node = ModelNodeConfig(
            name="registration",
            handler_class="omnibase_infra.nodes.PluginRegistration",
        )
        graph = ModelRuntimeNodeGraph(
            nodes=(node,),
            bootstrap_order=("registration",),
        )
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0
        assert graph.bootstrap_order == ("registration",)

    def test_graph_with_edges(self) -> None:
        """Graph with nodes and dependency edges."""
        nodes = (
            ModelNodeConfig(name="db", handler_class="db.Handler"),
            ModelNodeConfig(name="triage", handler_class="triage.Handler"),
        )
        edges = (ModelNodeEdge(source="db", target="triage"),)
        graph = ModelRuntimeNodeGraph(
            nodes=nodes,
            edges=edges,
            bootstrap_order=("db", "triage"),
        )
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "db"

    def test_frozen(self) -> None:
        """ModelRuntimeNodeGraph is immutable."""
        graph = ModelRuntimeNodeGraph(
            nodes=(ModelNodeConfig(name="a", handler_class="a.H"),),
            bootstrap_order=("a",),
        )
        with pytest.raises(ValidationError):
            graph.bootstrap_order = ("b",)  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields on graph are rejected."""
        with pytest.raises(ValidationError):
            ModelRuntimeNodeGraph(
                nodes=(ModelNodeConfig(name="a", handler_class="a.H"),),
                bootstrap_order=("a",),
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_export_from_models_init(self) -> None:
        """ModelRuntimeNodeGraph is accessible from runtime.models package."""
        from omnibase_infra.runtime import models as rt_models

        assert rt_models.ModelNodeConfig is ModelNodeConfig
        assert rt_models.ModelNodeEdge is ModelNodeEdge
        assert rt_models.ModelRuntimeNodeGraph is ModelRuntimeNodeGraph
