# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for onboarding graph loader (OMN-5267)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.onboarding.loader import (
    GraphValidationError,
    load_canonical_graph,
    load_graph_from_path,
)


class TestLoadCanonicalGraph:
    """Tests for the canonical onboarding graph."""

    def test_loads_successfully(self) -> None:
        graph = load_canonical_graph()
        assert graph.title == "onex_onboarding_canonical"

    def test_has_10_steps(self) -> None:
        graph = load_canonical_graph()
        assert len(graph.steps) == 10

    def test_dag_is_acyclic(self) -> None:
        """load_canonical_graph validates the DAG -- no exception means acyclic."""
        graph = load_canonical_graph()
        step_keys = {s.step_key for s in graph.steps}
        assert len(step_keys) == 10  # All unique

    def test_all_depends_on_reference_valid_step_keys(self) -> None:
        graph = load_canonical_graph()
        step_keys = {s.step_key for s in graph.steps}
        for step in graph.steps:
            for dep in step.depends_on:
                assert dep in step_keys, (
                    f"Step '{step.step_key}' depends on unknown '{dep}'"
                )

    def test_all_steps_have_verification(self) -> None:
        graph = load_canonical_graph()
        for step in graph.steps:
            assert step.verification is not None, (
                f"Step '{step.step_key}' missing verification"
            )
            assert step.verification.check_type in {
                "command_exit_0",
                "file_exists",
                "tcp_probe",
                "http_health",
                "python_import",
            }

    def test_all_steps_have_capabilities(self) -> None:
        graph = load_canonical_graph()
        for step in graph.steps:
            assert len(step.produces_capabilities) > 0

    def test_first_step_has_no_dependencies(self) -> None:
        graph = load_canonical_graph()
        first = graph.steps[0]
        assert first.step_key == "check_python"
        assert first.depends_on == []

    def test_expected_step_order(self) -> None:
        graph = load_canonical_graph()
        step_keys = [s.step_key for s in graph.steps]
        assert step_keys == [
            "check_python",
            "install_uv",
            "install_core",
            "create_first_node",
            "run_standalone_node",
            "start_docker_infra",
            "start_event_bus",
            "connect_node_to_bus",
            "configure_secrets",
            "start_omnidash",
        ]


class TestLoadGraphFromPath:
    """Tests for loading arbitrary graph files."""

    def test_rejects_cycle(self, tmp_path: Path) -> None:
        graph = {
            "title": "cycle_test",
            "steps": [
                {
                    "step_key": "a",
                    "name": "A",
                    "step_type": "action",
                    "depends_on": ["b"],
                },
                {
                    "step_key": "b",
                    "name": "B",
                    "step_type": "action",
                    "depends_on": ["a"],
                },
            ],
        }
        path = tmp_path / "cycle.yaml"
        path.write_text(yaml.dump(graph))
        with pytest.raises(GraphValidationError, match="Cycle detected"):
            load_graph_from_path(path)

    def test_rejects_invalid_reference(self, tmp_path: Path) -> None:
        graph = {
            "title": "bad_ref",
            "steps": [
                {
                    "step_key": "a",
                    "name": "A",
                    "step_type": "action",
                    "depends_on": ["nonexistent"],
                },
            ],
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(graph))
        with pytest.raises(GraphValidationError, match="unknown step"):
            load_graph_from_path(path)

    def test_rejects_empty_steps(self, tmp_path: Path) -> None:
        graph = {"title": "empty", "steps": []}
        path = tmp_path / "empty.yaml"
        path.write_text(yaml.dump(graph))
        with pytest.raises(GraphValidationError, match="no steps"):
            load_graph_from_path(path)

    def test_valid_linear_graph(self, tmp_path: Path) -> None:
        graph = {
            "title": "linear",
            "steps": [
                {"step_key": "a", "name": "A", "step_type": "action", "depends_on": []},
                {
                    "step_key": "b",
                    "name": "B",
                    "step_type": "action",
                    "depends_on": ["a"],
                },
                {
                    "step_key": "c",
                    "name": "C",
                    "step_type": "action",
                    "depends_on": ["b"],
                },
            ],
        }
        path = tmp_path / "linear.yaml"
        path.write_text(yaml.dump(graph))
        data = load_graph_from_path(path)
        assert len(data.steps) == 3
