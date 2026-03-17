# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Loader for onboarding graph YAML files (OMN-5267).

Parses a YAML graph definition into a ModelOnboardingGraph and validates
the DAG structure (acyclic, all depends_on references valid).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.onboarding.model_onboarding_graph import ModelOnboardingGraph


class GraphValidationError(Exception):
    """Raised when the onboarding graph fails validation."""


def _validate_dag(graph: ModelOnboardingGraph) -> None:
    """Validate that the steps form a valid DAG.

    Checks:
    1. All depends_on references point to valid step IDs.
    2. The graph is acyclic (no circular dependencies).
    """
    step_keys = {s.step_key for s in graph.steps}

    # Check all references are valid
    for step in graph.steps:
        for dep in step.depends_on:
            if dep not in step_keys:
                msg = f"Step '{step.step_key}' depends on unknown step '{dep}'"
                raise GraphValidationError(msg)

    # Check for cycles using DFS
    visited: set[str] = set()
    in_stack: set[str] = set()
    adj: dict[str, list[str]] = {s.step_key: s.depends_on for s in graph.steps}

    def _dfs(node: str) -> None:
        if node in in_stack:
            msg = f"Cycle detected involving step '{node}'"
            raise GraphValidationError(msg)
        if node in visited:
            return
        in_stack.add(node)
        for dep in adj.get(node, []):
            _dfs(dep)
        in_stack.remove(node)
        visited.add(node)

    for step_key in step_keys:
        _dfs(step_key)


def load_graph_from_path(path: Path) -> ModelOnboardingGraph:
    """Load and validate an onboarding graph from a YAML file path.

    Args:
        path: Path to the YAML graph file.

    Returns:
        Parsed and validated ModelOnboardingGraph.

    Raises:
        GraphValidationError: If the graph is invalid.
    """
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        msg = f"Expected YAML dict at top level, got {type(raw).__name__}"
        raise GraphValidationError(msg)

    graph = ModelOnboardingGraph.model_validate(raw)
    if not graph.steps:
        msg = "Graph has no steps"
        raise GraphValidationError(msg)
    _validate_dag(graph)
    return graph


def load_canonical_graph() -> ModelOnboardingGraph:
    """Load the built-in canonical onboarding graph.

    Returns:
        Parsed and validated canonical graph.
    """
    graphs_dir = Path(__file__).parent / "graphs"
    canonical_path = graphs_dir / "canonical.yaml"
    return load_graph_from_path(canonical_path)


__all__ = [
    "GraphValidationError",
    "load_canonical_graph",
    "load_graph_from_path",
]
