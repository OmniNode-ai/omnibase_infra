# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Policy resolver for onboarding graph traversal (OMN-5268).

Takes target capabilities and walks backwards through the graph to find
the minimal set of steps needed. Returns steps in topological order.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.onboarding.model_onboarding_graph import ModelOnboardingGraph
from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep


class UnsatisfiableTargetError(Exception):
    """Raised when target capabilities cannot be satisfied by the graph."""


def resolve_policy(
    graph: ModelOnboardingGraph,
    target_capabilities: list[str],
    skip_steps: list[str] | None = None,
) -> list[ModelOnboardingStep]:
    """Resolve a policy to a minimal set of steps in topological order.

    Algorithm: walk backwards from target capabilities through
    produces/required edges, collecting required steps, then
    topologically sort.

    Args:
        graph: The full onboarding graph.
        target_capabilities: Capabilities to achieve.
        skip_steps: Step keys to skip.

    Returns:
        List of steps in topological (execution) order.

    Raises:
        UnsatisfiableTargetError: If a target capability is not
            produced by any step in the graph.
    """
    skip = set(skip_steps or [])

    # Build lookup maps
    step_by_key: dict[str, ModelOnboardingStep] = {s.step_key: s for s in graph.steps}
    cap_to_step: dict[str, str] = {}
    for step in graph.steps:
        for cap in step.produces_capabilities:
            cap_to_step[cap] = step.step_key

    # Walk backwards from targets
    required_keys: set[str] = set()
    queue = list(target_capabilities)
    visited_caps: set[str] = set()

    while queue:
        cap = queue.pop(0)
        if cap in visited_caps:
            continue
        visited_caps.add(cap)

        step_key = cap_to_step.get(cap)
        if step_key is None:
            msg = f"Capability '{cap}' is not produced by any step in the graph"
            raise UnsatisfiableTargetError(msg)

        if step_key in skip:
            continue

        required_keys.add(step_key)
        step = step_by_key[step_key]

        # Add dependencies
        for dep_key in step.depends_on:
            if dep_key not in skip:
                required_keys.add(dep_key)
                dep_step = step_by_key[dep_key]
                for dep_cap in dep_step.produces_capabilities:
                    if dep_cap not in visited_caps:
                        queue.append(dep_cap)

        # Add required capabilities
        for req_cap in step.required_capabilities:
            if req_cap not in visited_caps:
                queue.append(req_cap)

    # Topological sort (preserve original order for stability)
    result = [s for s in graph.steps if s.step_key in required_keys]
    return result


def load_policy_yaml(path: Path) -> dict[str, list[str] | str | int | None]:
    """Load a policy YAML file.

    Args:
        path: Path to the policy YAML.

    Returns:
        Parsed policy dict with target_capabilities and other fields.
    """
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        msg = f"Expected dict, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def load_builtin_policies() -> dict[str, dict[str, list[str] | str | int | None]]:
    """Load all built-in policy YAML files.

    Returns:
        Dict mapping policy_name to policy data.
    """
    policies_dir = Path(__file__).parent / "policies"
    result: dict[str, dict[str, list[str] | str | int | None]] = {}
    for path in sorted(policies_dir.glob("*.yaml")):
        data = load_policy_yaml(path)
        name = data.get("policy_name")
        if isinstance(name, str):
            result[name] = data
    return result


__all__ = [
    "UnsatisfiableTargetError",
    "load_builtin_policies",
    "load_policy_yaml",
    "resolve_policy",
]
