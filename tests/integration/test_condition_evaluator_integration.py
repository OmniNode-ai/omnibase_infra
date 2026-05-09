# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for condition_evaluator against canonical graph data — OMN-10779."""

from __future__ import annotations

import pytest

from omnibase_infra.onboarding.condition_evaluator import (
    ConditionEvaluationError,
    evaluate_condition,
)
from omnibase_infra.onboarding.loader import load_canonical_graph

pytestmark = pytest.mark.integration


class TestConditionEvaluatorWithCanonicalGraph:
    """Verify evaluate_condition against step keys and capability names from the real graph."""

    def test_canonical_graph_loads(self) -> None:
        graph = load_canonical_graph()
        assert len(graph.steps) > 0

    def test_step_key_in_real_capability_list(self) -> None:
        graph = load_canonical_graph()
        all_capabilities = {
            cap for step in graph.steps for cap in step.produces_capabilities
        }
        # All canonical capabilities are known strings — verify membership check works
        for cap in all_capabilities:
            result = evaluate_condition(
                "current_cap in available_caps",
                {"current_cap": cap, "available_caps": list(all_capabilities)},
            )
            assert result is True, f"Expected {cap} in capability list"

    def test_none_condition_passes_for_all_steps(self) -> None:
        graph = load_canonical_graph()
        state: dict[str, object] = {}
        for step in graph.steps:
            assert evaluate_condition(None, state) is True

    def test_step_key_equality_condition(self) -> None:
        graph = load_canonical_graph()
        first_step = graph.steps[0]
        assert evaluate_condition(
            f"current_step == {first_step.step_key}",
            {"current_step": first_step.step_key},
        )

    def test_step_type_membership_condition(self) -> None:
        graph = load_canonical_graph()
        step_types = list({s.step_type for s in graph.steps})
        for step in graph.steps:
            result = evaluate_condition(
                "step_type in valid_types",
                {"step_type": step.step_type, "valid_types": step_types},
            )
            assert result is True

    def test_unknown_capability_not_in_list(self) -> None:
        graph = load_canonical_graph()
        all_capabilities = [
            cap for step in graph.steps for cap in step.produces_capabilities
        ]
        assert evaluate_condition(
            "cap not in known_caps",
            {"cap": "nonexistent_capability_xyz", "known_caps": all_capabilities},
        )

    def test_missing_state_key_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError):
            evaluate_condition("unknown_step_key == check_python", {})
