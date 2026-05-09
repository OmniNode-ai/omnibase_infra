# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for condition evaluation against onboarding policy YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.onboarding.condition_evaluator import evaluate_condition

POLICY_PATH = (
    Path(__file__).parents[3]
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "interactive_onboarding.yaml"
)


@pytest.fixture(scope="module")
def interactive_policy() -> dict[str, Any]:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def _transition_from(policy: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(
        transition
        for transition in policy["transitions"]
        if transition["from"] == step_id
    )


def _next_on_submit_step(transition: dict[str, Any], state: dict[str, object]) -> str:
    for branch in transition["on_submit"]:
        if evaluate_condition(branch["condition"], state):
            return str(branch["next"])
    pytest.fail(f"No matching branch for state: {state}")


def test_policy_not_in_response_routes_local_without_llm_to_terminal(
    interactive_policy: dict[str, Any],
) -> None:
    transition = _transition_from(interactive_policy, "configure_local_services")
    state: dict[str, object] = {
        "deployment_mode": "local",
        "response": ["kafka", "postgres"],
    }

    assert _next_on_submit_step(transition, state) == "write_config_local"


def test_policy_not_in_selected_services_routes_hybrid_without_llm_to_terminal(
    interactive_policy: dict[str, Any],
) -> None:
    transition = _transition_from(interactive_policy, "configure_aws_region")
    state: dict[str, object] = {
        "deployment_mode": "hybrid",
        "selected_local_services": ["kafka", "postgres"],
    }

    assert _next_on_submit_step(transition, state) == "write_config_hybrid"
