# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test: quoted-literal LHS support against the real policy YAML.

OMN-10797 adds the ability for `condition_evaluator` to treat a quoted-string
LHS as a literal value rather than a state key. The interactive onboarding
policy in `interactive_onboarding.yaml` depends on this for branches such as
``"llm_inference" in selected_local_services``. This test loads the *real*
deployed policy YAML and verifies that every branch condition is parseable
and produces the expected routing for representative state inputs.
"""

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
def policy() -> dict[str, Any]:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def transitions(policy: dict[str, Any]) -> dict[str, Any]:
    return {t["from"]: t for t in policy["transitions"]}


def _resolve_branch(transition: dict[str, Any], state: dict[str, object]) -> str:
    for branch in transition["on_submit"]:
        if evaluate_condition(branch["condition"], state):
            return str(branch["next"])
    pytest.fail(f"No matching branch for state={state} in transition {transition}")


class TestQuotedLhsAcrossPolicy:
    """Every transition with a quoted-LHS condition routes correctly end-to-end."""

    def test_local_without_llm_routes_to_write_local(
        self, transitions: dict[str, Any]
    ) -> None:
        # `"llm_inference" not in response` should match when response lacks llm_inference.
        t = transitions["configure_local_services"]
        state: dict[str, object] = {
            "deployment_mode": "local",
            "response": ["kafka", "postgres", "valkey"],
        }
        assert _resolve_branch(t, state) == "write_config_local"

    def test_local_with_llm_routes_to_configure_endpoint(
        self, transitions: dict[str, Any]
    ) -> None:
        # `"llm_inference" in response` should match when response contains llm_inference.
        t = transitions["configure_local_services"]
        state: dict[str, object] = {
            "deployment_mode": "local",
            "response": ["kafka", "llm_inference"],
        }
        assert _resolve_branch(t, state) == "configure_llm_endpoint"

    def test_hybrid_aws_with_llm_routes_to_configure_endpoint(
        self, transitions: dict[str, Any]
    ) -> None:
        t = transitions["configure_aws_region"]
        state: dict[str, object] = {
            "deployment_mode": "hybrid",
            "selected_local_services": ["llm_inference"],
        }
        assert _resolve_branch(t, state) == "configure_llm_endpoint"

    def test_hybrid_aws_without_llm_routes_to_write_hybrid(
        self, transitions: dict[str, Any]
    ) -> None:
        t = transitions["configure_aws_region"]
        state: dict[str, object] = {
            "deployment_mode": "hybrid",
            "selected_local_services": ["kafka", "postgres"],
        }
        assert _resolve_branch(t, state) == "write_config_hybrid"

    def test_hybrid_gcp_with_llm_routes_to_configure_endpoint(
        self, transitions: dict[str, Any]
    ) -> None:
        t = transitions["configure_gcp_project"]
        state: dict[str, object] = {
            "deployment_mode": "hybrid",
            "selected_local_services": ["llm_inference", "valkey"],
        }
        assert _resolve_branch(t, state) == "configure_llm_endpoint"

    def test_hybrid_gcp_without_llm_routes_to_write_hybrid(
        self, transitions: dict[str, Any]
    ) -> None:
        t = transitions["configure_gcp_project"]
        state: dict[str, object] = {
            "deployment_mode": "hybrid",
            "selected_local_services": ["postgres"],
        }
        assert _resolve_branch(t, state) == "write_config_hybrid"

    def test_step_visibility_quoted_literal_matches_state_list(
        self, policy: dict[str, Any]
    ) -> None:
        # The configure_llm_endpoint step has visibility condition
        # `"llm_inference" in selected_local_services` — verify that condition
        # on the loaded step parses against state.
        endpoint_step = next(
            s for s in policy["steps"] if s["id"] == "configure_llm_endpoint"
        )
        condition = endpoint_step["condition"]
        assert (
            evaluate_condition(
                condition,
                {"selected_local_services": ["kafka", "llm_inference"]},
            )
            is True
        )
        assert (
            evaluate_condition(
                condition,
                {"selected_local_services": ["kafka", "postgres"]},
            )
            is False
        )
