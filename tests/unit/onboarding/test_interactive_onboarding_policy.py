# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the interactive_onboarding branching policy contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

POLICY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "interactive_onboarding.yaml"
)

TERMINAL_STEPS = {"write_config_local", "write_config_cloud", "write_config_hybrid"}
DEPLOYMENT_MODES = ("local", "cloud", "hybrid")
CLOUD_PROVIDERS = ("aws", "gcp")


@pytest.fixture(scope="module")
def policy() -> dict[str, Any]:
    """Load the interactive onboarding policy YAML."""
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def step_ids(policy: dict[str, Any]) -> set[str]:
    return {s["id"] for s in policy["steps"]}


@pytest.fixture(scope="module")
def transitions_by_from(policy: dict[str, Any]) -> dict[str, Any]:
    return {t["from"]: t for t in policy["transitions"]}


class TestPolicyLoads:
    """Basic structural tests."""

    def test_policy_file_exists(self) -> None:
        assert POLICY_PATH.exists(), f"Policy file missing: {POLICY_PATH}"

    def test_parses_as_dict(self, policy: dict[str, Any]) -> None:
        assert isinstance(policy, dict)

    def test_has_required_top_level_keys(self, policy: dict[str, Any]) -> None:
        for key in ("policy_name", "steps", "transitions", "target_capabilities"):
            assert key in policy, f"Missing top-level key: {key}"

    def test_policy_name(self, policy: dict[str, Any]) -> None:
        assert policy["policy_name"] == "interactive_onboarding"

    def test_policy_type_is_interactive(self, policy: dict[str, Any]) -> None:
        assert policy["policy_type"] == "interactive"

    def test_produces_onboarding_config_written(self, policy: dict[str, Any]) -> None:
        assert "onboarding_config_written" in policy["target_capabilities"]

    def test_has_steps(self, policy: dict[str, Any]) -> None:
        assert len(policy["steps"]) >= 3

    def test_has_transitions(self, policy: dict[str, Any]) -> None:
        assert len(policy["transitions"]) >= 3


class TestStepStructure:
    """Every step must have required fields and valid types."""

    VALID_TYPES = {"choice", "multi_choice", "text", "action"}

    def test_all_steps_have_id(self, policy: dict[str, Any]) -> None:
        for step in policy["steps"]:
            assert "id" in step, f"Step missing 'id': {step}"

    def test_all_steps_have_prompt(self, policy: dict[str, Any]) -> None:
        for step in policy["steps"]:
            assert "prompt" in step, f"Step '{step.get('id')}' missing 'prompt'"

    def test_all_steps_have_valid_type(self, policy: dict[str, Any]) -> None:
        for step in policy["steps"]:
            assert step.get("type") in self.VALID_TYPES, (
                f"Step '{step['id']}' has invalid type '{step.get('type')}'"
            )

    def test_step_ids_are_unique(self, policy: dict[str, Any]) -> None:
        ids = [s["id"] for s in policy["steps"]]
        assert len(ids) == len(set(ids)), "Duplicate step IDs found"

    def test_terminal_steps_declared(self, step_ids: set[str]) -> None:
        for terminal in TERMINAL_STEPS:
            assert terminal in step_ids, (
                f"Terminal step '{terminal}' missing from steps"
            )

    def test_terminal_steps_produce_capability(self, policy: dict[str, Any]) -> None:
        for step in policy["steps"]:
            if step["id"] in TERMINAL_STEPS:
                caps = step.get("produces_capabilities", [])
                assert "onboarding_config_written" in caps, (
                    f"Terminal step '{step['id']}' must produce 'onboarding_config_written'"
                )

    def test_first_step_is_choose_deployment_mode(self, policy: dict[str, Any]) -> None:
        assert policy["steps"][0]["id"] == "choose_deployment_mode"

    def test_choose_deployment_mode_has_all_options(
        self, policy: dict[str, Any]
    ) -> None:
        first = policy["steps"][0]
        assert set(first["options"]) == {"local", "cloud", "hybrid"}


class TestTransitionTableCompleteness:
    """Every step must appear in the transition table, and every response must map
    to a valid next step."""

    def test_every_step_has_transition_entry(
        self, step_ids: set[str], transitions_by_from: dict[str, Any]
    ) -> None:
        for sid in step_ids:
            assert sid in transitions_by_from, (
                f"Step '{sid}' has no entry in transitions table"
            )

    def test_no_dangling_next_references(
        self, step_ids: set[str], transitions_by_from: dict[str, Any]
    ) -> None:
        """Every 'next' in the transition table must point to a valid step id."""
        for from_step, t in transitions_by_from.items():
            if t.get("terminal"):
                continue
            # Collect all 'next' values from responses and on_submit lists
            nexts: list[str] = []
            if "responses" in t:
                for resp in t["responses"].values():
                    if isinstance(resp, dict) and "next" in resp:
                        nexts.append(resp["next"])
            if "on_submit" in t:
                for branch in t["on_submit"]:
                    if "next" in branch:
                        nexts.append(branch["next"])
            for nxt in nexts:
                assert nxt in step_ids, (
                    f"Transition from '{from_step}' references unknown step '{nxt}'"
                )

    def test_terminal_steps_marked_terminal(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        for terminal in TERMINAL_STEPS:
            t = transitions_by_from[terminal]
            assert t.get("terminal") is True, (
                f"Step '{terminal}' is a terminal step but not marked terminal=true"
            )

    def test_non_terminal_steps_have_next(
        self, step_ids: set[str], transitions_by_from: dict[str, Any]
    ) -> None:
        for sid in step_ids:
            t = transitions_by_from[sid]
            if t.get("terminal"):
                continue
            has_next = "responses" in t or "on_submit" in t
            assert has_next, (
                f"Non-terminal step '{sid}' has neither 'responses' nor 'on_submit'"
            )


class TestLocalPath:
    """Simulate the local deployment path through the transition table."""

    def _follow_path(
        self, transitions_by_from: dict[str, Any], choices: dict[str, str]
    ) -> list[str]:
        """Walk through transitions using provided choices, return visited step ids."""
        visited = []
        current = "choose_deployment_mode"
        while current is not None:
            visited.append(current)
            t = transitions_by_from[current]
            if t.get("terminal"):
                break
            if "responses" in t:
                response = choices.get(current)
                assert response is not None, f"No choice provided for step '{current}'"
                nxt = t["responses"][response]["next"]
                current = nxt
            elif "on_submit" in t:
                # Take the first branch whose condition matches our choices context
                # (simplified: just pick the first applicable branch)
                response = choices.get(current)
                assert response is not None, f"No choice provided for step '{current}'"
                # For test purposes, pick the branch matching the response key
                matched = None
                for branch in t["on_submit"]:
                    cond = branch.get("condition", "")
                    if response in cond or "deployment_mode ==" in cond:
                        matched = branch
                        break
                assert matched is not None, (
                    f"No matching branch for step '{current}' with response '{response}'"
                )
                current = matched["next"]
            else:
                break
        return visited

    def test_local_no_llm_path_reaches_terminal(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        # choose_deployment_mode -> configure_local_services
        t = transitions_by_from["choose_deployment_mode"]
        assert "local" in t["responses"]
        nxt = t["responses"]["local"]["next"]
        assert nxt == "configure_local_services"

        # configure_local_services (no llm) -> write_config_local
        t2 = transitions_by_from["configure_local_services"]
        assert "on_submit" in t2
        # Find branch for local without llm
        local_no_llm = next(
            b
            for b in t2["on_submit"]
            if "deployment_mode == local" in b.get("condition", "")
            and "not in" in b.get("condition", "")
        )
        assert local_no_llm["next"] == "write_config_local"

    def test_local_with_llm_path_visits_configure_llm_endpoint(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        t = transitions_by_from["configure_local_services"]
        local_with_llm = next(
            b
            for b in t["on_submit"]
            if "deployment_mode == local" in b.get("condition", "")
            and "not in" not in b.get("condition", "")
        )
        assert local_with_llm["next"] == "configure_llm_endpoint"

        t2 = transitions_by_from["configure_llm_endpoint"]
        local_branch = next(
            b
            for b in t2["on_submit"]
            if "deployment_mode == local" in b.get("condition", "")
        )
        assert local_branch["next"] == "write_config_local"


class TestCloudPath:
    """Cloud path covers both AWS and GCP branches."""

    def test_cloud_aws_path_reaches_terminal(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        # choose_deployment_mode -> configure_cloud_provider
        t = transitions_by_from["choose_deployment_mode"]
        assert t["responses"]["cloud"]["next"] == "configure_cloud_provider"

        # configure_cloud_provider -> configure_aws_region
        t2 = transitions_by_from["configure_cloud_provider"]
        assert t2["responses"]["aws"]["next"] == "configure_aws_region"

        # configure_aws_region (cloud) -> write_config_cloud
        t3 = transitions_by_from["configure_aws_region"]
        cloud_branch = next(
            b
            for b in t3["on_submit"]
            if "deployment_mode == cloud" in b.get("condition", "")
        )
        assert cloud_branch["next"] == "write_config_cloud"

    def test_cloud_gcp_path_reaches_terminal(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        t2 = transitions_by_from["configure_cloud_provider"]
        assert t2["responses"]["gcp"]["next"] == "configure_gcp_project"

        t3 = transitions_by_from["configure_gcp_project"]
        cloud_branch = next(
            b
            for b in t3["on_submit"]
            if "deployment_mode == cloud" in b.get("condition", "")
        )
        assert cloud_branch["next"] == "write_config_cloud"


class TestHybridPath:
    """Hybrid path exercises both local services and cloud provider steps."""

    def test_hybrid_routes_through_local_services_then_cloud_provider(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        # choose_deployment_mode -> configure_local_services
        t = transitions_by_from["choose_deployment_mode"]
        assert t["responses"]["hybrid"]["next"] == "configure_local_services"

        # configure_local_services (hybrid) -> configure_cloud_provider
        t2 = transitions_by_from["configure_local_services"]
        hybrid_branch = next(
            b
            for b in t2["on_submit"]
            if "deployment_mode == hybrid" in b.get("condition", "")
        )
        assert hybrid_branch["next"] == "configure_cloud_provider"

    def test_hybrid_aws_no_llm_reaches_terminal(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        t3 = transitions_by_from["configure_aws_region"]
        hybrid_no_llm = next(
            b
            for b in t3["on_submit"]
            if "deployment_mode == hybrid" in b.get("condition", "")
            and "not in" in b.get("condition", "")
        )
        assert hybrid_no_llm["next"] == "write_config_hybrid"

    def test_hybrid_gcp_no_llm_reaches_terminal(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        t3 = transitions_by_from["configure_gcp_project"]
        hybrid_no_llm = next(
            b
            for b in t3["on_submit"]
            if "deployment_mode == hybrid" in b.get("condition", "")
            and "not in" in b.get("condition", "")
        )
        assert hybrid_no_llm["next"] == "write_config_hybrid"

    def test_hybrid_with_llm_visits_configure_llm_endpoint(
        self, transitions_by_from: dict[str, Any]
    ) -> None:
        t3 = transitions_by_from["configure_aws_region"]
        hybrid_llm = next(
            b
            for b in t3["on_submit"]
            if "deployment_mode == hybrid" in b.get("condition", "")
            and "llm_inference" in b.get("condition", "")
            and "not in" not in b.get("condition", "")
        )
        assert hybrid_llm["next"] == "configure_llm_endpoint"

        t4 = transitions_by_from["configure_llm_endpoint"]
        hybrid_branch = next(
            b
            for b in t4["on_submit"]
            if "deployment_mode == hybrid" in b.get("condition", "")
        )
        assert hybrid_branch["next"] == "write_config_hybrid"


class TestEnvOutput:
    """env_output section declares what gets written to .env per terminal step."""

    def test_env_output_exists(self, policy: dict[str, Any]) -> None:
        assert "env_output" in policy, "Policy missing 'env_output' section"

    def test_every_terminal_step_has_env_output(self, policy: dict[str, Any]) -> None:
        env_output = policy["env_output"]
        for terminal in TERMINAL_STEPS:
            assert terminal in env_output, (
                f"Terminal step '{terminal}' missing from env_output"
            )

    def test_local_env_output_has_required_keys(self, policy: dict[str, Any]) -> None:
        local = policy["env_output"]["write_config_local"]
        assert "ONEX_DEPLOYMENT_MODE" in local
        assert local["ONEX_DEPLOYMENT_MODE"] == "local"
        assert "KAFKA_BOOTSTRAP_SERVERS" in local
        assert "POSTGRES_HOST" in local

    def test_cloud_env_output_has_required_keys(self, policy: dict[str, Any]) -> None:
        cloud = policy["env_output"]["write_config_cloud"]
        assert "ONEX_DEPLOYMENT_MODE" in cloud
        assert cloud["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert "CLOUD_PROVIDER" in cloud

    def test_hybrid_env_output_has_both_local_and_cloud_keys(
        self, policy: dict[str, Any]
    ) -> None:
        hybrid = policy["env_output"]["write_config_hybrid"]
        assert hybrid["ONEX_DEPLOYMENT_MODE"] == "hybrid"
        assert "KAFKA_BOOTSTRAP_SERVERS" in hybrid
        assert "CLOUD_PROVIDER" in hybrid
