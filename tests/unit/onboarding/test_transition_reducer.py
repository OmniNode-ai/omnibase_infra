# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for TransitionReducer — OMN-10780."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # onex-any-ok: state dicts in test assertions

import pytest
import yaml

from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.transition_reducer import (
    InterpolationError,
    TransitionError,
    TransitionReducer,
)

pytestmark = pytest.mark.unit

POLICY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "interactive_onboarding.yaml"
)


@pytest.fixture
def policy() -> ModelInteractivePolicy:
    raw = yaml.safe_load(POLICY_PATH.read_text())
    return ModelInteractivePolicy.model_validate(raw)


@pytest.fixture
def reducer(policy: ModelInteractivePolicy) -> TransitionReducer:
    return TransitionReducer(policy)


# ------------------------------------------------------------------
# Basic single-step advance
# ------------------------------------------------------------------


class TestBasicAdvance:
    def test_choice_step_advances(self, reducer: TransitionReducer) -> None:
        step, state = reducer.advance("choose_deployment_mode", "local", {})
        assert step == "configure_local_services"
        assert state["deployment_mode"] == "local"

    def test_choice_step_cloud(self, reducer: TransitionReducer) -> None:
        step, state = reducer.advance("choose_deployment_mode", "cloud", {})
        assert step == "configure_cloud_provider"
        assert state["deployment_mode"] == "cloud"

    def test_choice_step_hybrid(self, reducer: TransitionReducer) -> None:
        step, state = reducer.advance("choose_deployment_mode", "hybrid", {})
        assert step == "configure_local_services"
        assert state["deployment_mode"] == "hybrid"

    def test_unknown_step_raises(self, reducer: TransitionReducer) -> None:
        with pytest.raises(TransitionError, match="Unknown step"):
            reducer.advance("nonexistent_step", "x", {})

    def test_terminal_step_raises(self, reducer: TransitionReducer) -> None:
        with pytest.raises(TransitionError, match="terminal"):
            reducer.advance("write_config_local", "x", {})

    def test_unknown_response_raises(self, reducer: TransitionReducer) -> None:
        with pytest.raises(TransitionError, match="No transition"):
            reducer.advance("choose_deployment_mode", "nonexistent", {})


# ------------------------------------------------------------------
# Conditional branching (on_submit)
# ------------------------------------------------------------------


class TestConditionalBranching:
    def test_first_matching_condition_wins(self, reducer: TransitionReducer) -> None:
        """Local mode + no llm_inference → write_config_local (first branch)."""
        state: dict[str, Any] = {"deployment_mode": "local"}
        step, _updated = reducer.advance(
            "configure_local_services", ["kafka", "postgres"], state
        )
        assert step == "write_config_local"

    def test_second_branch_matches_when_llm_selected(
        self, reducer: TransitionReducer
    ) -> None:
        """Local mode + llm_inference selected → configure_llm_endpoint."""
        state: dict[str, Any] = {"deployment_mode": "local"}
        step, _updated = reducer.advance(
            "configure_local_services",
            ["kafka", "postgres", "llm_inference"],
            state,
        )
        assert step == "configure_llm_endpoint"

    def test_hybrid_branch_matches(self, reducer: TransitionReducer) -> None:
        """Hybrid mode → configure_cloud_provider regardless of services."""
        state: dict[str, Any] = {"deployment_mode": "hybrid"}
        step, _updated = reducer.advance("configure_local_services", ["kafka"], state)
        assert step == "configure_cloud_provider"

    def test_cloud_provider_choice_aws(self, reducer: TransitionReducer) -> None:
        step, state = reducer.advance(
            "configure_cloud_provider", "aws", {"deployment_mode": "cloud"}
        )
        assert step == "configure_aws_region"
        assert state["cloud_provider"] == "aws"

    def test_cloud_provider_choice_gcp(self, reducer: TransitionReducer) -> None:
        step, state = reducer.advance(
            "configure_cloud_provider", "gcp", {"deployment_mode": "cloud"}
        )
        assert step == "configure_gcp_project"
        assert state["cloud_provider"] == "gcp"


# ------------------------------------------------------------------
# Multi-choice response stored as list (GPT #7)
# ------------------------------------------------------------------


class TestMultiChoiceState:
    def test_multi_choice_stored_as_list(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {"deployment_mode": "local"}
        _, updated = reducer.advance(
            "configure_local_services", ["kafka", "postgres"], state
        )
        assert isinstance(updated["selected_local_services"], list)
        assert updated["selected_local_services"] == ["kafka", "postgres"]

    def test_text_response_stored_as_string(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {"deployment_mode": "cloud", "cloud_provider": "aws"}
        _, updated = reducer.advance("configure_aws_region", "us-east-1", state)
        assert updated["aws_region"] == "us-east-1"
        assert isinstance(updated["aws_region"], str)


# ------------------------------------------------------------------
# Env output interpolation
# ------------------------------------------------------------------


class TestEnvOutputInterpolation:
    def test_basic_interpolation(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {
            "deployment_mode": "local",
            "llm_endpoint": "http://localhost:8000",
        }
        env = reducer.get_env_output("write_config_local", state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "local"
        assert env["LLM_CODER_URL"] == "http://localhost:8000"

    def test_empty_default_when_key_missing(self, reducer: TransitionReducer) -> None:
        """``{state.llm_endpoint|}`` with no llm_endpoint → empty string."""
        state: dict[str, Any] = {"deployment_mode": "local"}
        env = reducer.get_env_output("write_config_local", state)
        assert env["LLM_CODER_URL"] == ""

    def test_cloud_interpolation_with_aws(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {
            "cloud_provider": "aws",
            "aws_region": "us-west-2",
        }
        env = reducer.get_env_output("write_config_cloud", state)
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "us-west-2"
        assert env["GCP_PROJECT"] == ""  # optional, not set

    def test_cloud_interpolation_with_gcp(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {
            "cloud_provider": "gcp",
            "gcp_project": "my-project",
        }
        env = reducer.get_env_output("write_config_cloud", state)
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "my-project"
        assert env["AWS_REGION"] == ""  # optional, not set

    def test_literal_values_pass_through(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {"deployment_mode": "local"}
        env = reducer.get_env_output("write_config_local", state)
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert env["POSTGRES_HOST"] == "localhost"
        assert env["POSTGRES_PORT"] == "5436"

    def test_unknown_env_output_step_raises(self, reducer: TransitionReducer) -> None:
        with pytest.raises(TransitionError, match="No env_output"):
            reducer.get_env_output("choose_deployment_mode", {})


# ------------------------------------------------------------------
# Strict interpolation — unknown required keys raise (GPT #8)
# ------------------------------------------------------------------


class TestStrictInterpolation:
    def test_required_key_missing_raises(self, reducer: TransitionReducer) -> None:
        """``{state.cloud_provider}`` (no pipe) with missing key → error."""
        state: dict[str, Any] = {}  # cloud_provider missing, no default
        with pytest.raises(InterpolationError, match="cloud_provider"):
            reducer.get_env_output("write_config_cloud", state)

    def test_optional_key_with_pipe_does_not_raise(
        self, reducer: TransitionReducer
    ) -> None:
        """``{state.aws_region|}`` with missing key → empty string, no error."""
        state: dict[str, Any] = {"cloud_provider": "aws"}
        env = reducer.get_env_output("write_config_cloud", state)
        assert env["AWS_REGION"] == ""


# ------------------------------------------------------------------
# Graph validation at construction (GPT #4)
# ------------------------------------------------------------------


class TestGraphValidation:
    def test_valid_policy_constructs(self, policy: ModelInteractivePolicy) -> None:
        reducer = TransitionReducer(policy)
        assert reducer is not None

    def test_unreachable_step_raises(self) -> None:
        """A step with no incoming transitions (except start) is unreachable."""
        raw = yaml.safe_load(POLICY_PATH.read_text())
        # Add an orphan step that nothing transitions to
        raw["steps"].append({"id": "orphan_step", "prompt": "orphan", "type": "text"})
        policy = ModelInteractivePolicy.model_validate(raw)
        with pytest.raises(TransitionError, match=r"[Uu]nreachable"):
            TransitionReducer(policy)


# ------------------------------------------------------------------
# is_terminal
# ------------------------------------------------------------------


class TestIsTerminal:
    def test_terminal_steps_identified(self, reducer: TransitionReducer) -> None:
        assert reducer.is_terminal("write_config_local") is True
        assert reducer.is_terminal("write_config_cloud") is True
        assert reducer.is_terminal("write_config_hybrid") is True

    def test_non_terminal_steps(self, reducer: TransitionReducer) -> None:
        assert reducer.is_terminal("choose_deployment_mode") is False
        assert reducer.is_terminal("configure_cloud_provider") is False


# ------------------------------------------------------------------
# Full path traversals with real policy
# ------------------------------------------------------------------


class TestRealPolicyTraversal:
    def test_local_path_no_llm(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "local", state)
        assert step == "configure_local_services"
        assert state["deployment_mode"] == "local"

        step, state = reducer.advance(
            "configure_local_services", ["kafka", "postgres"], state
        )
        assert step == "write_config_local"
        assert isinstance(state["selected_local_services"], list)
        assert reducer.is_terminal(step)

        env = reducer.get_env_output(step, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "local"
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"

    def test_local_path_with_llm(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "local", state)
        step, state = reducer.advance(
            step, ["kafka", "postgres", "llm_inference"], state
        )
        assert step == "configure_llm_endpoint"

        step, state = reducer.advance(step, "http://localhost:8000", state)
        assert step == "write_config_local"
        assert reducer.is_terminal(step)

        env = reducer.get_env_output(step, state)
        assert env["LLM_CODER_URL"] == "http://localhost:8000"

    def test_cloud_path_aws(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "cloud", state)
        assert step == "configure_cloud_provider"

        step, state = reducer.advance(step, "aws", state)
        assert step == "configure_aws_region"

        step, state = reducer.advance(step, "us-east-1", state)
        assert step == "write_config_cloud"
        assert reducer.is_terminal(step)

        env = reducer.get_env_output(step, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "us-east-1"

    def test_cloud_path_gcp(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "cloud", state)
        step, state = reducer.advance(step, "gcp", state)
        assert step == "configure_gcp_project"

        step, state = reducer.advance(step, "my-gcp-project", state)
        assert step == "write_config_cloud"
        assert reducer.is_terminal(step)

        env = reducer.get_env_output(step, state)
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "my-gcp-project"

    def test_hybrid_path_aws_with_llm(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "hybrid", state)
        assert step == "configure_local_services"

        step, state = reducer.advance(
            step, ["kafka", "postgres", "llm_inference"], state
        )
        assert step == "configure_cloud_provider"

        step, state = reducer.advance(step, "aws", state)
        assert step == "configure_aws_region"

        step, state = reducer.advance(step, "us-west-2", state)
        assert step == "configure_llm_endpoint"

        step, state = reducer.advance(step, "http://my-llm:8000", state)
        assert step == "write_config_hybrid"
        assert reducer.is_terminal(step)

        env = reducer.get_env_output(step, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "hybrid"
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "us-west-2"
        assert env["LLM_CODER_URL"] == "http://my-llm:8000"
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"

    def test_hybrid_path_gcp_no_llm(self, reducer: TransitionReducer) -> None:
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "hybrid", state)
        step, state = reducer.advance(step, ["kafka", "postgres"], state)
        assert step == "configure_cloud_provider"

        step, state = reducer.advance(step, "gcp", state)
        assert step == "configure_gcp_project"

        step, state = reducer.advance(step, "my-project", state)
        assert step == "write_config_hybrid"
        assert reducer.is_terminal(step)

        env = reducer.get_env_output(step, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "hybrid"
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "my-project"
        assert env["LLM_CODER_URL"] == ""  # not set, optional
