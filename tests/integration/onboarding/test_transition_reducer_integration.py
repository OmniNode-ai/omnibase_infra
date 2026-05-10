# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests — full traversal of all paths through real interactive policy.

OMN-10780: exercises local, cloud (AWS/GCP), and hybrid (AWS/GCP, with/without LLM)
paths end-to-end using the real ``interactive_onboarding.yaml`` policy file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.transition_reducer import TransitionReducer

pytestmark = pytest.mark.integration

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


def _walk(
    reducer: TransitionReducer,
    start: str,
    steps: list[tuple[str, str | list[str]]],
) -> tuple[str, dict[str, Any]]:
    """Walk through a sequence of (step_id, response) pairs."""
    state: dict[str, Any] = {}
    current = start
    for expected_step, response in steps:
        assert current == expected_step, (
            f"Expected step '{expected_step}', got '{current}'"
        )
        current, state = reducer.advance(current, response, state)
    return current, state


# ------------------------------------------------------------------
# Local paths
# ------------------------------------------------------------------


class TestLocalPaths:
    def test_local_minimal(self, reducer: TransitionReducer) -> None:
        """local → services (no llm) → write_config_local."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "local"),
                ("configure_local_services", ["kafka", "postgres", "valkey"]),
            ],
        )
        assert terminal == "write_config_local"
        assert reducer.is_terminal(terminal)

        env = reducer.get_env_output(terminal, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "local"
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert env["POSTGRES_HOST"] == "localhost"
        assert env["POSTGRES_PORT"] == "5436"
        assert env["VALKEY_HOST"] == "localhost"
        assert env["VALKEY_PORT"] == "16379"
        assert env["LLM_CODER_URL"] == ""

    def test_local_with_llm(self, reducer: TransitionReducer) -> None:
        """local → services (with llm) → llm endpoint → write_config_local."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "local"),
                (
                    "configure_local_services",
                    ["kafka", "postgres", "llm_inference"],
                ),
                ("configure_llm_endpoint", "http://localhost:8000"),
            ],
        )
        assert terminal == "write_config_local"
        assert reducer.is_terminal(terminal)

        env = reducer.get_env_output(terminal, state)
        assert env["LLM_CODER_URL"] == "http://localhost:8000"

    def test_local_all_services_with_llm(self, reducer: TransitionReducer) -> None:
        """local → all services → llm endpoint → write_config_local."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "local"),
                (
                    "configure_local_services",
                    ["kafka", "postgres", "valkey", "llm_inference"],
                ),
                ("configure_llm_endpoint", "http://192.168.86.201:8000"),
            ],
        )
        assert terminal == "write_config_local"
        env = reducer.get_env_output(terminal, state)
        assert env["LLM_CODER_URL"] == "http://192.168.86.201:8000"


# ------------------------------------------------------------------
# Cloud paths
# ------------------------------------------------------------------


class TestCloudPaths:
    def test_cloud_aws(self, reducer: TransitionReducer) -> None:
        """cloud → aws → region → write_config_cloud."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "cloud"),
                ("configure_cloud_provider", "aws"),
                ("configure_aws_region", "us-east-1"),
            ],
        )
        assert terminal == "write_config_cloud"
        assert reducer.is_terminal(terminal)

        env = reducer.get_env_output(terminal, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "us-east-1"
        assert env["GCP_PROJECT"] == ""

    def test_cloud_gcp(self, reducer: TransitionReducer) -> None:
        """cloud → gcp → project → write_config_cloud."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "cloud"),
                ("configure_cloud_provider", "gcp"),
                ("configure_gcp_project", "my-gcp-project-123"),
            ],
        )
        assert terminal == "write_config_cloud"
        assert reducer.is_terminal(terminal)

        env = reducer.get_env_output(terminal, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "my-gcp-project-123"
        assert env["AWS_REGION"] == ""


# ------------------------------------------------------------------
# Hybrid paths
# ------------------------------------------------------------------


class TestHybridPaths:
    def test_hybrid_aws_with_llm(self, reducer: TransitionReducer) -> None:
        """hybrid → services (with llm) → aws → region → llm → write_config_hybrid."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "hybrid"),
                (
                    "configure_local_services",
                    ["kafka", "postgres", "llm_inference"],
                ),
                ("configure_cloud_provider", "aws"),
                ("configure_aws_region", "eu-west-1"),
                ("configure_llm_endpoint", "http://my-llm:8080"),
            ],
        )
        assert terminal == "write_config_hybrid"
        assert reducer.is_terminal(terminal)

        env = reducer.get_env_output(terminal, state)
        assert env["ONEX_DEPLOYMENT_MODE"] == "hybrid"
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "eu-west-1"
        assert env["GCP_PROJECT"] == ""
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert env["LLM_CODER_URL"] == "http://my-llm:8080"

    def test_hybrid_aws_no_llm(self, reducer: TransitionReducer) -> None:
        """hybrid → services (no llm) → aws → region → write_config_hybrid."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "hybrid"),
                ("configure_local_services", ["kafka", "postgres"]),
                ("configure_cloud_provider", "aws"),
                ("configure_aws_region", "ap-southeast-1"),
            ],
        )
        assert terminal == "write_config_hybrid"
        assert reducer.is_terminal(terminal)

        env = reducer.get_env_output(terminal, state)
        assert env["AWS_REGION"] == "ap-southeast-1"
        assert env["LLM_CODER_URL"] == ""

    def test_hybrid_gcp_with_llm(self, reducer: TransitionReducer) -> None:
        """hybrid → services (with llm) → gcp → project → llm → write_config_hybrid."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "hybrid"),
                (
                    "configure_local_services",
                    ["kafka", "llm_inference"],
                ),
                ("configure_cloud_provider", "gcp"),
                ("configure_gcp_project", "hybrid-project"),
                ("configure_llm_endpoint", "http://inference:9000"),
            ],
        )
        assert terminal == "write_config_hybrid"

        env = reducer.get_env_output(terminal, state)
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "hybrid-project"
        assert env["LLM_CODER_URL"] == "http://inference:9000"

    def test_hybrid_gcp_no_llm(self, reducer: TransitionReducer) -> None:
        """hybrid → services (no llm) → gcp → project → write_config_hybrid."""
        terminal, state = _walk(
            reducer,
            "choose_deployment_mode",
            [
                ("choose_deployment_mode", "hybrid"),
                ("configure_local_services", ["valkey"]),
                ("configure_cloud_provider", "gcp"),
                ("configure_gcp_project", "another-project"),
            ],
        )
        assert terminal == "write_config_hybrid"

        env = reducer.get_env_output(terminal, state)
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "another-project"
        assert env["AWS_REGION"] == ""
        assert env["LLM_CODER_URL"] == ""


# ------------------------------------------------------------------
# State accumulation across full paths
# ------------------------------------------------------------------


class TestStateAccumulation:
    def test_state_accumulates_across_steps(self, reducer: TransitionReducer) -> None:
        """Verify state dict grows as we advance through steps."""
        state: dict[str, Any] = {}

        step, state = reducer.advance("choose_deployment_mode", "hybrid", state)
        assert "deployment_mode" in state

        step, state = reducer.advance(
            step, ["kafka", "postgres", "llm_inference"], state
        )
        assert "selected_local_services" in state
        assert "deployment_mode" in state  # preserved

        step, state = reducer.advance(step, "aws", state)
        assert "cloud_provider" in state
        assert "deployment_mode" in state
        assert "selected_local_services" in state

        step, state = reducer.advance(step, "us-east-1", state)
        assert "aws_region" in state

        step, state = reducer.advance(step, "http://llm:8000", state)
        assert "llm_endpoint" in state
        assert step == "write_config_hybrid"

        # All keys accumulated
        assert set(state.keys()) == {
            "deployment_mode",
            "selected_local_services",
            "cloud_provider",
            "aws_region",
            "llm_endpoint",
        }
