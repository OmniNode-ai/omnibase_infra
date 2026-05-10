# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests — full traversal of all paths through the interactive executor.

OMN-10782: exercises local, cloud (AWS/GCP), and hybrid (AWS/GCP, with/without LLM)
paths end-to-end using the real ``interactive_onboarding.yaml`` policy file
driven by ``InteractiveExecutor`` + ``AdapterFakeInput``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.interactive_executor import InteractiveExecutor
from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy

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


# ------------------------------------------------------------------
# Local paths
# ------------------------------------------------------------------


class TestLocalPaths:
    @pytest.mark.asyncio
    async def test_local_minimal(self, policy: ModelInteractivePolicy) -> None:
        """local → services (no llm) → write_config_local."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": ["kafka", "postgres", "valkey"],
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_local"
        assert result.policy_name == "interactive_onboarding"

        env = result.env_dict
        assert env["ONEX_DEPLOYMENT_MODE"] == "local"
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert env["POSTGRES_HOST"] == "localhost"
        assert env["POSTGRES_PORT"] == "5436"
        assert env["VALKEY_HOST"] == "localhost"
        assert env["VALKEY_PORT"] == "16379"
        assert env["LLM_CODER_URL"] == ""

        # Step order verification
        step_keys = [sr.step_key for sr in result.step_results]
        assert step_keys == ["choose_deployment_mode", "configure_local_services"]

    @pytest.mark.asyncio
    async def test_local_with_llm(self, policy: ModelInteractivePolicy) -> None:
        """local → services (with llm) → llm endpoint → write_config_local."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": ["kafka", "postgres", "llm_inference"],
                "configure_llm_endpoint": "http://localhost:8000",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_local"
        assert result.env_dict["LLM_CODER_URL"] == "http://localhost:8000"

        step_keys = [sr.step_key for sr in result.step_results]
        assert step_keys == [
            "choose_deployment_mode",
            "configure_local_services",
            "configure_llm_endpoint",
        ]

    @pytest.mark.asyncio
    async def test_local_all_services_with_llm(
        self, policy: ModelInteractivePolicy
    ) -> None:
        """local → all 4 services → llm endpoint → write_config_local."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": [
                    "kafka",
                    "postgres",
                    "valkey",
                    "llm_inference",
                ],
                "configure_llm_endpoint": "http://192.168.86.201:8000",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_local"
        assert result.env_dict["LLM_CODER_URL"] == "http://192.168.86.201:8000"


# ------------------------------------------------------------------
# Cloud paths
# ------------------------------------------------------------------


class TestCloudPaths:
    @pytest.mark.asyncio
    async def test_cloud_aws(self, policy: ModelInteractivePolicy) -> None:
        """cloud → aws → region → write_config_cloud."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "cloud",
                "configure_cloud_provider": "aws",
                "configure_aws_region": "us-east-1",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_cloud"

        env = result.env_dict
        assert env["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "us-east-1"
        assert env["GCP_PROJECT"] == ""

        step_keys = [sr.step_key for sr in result.step_results]
        assert step_keys == [
            "choose_deployment_mode",
            "configure_cloud_provider",
            "configure_aws_region",
        ]

    @pytest.mark.asyncio
    async def test_cloud_gcp(self, policy: ModelInteractivePolicy) -> None:
        """cloud → gcp → project → write_config_cloud."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "cloud",
                "configure_cloud_provider": "gcp",
                "configure_gcp_project": "my-gcp-project-123",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_cloud"

        env = result.env_dict
        assert env["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "my-gcp-project-123"
        assert env["AWS_REGION"] == ""


# ------------------------------------------------------------------
# Hybrid paths
# ------------------------------------------------------------------


class TestHybridPaths:
    @pytest.mark.asyncio
    async def test_hybrid_aws_with_llm(self, policy: ModelInteractivePolicy) -> None:
        """hybrid → services (with llm) → aws → region → llm → write_config_hybrid."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "hybrid",
                "configure_local_services": ["kafka", "postgres", "llm_inference"],
                "configure_cloud_provider": "aws",
                "configure_aws_region": "eu-west-1",
                "configure_llm_endpoint": "http://my-llm:8080",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_hybrid"

        env = result.env_dict
        assert env["ONEX_DEPLOYMENT_MODE"] == "hybrid"
        assert env["CLOUD_PROVIDER"] == "aws"
        assert env["AWS_REGION"] == "eu-west-1"
        assert env["GCP_PROJECT"] == ""
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert env["LLM_CODER_URL"] == "http://my-llm:8080"

        step_keys = [sr.step_key for sr in result.step_results]
        assert step_keys == [
            "choose_deployment_mode",
            "configure_local_services",
            "configure_cloud_provider",
            "configure_aws_region",
            "configure_llm_endpoint",
        ]

    @pytest.mark.asyncio
    async def test_hybrid_aws_no_llm(self, policy: ModelInteractivePolicy) -> None:
        """hybrid → services (no llm) → aws → region → write_config_hybrid."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "hybrid",
                "configure_local_services": ["kafka", "postgres"],
                "configure_cloud_provider": "aws",
                "configure_aws_region": "ap-southeast-1",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_hybrid"
        assert result.env_dict["AWS_REGION"] == "ap-southeast-1"
        assert result.env_dict["LLM_CODER_URL"] == ""

    @pytest.mark.asyncio
    async def test_hybrid_gcp_with_llm(self, policy: ModelInteractivePolicy) -> None:
        """hybrid → services (with llm) → gcp → project → llm → write_config_hybrid."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "hybrid",
                "configure_local_services": ["kafka", "llm_inference"],
                "configure_cloud_provider": "gcp",
                "configure_gcp_project": "hybrid-project",
                "configure_llm_endpoint": "http://inference:9000",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_hybrid"

        env = result.env_dict
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "hybrid-project"
        assert env["LLM_CODER_URL"] == "http://inference:9000"

    @pytest.mark.asyncio
    async def test_hybrid_gcp_no_llm(self, policy: ModelInteractivePolicy) -> None:
        """hybrid → services (no llm) → gcp → project → write_config_hybrid."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "hybrid",
                "configure_local_services": ["valkey"],
                "configure_cloud_provider": "gcp",
                "configure_gcp_project": "another-project",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_hybrid"

        env = result.env_dict
        assert env["CLOUD_PROVIDER"] == "gcp"
        assert env["GCP_PROJECT"] == "another-project"
        assert env["AWS_REGION"] == ""
        assert env["LLM_CODER_URL"] == ""


# ------------------------------------------------------------------
# Provenance and step ordering
# ------------------------------------------------------------------


class TestProvenance:
    @pytest.mark.asyncio
    async def test_step_results_ordered(self, policy: ModelInteractivePolicy) -> None:
        """Step results are in execution order."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "hybrid",
                "configure_local_services": ["kafka", "postgres", "llm_inference"],
                "configure_cloud_provider": "aws",
                "configure_aws_region": "us-east-1",
                "configure_llm_endpoint": "http://llm:8000",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        # Verify ordering matches the actual traversal
        assert len(result.step_results) == 5
        assert result.step_results[0].step_key == "choose_deployment_mode"
        assert result.step_results[1].step_key == "configure_local_services"
        assert result.step_results[2].step_key == "configure_cloud_provider"
        assert result.step_results[3].step_key == "configure_aws_region"
        assert result.step_results[4].step_key == "configure_llm_endpoint"

    @pytest.mark.asyncio
    async def test_step_titles_populated(self, policy: ModelInteractivePolicy) -> None:
        """Each step_result has the correct step_title from the policy."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": ["kafka"],
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.step_results[0].step_title == "How will you deploy ONEX?"
        assert result.step_results[1].step_title == "Which local services do you need?"

    @pytest.mark.asyncio
    async def test_policy_name_populated(self, policy: ModelInteractivePolicy) -> None:
        """Result includes policy_name from the executed policy."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "cloud",
                "configure_cloud_provider": "aws",
                "configure_aws_region": "us-east-1",
            }
        )
        result = await InteractiveExecutor(policy, adapter).execute()
        assert result.policy_name == "interactive_onboarding"
