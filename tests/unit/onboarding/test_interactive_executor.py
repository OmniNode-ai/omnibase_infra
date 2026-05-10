# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for InteractiveExecutor — OMN-10782."""

from __future__ import annotations

from pathlib import Path
from typing import Any  # onex-any-ok: state dicts in test assertions

import pytest
import yaml

from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.interactive_executor import InteractiveExecutor
from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.model_interactive_result import ModelInteractiveResult
from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.transition_reducer import TransitionError

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


def _make_simple_policy() -> ModelInteractivePolicy:
    """Build a minimal 2-step policy for isolated unit tests."""
    raw: dict[str, Any] = {
        "policy_name": "test_simple",
        "description": "Minimal 2-step test policy",
        "version": {"major": 1, "minor": 0, "patch": 0},
        "policy_type": "interactive",
        "target_capabilities": ["test"],
        "max_estimated_minutes": 1,
        "steps": [
            {
                "id": "step_one",
                "prompt": "Pick A or B",
                "type": "choice",
                "options": ["a", "b"],
            },
            {
                "id": "step_terminal",
                "prompt": "Done",
                "type": "action",
                "action": "write_env",
            },
        ],
        "transitions": [
            {
                "from": "step_one",
                "responses": {
                    "a": {"next": "step_terminal", "set_state": {"picked": "a"}},
                    "b": {"next": "step_terminal", "set_state": {"picked": "b"}},
                },
            },
            {"from": "step_terminal", "terminal": True},
        ],
        "env_output": {
            "step_terminal": {
                "MY_VAR": "{state.picked}",
            },
        },
    }
    return ModelInteractivePolicy.model_validate(raw)


def _make_multi_choice_policy() -> ModelInteractivePolicy:
    """Build a policy with a multi_choice step for testing list responses."""
    raw: dict[str, Any] = {
        "policy_name": "test_multi_choice",
        "description": "Multi-choice test policy",
        "version": {"major": 1, "minor": 0, "patch": 0},
        "policy_type": "interactive",
        "target_capabilities": ["test"],
        "max_estimated_minutes": 1,
        "steps": [
            {
                "id": "select_items",
                "prompt": "Pick items",
                "type": "multi_choice",
                "options": ["x", "y", "z"],
            },
            {
                "id": "done",
                "prompt": "Done",
                "type": "action",
                "action": "write_env",
            },
        ],
        "transitions": [
            {
                "from": "select_items",
                "on_submit": [
                    {
                        "condition": None,
                        "next": "done",
                        "set_state": {"items": "{response}"},
                    },
                ],
            },
            {"from": "done", "terminal": True},
        ],
        "env_output": {
            "done": {
                "SELECTED": "multi",
            },
        },
    }
    return ModelInteractivePolicy.model_validate(raw)


# ------------------------------------------------------------------
# Basic 2-step policy tests
# ------------------------------------------------------------------


class TestSimplePolicy:
    @pytest.mark.asyncio
    async def test_executor_two_step_policy(self) -> None:
        """Executor drives a simple 2-step policy to completion."""
        policy = _make_simple_policy()
        adapter = AdapterFakeInput(responses={"step_one": "a"})

        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert isinstance(result, ModelInteractiveResult)
        assert result.completed is True
        assert result.terminal_step == "step_terminal"
        assert result.policy_name == "test_simple"

    @pytest.mark.asyncio
    async def test_executor_produces_correct_env_dict(self) -> None:
        """Executor produces correct env_dict at terminal step."""
        policy = _make_simple_policy()
        adapter = AdapterFakeInput(responses={"step_one": "b"})

        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.env_dict == {"MY_VAR": "b"}

    @pytest.mark.asyncio
    async def test_executor_records_step_results(self) -> None:
        """Executor records all step results with step_key mapping."""
        policy = _make_simple_policy()
        adapter = AdapterFakeInput(responses={"step_one": "a"})

        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert len(result.step_results) == 1  # terminal action step not in results
        step_result = result.step_results[0]
        assert step_result.step_key == "step_one"
        assert step_result.step_title == "Pick A or B"
        assert step_result.response == "a"


# ------------------------------------------------------------------
# Action step tests
# ------------------------------------------------------------------


class TestActionSteps:
    @pytest.mark.asyncio
    async def test_action_step_calls_notify_action(self) -> None:
        """Action steps call adapter.notify_action, not collect_*."""
        policy = _make_simple_policy()
        adapter = AdapterFakeInput(responses={"step_one": "a"})

        # Wrap with a mock to verify notify_action is called
        original_notify = adapter.notify_action
        notify_calls: list[ModelInteractiveStep] = []

        async def tracking_notify(step: ModelInteractiveStep) -> None:
            notify_calls.append(step)
            await original_notify(step)

        adapter.notify_action = tracking_notify  # type: ignore[assignment]

        executor = InteractiveExecutor(policy, adapter)
        await executor.execute()

        # Terminal action step triggers notify_action
        assert len(notify_calls) == 1
        assert notify_calls[0].id == "step_terminal"


# ------------------------------------------------------------------
# Multi-choice tests
# ------------------------------------------------------------------


class TestMultiChoiceSteps:
    @pytest.mark.asyncio
    async def test_multi_choice_stores_list_response(self) -> None:
        """Multi-choice steps store list responses in step_results."""
        policy = _make_multi_choice_policy()
        adapter = AdapterFakeInput(responses={"select_items": ["x", "z"]})

        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.completed is True
        assert len(result.step_results) == 1
        assert result.step_results[0].response == ["x", "z"]
        assert result.step_results[0].step_key == "select_items"


# ------------------------------------------------------------------
# Real policy traversals with FakeInputAdapter
# ------------------------------------------------------------------


class TestRealPolicyLocal:
    @pytest.mark.asyncio
    async def test_local_no_llm(self, policy: ModelInteractivePolicy) -> None:
        """Local path without LLM: 2 input steps → write_config_local."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": ["kafka", "postgres"],
            }
        )
        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_local"
        assert result.policy_name == "interactive_onboarding"

        # Env dict assertions
        assert result.env_dict["ONEX_DEPLOYMENT_MODE"] == "local"
        assert result.env_dict["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert result.env_dict["POSTGRES_HOST"] == "localhost"
        assert result.env_dict["LLM_CODER_URL"] == ""  # not selected

        # Step results: 2 input steps recorded
        assert len(result.step_results) == 2
        assert result.step_results[0].step_key == "choose_deployment_mode"
        assert result.step_results[1].step_key == "configure_local_services"

    @pytest.mark.asyncio
    async def test_local_with_llm(self, policy: ModelInteractivePolicy) -> None:
        """Local path with LLM: 3 input steps → write_config_local."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": ["kafka", "postgres", "llm_inference"],
                "configure_llm_endpoint": "http://localhost:8000",
            }
        )
        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_local"
        assert result.env_dict["LLM_CODER_URL"] == "http://localhost:8000"
        assert len(result.step_results) == 3


class TestRealPolicyCloud:
    @pytest.mark.asyncio
    async def test_cloud_aws(self, policy: ModelInteractivePolicy) -> None:
        """Cloud AWS path: 3 input steps → write_config_cloud."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "cloud",
                "configure_cloud_provider": "aws",
                "configure_aws_region": "us-east-1",
            }
        )
        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_cloud"
        assert result.env_dict["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert result.env_dict["CLOUD_PROVIDER"] == "aws"
        assert result.env_dict["AWS_REGION"] == "us-east-1"
        assert result.env_dict["GCP_PROJECT"] == ""

    @pytest.mark.asyncio
    async def test_cloud_gcp(self, policy: ModelInteractivePolicy) -> None:
        """Cloud GCP path: 3 input steps → write_config_cloud."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "cloud",
                "configure_cloud_provider": "gcp",
                "configure_gcp_project": "my-project",
            }
        )
        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_cloud"
        assert result.env_dict["CLOUD_PROVIDER"] == "gcp"
        assert result.env_dict["GCP_PROJECT"] == "my-project"


class TestRealPolicyHybrid:
    @pytest.mark.asyncio
    async def test_hybrid_aws_with_llm(self, policy: ModelInteractivePolicy) -> None:
        """Hybrid AWS+LLM path: 5 input steps → write_config_hybrid."""
        adapter = AdapterFakeInput(
            responses={
                "choose_deployment_mode": "hybrid",
                "configure_local_services": ["kafka", "postgres", "llm_inference"],
                "configure_cloud_provider": "aws",
                "configure_aws_region": "us-west-2",
                "configure_llm_endpoint": "http://my-llm:8000",
            }
        )
        executor = InteractiveExecutor(policy, adapter)
        result = await executor.execute()

        assert result.completed is True
        assert result.terminal_step == "write_config_hybrid"
        assert result.env_dict["ONEX_DEPLOYMENT_MODE"] == "hybrid"
        assert result.env_dict["CLOUD_PROVIDER"] == "aws"
        assert result.env_dict["AWS_REGION"] == "us-west-2"
        assert result.env_dict["LLM_CODER_URL"] == "http://my-llm:8000"
        assert result.env_dict["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        assert len(result.step_results) == 5


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_missing_start_step_raises(self) -> None:
        """Policy with no start_step raises at construction time.

        The TransitionReducer validates graph integrity at __init__ and
        raises TransitionError when start_step is None.  This is defense-
        in-depth — the Pydantic validator normally prevents this state.
        """
        raw: dict[str, Any] = {
            "policy_name": "broken",
            "description": "test",
            "version": {"major": 1, "minor": 0, "patch": 0},
            "policy_type": "interactive",
            "target_capabilities": [],
            "max_estimated_minutes": 1,
            "steps": [
                {"id": "s1", "prompt": "p", "type": "choice", "options": ["a"]},
            ],
            "transitions": [{"from": "s1", "terminal": True}],
            "env_output": {"s1": {"X": "1"}},
        }
        policy = ModelInteractivePolicy.model_validate(raw)
        # start_step is auto-set to "s1" by validator; force override for test
        object.__setattr__(policy, "start_step", None)

        adapter = AdapterFakeInput(responses={})

        # TransitionReducer catches the invalid graph at construction time
        with pytest.raises(TransitionError, match="no start_step"):
            InteractiveExecutor(policy, adapter)
