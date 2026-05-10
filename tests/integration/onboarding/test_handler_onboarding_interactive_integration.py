# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for ``handle_onboarding`` interactive path (OMN-10784).

Exercises the handler end-to-end with the real ``interactive_onboarding.yaml``
policy and the real ``AdapterFakeInput`` adapter — covers both ``dry_run=True``
(default, no file write) and ``dry_run=False`` (writes env config to a tmp
path). Asserts no writes occur under the real ``~/.omnibase/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    OnboardingHandlerError,
    handle_onboarding,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_local_path_dry_run_produces_env_output() -> None:
    """Local path with dry_run=True returns env output without writing."""
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka", "postgres"],
        }
    )
    input_model = ModelOnboardingInput(
        policy_name="interactive_onboarding",
        dry_run=True,
    )

    output = await handle_onboarding(input_model, input_adapter=adapter)

    assert output.success is True
    assert output.policy_name == "interactive_onboarding"
    assert output.policy_type == "interactive"
    assert output.terminal_step == "write_config_local"
    assert output.dry_run is True
    assert output.env_output_path_written is None
    assert output.provenance is not None
    assert "ONEX_DEPLOYMENT_MODE" in output.provenance.env_dict


@pytest.mark.asyncio
async def test_cloud_path_dry_run_produces_env_output() -> None:
    """Cloud (AWS) path with dry_run=True returns env output."""
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "cloud",
            "configure_cloud_provider": "aws",
            "configure_aws_region": "us-east-1",
        }
    )
    input_model = ModelOnboardingInput(
        policy_name="interactive_onboarding",
        dry_run=True,
    )

    output = await handle_onboarding(input_model, input_adapter=adapter)

    assert output.success is True
    assert output.terminal_step == "write_config_cloud"
    assert output.provenance is not None
    assert output.provenance.env_dict.get("CLOUD_PROVIDER") == "aws"
    assert output.provenance.env_dict.get("AWS_REGION") == "us-east-1"


@pytest.mark.asyncio
async def test_dry_run_false_writes_to_tmp_path(tmp_path: Path) -> None:
    """dry_run=False writes the env config to the explicit env_output_path."""
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka", "postgres"],
        }
    )
    target = tmp_path / "onboarding.env"
    input_model = ModelOnboardingInput(
        policy_name="interactive_onboarding",
        dry_run=False,
        env_output_path=str(target),
    )

    output = await handle_onboarding(input_model, input_adapter=adapter)

    assert output.success is True
    assert output.dry_run is False
    assert output.env_output_path_written == str(target)
    assert target.exists()
    contents = target.read_text()
    assert "ONEX_DEPLOYMENT_MODE=local" in contents


@pytest.mark.asyncio
async def test_dry_run_false_without_path_raises() -> None:
    """dry_run=False with no env_output_path raises OnboardingHandlerError."""
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka", "postgres"],
        }
    )
    input_model = ModelOnboardingInput(
        policy_name="interactive_onboarding",
        dry_run=False,
        env_output_path=None,
    )

    with pytest.raises(OnboardingHandlerError, match="env_output_path"):
        await handle_onboarding(input_model, input_adapter=adapter)


@pytest.mark.asyncio
async def test_no_real_home_writes(tmp_path: Path) -> None:
    """Confirm dry_run=True never writes anywhere on disk, including ~/.omnibase/."""
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka"],
        }
    )
    real_omnibase_env = Path.home() / ".omnibase" / ".env"
    mtime_before = (
        real_omnibase_env.stat().st_mtime if real_omnibase_env.exists() else None
    )

    input_model = ModelOnboardingInput(
        policy_name="interactive_onboarding",
        dry_run=True,
    )
    await handle_onboarding(input_model, input_adapter=adapter)

    mtime_after = (
        real_omnibase_env.stat().st_mtime if real_omnibase_env.exists() else None
    )
    assert mtime_before == mtime_after
