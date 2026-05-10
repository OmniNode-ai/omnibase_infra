# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Proof of Life — end-to-end interactive onboarding integration (OMN-10786).

Three parametrized integration tests covering local, cloud, and hybrid
deployment paths. Each test:

1. Loads the real ``interactive_onboarding.yaml`` policy from disk.
2. Drives the full transition table via ``InteractiveExecutor`` +
   ``AdapterFakeInput`` — no mocking of executor internals.
3. Asserts the produced env_dict, visited steps, terminal step, and
   provenance fields.
4. Round-trips the env_dict through ``ConfigWriter`` to a tmp_path and
   reads back to verify content.
5. Asserts no writes occur under the real ``~/.omnibase/`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.config_writer import ConfigWriter
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


@dataclass(frozen=True)
class _PathSpec:
    """A single onboarding path under test."""

    label: str
    responses: dict[str, str | list[str]]
    expected_terminal: str
    expected_visited: list[str]
    expected_env: dict[str, str]


_LOCAL = _PathSpec(
    label="local-no-llm",
    responses={
        "choose_deployment_mode": "local",
        "configure_local_services": ["kafka", "postgres"],
    },
    expected_terminal="write_config_local",
    expected_visited=["choose_deployment_mode", "configure_local_services"],
    expected_env={
        "ONEX_DEPLOYMENT_MODE": "local",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:19092",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5436",
        "VALKEY_HOST": "localhost",
        "VALKEY_PORT": "16379",
        "LLM_CODER_URL": "",
    },
)

_CLOUD = _PathSpec(
    label="cloud-aws",
    responses={
        "choose_deployment_mode": "cloud",
        "configure_cloud_provider": "aws",
        "configure_aws_region": "us-east-1",
    },
    expected_terminal="write_config_cloud",
    expected_visited=[
        "choose_deployment_mode",
        "configure_cloud_provider",
        "configure_aws_region",
    ],
    expected_env={
        "ONEX_DEPLOYMENT_MODE": "cloud",
        "CLOUD_PROVIDER": "aws",
        "AWS_REGION": "us-east-1",
        "GCP_PROJECT": "",
    },
)

_HYBRID = _PathSpec(
    label="hybrid-gcp-with-llm",
    responses={
        "choose_deployment_mode": "hybrid",
        "configure_local_services": ["kafka", "llm_inference"],
        "configure_cloud_provider": "gcp",
        "configure_gcp_project": "my-project",
        "configure_llm_endpoint": "http://localhost:8000",
    },
    expected_terminal="write_config_hybrid",
    expected_visited=[
        "choose_deployment_mode",
        "configure_local_services",
        "configure_cloud_provider",
        "configure_gcp_project",
        "configure_llm_endpoint",
    ],
    expected_env={
        "ONEX_DEPLOYMENT_MODE": "hybrid",
        "CLOUD_PROVIDER": "gcp",
        "GCP_PROJECT": "my-project",
        "AWS_REGION": "",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:19092",
        "LLM_CODER_URL": "http://localhost:8000",
    },
)


@pytest.fixture
def policy() -> ModelInteractivePolicy:
    raw = yaml.safe_load(POLICY_PATH.read_text())
    return ModelInteractivePolicy.model_validate(raw)


@pytest.mark.asyncio
@pytest.mark.serial
@pytest.mark.parametrize(
    "spec",
    [_LOCAL, _CLOUD, _HYBRID],
    ids=lambda s: s.label,
)
async def test_full_transition_table_e2e(
    spec: _PathSpec,
    policy: ModelInteractivePolicy,
    tmp_path: Path,
) -> None:
    """For each deployment path: traverse → assert env → write to tmp → read back.

    OMN-10786 Proof of Life. Each path drives the full transition table
    end-to-end with the real policy, asserts terminal step, visited steps,
    env output, provenance, and round-trips through ConfigWriter.
    """
    real_omnibase_env = Path.home() / ".omnibase" / ".env"
    mtime_before = (
        real_omnibase_env.stat().st_mtime if real_omnibase_env.exists() else None
    )

    adapter = AdapterFakeInput(responses=dict(spec.responses))
    result = await InteractiveExecutor(policy, adapter).execute()

    # 1. Path traversal
    assert result.completed is True
    assert result.terminal_step == spec.expected_terminal

    # 2. Step ordering
    visited = [sr.step_key for sr in result.step_results]
    assert visited == spec.expected_visited

    # 3. Env output
    for key, expected_val in spec.expected_env.items():
        assert result.env_dict.get(key) == expected_val, (
            f"{spec.label}: env[{key}] expected {expected_val!r}, "
            f"got {result.env_dict.get(key)!r}"
        )

    # 4. Provenance — ModelInteractiveResult carries the policy_name so the
    #    caller can correlate the run back to the source policy. policy_type
    #    + policy_version live on the policy itself; assert against the
    #    fixture.
    assert result.policy_name == "interactive_onboarding"
    assert result.policy_name == policy.policy_name
    assert policy.policy_type == "interactive"
    assert policy.version == {"major": 1, "minor": 0, "patch": 0}

    # 5. Config writer round-trip
    target = tmp_path / f"{spec.label}.env"
    writer = ConfigWriter()
    writer.write(result.env_dict, target)
    assert target.exists()

    written_lines = target.read_text().splitlines()
    written_pairs = dict(
        line.split("=", 1) for line in written_lines if "=" in line and line.strip()
    )
    for key, expected_val in spec.expected_env.items():
        assert written_pairs.get(key) == expected_val, (
            f"{spec.label}: round-trip env[{key}] expected {expected_val!r}, "
            f"got {written_pairs.get(key)!r}"
        )

    # 6. No real ~/.omnibase/.env writes
    mtime_after = (
        real_omnibase_env.stat().st_mtime if real_omnibase_env.exists() else None
    )
    assert mtime_before == mtime_after, (
        f"{spec.label}: real ~/.omnibase/.env was modified during the test"
    )
