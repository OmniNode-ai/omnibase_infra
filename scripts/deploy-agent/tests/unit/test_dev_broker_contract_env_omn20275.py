# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev refresh renders broker policy afresh, with a private contract overlay."""

from pathlib import Path

import pytest
from deploy_agent import executor
from deploy_agent.events import EnumRuntimeLane

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def policy_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(executor, "REPO_DIR", str(ROOT))
    monkeypatch.setattr(
        executor, "operator_store_overlay", lambda: {"DEV_REDPANDA_MEMORY": "24G"}
    )
    monkeypatch.setattr(executor, "_lane_env_overlay", dict)
    monkeypatch.setattr(executor, "active_dev_instance", lambda: "dev-201")
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(tmp_path))
    monkeypatch.setenv("DEV_REDPANDA_MEMORY", "3G")
    monkeypatch.setenv("DEV_REDPANDA_MEM_LIMIT", "4g")
    return tmp_path


def test_refresh_reads_contract_overlay_again_without_settings_change(
    policy_env: Path,
) -> None:
    with executor.active_deploy_lane(EnumRuntimeLane.DEV):
        env = executor._compose_env()
        assert env["DEV_REDPANDA_MEMORY"] == "12G"
        assert env["DEV_REDPANDA_MEM_LIMIT"] == "16g"
        overlay = policy_env / "runtime_policy.contract.yaml"
        overlay.write_text(
            "dev_redpanda_memory: 12288M\ndev_redpanda_mem_limit: 16384m\n"
        )
        env = executor._compose_env({"DEV_REDPANDA_MEMORY": "30G"})
        assert env["DEV_REDPANDA_MEMORY"] == "12288M"
        assert env["DEV_REDPANDA_MEM_LIMIT"] == "16384m"
        overlay.write_text("dev_redpanda_memory: 13G\ndev_redpanda_mem_limit: 16g\n")
        with pytest.raises(RuntimeError, match=r"broker.*policy"):
            executor._compose_env()


@pytest.mark.parametrize("lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST])
def test_other_lanes_do_not_read_dev_broker_contract_overlay(
    policy_env: Path, lane: EnumRuntimeLane
) -> None:
    (policy_env / "runtime_policy.contract.yaml").write_text("invalid: true\n")
    with executor.active_deploy_lane(lane):
        executor._compose_env()


def test_other_dev_instances_do_not_read_dev_201_broker_overlay(
    policy_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "active_dev_instance", lambda: "other-instance")
    (policy_env / "runtime_policy.contract.yaml").write_text("invalid: true\n")
    with executor.active_deploy_lane(EnumRuntimeLane.DEV):
        executor._compose_env()
