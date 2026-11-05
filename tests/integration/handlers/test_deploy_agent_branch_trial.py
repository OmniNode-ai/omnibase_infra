# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-consumer routing and the agent's own HTTP readback composition."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent import agent as agent_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore
from yaml import safe_load

pytestmark = pytest.mark.integration


def test_host_consumer_contract_does_not_register_a_second_rebuild_consumer():
    contract = safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "src/omnibase_infra/nodes/node_deploy_agent_branch_trial_effect/contract.yaml"
        ).read_text()
    )
    # The signed DeployConsumer owns this command and calls the host handler.
    # A runtime subscription would bypass that admission and has no host state.
    assert not contract["event_bus"].get("subscribe_topics")
    assert contract["runtime_profiles"] == ["local-dev"]
    assert contract["handler_routing"]["handlers"][0]["handler"]["name"] == (
        "HandlerDeployAgentBranchTrial"
    )


@pytest.mark.asyncio
async def test_agent_startup_wires_the_real_branch_trial_readback(
    tmp_path, monkeypatch
):
    """Use run()'s callback, rather than a test-authored replacement for it."""
    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    agent._kafka_config = SimpleNamespace(bootstrap_servers="unused.invalid:1")
    agent._allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    agent._router = None
    agent._state = "idle"
    agent._settle_guard = threading.Lock()
    agent._settle_future = None
    agent.job_store = JobStore(tmp_path)
    agent._accept_backlog = SimpleNamespace(latest=lambda: None)
    agent._retry_pending_publishes = Mock()
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent_mod, "record_loaded_code_sha", lambda _: "a" * 40)
    record = {"status": "expired", "loaded_code_sha": "a" * 40}
    root = tmp_path / "agent-branch-trial"
    root.mkdir()
    (root / "receipt.json").write_text(json.dumps(record))
    apps = []

    class AppCapturedError(Exception):
        pass

    def capture_app(**kwargs):
        apps.append(create_health_app(**kwargs))
        raise AppCapturedError

    monkeypatch.setattr(agent_mod, "create_health_app", capture_app)
    with pytest.raises(AppCapturedError):
        await agent.run()
    async with TestClient(TestServer(apps[0])) as client:
        response = await client.get("/agent-branch-trial")
        assert response.status == 200
        assert await response.json() == record
        health = await client.get("/health")
        assert health.status == 200
        assert (await health.json())["state"] == "idle"
