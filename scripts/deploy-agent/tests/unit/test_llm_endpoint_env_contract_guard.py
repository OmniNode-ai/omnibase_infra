# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deploy-agent guard for LLM endpoint env contract drift."""

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import ModelRebuildRequested, Scope
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobStore


def _completed(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


@pytest.mark.unit
def test_validate_llm_endpoint_env_contract_checks_process_and_env_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    repo.mkdir()
    (repo / ".env").write_text("LLM_EMBEDDING_URL=http://192.168.86.201:8100\n")
    (home / ".omnibase").mkdir(parents=True)
    (home / ".omnibase" / ".env").write_text(
        "LLM_EMBEDDING_URL=http://192.168.86.201:8100\n"
    )
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(repo))
    monkeypatch.setenv("HOME", str(home))

    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        timeout: int,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return _completed()

    monkeypatch.setattr(executor_mod, "_run", fake_run)

    DeployExecutor().validate_llm_endpoint_env_contract()

    assert len(captured) == 3
    assert "--env-file" not in captured[0]
    env_files = [
        command[command.index("--env-file") + 1]
        for command in captured[1:]
        if "--env-file" in command
    ]
    assert env_files == [str(home / ".omnibase" / ".env"), str(repo / ".env")]


@pytest.mark.unit
def test_validate_llm_endpoint_env_contract_fails_on_stale_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_file = repo / ".env"
    env_file.write_text("LLM_EMBEDDING_URL=http://192.168.86.200:8100\n")
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(repo))
    monkeypatch.setenv("OMNIBASE_ENV_FILE", str(env_file))

    def fake_run(
        cmd: list[str],
        timeout: int,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if "--env-file" in cmd:
            return _completed(
                returncode=1,
                stderr="LLM_EMBEDDING_URL does not match canonical running slot",
            )
        return _completed()

    monkeypatch.setattr(executor_mod, "_run", fake_run)

    with pytest.raises(RuntimeError, match=str(env_file)):
        DeployExecutor().validate_llm_endpoint_env_contract()


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def preflight(self, **kwargs: object) -> None:
        self.calls.append("preflight")

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self.calls.append("git_pull")
        return "abc123"

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: object) -> None:
        self.calls.append("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self.calls.append("rebuild_scope")
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self.calls.append("verify")
        return []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runtime_deploy_validates_llm_env_before_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload: False)

    await agent._run_deploy(cmd)

    assert fake_executor.calls.index("seed_infisical") < fake_executor.calls.index(
        "validate_llm_endpoint_env_contract"
    )
    assert fake_executor.calls.index(
        "validate_llm_endpoint_env_contract"
    ) < fake_executor.calls.index("rebuild_scope")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_core_deploy_does_not_validate_runtime_llm_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.CORE,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload: False)

    await agent._run_deploy(cmd)

    assert "validate_llm_endpoint_env_contract" not in fake_executor.calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_full_deploy_validates_llm_env_before_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.FULL,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload: False)

    await agent._run_deploy(cmd)

    assert fake_executor.calls.index("seed_infisical") < fake_executor.calls.index(
        "validate_llm_endpoint_env_contract"
    )
    assert fake_executor.calls.index(
        "validate_llm_endpoint_env_contract"
    ) < fake_executor.calls.index("rebuild_scope")
