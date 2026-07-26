# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15181: wire assert_prod_request_has_stability_digest into the live path.

``assert_prod_request_has_stability_digest`` (executor.py) existed and was
unit-tested in isolation (test_executor_prod_digest.py) but was never called
from ``agent.py``/``consumer.py`` — dead code that never fired on a real
prod deploy. These tests drive it through ``DeployAgent._run_deploy`` (the
actual consume/execute path) using the same ``_FakeExecutor`` pattern as
``test_llm_endpoint_env_contract_guard.py``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "c" * 64
_OTHER_DIGEST = "sha256:" + "d" * 64


class _FakeExecutor:
    """Records call order; resolve_stability_ready_digest is injectable."""

    def __init__(self, *, stability_ready_digest: str | None) -> None:
        self.calls: list[str] = []
        self._stability_ready_digest = stability_ready_digest

    def resolve_stability_ready_digest(self) -> str | None:
        self.calls.append("resolve_stability_ready_digest")
        return self._stability_ready_digest

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
        return ["omninode-prod-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self.calls.append("verify")
        return []

    def deploy_and_verify(self, **kwargs: object) -> list[object]:
        self.calls.append("deploy_and_verify")
        return []


def _prod_cmd(image_digest: str = _DIGEST) -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_digest=image_digest,
    )


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cmd: ModelRebuildRequested,
    fake_executor: _FakeExecutor,
) -> DeployAgent:
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    return agent


@pytest.mark.asyncio
async def test_prod_deploy_with_mismatched_stability_digest_is_rejected_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prod request whose digest != the live stability-ready digest must be
    rejected by the guard BEFORE any deploy effect — preflight must never run."""
    cmd = _prod_cmd(image_digest=_DIGEST)
    fake_executor = _FakeExecutor(stability_ready_digest=_OTHER_DIGEST)
    agent = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    await agent._run_deploy(cmd)

    assert fake_executor.calls == ["resolve_stability_ready_digest"], (
        "guard must fire before preflight/git_pull/rebuild_scope — no deploy "
        f"effect may run on a rejected request; got {fake_executor.calls}"
    )
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert any("stability" in e for e in job.errors), job.errors


@pytest.mark.asyncio
async def test_prod_deploy_with_no_stability_ready_digest_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No resolvable stability-test digest at all (e.g. inspect failed) must
    fail closed exactly like a mismatch — never proceed on an unproven digest."""
    cmd = _prod_cmd(image_digest=_DIGEST)
    fake_executor = _FakeExecutor(stability_ready_digest=None)
    agent = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    await agent._run_deploy(cmd)

    assert fake_executor.calls == ["resolve_stability_ready_digest"]
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"


@pytest.mark.asyncio
async def test_prod_deploy_with_matching_stability_digest_proceeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prod request whose digest matches the live stability-ready digest must
    pass the guard and proceed through the normal deploy sequence."""
    cmd = _prod_cmd(image_digest=_DIGEST)
    fake_executor = _FakeExecutor(stability_ready_digest=_DIGEST)
    agent = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    await agent._run_deploy(cmd)

    assert fake_executor.calls[0] == "resolve_stability_ready_digest"
    assert "preflight" in fake_executor.calls
    assert fake_executor.calls.index(
        "resolve_stability_ready_digest"
    ) < fake_executor.calls.index("preflight")
    assert "rebuild_scope" in fake_executor.calls
    assert "deploy_and_verify" in fake_executor.calls
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "success"


@pytest.mark.asyncio
async def test_dev_lane_deploy_never_invokes_stability_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is prod-only — dev/stability-test deploys must not even
    resolve a stability-ready digest (no behavior change for non-prod lanes)."""
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    fake_executor = _FakeExecutor(stability_ready_digest=None)
    agent = _make_agent(tmp_path, monkeypatch, cmd, fake_executor)

    await agent._run_deploy(cmd)

    assert "resolve_stability_ready_digest" not in fake_executor.calls
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "success"
