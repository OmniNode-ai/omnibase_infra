# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15181: a verify-digest failure must produce a truthful completed event.

Confirms two properties of the failure path through deploy_and_verify() /
DeployAgent._run_deploy():

1. No phantom auto-rollback and no half-recreated container goes unreported —
   a digest mismatch must not trigger any further container mutation (no
   second rebuild_scope/_compose_up/_pull_pinned_image call; grep of
   deploy_agent/ confirms no rollback code path exists anywhere in this
   package today, so this is enforced by construction, not disabled here).
2. The completed job carries a TRUTHFUL status: JobState.status == "failed",
   job.errors names the real cause, and phase_results marks
   Phase.VERIFICATION FAILED rather than leaving it unset/absent — an absent
   VERIFICATION key would let ModelRebuildCompleted.status (computed purely
   from phase_results, see test_events_lane_digest.py) report "success" for a
   deploy that never passed verification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildCompleted,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor, DigestMismatchError
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "c" * 64
_OTHER_DIGEST = "sha256:" + "d" * 64


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class _FakeExecutorRealDeployAndVerify:
    """Uses the REAL DeployExecutor.deploy_and_verify (not mocked) so the
    phase-tracking fix under test actually runs; everything upstream of
    verification is faked to isolate the failure path."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._real = DeployExecutor()

    def resolve_stability_ready_digest(self) -> str | None:
        self.calls.append("resolve_stability_ready_digest")
        return _DIGEST

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

    def deploy_and_verify(self, **kwargs: object) -> list[object]:
        self.calls.append("deploy_and_verify")
        return self._real.deploy_and_verify(**kwargs)


@pytest.mark.asyncio
async def test_digest_mismatch_produces_truthful_failed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_digest=_DIGEST,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")

    published: list[dict] = []
    monkeypatch.setattr(
        agent_mod,
        "publish_result",
        lambda payload, config: published.append(payload) or True,
    )

    fake_executor = _FakeExecutorRealDeployAndVerify()
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]

    # The running container reports a DIFFERENT digest than requested —
    # real docker-inspect mismatch, not a mocked-away success.
    def fake_run(
        run_cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if run_cmd[:2] == ["docker", "inspect"]:
            return _ok(stdout=f"sha256:imageid {_OTHER_DIGEST}\n")
        return _ok()

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        await agent._run_deploy(cmd)

    # 1. No phantom rollback / re-recreate: exactly one rebuild_scope call,
    #    exactly one deploy_and_verify call, nothing after the failure.
    assert fake_executor.calls == [
        "resolve_stability_ready_digest",
        "preflight",
        "git_pull",
        "compose_gen",
        "seed_infisical",
        "validate_llm_endpoint_env_contract",
        "rebuild_scope",
        "deploy_and_verify",
    ], fake_executor.calls

    # 2. Truthful status at the JobState level.
    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "failed"
    assert any("digest" in e.lower() for e in job.errors), job.errors

    # 3. Truthful status in phase_results: VERIFICATION must be FAILED, not
    #    silently absent (the OMN-15181 "phantom success" gap).
    assert job.phase_results[Phase.VERIFICATION] == PhaseStatus.FAILED

    # 4. The completed event actually published must reconstruct as "failed",
    #    not "success" — this is the truthful-status contract downstream
    #    consumers rely on.
    assert published, "completion event must still be published on a failure"
    payload = published[0]
    reconstructed = ModelRebuildCompleted(
        correlation_id=payload["correlation_id"],
        requested_git_ref=payload["requested_git_ref"],
        git_sha=payload["git_sha"],
        started_at=payload["started_at"],
        completed_at=payload["completed_at"],
        duration_seconds=payload["duration_seconds"],
        scope=payload["scope"],
        runtime_lane=payload["runtime_lane"],
        image_ref=payload["image_ref"],
        image_digest=payload["image_digest"],
        services_restarted=payload["services_restarted"],
        phase_results=payload["phase_results"],
        errors=payload["errors"],
        health_checks=payload["health_checks"],
    )
    assert reconstructed.status == "failed", (
        "a digest-mismatch deploy must never reconstruct as status=success "
        f"downstream; phase_results={payload['phase_results']!r}"
    )


@pytest.mark.asyncio
async def test_digest_match_still_marks_verification_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: the new IN_PROGRESS/FAILED phase tracking must not
    break the existing happy path — VERIFICATION still reaches SUCCESS when
    the digest matches and health checks pass."""
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.PROD,
        image_digest=_DIGEST,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)

    fake_executor = _FakeExecutorRealDeployAndVerify()
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]

    health_payload = (
        '{"status": "healthy", "details": '
        '{"is_running": true, "config_prefetch_status": "ok"}}'
    )

    def fake_run(
        run_cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if run_cmd[:2] == ["docker", "inspect"]:
            # verify_running_image_digest now compares the container's
            # `.Image` field (an exact image id) to expected_digest, not a
            # substring match against a RepoDigests-style value.
            return _ok(stdout=f"{_DIGEST}\n")
        if run_cmd[:2] == ["docker", "ps"]:
            return _ok()
        if "omnidash_analytics" in run_cmd:
            return _ok(stdout="t\n")
        if any("/health" in tok for tok in run_cmd):
            return _ok(stdout=health_payload)
        return _ok()

    with patch("deploy_agent.executor._run", side_effect=fake_run):
        await agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None
    assert job.status == "success"
    assert job.phase_results[Phase.VERIFICATION] == PhaseStatus.SUCCESS


def test_no_rollback_code_path_exists_in_deploy_agent_package() -> None:
    """Static confirmation backing claim (1) above: grep the shipped package
    source for any rollback CALLABLE (function/method/class), ignoring prose
    in comments/docstrings that merely discusses the absence of one. A future
    rollback feature MUST update this test deliberately rather than silently
    reintroducing an unreported phantom-recovery path."""
    import re

    import deploy_agent

    callable_pattern = re.compile(r"(?:def|class)\s+\w*rollback\w*", re.IGNORECASE)
    call_pattern = re.compile(r"\brollback\w*\s*\(", re.IGNORECASE)

    pkg_dir = Path(deploy_agent.__file__).parent
    hits: list[str] = []
    for py_file in pkg_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if callable_pattern.search(text) or call_pattern.search(text):
            hits.append(py_file.name)
    assert hits == [], (
        f"unexpected rollback callable found in {hits}; if this is "
        "intentional, this test's assumption (no rollback path) must be "
        "revisited alongside the OMN-15181 failure-semantics guarantee"
    )
