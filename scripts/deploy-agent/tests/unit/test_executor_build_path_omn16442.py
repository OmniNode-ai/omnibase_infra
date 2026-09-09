# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's build path on the dev lane (OMN-16442, OMN-17291).

Two defects, both proven live on 2026-09-08 and both fail-closed BEFORE any
docker side effect, left the operator entry point with no reachable build
source on the dev lane:

1. ``assert_release_build_promoted`` applied the PROD promotion-lineage
   assertion (clean tree AND HEAD ancestor-of/equal-to ``origin/main``)
   lane-blind, and ``trigger.py`` defaulted ``--build-source`` to ``release``.
   Command ``c73cc38a`` was refused on DIRTY_TREE, and would have been refused
   on NOT_PROMOTED next: a dev head is by construction not an ancestor of a
   release-synced ``origin/main``, so release-mode on the dev lane can never
   pass on any day.
2. ``_stage_workspace`` invoked ``stage_workspace.sh`` with ``DEPLOY_REF``
   unset, so the OMN-17291 guard refused the ambient host tree. Command
   ``a5635af0``: ``Workspace staging failed (exit=5): ERROR: DEPLOY_REF unset``.
   The envelope CARRIED the pin (``git_ref=origin/dev``) and ``self_update``
   had already used it one line earlier; it was dropped between the consumer
   and the staging step.

These tests pin the fix from both ends: the lineage assertion is scoped to the
lanes whose artifacts can reach prod (prod unchanged, stability-test still
gated, dev exempt), and the accepted command's ``git_ref`` reaches
``stage_workspace.sh`` as ``DEPLOY_REF``. The OMN-17291 guard itself is
untouched and is never opted out of.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent import trigger as trigger_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobStore

pytestmark = [pytest.mark.unit, pytest.mark.promotion_guard]


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    pass


def _ok() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


class _GuardStub:
    """Stand-in for the scripts/ promotion-lineage guard module."""

    class ProdLineageError(RuntimeError):
        pass

    def __init__(self, *, raises: bool) -> None:
        self.calls: list[Path] = []
        self._raises = raises

    def assert_prod_build_promoted(self, repo_dir: Path) -> str:
        self.calls.append(Path(repo_dir))
        if self._raises:
            raise self.ProdLineageError(
                "prod build rejected: working tree has uncommitted or untracked changes"
            )
        return "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------
# Defect 1 — the prod promotion-lineage assertion is scoped to prod-bound lanes
# ---------------------------------------------------------------------------


def test_prod_release_build_still_runs_the_lineage_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prod behaviour is byte-for-byte what it was: gate runs, build proceeds."""
    guard = _GuardStub(raises=False)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    executor_mod.assert_release_build_promoted(
        BuildSource.RELEASE, runtime_lane=EnumRuntimeLane.PROD
    )

    assert guard.calls == [Path(executor_mod.REPO_DIR)]


def test_prod_release_build_still_fails_closed_when_the_gate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _GuardStub(raises=True)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    with pytest.raises(_GuardStub.ProdLineageError):
        executor_mod.assert_release_build_promoted(
            BuildSource.RELEASE, runtime_lane=EnumRuntimeLane.PROD
        )


def test_stability_test_release_build_is_still_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stability is NOT exempt: it is where a prod grant's proven digest is built.

    The exemption is the dev lane and only the dev lane. Widening it to
    "not prod" would drop the lineage requirement from the lane that supplies
    the ``stability-proven`` premise of every prod promotion grant
    (CLAUDE.md rule 12, OMN-15243).
    """
    guard = _GuardStub(raises=True)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    with pytest.raises(_GuardStub.ProdLineageError):
        executor_mod.assert_release_build_promoted(
            BuildSource.RELEASE, runtime_lane=EnumRuntimeLane.STABILITY_TEST
        )


def test_dev_release_build_does_not_run_the_prod_lineage_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dev lane is exempt — the guard is never even loaded."""
    guard = _GuardStub(raises=True)  # would refuse if consulted
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    executor_mod.assert_release_build_promoted(
        BuildSource.RELEASE, runtime_lane=EnumRuntimeLane.DEV
    )

    assert guard.calls == []


def test_undeclared_lane_fails_closed_and_runs_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No lane argument means the gate applies — an omission never exempts."""
    guard = _GuardStub(raises=True)
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    with pytest.raises(_GuardStub.ProdLineageError):
        executor_mod.assert_release_build_promoted(BuildSource.RELEASE)


def test_compose_build_passes_its_lane_to_the_lineage_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dev release-mode compose build reaches docker instead of refusing."""
    guard = _GuardStub(raises=True)  # would refuse if consulted
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", lambda: guard)

    captured: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return _ok()

    monkeypatch.setattr(executor_mod, "_run", fake_run)
    monkeypatch.delenv("OMNI_HOME", raising=False)

    DeployExecutor()._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.RELEASE,
        runtime_lane=EnumRuntimeLane.DEV,
    )

    assert guard.calls == []
    assert any("build" in cmd for cmd in captured)


# ---------------------------------------------------------------------------
# Defect 2 — the accepted command's git_ref reaches stage_workspace.sh
# ---------------------------------------------------------------------------


def _install_recording_stage_script(repo: Path) -> Path:
    """Write a stand-in stage_workspace.sh that records the env it was given."""
    script_dir = repo / "scripts" / "runtime_build"
    script_dir.mkdir(parents=True)
    record = repo / "staging-env.json"
    script = script_dir / "stage_workspace.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'python3 -c "import json,os,sys; '
        "json.dump({'DEPLOY_REF': os.environ.get('DEPLOY_REF'), "
        "'OMNI_HOME': os.environ.get('OMNI_HOME'), "
        "'ALLOW_UNPINNED_DEPLOY_SOURCE': "
        "os.environ.get('ALLOW_UNPINNED_DEPLOY_SOURCE')}, "
        f"open({str(record)!r}, 'w'))\"\n"
    )
    script.chmod(0o755)
    return record


def test_dev_workspace_build_stages_with_the_envelopes_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin the operator supplied reaches the staging script as DEPLOY_REF."""
    repo = tmp_path / "repo"
    repo.mkdir()
    record = _install_recording_stage_script(repo)
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(repo))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "omni_home"))
    monkeypatch.delenv("ALLOW_UNPINNED_DEPLOY_SOURCE", raising=False)
    monkeypatch.setattr(
        executor_mod,
        "_run",
        lambda cmd, timeout, **kwargs: _ok(),
    )

    DeployExecutor()._compose_build(
        Scope.RUNTIME,
        "abc1234",
        _noop_phase_update,
        build_source=BuildSource.WORKSPACE,
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref="origin/dev",
    )

    staged = json.loads(record.read_text())
    assert staged["DEPLOY_REF"] == "origin/dev"
    assert staged["OMNI_HOME"] == str(tmp_path / "omni_home")
    # The OMN-17291 guard is satisfied by supplying the pin, never opted out of.
    assert staged["ALLOW_UNPINNED_DEPLOY_SOURCE"] is None


def test_workspace_build_without_a_ref_exports_nothing_and_lets_the_guard_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pin to offer means no DEPLOY_REF — never a substituted fallback ref."""
    repo = tmp_path / "repo"
    repo.mkdir()
    record = _install_recording_stage_script(repo)
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "omni_home"))
    monkeypatch.delenv("DEPLOY_REF", raising=False)

    DeployExecutor._stage_workspace(str(repo), str(tmp_path / "omni_home"))

    staged = json.loads(record.read_text())
    assert staged["DEPLOY_REF"] is None


def test_rebuild_scope_threads_lane_and_ref_into_compose_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_compose_build(
        self: DeployExecutor,
        scope: Scope,
        git_sha: str,
        on_phase_update: object,
        **kwargs: object,
    ) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(DeployExecutor, "_compose_build", fake_compose_build)
    monkeypatch.setattr(DeployExecutor, "_compose_up", lambda *a, **k: None)

    # No self_update stub: OMN-16442 (#3331) removed self-update from this
    # path entirely, so reaching it would be a defect rather than a call to
    # neutralise. test_self_update_job_boundary_omn16442.py owns that.
    DeployExecutor().rebuild_scope(
        Scope.RUNTIME,
        [],
        _noop_phase_update,
        git_sha="abc1234",
        git_ref="origin/dev",
        build_source=BuildSource.WORKSPACE,
        lane=EnumRuntimeLane.DEV,
    )

    assert captured == [
        {
            "build_source": BuildSource.WORKSPACE,
            "runtime_lane": EnumRuntimeLane.DEV,
            "git_ref": "origin/dev",
        }
    ]


class _RefRecordingExecutor:
    def __init__(self) -> None:
        self.rebuild_kwargs: dict[str, object] = {}
        # OMN-18057: the agent reads residue from the executor when it
        # builds the terminal event.
        self.container_residue: list[object] = []
        # OMN-17135: and the sibling SHAs the build vendored, since the
        # command's git_ref pins omnibase_infra alone.
        self.sibling_source_refs: dict[str, str] = {}

    def preflight(self, **kwargs: object) -> None:
        pass

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        return "abc1234"

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        pass

    def seed_infisical(self, **kwargs: object) -> None:
        pass

    def validate_llm_endpoint_env_contract(self) -> None:
        pass

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self.rebuild_kwargs = kwargs
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        return []


@pytest.mark.asyncio
async def test_agent_hands_the_commands_git_ref_to_rebuild_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The consumer already resolved the pin; the build path must receive it."""
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
        git_ref="origin/dev",
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))

    fake_executor = _RefRecordingExecutor()
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    agent.executor = fake_executor  # type: ignore[assignment]
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)

    await agent._run_deploy(cmd)

    assert fake_executor.rebuild_kwargs["git_ref"] == "origin/dev"
    assert fake_executor.rebuild_kwargs["lane"] == EnumRuntimeLane.DEV


# ---------------------------------------------------------------------------
# The trigger's default build source is derived from the lane
# ---------------------------------------------------------------------------


def test_trigger_default_build_source_is_workspace_on_dev() -> None:
    assert (
        trigger_mod._resolve_build_source(None, EnumRuntimeLane.DEV)
        is BuildSource.WORKSPACE
    )


def test_trigger_default_build_source_is_release_off_the_dev_lane() -> None:
    assert (
        trigger_mod._resolve_build_source(None, EnumRuntimeLane.STABILITY_TEST)
        is BuildSource.RELEASE
    )
    assert (
        trigger_mod._resolve_build_source(None, EnumRuntimeLane.PROD)
        is BuildSource.RELEASE
    )


def test_trigger_has_no_literal_build_source_default() -> None:
    """Every lane has a named entry: the map is the default, not a fallback."""
    assert set(trigger_mod.DEFAULT_BUILD_SOURCE_FOR_LANE) == set(EnumRuntimeLane)


def test_trigger_refuses_release_on_the_dev_lane_naming_the_lane() -> None:
    with pytest.raises(trigger_mod.TriggerRefusedError) as excinfo:
        trigger_mod._resolve_build_source("release", EnumRuntimeLane.DEV)

    message = str(excinfo.value)
    assert "release" in message
    assert "dev" in message
    assert "origin/main" in message


def test_trigger_honours_an_explicit_build_source_where_it_can_be_satisfied() -> None:
    assert (
        trigger_mod._resolve_build_source("workspace", EnumRuntimeLane.DEV)
        is BuildSource.WORKSPACE
    )
    assert (
        trigger_mod._resolve_build_source("workspace", EnumRuntimeLane.STABILITY_TEST)
        is BuildSource.WORKSPACE
    )
    assert (
        trigger_mod._resolve_build_source("release", EnumRuntimeLane.STABILITY_TEST)
        is BuildSource.RELEASE
    )


def test_trigger_dev_command_defaults_to_a_workspace_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through main(): no --build-source, dev lane, dry run."""
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv(trigger_mod.ENV_HMAC_SECRET, "unit-test-signing-key")

    exit_code = trigger_mod.main(["--dry-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "build_source:   workspace" in out
    assert '"build_source": "workspace"' in out
    assert '"git_ref": "origin/dev"' in out


def test_trigger_dev_release_request_refuses_before_signing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv(trigger_mod.ENV_HMAC_SECRET, "unit-test-signing-key")

    exit_code = trigger_mod.main(["--build-source", "release", "--dry-run"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "dev" in captured.err
    # Refused before signing: nothing about a command was printed.
    assert "payload (sig masked)" not in captured.out
