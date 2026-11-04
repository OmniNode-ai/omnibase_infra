# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-only agent trials use a trusted parent and read the child's loaded SHA."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent.events import EnumRuntimeLane

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.deploy_agent_branch_trial_effect import effect
from omnibase_infra.handlers.deploy_agent_branch_trial_effect.effect import (
    supervise_trial,
)
from omnibase_infra.models.model_deploy_agent_branch_trial_request import (
    ModelDeployAgentBranchTrialRequest,
)
from omnibase_infra.nodes.node_deploy_agent_branch_trial_effect.runtime import (
    queue_from_host,
)

NON_DEV_LANES = tuple(
    lane.value for lane in EnumRuntimeLane if lane is not EnumRuntimeLane.DEV
)


@pytest.fixture(autouse=True)
def configured_host(tmp_path, monkeypatch):
    for name, value in {
        "DEPLOY_AGENT_ALLOWED_LANES": "dev",
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        "DEPLOY_AGENT_DIR": str(tmp_path / "scripts/deploy-agent"),
        "DEPLOY_AGENT_STATE_DIR": str(tmp_path / "state"),
        "DEPLOY_AGENT_TRIAL_CONTRACT": str(
            Path(__file__).resolve().parents[3]
            / "src/omnibase_infra/nodes/node_deploy_agent_branch_trial_effect/contract.yaml"
        ),
        "DEPLOY_AGENT_PYTHON": sys.executable,
        "DEPLOY_AGENT_PORT": "1",
        "HOME": str(tmp_path / "home"),
    }.items():
        monkeypatch.setenv(name, value)
    effect.configure_host(dict(os.environ))


@pytest.mark.parametrize("target", [*NON_DEV_LANES, "other", "unknown"])
def test_non_dev_request_is_refused_before_any_host_effect(
    tmp_path, monkeypatch, target
):
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    with pytest.raises(ValueError, match="dev"):
        request = ModelDeployAgentBranchTrialRequest(
            correlation_id=uuid4(),
            target=target,
            branch="fix-agent",
            head="a" * 40,
            duration_seconds=30,
        )
        queue_from_host(tmp_path / "state", request)
    assert list(tmp_path.iterdir()) == []


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


_AGENT = """import http.server,json,os,subprocess,sys
if CRASH:
    raise SystemExit(23)
sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body=json.dumps({"loaded_code_sha":sha,"state":"idle","active_job":None,"pending_publish_count":0}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self,*a): pass
http.server.HTTPServer.allow_reuse_address=True
http.server.HTTPServer(("127.0.0.1",int(os.environ["DEPLOY_AGENT_PORT"])),Handler).serve_forever()
"""


@pytest.fixture
def trial_host(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "dev")
    _git(origin, "config", "user.email", "test@example.invalid")
    _git(origin, "config", "user.name", "Test")
    agent = origin / "scripts/deploy-agent/deploy_agent"
    agent.mkdir(parents=True)
    (agent / "__init__.py").touch()
    (agent / "__main__.py").write_text(_AGENT.replace("CRASH", "False"))
    source = Path(__file__).resolve().parents[3]
    for name in ("events.py", "tracking_ref.py"):
        shutil.copyfile(
            source / "scripts/deploy-agent/deploy_agent" / name, agent / name
        )
    source_package = source / "src/omnibase_infra"
    target_package = origin / "src/omnibase_infra"
    for name in (
        "model_deploy_agent_branch_trial_request.py",
        "model_deploy_agent_branch_trial_result.py",
        "model_deploy_agent_branch_trial_host.py",
        "model_deploy_agent_branch_trial_policy.py",
    ):
        destination = target_package / "models" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_package / "models" / name, destination)
    shutil.copytree(
        source_package / "handlers/deploy_agent_branch_trial_effect",
        target_package / "handlers/deploy_agent_branch_trial_effect",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(
        source_package / "handlers/handler_deploy_agent_branch_trial.py",
        target_package / "handlers/handler_deploy_agent_branch_trial.py",
    )
    shutil.copytree(
        source_package / "nodes/node_deploy_agent_branch_trial_effect",
        target_package / "nodes/node_deploy_agent_branch_trial_effect",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "baseline")
    baseline = _git(origin, "rev-parse", "HEAD")
    _git(origin, "switch", "-c", "trial-head")
    (agent / "__main__.py").write_text(_AGENT.replace("CRASH", "True"))
    _git(origin, "commit", "-am", "failed trial")
    failed = _git(origin, "rev-parse", "HEAD")
    _git(origin, "switch", "dev")
    clone = tmp_path / "agent"
    subprocess.run(
        ["git", "clone", str(origin), str(clone)],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_PORT", str(port))
    monkeypatch.setenv("DEPLOY_AGENT_PYTHON", sys.executable)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone / "scripts/deploy-agent"))
    effect.configure_host(dict(os.environ))
    return clone, tmp_path / "state", baseline, failed, origin


def test_failed_head_returns_to_origin_dev_with_loaded_code_readback(trial_host):
    clone, state, baseline, failed, _ = trial_host
    request = ModelDeployAgentBranchTrialRequest(
        correlation_id=uuid4(),
        target="dev",
        branch="trial-head",
        head=failed,
        duration_seconds=10,
    )
    result = supervise_trial(request, repo_root=clone, state_dir=state)
    assert result.status == "failed"
    assert result.return_ref == "origin/dev"
    assert result.restored_loaded_code_sha == baseline
    assert result.loaded_code_sha is None
    assert "exit" in result.error
    assert _git(clone, "rev-parse", "HEAD") == baseline
    assert _git(clone, "rev-parse", "origin/dev") == baseline
    record = json.loads((state / "agent-branch-trial/receipt.json").read_text())
    assert record["restored_loaded_code_sha"] == baseline


def test_healthy_head_is_observed_then_expires_and_returns(trial_host):
    clone, state, baseline, _, origin = trial_host
    _git(origin, "switch", "trial-head")
    path = origin / "scripts/deploy-agent/deploy_agent/__main__.py"
    path.write_text(_AGENT.replace("CRASH", "False") + "\n# healthy branch\n")
    _git(origin, "commit", "-am", "healthy trial")
    head = _git(origin, "rev-parse", "HEAD")
    request = ModelDeployAgentBranchTrialRequest(
        correlation_id=uuid4(),
        target="dev",
        branch="trial-head",
        head=head,
        duration_seconds=3,
    )
    started = time.monotonic()
    result = supervise_trial(request, repo_root=clone, state_dir=state)
    assert result.status == "expired"
    assert result.loaded_code_sha == head
    assert result.observed_at is not None
    assert result.restored_loaded_code_sha == baseline
    assert time.monotonic() - started < 10


def test_moving_branch_cannot_substitute_a_different_head(trial_host):
    clone, state, baseline, _, _ = trial_host
    request = ModelDeployAgentBranchTrialRequest(
        correlation_id=uuid4(),
        target="dev",
        branch="trial-head",
        head=baseline,
        duration_seconds=10,
    )
    result = supervise_trial(request, repo_root=clone, state_dir=state)
    assert result.status == "failed"
    assert "head" in result.error
    assert result.restored_loaded_code_sha == baseline


def test_signed_command_adapter_queues_the_effect_without_a_runtime_rebuild(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from deploy_agent import agent as agent_mod
    from deploy_agent.auth import verify_command
    from deploy_agent.events import (
        BuildSource,
        EnumRuntimeLane,
        ModelRebuildRequested,
        Scope,
    )
    from deploy_agent.job_state import JobStore
    from deploy_agent.trigger import build_rebuild_command, command_to_signed_envelope

    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    root = tmp_path / "agent-branch-trial"
    root.mkdir()
    (root / "supervisor.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "observed_at": time.time(),
                "target": "dev",
                "return_ref": "origin/dev",
                "valid_until": time.time() + 5,
            }
        )
    )
    command = build_rebuild_command(
        correlation_id=uuid4(),
        requested_by="test/branch-trial",
        scope=Scope.AGENT_TRIAL,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
        git_ref="a" * 40,
        services=[],
        agent_branch="fix-agent",
        agent_trial_seconds=30,
    )
    envelope = command_to_signed_envelope(command, "test-secret")
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "test-secret")
    assert verify_command(envelope)
    command = ModelRebuildRequested.model_validate(
        {k: v for k, v in envelope.items() if k != "_signature"}
    )
    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    agent.job_store = JobStore(tmp_path)
    agent.job_store.accept(
        correlation_id=command.correlation_id, command=command.model_dump(mode="json")
    )
    agent._state = "idle"
    agent._run_deploy = Mock(
        side_effect=AssertionError("must not touch runtime or broker")
    )
    agent._self_update_post_terminal = Mock(
        side_effect=AssertionError("queued trial must keep dev supervisor")
    )
    agent._publish_trial_result = Mock()
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent_mod, "single_flight_lock", nullcontext)
    agent._execute_command(command)
    agent._run_deploy.assert_not_called()
    agent._publish_trial_result.assert_called_once()
    assert agent._publish_trial_result.call_args.args[0]["status"] == "queued"
    assert agent._publish_trial_result.call_args.args[0]["loaded_code_sha"] is None
    assert agent.job_store.load(command.correlation_id).status == "success"
    assert (
        json.loads((root / "request.json").read_text())["request"]["head"]
        == command.git_ref
    )
    assert json.loads((root / "receipt.json").read_text())["loaded_code_sha"] is None


@pytest.mark.parametrize("target", NON_DEV_LANES)
def test_command_schema_refuses_non_dev_even_with_prod_digest(target):
    from deploy_agent.events import ModelRebuildRequested

    with pytest.raises(ValueError, match="dev"):
        ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="test/branch-trial",
            scope="agent-trial",
            runtime_lane=target,
            git_ref="a" * 40,
            image_digest="sha256:" + "b" * 64,
            agent_branch="fix-agent",
            agent_trial_seconds=30,
        )


def test_image_prune_install_uses_the_pinned_nodes_manifest_and_restores_units(
    tmp_path, monkeypatch
):
    from omnibase_infra.handlers.deploy_agent_branch_trial_effect import effect

    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    tree = tmp_path / "head"
    node = tree / "src/omnibase_infra/nodes/node_deploy_agent_image_prune_effect"
    node.mkdir(parents=True)
    (node / "install.json").write_text(
        json.dumps(
            {
                "operation": "deploy_agent.images.install",
                "repository": ".",
                "policy_path": "src/omnibase_infra/nodes/node_deploy_agent_image_prune_effect/contract.yaml",
            }
        )
    )
    (node / "contract.yaml").write_text("name: node_deploy_agent_image_prune_effect\n")
    dispatch = tmp_path / "registry/.onex-dispatch-venv/bin"
    dispatch.mkdir(parents=True)
    (dispatch / "python").symlink_to(sys.executable)
    monkeypatch.setenv("UNIT_DRIFT_REGISTRY_ROOT", str(dispatch.parents[1]))
    effect.configure_host(dict(os.environ))
    calls = []
    unit_dir = home / ".config/systemd/user"
    unit_dir.mkdir(parents=True)
    service = unit_dir / "onex-deploy-agent-image-prune.service"
    service.write_text("old service\n")

    def action(argv, *, cwd, deadline):
        calls.append(argv)
        assert "prod" not in " ".join(argv)
        assert "redpanda" not in " ".join(argv)
        if argv[0] == "systemctl":
            if "is-enabled" in argv or "is-active" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            if "show" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, "Result=success\nExecMainStatus=0\n", ""
                )
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[1:3] == ["node", "node_deploy_agent_image_prune_effect"]
        assert str(node / "install.json") in argv
        assert str(node / "contract.yaml") in argv
        service.write_text("trial service\n")
        (unit_dir / "onex-deploy-agent-image-prune.timer").write_text("trial timer\n")
        env = home / ".config/onex/deploy-agent-image-prune.env"
        env.parent.mkdir(parents=True)
        env.write_text("trial source\n")
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "status": "success",
                    "result": {"success": True, "installed_revision": "b" * 64},
                }
            ),
            "",
        )

    monkeypatch.setattr(effect, "_action", action)
    root = tmp_path / "state"
    root.mkdir()
    revision, observation = effect._install_image_prune(tree, root, time.time() + 30)
    assert revision == "b" * 64
    assert "ExecMainStatus=0" in observation
    assert service.read_text() == "trial service\n"
    effect._restore_units(root, time.time() + 30)
    assert service.read_text() == "old service\n"
    assert not (unit_dir / "onex-deploy-agent-image-prune.timer").exists()
    assert not (home / ".config/onex/deploy-agent-image-prune.env").exists()
    assert any(
        "start" in call and call[-1].endswith("image-prune.service") for call in calls
    )
    assert any(
        "disable" in call and call[-1].endswith("image-prune.timer") for call in calls
    )


def test_install_manifest_cannot_redirect_the_host_action(tmp_path, monkeypatch):
    from omnibase_infra.handlers.deploy_agent_branch_trial_effect import effect

    node = tmp_path / "src/omnibase_infra/nodes/node_deploy_agent_image_prune_effect"
    node.mkdir(parents=True)
    (node / "install.json").write_text(json.dumps({"operation": "prod.promote"}))
    with pytest.raises(ValueError, match="manifest"):
        effect._install_image_prune(tmp_path, tmp_path / "state", time.time() + 30)
    assert not (tmp_path / "state").exists()


def test_real_launcher_parent_survives_a_failed_head_and_keeps_dev_running(
    trial_host, monkeypatch
):
    from omnibase_infra.handlers.deploy_agent_branch_trial_effect import effect

    clone, state, baseline, failed, _ = trial_host
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone / "scripts/deploy-agent"))
    monkeypatch.setenv("DEPLOY_AGENT_STATE_DIR", str(state))
    monkeypatch.setenv("DEPLOY_SOURCE_CLONE_ROOT", str(clone.parent))
    monkeypatch.delenv("DEPLOY_AGENT_SUPERVISED", raising=False)
    store = clone.parent / "operator.env"
    store.write_text("")
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(store))
    monkeypatch.setenv(
        "DEPLOY_AGENT_ENV_PROTECTED",
        "DEPLOY_AGENT_DIR DEPLOY_AGENT_STATE_DIR DEPLOY_AGENT_TRACKING_REF DEPLOY_AGENT_ALLOWED_LANES DEPLOY_AGENT_PORT DEPLOY_AGENT_PYTHON DEPLOY_SOURCE_CLONE_ROOT",
    )
    launcher = (
        Path(__file__).resolve().parents[3]
        / "scripts/deploy-agent/deploy/deploy-agent-launch.sh"
    )
    parent = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=clone / "scripts/deploy-agent",
        env=os.environ.copy(),
        start_new_session=True,
    )
    try:
        effect._wait_loaded(parent, baseline, time.time() + 10)
        request = ModelDeployAgentBranchTrialRequest(
            correlation_id=uuid4(),
            target="dev",
            branch="trial-head",
            head=failed,
            duration_seconds=10,
        )
        queued = queue_from_host(state, request)
        assert queued.status == "queued"
        deadline = time.time() + 15
        record = {}
        while time.time() < deadline:
            record = effect.read_receipt(state)
            if record.get("restored_loaded_code_sha"):
                break
            assert parent.poll() is None
            time.sleep(0.05)
        assert record["status"] == "failed"
        assert record["restored_loaded_code_sha"] == baseline
        assert parent.poll() is None
        assert effect._health()["loaded_code_sha"] == baseline
    finally:
        effect._stop(parent)


def test_normal_rebuild_wire_payload_has_no_agent_trial_fields():
    from deploy_agent.events import BuildSource, EnumRuntimeLane, Scope
    from deploy_agent.trigger import build_rebuild_command, command_to_signed_envelope

    command = build_rebuild_command(
        correlation_id=uuid4(),
        requested_by="test/normal-command",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
        git_ref="a" * 40,
        services=[],
    )
    envelope = command_to_signed_envelope(command, "test-secret")
    assert (
        not {"agent_branch", "agent_trial_seconds", "agent_installer"} & envelope.keys()
    )


def test_signed_consumer_admits_trial_and_rejects_a_bad_signature(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from deploy_agent.consumer import DeployConsumer
    from deploy_agent.events import (
        TOPIC_REBUILD_REQUESTED,
        BuildSource,
        EnumRuntimeLane,
        Scope,
    )
    from deploy_agent.job_state import JobStore
    from deploy_agent.trigger import build_rebuild_command, command_to_signed_envelope

    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "test-secret")
    command = build_rebuild_command(
        correlation_id=uuid4(),
        requested_by="test/signed-consumer",
        scope=Scope.AGENT_TRIAL,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
        git_ref="a" * 40,
        services=[],
        agent_branch="fix-agent",
        agent_trial_seconds=30,
    )
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = JobStore(tmp_path)
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    signed = command_to_signed_envelope(command, "test-secret")
    msg = SimpleNamespace(
        value=signed, topic=TOPIC_REBUILD_REQUESTED, partition=0, offset=3
    )
    accepted, reason = consumer._process_message(msg)
    assert accepted == command
    assert reason is None
    assert consumer.job_store.load(command.correlation_id) is not None
    msg.value = {**signed, "_signature": "invalid"}
    rejected, reason = consumer._process_message(msg)
    assert rejected is None
    assert reason is not None


def test_recovery_reads_the_dev_head_after_a_concurrent_refresh(
    trial_host, monkeypatch
):
    clone, state, baseline, failed, origin = trial_host
    real_stop = effect._stop
    refreshed = []

    def refresh_after_stop(child):
        real_stop(child)
        if not refreshed:
            path = origin / "scripts/deploy-agent/deploy_agent/__main__.py"
            path.write_text(_AGENT.replace("CRASH", "False") + "\n# new dev head\n")
            _git(origin, "commit", "-am", "advance dev")
            refreshed.append(_git(origin, "rev-parse", "HEAD"))
            _git(clone, "fetch", "origin", "dev")

    monkeypatch.setattr(effect, "_stop", refresh_after_stop)
    request = ModelDeployAgentBranchTrialRequest(
        correlation_id=uuid4(),
        target="dev",
        branch="trial-head",
        head=failed,
        duration_seconds=10,
    )
    result = supervise_trial(request, repo_root=clone, state_dir=state)
    assert result.status == "failed"
    assert refreshed[0] != baseline
    assert result.restored_loaded_code_sha == refreshed[0]
    assert _git(clone, "rev-parse", "HEAD") == refreshed[0]


def test_queue_acknowledgement_uses_the_contract_topic_with_cached_handler(
    monkeypatch,
):
    from unittest.mock import Mock

    from deploy_agent import agent as agent_mod

    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    agent._kafka_config = Mock()
    publish = Mock(return_value=True)
    monkeypatch.setattr(agent_mod, "publish_result", publish)
    payload = {"correlation_id": str(uuid4()), "status": "queued"}

    agent._publish_trial_result(payload)

    contract = Path(os.environ["DEPLOY_AGENT_TRIAL_CONTRACT"])
    from yaml import safe_load

    assert publish.call_args.args == (payload, agent._kafka_config)
    assert publish.call_args.kwargs == {
        "topic": safe_load(contract.read_text())["terminal_event"]
    }
