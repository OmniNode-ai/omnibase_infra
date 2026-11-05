# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deploy regression for h053's UID 0, mode 0755 tmp binds (OMN-19998).

Run the shipped stand-up path with a modeled host filesystem: SSH executes
the real remote shell, Compose renders the shipped h053 volumes, and install
and stat model privileged ownership changes without touching a live runner.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from omnibase_infra.models.model_deploy_runners_result import ModelDeployRunnersResult
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/deploy-runners.sh"
COMPOSE = ROOT / "docker/docker-compose.runners-hp053-ci-runner.yml"

HOST_TOOL = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
if tool == "uv":
    args = args[args.index("python") + 1:]
    os.execv(os.environ["TMP_NODE_PYTHON"], [os.environ["TMP_NODE_PYTHON"], *args])
state_path = Path(os.environ["TMP_HOST_STATE"])
state = json.loads(state_path.read_text())
with open(os.environ["TMP_HOST_CALLS"], "a") as log:
    log.write(tool + " " + " ".join(args) + "\n")
if tool == "ssh":
    sys.exit(subprocess.run(["bash", "-c", args[-1]], check=False).returncode)
if tool == "rsync":
    sys.exit(0)
if tool == "docker":
    if state.get("render_error"):
        sys.exit(17)
    if "config" not in args:
        sys.exit("unexpected container operation")
    if "--format" in args:
        print(json.dumps(state["compose"]))
elif tool == "sudo":
    if state.get("install_error"):
        sys.exit(19)
    assert args[0:2] == ["-n", "install"], args
    assert "-d" in args, args
    owner = int(args[args.index("-o") + 1])
    mode = args[args.index("-m") + 1].lstrip("0")
    if not state.get("ignore_install"):
        state["dirs"][args[-1]] = {"uid": owner, "mode": mode}
    state_path.write_text(json.dumps(state))
elif tool == "stat":
    entry = state["dirs"].get(args[-1])
    if entry is None:
        sys.exit(1)
    print("directory:" + str(entry["uid"]) + ":" + entry["mode"])
else:
    sys.exit("unsupported host tool " + tool)
"""


@pytest.fixture
def host(tmp_path: Path) -> tuple[Path, Path, Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("ssh", "rsync", "docker", "sudo", "stat", "uv"):
        tool = bindir / name
        tool.write_text(HOST_TOOL)
        tool.chmod(0o755)

    compose = yaml.safe_load(COMPOSE.read_text())
    services = {}
    dirs = {}
    for name, service in compose["services"].items():
        volumes = []
        image_home = service["environment"]["TMPDIR"].rsplit("/", 1)[0]
        for volume in service["volumes"]:
            if isinstance(volume, str):
                resolved = volume.replace(image_home, "/runner")
                source, target, *_ = resolved.split(":")
                volumes.append({"type": "bind", "source": source, "target": target})
                if target == "/runner/tmp":
                    dirs[source] = {"uid": 0, "mode": "755"}
        services[name] = {
            "environment": {"TMPDIR": "/runner/tmp"},
            "volumes": volumes,
        }
    # Other services must never have their directories prepared by this pool.
    services["other-pool-1"] = {
        "environment": {"TMPDIR": "/runner/tmp"},
        "volumes": [{"type": "bind", "source": "/other/tmp", "target": "/runner/tmp"}],
    }
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"compose": {"services": services}, "dirs": dirs}))
    calls = tmp_path / "calls.txt"
    return bindir, state, calls


def deploy(
    host: tuple[Path, Path, Path], *flags: str
) -> subprocess.CompletedProcess[str]:
    bindir, state, calls = host
    # Keep the real main and secondary_deploy; replace unrelated deployment
    # edges so no build, registry write, unit install or runner restart runs.
    script = SCRIPT.read_text().removesuffix('main "$@"\n')
    script += r"""
rsync_artifacts() { :; }
install_runner_slice() { :; }
secondary_registry_states() {
    local name
    for name in $(secondary_target_names); do printf '%s online\n' "$name"; done
}
fetch_registration_token() { echo FETCH >> "$TMP_HOST_CALLS"; echo fixture-token; }
relabel_sync_compose() { :; }
relabel_registry_snapshot() { echo REGISTRY >> "$TMP_HOST_CALLS"; return 1; }
main "$@"
"""
    fixture_root = state.parent / "repo"
    (fixture_root / "scripts").mkdir(parents=True, exist_ok=True)
    for name in ("config", "docker", "src"):
        target = fixture_root / name
        if not target.exists():
            target.symlink_to(ROOT / name, target_is_directory=True)
    harness = fixture_root / "scripts/deploy-runners.sh"
    harness.write_text(script)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "HOME": str(state.parent),
        "TMP_HOST_STATE": str(state),
        "TMP_HOST_CALLS": str(calls),
        "TMP_NODE_PYTHON": sys.executable,
        "ONEX_STATE_DIR": str(state.parent / "node-state"),
    }
    return subprocess.run(
        ["bash", str(harness), "--host=h053", "--skip-build", *flags],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_deploy_repairs_recorded_root_owned_0755_tmp_binds(host) -> None:
    _, state, calls = host
    before = json.loads(state.read_text())["dirs"]
    assert len(before) == 8
    assert all(entry == {"uid": 0, "mode": "755"} for entry in before.values())
    result = deploy(host)
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads(state.read_text())["dirs"]
    assert all(entry == {"uid": 1001, "mode": "1777"} for entry in after.values())
    assert (
        sum(
            line.startswith("sudo -n install ")
            for line in calls.read_text().splitlines()
        )
        == 8
    )
    assert "/other/tmp" not in calls.read_text()
    assert "RUNNER-TMP-READY" in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("install_error", "RUNNER-TMP-PREPARE"),
        ("ignore_install", "RUNNER-TMP-VERIFY"),
        ("render_error", "RUNNER-TMP-DISCOVERY"),
    ],
)
def test_refuses_before_registration_when_tmp_cannot_be_prepared(
    host, fault, error
) -> None:
    _, state, calls = host
    data = json.loads(state.read_text())
    data[fault] = True
    state.write_text(json.dumps(data))
    result = deploy(host)
    assert result.returncode != 0, result.stdout + result.stderr
    assert error in result.stdout + result.stderr
    assert "h053" in result.stdout + result.stderr
    assert "FETCH" not in calls.read_text()


def test_every_deploy_rechecks_existing_directories(host) -> None:
    _, state, calls = host
    assert deploy(host).returncode == 0
    data = json.loads(state.read_text())
    data["dirs"]["/data/scratch/runners/1/tmp"] = {"uid": 0, "mode": "755"}
    state.write_text(json.dumps(data))
    result = deploy(host)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(state.read_text())["dirs"]["/data/scratch/runners/1/tmp"] == {
        "uid": 1001,
        "mode": "1777",
    }
    assert (
        sum(
            line.startswith("sudo -n install ")
            for line in calls.read_text().splitlines()
        )
        == 16
    )


def test_dry_run_leaves_tmp_directories_unchanged(host) -> None:
    _, state, calls = host
    before = state.read_text()
    result = deploy(host, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert state.read_text() == before
    assert not calls.exists()


def test_missing_directories_are_created(host) -> None:
    _, state, _ = host
    data = json.loads(state.read_text())
    data["dirs"] = {}
    state.write_text(json.dumps(data))
    result = deploy(host)
    assert result.returncode == 0, result.stdout + result.stderr
    dirs = json.loads(state.read_text())["dirs"]
    assert len(dirs) == 8
    assert all(entry == {"uid": 1001, "mode": "1777"} for entry in dirs.values())


def test_relabel_prepares_tmp_before_registry_and_recreation(host) -> None:
    _, state, calls = host
    result = deploy(host, "--relabel")
    # The fixture stops at the registry edge, after real relabel preparation.
    assert result.returncode == 1
    assert "could not read the org runner registry" in result.stderr
    dirs = json.loads(state.read_text())["dirs"]
    assert all(entry == {"uid": 1001, "mode": "1777"} for entry in dirs.values())
    assert calls.read_text().splitlines()[-1] == "REGISTRY"


def test_tmpfs_pool_needs_no_host_directory_and_environment_stays_private(host) -> None:
    _, state, calls = host
    data = json.loads(state.read_text())
    for service in data["compose"]["services"].values():
        service["volumes"] = []
        service["environment"]["RUNNER_TOKEN"] = "FIXTURE-SECRET-MUST-NOT-PRINT"
    state.write_text(json.dumps(data))
    result = deploy(host)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sudo -n install" not in calls.read_text()
    assert "FIXTURE-SECRET-MUST-NOT-PRINT" not in result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("install_error", [False, True])
async def test_command_and_terminal_event_chain(
    host, monkeypatch, install_error
) -> None:
    bindir, state, calls = host
    data = json.loads(state.read_text())
    data["install_error"] = install_error
    state.write_text(json.dumps(data))
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("TMP_HOST_STATE", str(state))
    monkeypatch.setenv("TMP_HOST_CALLS", str(calls))
    contract = ROOT / "src/omnibase_infra/nodes/node_deploy_runners/contract.yaml"
    manifest = discover_contracts_from_paths([contract])
    assert manifest.errors == ()
    declaration = manifest.contracts[0]
    assert declaration.event_bus is not None
    (command_topic,) = declaration.event_bus.subscribe_topics
    (terminal_topic,) = declaration.event_bus.publish_topics
    correlation = uuid4()
    result_future = asyncio.get_running_loop().create_future()

    async def collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation and not result_future.done():
            result_future.set_result(
                ModelDeployRunnersResult.model_validate(envelope.payload)
            )

    bus = EventBusInmemory(environment="dev", group=str(uuid4()))
    await bus.start()
    try:
        await bus.subscribe(terminal_topic, group_id=str(uuid4()), on_message=collect)
        engine = MessageDispatchEngine()
        await wire_from_manifest(manifest, engine, event_bus=bus, environment="dev")
        engine.freeze()
        request = ModelDeployRunnersRequest(
            host="h053",
            host_dir=str(state.parent),
            compose_file=str(COMPOSE),
            prefix="hp053-ci-runner",
            count=8,
        )
        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=correlation,
            event_type=request.operation,
            payload=request.model_dump(mode="json"),
        )
        await bus.publish(
            command_topic, None, envelope.model_dump_json().encode(), None
        )
        result = await asyncio.wait_for(result_future, timeout=30)
        assert result.success is not install_error
        if install_error:
            assert "RUNNER-TMP-PREPARE" in result.error_message
            assert result.prepared == ()
        else:
            assert len(result.prepared) == 8
            assert result.error_message is None
    finally:
        await bus.close()
