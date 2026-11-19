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

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from omnibase_infra.models.model_deploy_runners_result import ModelDeployRunnersResult
from omnibase_infra.models.model_runner_lab_host_table_policy import (
    ModelRunnerLabHostTablePolicy,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/deploy-runners.sh"
COMPOSE = ROOT / "docker/docker-compose.runners-hp053-ci-runner.yml"
POOL_COUNT = 5

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
# OMN-26098: the lab host table copy is not a TMPDIR bind; its sudo calls stay out of the bind counts.
lab_table_call = (
    tool == "sudo" and (args[1] in ("sh", "chown", "chmod") or "lab-host-table" in args[-1])
) or (tool == "ssh" and "lab-host-table" in args[-1])
if not lab_table_call:
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
    if args[1] == "sh":  # OMN-26098: root writes the 0440 lab host table copy
        Path(args[-1]).chmod(0o644)
        sys.exit(subprocess.run(["sh", *args[2:]], check=False).returncode)
    if args[1] == "chown":
        state["files"][args[-1]] = int(args[2].split(":")[0])
        state_path.write_text(json.dumps(state))
        sys.exit(0)
    if args[1] == "chmod":
        sys.exit(subprocess.run(["chmod", *args[2:]], check=False).returncode)
    if state.get("install_error") and "lab-host-table" not in args[-1]:
        sys.exit(19)
    assert args[1] == "install", args
    if "lab-host-table" in args[-1]:
        kept, skip = [], False
        for arg in args[2:]:
            if skip:
                skip = False
            elif arg in ("-o", "-g"):
                skip = True
            else:
                kept.append(arg)
        sys.exit(subprocess.run(["install", *kept], check=False).returncode)
    assert "-d" in args, args
    owner = int(args[args.index("-o") + 1])
    mode = args[args.index("-m") + 1].lstrip("0")
    if not state.get("ignore_install"):
        state["dirs"][args[-1]] = {"uid": owner, "mode": mode}
    state_path.write_text(json.dumps(state))
elif tool == "stat" and args[-1] in state["files"]:
    mode = os.stat(args[-1]).st_mode & 0o7777
    print("regular file:" + str(state["files"][args[-1]]) + ":" + format(mode, "o"))
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

    # The bind-repair path is modeled with a synthetic pool of the same size; the
    # shipped h053 pool's own binds are asserted in test_hp053_pool_in_memory_omn19998.
    services = {}
    dirs = {}
    for number in range(1, POOL_COUNT + 1):
        source = "/".join(("/data/scratch/runners", str(number), "tmp"))
        services[f"hp053-ci-runner-{number}"] = {
            "environment": {"TMPDIR": "/runner/tmp"},
            "volumes": [{"type": "bind", "source": source, "target": "/runner/tmp"}],
        }
        dirs[source] = {"uid": 0, "mode": "755"}
    # Other services must never have their directories prepared by this pool.
    services["other-pool-1"] = {
        "environment": {"TMPDIR": "/runner/tmp"},
        "volumes": [{"type": "bind", "source": "/other/tmp", "target": "/runner/tmp"}],
    }
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"compose": {"services": services}, "dirs": dirs, "files": {}})
    )
    # OMN-26098: the runner host's overlay projection and host-generated env file, under the fake host's HOME.
    projection = tmp_path / ModelRunnerLabHostTablePolicy().projection_home_relative
    projection.parent.mkdir(parents=True)
    projection.write_text("hosts: []\n")
    env_file = tmp_path / ".omnibase/runners/docker/.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("LOCAL_LLM_SHARED_SECRET=x\n")
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
    assert len(before) == POOL_COUNT
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
        == POOL_COUNT
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
        == 2 * POOL_COUNT
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
    assert len(dirs) == POOL_COUNT
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
    monkeypatch.setenv("HOME", str(state.parent))
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
            host_dir=str(state.parent / ".omnibase/runners"),
            compose_file=str(COMPOSE),
            prefix="hp053-ci-runner",
            count=POOL_COUNT,
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
            assert len(result.prepared) == POOL_COUNT
            assert result.error_message is None
    finally:
        await bus.close()


# OMN-25674: the disk-admission pause directory is a compose bind source too.
# Docker creates it root-owned mode 755 and the job-started hook writes its
# pause marker as `runner` (Permission denied on hp051-ci-runner-3.tmp, 12
# consecutive admission failures against a threshold of 3), so the same effect
# hands it to the runner user.
PAUSE_SOURCE = "/srv/runners/docker/state/disk-admission-pause"
PAUSE_TARGET = "/".join(("", "home", "runner", ".onex-disk-admission-pause"))


def _with_pause_bind(state: Path, *, extra_bind: str | None = None) -> None:
    data = json.loads(state.read_text())
    for name, service in data["compose"]["services"].items():
        if name.startswith("hp053-ci-runner-"):
            service["volumes"].append(
                {"type": "bind", "source": PAUSE_SOURCE, "target": PAUSE_TARGET}
            )
            if extra_bind:
                service["volumes"].append(
                    {"type": "bind", "source": extra_bind, "target": "/runner/keys"}
                )
    data["dirs"][PAUSE_SOURCE] = {"uid": 0, "mode": "755"}
    state.write_text(json.dumps(data))


def test_deploy_hands_the_root_owned_pause_dir_to_runner(host) -> None:
    _, state, calls = host
    _with_pause_bind(state, extra_bind="/etc/hostkeys")
    result = deploy(host)
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads(state.read_text())["dirs"]
    assert after[PAUSE_SOURCE] == {"uid": 1001, "mode": "755"}
    installs = [
        line
        for line in calls.read_text().splitlines()
        if line.startswith("sudo -n install ")
    ]
    # One install per distinct bind source: the pool's TMPDIRs and the one pause dir.
    assert len(installs) == POOL_COUNT + 1
    assert sum(PAUSE_SOURCE in line for line in installs) == 1
    assert "/etc/hostkeys" not in calls.read_text()


def test_pause_dir_that_cannot_be_prepared_refuses_before_registration(host) -> None:
    _, state, calls = host
    _with_pause_bind(state)
    data = json.loads(state.read_text())
    data["ignore_install"] = True
    state.write_text(json.dumps(data))
    result = deploy(host)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "RUNNER-TMP-VERIFY" in result.stdout + result.stderr
    assert "FETCH" not in calls.read_text()
