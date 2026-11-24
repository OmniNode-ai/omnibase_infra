# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Secondary rolling deployments through the installer entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts/deploy-runners.sh"
FLEET = ROOT / "config/runner_fleet.yaml"


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.touch(mode=0o600)
    path.write_text("fixture-token\n")
    return path


def _ip_addressed_ci_pool() -> tuple[str, str]:
    """Return the fleet's address-keyed secondary host and its CI pool prefix."""
    fleet = load_runner_fleet_config(FLEET)
    return next(
        (host.host, pool.runner_name_prefix)
        for host in fleet.hosts
        if host.host[0].isdigit()
        for pool in host.pools
        if pool.runner_name_prefix.endswith("-ci-runner")
    )


@pytest.fixture
def h202() -> str:
    return _ip_addressed_ci_pool()[0]


def _installer(
    tmp_path: Path, *args: str, extra_env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], str]:
    # A test invocation must never read or mutate a live runner or ask for a token.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    probe = tmp_path / "unexpected-effects"
    for command in ("ssh", "rsync", "docker", "gh"):
        stub = bin_dir / command
        stub.write_text(
            '#!/bin/sh\nprintf "%s\\n" "${0##*/}" >> "$ROLL_TEST_PROBE"\nexit 89\n'
        )
        stub.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\nshift 4\nshift\nexec "$ROLL_TEST_PYTHON" "$@"\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(INSTALLER), "--rolling", *args],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUNNER_FLEET_CONFIG_PATH": str(FLEET),
            "ROLL_TEST_PROBE": str(probe),
            "ROLL_TEST_PYTHON": sys.executable,
            "PYTHONPATH": str(ROOT / "src"),
            "ONEX_STATE_DIR": str(tmp_path / "state"),
            **(extra_env or {}),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result, probe.read_text() if probe.exists() else ""


def test_secondary_requires_token_file_before_remote_calls(tmp_path: Path) -> None:
    result, effects = _installer(tmp_path, "--host=h053")
    assert result.returncode != 0
    assert "--token-file" in result.stdout + result.stderr
    assert effects == ""


@pytest.mark.parametrize(
    "pool", ["hp053-ci-runner", _ip_addressed_ci_pool()[1]], ids=["h053", "h202"]
)
def test_secondary_dry_run_plans_only_the_requested_service(
    tmp_path: Path, token_file: Path, h202: str, pool: str
) -> None:
    host = "h053" if pool == "hp053-ci-runner" else h202
    pool_args = () if host == "h053" else (f"--pool={pool}",)
    target = f"{pool}-2"
    result, effects = _installer(
        tmp_path,
        f"--host={host}",
        *pool_args,
        "--dry-run",
        f"--token-file={token_file}",
        f"--only={target}",
        "--limit=1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert target in result.stdout + result.stderr
    assert f"{pool}-1" not in result.stdout + result.stderr
    assert "fixture-token" not in result.stdout + result.stderr
    assert effects == ""


def test_secondary_rejects_another_pool_before_effects(
    tmp_path: Path, token_file: Path
) -> None:
    result, effects = _installer(
        tmp_path,
        "--host=h053",
        "--dry-run",
        f"--token-file={token_file}",
        "--only=hp053-ci-mq-runner-1",
    )
    assert result.returncode != 0
    assert "--only" in result.stdout + result.stderr
    assert effects == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("busy", "readback", "limit", "recreated", "success"),
    [
        (True, True, 2, ["-2", "-1"], True),
        (False, True, 1, ["-1"], True),
        (False, False, 2, ["-1"], False),
    ],
)
async def test_roll_waits_for_readback_before_touching_next_runner(
    monkeypatch: pytest.MonkeyPatch,
    token_file: Path,
    busy: bool,
    readback: bool,
    limit: int,
    recreated: list[str],
    success: bool,
) -> None:
    from omnibase_infra.handlers.handler_runner_image_roll import HandlerRunnerImageRoll
    from omnibase_infra.models.model_runner_image_roll_request import (
        ModelRunnerImageRollRequest,
    )

    handler = HandlerRunnerImageRoll()
    first = "hp053-ci-runner-1"
    events: list[tuple[str, str]] = []
    waited = False
    checked: dict[str, int] = {}

    async def prepare(*_args: object) -> str:
        return "sha256:new"

    async def idle(_pool: object, name: str, _org: str) -> tuple[str, str] | None:
        checked[name] = checked.get(name, 0) + 1
        return (
            None
            if name == first and busy and checked[name] == 1
            else ("before", "sha256:old")
        )

    async def recreate(
        _pool: object,
        name: str,
        previous_id: str,
        token: bytes,
        _tag: str,
        _vars: tuple[str, ...],
    ) -> bool:
        assert previous_id == "before"
        assert token == b"fixture-token"
        events.append(("recreate", name))
        return True

    async def online(
        _pool: object, name: str, previous_id: str, image: str, _org: str
    ) -> bool:
        assert previous_id == "before"
        assert image == "sha256:new"
        events.append(("readback", name))
        return readback

    async def retry() -> None:
        nonlocal waited
        waited = True

    monkeypatch.setattr(handler, "_prepare_image", prepare)
    monkeypatch.setattr(handler, "_idle_container", idle)
    monkeypatch.setattr(handler, "_recreate", recreate)
    monkeypatch.setattr(handler, "_wait_online", online)
    monkeypatch.setattr(handler, "_retry_pause", retry)
    result = await handler._roll(
        ModelRunnerImageRollRequest(
            host="h053",
            prefix="hp053-ci-runner",
            repo_root=ROOT,
            fleet_config=FLEET,
            token_path=token_file,
            limit=limit,
        ),
        targets=(first, "hp053-ci-runner-2"),
        image_tag="test-image",
        token=b"fixture-token",
        org="Fixture",
        token_vars=("RUNNER_TOKEN",),
    )
    assert result.success is success
    assert waited is busy
    assert events == [
        (event, "hp053-ci-runner" + name)
        for name in recreated
        for event in ("recreate", "readback")
    ]
    if not success:
        assert "HALT" in (result.error_message or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("process_state", ["worker", "unreadable", "idle"])
async def test_remote_worker_check_gates_the_single_compose_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_file: Path,
    process_state: str,
) -> None:
    from omnibase_infra.handlers.handler_runner_image_roll import HandlerRunnerImageRoll
    from omnibase_infra.models.model_runner_image_roll_request import (
        ModelRunnerImageRollRequest,
    )

    handler = HandlerRunnerImageRoll()
    docker_dir = tmp_path / ".omnibase/runners/docker"
    docker_dir.mkdir(parents=True)
    docker = tmp_path / "docker"
    effects = tmp_path / "effects"
    docker.write_text("""#!/bin/bash
set -eu
case "$1" in
    ps) ;;
    inspect) echo old-id ;;
    top)
        [[ "$PROCESS_STATE" != unreadable ]] || exit 1
        echo Runner.Listener
        if [[ "$PROCESS_STATE" == worker ]]; then echo Runner.Worker; fi
        ;;
    stop) printf '%s\\n' "$*" >> "$EFFECTS" ;;
    compose)
        [[ "$RUNNER_TOKEN" == fixture-token ]]
        [[ "$*" != *fixture-token* ]]
        printf '%s\\n' "$*" >> "$EFFECTS"
        cat >> "$EFFECTS"
        ;;
    *) exit 99 ;;
esac
""")
    docker.chmod(0o755)

    async def remote(
        _host: str, script: str, *, stdin: bytes | None = None, timeout: int = 30
    ) -> tuple[int, str]:
        assert b"fixture-token" not in script.encode()
        result = subprocess.run(
            ["bash", "-c", script],
            input=stdin,
            capture_output=True,
            check=False,
            timeout=timeout,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "HOME": str(tmp_path),
                "PROCESS_STATE": process_state,
                "EFFECTS": str(effects),
            },
        )
        return result.returncode, result.stdout.decode()

    monkeypatch.setattr(handler, "_remote", remote)
    target = "hp053-ci-runner-2"
    recreated = await handler._recreate(
        ModelRunnerImageRollRequest(
            host="h053",
            prefix="hp053-ci-runner",
            repo_root=ROOT,
            fleet_config=FLEET,
            token_path=token_file,
        ),
        target,
        "old-id",
        b"fixture-token",
        "fixture-image:v14",
        ("RUNNER_TOKEN",),
    )
    assert recreated is (process_state == "idle")
    if recreated:
        stop, compose, override = effects.read_text().splitlines()
        assert stop == "stop -t 30 old-id"
        assert compose.endswith(
            f"-f - up -d --force-recreate --no-deps --no-build --pull never {target}"
        )
        assert '"fixture-image:v14"' in override
        assert "fixture-token" not in effects.read_text()
    else:
        assert not effects.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "idle"),
    [
        (("online", False), True),
        (("online", True), False),
        (("online", None), False),
        (("offline", False), False),
        (("unknown", None), False),
    ],
)
async def test_only_online_not_busy_is_eligible(
    monkeypatch: pytest.MonkeyPatch,
    token_file: Path,
    state: tuple[str, bool | None],
    idle: bool,
) -> None:
    from omnibase_infra.handlers.handler_runner_image_roll import HandlerRunnerImageRoll
    from omnibase_infra.models.model_runner_image_roll_request import (
        ModelRunnerImageRollRequest,
    )

    handler = HandlerRunnerImageRoll()

    async def container(*_args: object) -> tuple[str, str]:
        return ("before", "old-image")

    async def registry(*_args: object) -> tuple[str, bool | None]:
        return state

    monkeypatch.setattr(handler, "_container", container)
    monkeypatch.setattr(handler, "_registry", registry)
    result = await handler._idle_container(
        ModelRunnerImageRollRequest(
            host="h053",
            prefix="hp053-ci-runner",
            repo_root=ROOT,
            fleet_config=FLEET,
            token_path=token_file,
        ),
        "hp053-ci-runner-1",
        "Fixture",
    )
    assert (result is not None) is idle


def test_bound_runner_image_registers_uv_libpython_with_the_loader() -> None:
    dockerfile = (ROOT / "docker/runners/Dockerfile").read_text()
    assert "ld.so.conf.d/onex-uv-python-libpython.conf" in dockerfile
    assert "&& ldconfig" in dockerfile


def test_secondary_dry_run_accepts_a_registry_snapshot_and_reads_no_registry(
    tmp_path: Path, token_file: Path
) -> None:
    """OMN-24139: the declared registry file reaches the roll; no gh call is made."""
    snapshot = tmp_path / "registry.json"
    snapshot.write_text('[{"runners": []}]')
    result, effects = _installer(
        tmp_path,
        "--host=h053",
        "--dry-run",
        f"--token-file={token_file}",
        "--only=hp053-ci-runner-1",
        extra_env={"RUNNER_REGISTRY_SNAPSHOT": str(snapshot)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert effects == ""
