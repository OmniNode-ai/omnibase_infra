# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26098: the runner installer copies the overlay projection for the runners and writes RUNNER_LAB_HOST_TABLE.

Every runner compose mounts ``${RUNNER_LAB_HOST_TABLE}`` read-only at ``/etc/onex/lab_run_hosts.yaml`` with
``create_host_path: false``. The host's ``docker/.env`` was set by hand to a file a later change deleted, so every
container create failed. The deploy now owns both halves: the copy of the projection and the key.

The remote script runs for real under bash in a temporary home; only ``sudo`` is a shim that records the owner it
was asked for (the test user cannot chown) and runs the rest.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_deploy_runners import HandlerDeployRunners
from omnibase_infra.handlers.handler_runner_image_roll import (
    HandlerRunnerImageRoll,
    RunnerRollError,
)
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from omnibase_infra.models.model_runner_image_roll_request import (
    ModelRunnerImageRollRequest,
)
from omnibase_infra.models.model_runner_lab_host_table_policy import (
    ModelRunnerLabHostTablePolicy,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
KEY = "RUNNER_LAB_HOST_TABLE"
STALE = f"{KEY}=/srv/deleted/lab_run_hosts.yaml\n"
PROJECTION_TEXT = "hosts:\n  - host: h051\n  - host: h053\n"

SUDO_SHIM = r"""#!/usr/bin/env bash
# Records what the real sudo would have set, then runs the rest unprivileged.
shift  # -n
cmd="$1"; shift
case "$cmd" in
  install)
    args=()
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o) echo "owner $2" >> "$SHIM_LOG"; shift 2 ;;
        -g) shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    exec install "${args[@]}" ;;
  sh) chmod u+w "${@: -1}"; exec sh "$@" ;;  # root writes a 0440 file; the test user needs the bit
  chown) echo "owner ${1%%:*}" >> "$SHIM_LOG"; exit 0 ;;
  *) exec "$cmd" "$@" ;;
esac
"""

COMPOSE_WITH_TABLE = (
    "services:\n  x:\n    environment:\n      RUNNER_TOKEN: ${RUNNER_TOKEN_X:?}\n"
    f"    volumes:\n      - type: bind\n        source: ${{{KEY}:?must name the table}}\n"
)


class LocalHost:
    """The runner host is this machine with a temporary home; ssh is a local bash."""

    def __init__(self, home: Path, shim_dir: Path) -> None:
        self.home = home
        self.env = {
            **os.environ,
            "HOME": str(home),
            "PATH": f"{shim_dir}:{os.environ['PATH']}",
            "SHIM_LOG": str(shim_dir / "log"),
        }
        self.shim_log = shim_dir / "log"
        self.commands: list[str] = []

    def run(self, command: str, stdin: bytes | None = None) -> tuple[int, str]:
        self.commands.append(command)
        stat_match = re.search(r"stat -c '%F:%u:%a' -- (\S+)", command)
        if stat_match is not None:
            path = Path(stat_match.group(1).strip("'"))
            owners = self.shim_log.read_text().split()[1::2]
            return (
                0,
                f"regular file:{owners[-1]}:{stat.S_IMODE(path.stat().st_mode):o}\n",
            )
        done = subprocess.run(
            ["bash", "-c", command],
            env=self.env,
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        return done.returncode, done.stdout.decode()

    def write(self, word: str, data: str) -> None:
        self.commands.append(f"write {word}")
        path = shlex.split(word.replace('"$HOME"', str(self.home)))[0]
        with open(
            path, "w", encoding="utf-8", newline=""
        ) as handle:  # same inode, mode kept
            handle.write(data)


class LocalHostHandler(HandlerDeployRunners):
    """The deploy effect on that host; only Compose rendering is answered from a fixed document."""

    def __init__(self, home: Path, shim_dir: Path) -> None:
        self.host = LocalHost(home, shim_dir)
        self.home = home
        self.shim_log = self.host.shim_log

    @property
    def commands(self) -> list[str]:
        return self.host.commands

    async def _remote(self, host: str, command: str, timeout: int) -> str:
        if "compose" in command and "config" in command:
            self.host.commands.append("compose config")
            return json.dumps({"services": {"x-1": {"environment": {}, "volumes": []}}})
        rc, output = self.host.run(command)
        if rc:
            raise OSError(f"SSH exited {rc}")
        return output

    async def _write_remote(
        self, host: str, word: str, data: str, timeout: int
    ) -> None:
        self.host.write(word, data)


@pytest.fixture
def lab(tmp_path: Path) -> LocalHostHandler:
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    sudo = shim_dir / "sudo"
    sudo.write_text(SUDO_SHIM, encoding="utf-8")
    sudo.chmod(0o755)
    (shim_dir / "log").write_text("")
    home = tmp_path / "home"
    (home / ".omnibase/runners/docker").mkdir(parents=True)
    return LocalHostHandler(home, shim_dir)


def _projection(home: Path, text: str = PROJECTION_TEXT) -> Path:
    path = home / ModelRunnerLabHostTablePolicy().projection_home_relative
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def _env_file(home: Path, content: str) -> Path:
    path = home / ".omnibase/runners/docker/.env"
    path.write_bytes(content.encode())
    path.chmod(0o600)
    return path


def _request(home: Path, compose: Path) -> ModelDeployRunnersRequest:
    return ModelDeployRunnersRequest(
        host="h051",
        host_dir=str(home / ".omnibase/runners"),
        compose_file=str(compose),
        prefix="x",
        count=1,
    )


@pytest.fixture
def compose(tmp_path: Path) -> Path:
    path = tmp_path / "compose.yml"
    path.write_text(COMPOSE_WITH_TABLE, encoding="utf-8")
    return path


def test_lab_host_table_key_is_written_from_the_projection_and_the_copy_is_readable_by_the_runner(
    lab: LocalHostHandler, compose: Path
) -> None:
    _projection(lab.home)
    env = _env_file(lab.home, "A=1\n" + STALE + "SECRET=pa ss\n")

    result = asyncio.run(lab.handle(_request(lab.home, compose)))

    assert result.success is True, result.error_message
    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    assert env.read_bytes().decode() == f"A=1\n{KEY}={copy}\nSECRET=pa ss\n"
    assert copy.read_text(encoding="utf-8") == PROJECTION_TEXT
    # The Runner.Worker user: the image creates uid 1001 and the entrypoint drops to it with gosu.
    dockerfile = (DOCKER / "runners" / "Dockerfile").read_text(encoding="utf-8")
    declared = re.search(r"useradd --uid (\d+) --gid runner", dockerfile)
    assert declared is not None
    image_uid = int(declared.group(1))
    assert "gosu runner" in (DOCKER / "runners" / "entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert ModelRunnerLabHostTablePolicy().reader_uid == image_uid
    owners = lab.shim_log.read_text().split()[1::2]
    assert set(owners) == {str(image_uid)}
    # Owner and group read, nobody writes, no other host user reads; the projection itself stays 0600.
    assert stat.S_IMODE(copy.stat().st_mode) == 0o440
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_lab_host_table_key_replaces_a_stale_value_and_leaves_every_other_byte(
    lab: LocalHostHandler, compose: Path
) -> None:
    _projection(lab.home)
    other = "# comment\r\nA=1\n\nZ=\n  INDENTED=1\nRUNNER_LAB_HOST_TABLE_OTHER=keep\n"
    env = _env_file(lab.home, other + STALE + "TAIL=no newline")

    result = asyncio.run(lab.handle(_request(lab.home, compose)))

    assert result.success is True, result.error_message
    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    assert env.read_bytes().decode() == other + f"{KEY}={copy}\n" + "TAIL=no newline"


def test_lab_host_table_key_is_appended_once_when_absent_and_a_second_deploy_changes_nothing(
    lab: LocalHostHandler, compose: Path
) -> None:
    _projection(lab.home)
    env = _env_file(lab.home, "A=1")

    first = asyncio.run(lab.handle(_request(lab.home, compose)))
    after_first = env.read_bytes()
    lab.host.commands.clear()
    second = asyncio.run(lab.handle(_request(lab.home, compose)))

    assert first.success is True and second.success is True
    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    assert after_first == f"A=1\n{KEY}={copy}\n".encode()
    assert env.read_bytes() == after_first
    assert not any(command.startswith("write ") for command in lab.commands)


def test_lab_host_table_copy_is_refreshed_on_every_deploy_keeping_its_inode(
    lab: LocalHostHandler, compose: Path
) -> None:
    projection = _projection(lab.home)
    _env_file(lab.home, "A=1\n")
    asyncio.run(lab.handle(_request(lab.home, compose)))
    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    inode = copy.stat().st_ino

    projection.write_text(PROJECTION_TEXT + "  - host: h999\n", encoding="utf-8")
    asyncio.run(lab.handle(_request(lab.home, compose)))

    assert copy.read_text(encoding="utf-8").endswith("h999\n")
    assert copy.stat().st_ino == inode  # a running container's bind sees the refresh


@pytest.mark.parametrize("empty", [False, True])
def test_lab_host_table_missing_projection_refuses_naming_the_path_and_writes_nothing(
    lab: LocalHostHandler, compose: Path, empty: bool
) -> None:
    if empty:
        _projection(lab.home, "")
    env = _env_file(lab.home, "A=1\n" + STALE)

    result = asyncio.run(lab.handle(_request(lab.home, compose)))

    projection = lab.home / ModelRunnerLabHostTablePolicy().projection_home_relative
    assert result.success is False
    assert result.error_message is not None
    assert result.error_message.startswith("RUNNER-LAB-HOST-TABLE-PROJECTION host=h051")
    assert f"path={projection}" in result.error_message
    assert env.read_bytes().decode() == "A=1\n" + STALE  # no stale fallback, no edit
    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    assert not copy.exists()  # never an empty file
    assert not any("compose" in command for command in lab.commands)


def test_lab_host_table_projection_path_override_comes_from_the_request(
    lab: LocalHostHandler, compose: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere.yaml"
    elsewhere.write_text("hosts: []\n", encoding="utf-8")
    _env_file(lab.home, "A=1\n")

    result = asyncio.run(
        lab.handle(
            _request(lab.home, compose).model_copy(
                update={"lab_host_table_projection": str(elsewhere)}
            )
        )
    )

    assert result.success is True, result.error_message
    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    assert copy.read_text(encoding="utf-8") == "hosts: []\n"


def test_lab_host_table_a_compose_without_the_mount_is_left_alone(
    lab: LocalHostHandler, tmp_path: Path
) -> None:
    plain = tmp_path / "plain.yml"
    plain.write_text(COMPOSE_WITH_TABLE.replace(KEY, "OTHER"), encoding="utf-8")
    env = _env_file(lab.home, "A=1\n" + STALE)

    result = asyncio.run(lab.handle(_request(lab.home, plain)))

    assert result.success is True, result.error_message
    assert env.read_bytes().decode() == "A=1\n" + STALE


def test_lab_host_table_env_key_replacement_is_a_pure_function_of_the_text() -> None:
    def new(content: str) -> str:
        return ModelRunnerLabHostTablePolicy().with_env_value(content, "/p")

    assert new("") == f"{KEY}=/p\n"
    assert new(f"{KEY}=/a\n{KEY}=/b\nB=2\n") == f"{KEY}=/p\nB=2\n"
    assert new(f"B=2\n{KEY}=/a") == f"B=2\n{KEY}=/p"
    assert new(f"B=2\rX\n{KEY}=/a\r\n") == f"B=2\rX\n{KEY}=/p\n"


class LocalHostRoll(HandlerRunnerImageRoll):
    """The secondary pool image roll, with its ssh seam run against the temporary home."""

    def __init__(self, host: LocalHost) -> None:
        self.host = host

    async def _remote(
        self,
        host: str,
        script: str,
        *,
        stdin: bytes | None = None,
        timeout: int = 30,
    ) -> tuple[int, str]:
        return self.host.run(script, stdin)


def _roll_request(tmp_path: Path, compose_text: str) -> ModelRunnerImageRollRequest:
    docker = tmp_path / "repo" / "docker"
    docker.mkdir(parents=True)
    (docker / "docker-compose.runners-x.yml").write_text(compose_text, encoding="utf-8")
    return ModelRunnerImageRollRequest(
        host="h053",
        prefix="x",
        repo_root=tmp_path / "repo",
        fleet_config=tmp_path / "fleet.yaml",
        token_path=tmp_path / "token",
    )


def test_lab_host_table_secondary_roll_writes_the_key_before_it_recreates_anything(
    lab: LocalHostHandler, compose: Path, tmp_path: Path
) -> None:
    """The roll recreates from the host's installed Compose, so it must not meet a stale key either."""
    _projection(lab.home)
    env = _env_file(lab.home, "A=1\n" + STALE)
    roll = LocalHostRoll(lab.host)

    asyncio.run(
        roll._converge_lab_host_table(_roll_request(tmp_path, COMPOSE_WITH_TABLE))
    )

    copy = lab.home / ModelRunnerLabHostTablePolicy().copy_home_relative
    assert env.read_bytes().decode() == f"A=1\n{KEY}={copy}\n"


def test_lab_host_table_secondary_roll_halts_on_a_missing_projection_naming_the_path(
    lab: LocalHostHandler, tmp_path: Path
) -> None:
    env = _env_file(lab.home, "A=1\n" + STALE)
    roll = LocalHostRoll(lab.host)

    with pytest.raises(RunnerRollError) as refusal:
        asyncio.run(
            roll._converge_lab_host_table(_roll_request(tmp_path, COMPOSE_WITH_TABLE))
        )

    projection = lab.home / ModelRunnerLabHostTablePolicy().projection_home_relative
    assert str(refusal.value).startswith("RUNNER-LAB-HOST-TABLE-PROJECTION host=h053")
    assert f"path={projection}" in str(refusal.value)
    assert env.read_bytes().decode() == "A=1\n" + STALE


def test_lab_host_table_secondary_roll_of_a_pool_without_the_mount_touches_nothing(
    lab: LocalHostHandler, tmp_path: Path
) -> None:
    env = _env_file(lab.home, "A=1\n" + STALE)
    roll = LocalHostRoll(lab.host)

    asyncio.run(
        roll._converge_lab_host_table(
            _roll_request(tmp_path, COMPOSE_WITH_TABLE.replace(KEY, "OTHER"))
        )
    )

    assert env.read_bytes().decode() == "A=1\n" + STALE
    assert lab.host.commands == []
