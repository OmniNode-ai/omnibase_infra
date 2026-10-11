# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The nightly sweep unit runs against a broker that exists (OMN-20798).

The installed unit on the .201 host ran an untracked script that piped one
command into ``docker exec -i omnibase-infra-redpanda rpk topic produce``. On
2026-10-09 that failed with ``OCI runtime exec failed: ... setns process: exit
status 1`` (the container was being recreated), and the run before it had hung
from 2026-10-07 03:00 to 2026-10-09 10:01 with no timeout. Separately the dev
broker has since required SASL, which the script's bare ``rpk`` never spoke.

The tracked unit now carries the whole mechanism, so the file the host-drift
check compares is the file that runs. These tests parse the unit the way systemd
does (quotes, ``\\'``, ``$$``, ``%%`` and ``${VAR}`` substitution from
``Environment=``) and run the resulting commands against a ``docker`` shim:

  1. nothing in the unit points at an untracked host script;
  2. the start is bounded by ``TimeoutStartSec``;
  3. a broker container that never becomes healthy fails the unit before any
     produce is attempted, and names the container;
  4. a healthy broker gets one command on the build-loop-start topic, with a
     fresh correlation id per run and the SASL pair named by variable, never by
     value;
  5. a produce that fails fails the unit.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_UNIT = _REPO / "scripts" / "systemd" / "onex-nightly-sweep.service"
_TOPIC = "onex.cmd.omnibase-infra.build-loop-start.v1"

_ESCAPES = {"\\": "\\", "'": "'", '"': '"', "n": "\n", "t": "\t", "s": " "}


def _unit_lines(key: str) -> list[str]:
    return [
        line.split("=", 1)[1].strip()
        for line in _UNIT.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"{key}=")
    ]


def _unit_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for value in _unit_lines("Environment"):
        name, _, rest = value.partition("=")
        env[name] = rest
    return env


def _systemd_argv(line: str, env: dict[str, str]) -> list[str]:
    """Split an ExecStart= value into argv the way systemd does."""
    words: list[str] = []
    current: list[str] = []
    in_word = False
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            current.append(_ESCAPES.get(line[i + 1], line[i + 1]))
            in_word = True
            i += 2
            continue
        if quote:
            if ch == quote:
                quote = ""
            else:
                current.append(ch)
        elif ch in "'\"":
            quote = ch
            in_word = True
        elif ch.isspace():
            if in_word:
                words.append("".join(current))
                current, in_word = [], False
        else:
            current.append(ch)
            in_word = True
        i += 1
    assert not quote, f"unterminated quote in: {line}"
    if in_word:
        words.append("".join(current))

    def expand(word: str) -> str:
        word = word.replace("$$", "\0")
        word = re.sub(
            r"\$\{(\w+)\}",
            lambda m: env[m.group(1)],
            word,
        )
        return word.replace("\0", "$").replace("%%", "%")

    return [expand(w) for w in words]


def _shim(bin_dir: Path, name: str, body: str) -> None:
    shim = bin_dir / name
    shim.write_text(f"#!/usr/bin/env bash\n{body}\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class _Host:
    def __init__(self, tmp_path: Path, *, health: str, produce_rc: int = 0) -> None:
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.calls = tmp_path / "calls.log"
        self.calls.write_text("")
        self.produced = tmp_path / "produced.ndjson"
        _shim(
            self.bin,
            "docker",
            f'echo "docker $*" >> "{self.calls}"\n'
            'case "$1" in\n'
            f'  inspect) echo "{health}" ;;\n'
            f'  exec) cat >> "{self.produced}"; exit {produce_rc} ;;\n'
            "esac\n"
            "exit 0",
        )
        # The wait loop polls every couple of seconds; the test must not.
        _shim(self.bin, "sleep", "exit 0")

    def run(self, key: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}:{env['PATH']}"
        argv = _systemd_argv(_unit_lines(key)[0], _unit_env())
        return subprocess.run(
            argv, capture_output=True, text=True, env=env, timeout=60, check=False
        )

    def exec_lines(self) -> list[str]:
        return [
            line
            for line in self.calls.read_text().splitlines()
            if line.startswith("docker exec")
        ]


def test_nothing_in_the_unit_points_at_an_untracked_host_script() -> None:
    directives = [
        line
        for line in _UNIT.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    assert not [line for line in directives if ".local/bin" in line]
    for value in _unit_lines("ExecStart"):
        assert value.startswith("/bin/bash -c "), value


def test_the_start_is_bounded() -> None:
    (timeout,) = _unit_lines("TimeoutStartSec")
    assert 0 < int(timeout) <= 600, timeout


def test_an_unhealthy_broker_fails_the_unit_before_any_produce(
    tmp_path: Path,
) -> None:
    host = _Host(tmp_path, health="starting")
    pre = host.run("ExecStartPre")
    assert pre.returncode != 0
    assert _unit_env()["ONEX_SWEEP_BROKER_CONTAINER"] in pre.stderr
    assert not host.exec_lines()


def test_a_healthy_broker_gets_one_command_per_run_with_a_fresh_id(
    tmp_path: Path,
) -> None:
    host = _Host(tmp_path, health="healthy")
    assert host.run("ExecStartPre").returncode == 0
    first = host.run("ExecStart")
    second = host.run("ExecStart")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    lines = host.exec_lines()
    assert len(lines) == 2, lines
    container = _unit_env()["ONEX_SWEEP_BROKER_CONTAINER"]
    for line in lines:
        assert f"exec -i {container} sh -c" in line
        assert _TOPIC in line
        assert "${DEV_KAFKA_SASL_USERNAME}" in line
        assert "${DEV_KAFKA_SASL_PASSWORD}" in line

    commands = [json.loads(row) for row in host.produced.read_text().splitlines()]
    assert len(commands) == 2
    for command in commands:
        assert command["event_type"] == "omnibase-infra.build-loop-start"
        assert command["correlation_id"] == command["payload"]["correlation_id"]
        assert command["payload"]["max_cycles"] == 1
        assert command["payload"]["dry_run"] is False
    assert commands[0]["correlation_id"] != commands[1]["correlation_id"]


def test_a_failed_produce_fails_the_unit(tmp_path: Path) -> None:
    host = _Host(tmp_path, health="healthy", produce_rc=1)
    result = host.run("ExecStart")
    assert result.returncode != 0
