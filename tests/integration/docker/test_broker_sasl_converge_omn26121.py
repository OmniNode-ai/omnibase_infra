# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26121: ``converge_broker_sasl`` drives a real ``docker`` executable.

The stability roll must run the overlay's ``redpanda-scram-user`` and
``redpanda-sasl-enable`` one-shots, in that order and each to exit 0, before it
recreates a client. This module runs the shipped function with real subprocess
calls against a small executable named ``docker`` found on ``PATH`` (a stand-in
for the daemon that records calls in a state file), so a hosted runner needs no
daemon, credential or lab host.
"""

from __future__ import annotations

import os
import stat
import subprocess
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.nodes.node_setup_local_provision_effect.runtime_cost_estimator_build import (
    BROKER_SASL_ONESHOTS,
    converge_broker_sasl,
)

SCRAM, ENABLE = BROKER_SASL_ONESHOTS

FAKE_DOCKER = """\
#!/bin/sh
state="$FAKE_DOCKER_STATE"
printf '%s\\n' "$*" >> "$state/calls.log"
case "$*" in
  *"config --services"*) printf 'redpanda\\n%s\\nwriter\\n' "$FAKE_DECLARED" ;;
  *" ps "*) for last; do :; done; printf 'id-%s\\n' "$last" ;;
  "wait id-"*) service="${2#id-}"
    if [ "$service" = "$FAKE_FAIL" ]; then echo 1; else echo 0; fi ;;
esac
exit 0
"""


class _Phase(Enum):
    RUNTIME = "runtime"


def _execution() -> SimpleNamespace:
    def run(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=kwargs.get("env"),
        )

    return SimpleNamespace(
        _run=run,
        _compose_argv=lambda lane: ["docker", "compose", "-p", lane],
        RUNTIME_HEALTH_PROBE_TIMEOUT_SECONDS=30,
        PHASE_TIMEOUTS={_Phase.RUNTIME: 30},
        Phase=_Phase,
    )


def _environment(tmp_path: Path, declared: str, fail: str = "") -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DOCKER_STATE": str(tmp_path),
        "FAKE_DECLARED": declared,
        "FAKE_FAIL": fail,
    }


def _calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.integration
def test_both_one_shots_run_in_order_each_waited_to_exit_0(tmp_path: Path) -> None:
    env = _environment(tmp_path, f"{SCRAM}\n{ENABLE}")
    ran = converge_broker_sasl(_execution(), "lane", env)
    assert ran == (SCRAM, ENABLE)
    started = [c for c in _calls(tmp_path) if " up " in c or c.startswith("wait")]
    assert [c.split()[-1] for c in started] == [
        SCRAM,
        f"id-{SCRAM}",
        ENABLE,
        f"id-{ENABLE}",
    ]


@pytest.mark.integration
def test_a_non_zero_one_shot_raises_before_the_next_starts(tmp_path: Path) -> None:
    env = _environment(tmp_path, f"{SCRAM}\n{ENABLE}", fail=SCRAM)
    with pytest.raises(RuntimeError, match=f"{SCRAM} exited 1"):
        converge_broker_sasl(_execution(), "lane", env)
    assert not any(c.endswith(ENABLE) and " up " in c for c in _calls(tmp_path))


@pytest.mark.integration
def test_a_lane_declaring_neither_one_shot_starts_nothing(tmp_path: Path) -> None:
    env = _environment(tmp_path, "")
    assert converge_broker_sasl(_execution(), "lane", env) == ()
    assert not any(" up " in c for c in _calls(tmp_path))
