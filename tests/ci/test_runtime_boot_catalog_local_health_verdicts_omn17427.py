# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed laptop boot names its cause at the end of its own log (OMN-17427).

On 2026-10-13 twelve merge-group runs of ``boot-catalog-local`` failed on the
same step, ``Boot the laptop profile (make up-local)``, after 2115-2240 s: the
``omninode-runtime`` container never passed its healthcheck inside its 1800 s
``start_period``, so ``projection-api`` (``depends_on: omninode-runtime:
service_healthy``) was never started and ``docker compose up`` gave up. The
runtime's own verdict -- its ``/health`` body and the healthcheck output
Docker keeps -- was never printed, and the kernel log tails the job does print
fall outside the last 64 KB of a 660 KB log, which is all a job log read
returns. The same merged tree booted green at 08:17 and red at 10:13, so the
cause lay outside the tree, and the log could not say what it was.

The job now prints, for both kernels, the Docker health state, the last
healthcheck outputs and the ``/health`` body as the last step before
teardown, so the reason is inside the end of the log on every run. These
tests run that step's own ``run:`` text against a stub Docker.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "reusable-runtime-boot.yml"
_JOB = "boot-catalog-local"
_VERDICTS = "Kernel health verdicts (always, last before teardown)"
_LOG_TAIL = "Container state and log tail (always)"
_TEARDOWN = "Tear down the laptop profile and its volumes (always)"
_KERNELS = ("omninode-runtime", "runtime-effects")
# GitHub stamps every job log line with a timestamp of this width.
_LINE_STAMP_BYTES = len("2026-10-13T12:43:24.8642089Z ")
# The step's group also echoes its `env:` block; the recorded teardown group's
# echo of the job env is about 600 bytes.
_ENV_ECHO_BYTES = 2000


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return cast("list[dict[str, Any]]", workflow["jobs"][_JOB]["steps"])


def _step(name: str) -> dict[str, Any]:
    hits = [s for s in _steps() if s.get("name") == name]
    assert len(hits) == 1, f"expected exactly one step named {name!r} in {_JOB}"
    return hits[0]


_DOCKER_STUB = """#!/usr/bin/env bash
set -u
[ -n "${STUB_FAIL:-}" ] && { echo "stub docker: daemon unreachable" >&2; exit 1; }
[ -n "${STUB_FLOOD:-}" ] && { yes "$(printf '%0100d' 0)" | head -c 200000; exit 0; }
case "$1" in
  info) echo "docker host stub-host, server 28.0.0, 32 cpus" ;;
  ps) printf 'NAMES\\tSTATUS\\nomnibase-infra-local-omninode-runtime\\tUp 31 minutes (unhealthy)\\n' ;;
  inspect)
    container="${!#}"
    case "$*" in
      *Health.Log*) echo "2026-10-13T12:42:51Z exit=1 FAIL runtime_degraded: discovery_errors: 2 contract(s) failed to load ($container)" ;;
      *) echo "status=running health=unhealthy restarts=0 ($container)" ;;
    esac ;;
  exec) echo "{\\"status\\": \\"degraded\\", \\"container\\": \\"$2\\"}" ;;
  *) echo "stub docker: unexpected $*" >&2; exit 2 ;;
esac
"""


def _run_verdicts(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(_DOCKER_STUB, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    step = _step(_VERDICTS)
    # GitHub runs a bash step as `bash -e -o pipefail`; run it the same way.
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=_REPO,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "LOCAL_PROJECT": "omnibase-infra-local",
            "RUNNER_NAME": "stub-runner-7",
            **env,
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_verdicts_step_is_always_run_last_before_teardown() -> None:
    names = [s.get("name") for s in _steps()]
    assert _VERDICTS in names, f"{_JOB} lacks the step {_VERDICTS!r}"
    assert names.index(_LOG_TAIL) < names.index(_VERDICTS)
    assert names.index(_VERDICTS) + 1 == names.index(_TEARDOWN), (
        "the verdicts must be the last output before teardown, so they sit "
        "inside the end of the job log"
    )
    assert _step(_VERDICTS).get("if") == "always()"


def test_the_verdicts_name_the_runner_and_each_kernels_health(tmp_path: Path) -> None:
    proc = _run_verdicts(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "stub-runner-7" in out
    assert "stub-host" in out
    assert "(unhealthy)" in out
    for kernel in _KERNELS:
        container = f"omnibase-infra-local-{kernel}"
        assert f"health=unhealthy restarts=0 ({container})" in out
        assert (
            f"runtime_degraded: discovery_errors: 2 contract(s) failed to load ({container})"
            in out
        )
        assert f'"container": "{container}"' in out


def test_the_verdicts_never_fail_the_job_when_docker_cannot_answer(
    tmp_path: Path,
) -> None:
    proc = _run_verdicts(tmp_path, STUB_FAIL="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "daemon unreachable" in proc.stderr


@pytest.mark.live_contact("tests/ci/fixtures/omn17427_catalog_local_boot_log_tail.json")
def test_the_verdicts_fit_inside_the_log_tail_a_failed_boot_leaves(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    recording = cast("dict[str, Any]", recorded_response["response"])
    # The recorded read of a failed boot is the tail of a longer log, and no
    # kernel's log group reaches it: the old dump could not name the cause.
    assert recording["truncated"] is True
    assert recording["total_bytes"] > recording["tail_bytes"]
    headers = cast("list[str]", recording["group_headers"])
    for kernel in _KERNELS:
        assert not any(f"{kernel} logs (tail)" in header for header in headers)

    # At its largest -- every docker call flooding its output -- the verdicts
    # step, plus everything the recorded log printed after the dump step,
    # still fits inside that same read.
    proc = _run_verdicts(tmp_path, STUB_FLOOD="1")
    assert proc.returncode == 0, proc.stderr
    printed = proc.stdout + proc.stderr
    script = cast("str", _step(_VERDICTS)["run"])
    lines = printed.count("\n") + script.count("\n") + 2
    verdict_bytes = (
        len(printed.encode())
        + len(script.encode())
        + _ENV_ECHO_BYTES
        + lines * _LINE_STAMP_BYTES
    )
    after = len(cast("str", recording["after_log_tail_dump"]).encode())
    assert verdict_bytes + after <= recording["tail_bytes"], (
        f"verdicts {verdict_bytes} B + teardown {after} B exceed the "
        f"{recording['tail_bytes']} B a job log read returns"
    )
