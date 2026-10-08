# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execute the laptop-profile health step against a stub Docker (OMN-19972).

OMN-19972's demo half adds the projection API, the LLM call and cost writer and
the consumer-health projection to the ``local`` bundle, and its acceptance says
the ``runtime-boot (mode=catalog-local)`` job asserts each added service
healthy -- falsified by the step passing with a service stopped.

The step derives the services to wait for from the ``local`` render itself
(every service with a healthcheck that is not a one-shot), so a service added
to the bundle later is covered without editing the workflow. These tests run
the step's own ``run:`` text, unmodified: the derivation runs for real through
``uv run``, and ``docker`` and ``sleep`` are stubbed on PATH.
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
_STEP = "Every health-checked laptop service reports healthy"
_PROJECT = "omnibase-infra-local"
_ADDED = (
    "projection-api",
    "omnimarket-projection-llm-cost",
    "consumer-health-projection",
)

_DOCKER_STUB = """#!/usr/bin/env bash
# Stub of `docker inspect --format ... <container>` for the health step.
set -u
[ "$1" = "inspect" ] || { echo "stub docker: unexpected $*" >&2; exit 2; }
container="${!#}"
echo "$container" >> "$STUB_SEEN"
if [ "$container" = "$STUB_MISSING" ]; then
  echo "Error: No such object: $container" >&2
  exit 1
fi
case "$*" in
  *RestartCount*) echo 0 ;;
  *Health.Status*)
    if [ "$container" = "$STUB_UNHEALTHY" ]; then echo unhealthy;
    elif [[ " $STUB_STARTING " == *" $container "* ]]; then echo starting;
    else echo healthy; fi ;;
  *) echo "stub docker: unexpected inspect $*" >&2; exit 2 ;;
esac
"""


def _step_script() -> str:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][_JOB]["steps"]
    matches = [s for s in steps if s.get("name") == _STEP]
    assert len(matches) == 1, f"expected exactly one step named {_STEP!r}"
    run = matches[0]["run"]
    assert isinstance(run, str)
    return run


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_step(
    tmp_path: Path, *, unhealthy: str = "", missing: str = "", starting: str = ""
) -> tuple[subprocess.CompletedProcess[str], list[str], int]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "docker", _DOCKER_STUB)
    sleeps = tmp_path / "sleeps.txt"
    sleeps.write_text("", encoding="utf-8")
    _write_executable(
        bin_dir / "sleep", f'#!/usr/bin/env bash\necho "$1" >> {sleeps}\nexit 0\n'
    )
    script = tmp_path / "step.sh"
    script.write_text(_step_script(), encoding="utf-8")
    seen = tmp_path / "seen.txt"
    seen.write_text("", encoding="utf-8")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LOCAL_PROJECT": _PROJECT,
        "STUB_SEEN": str(seen),
        "STUB_UNHEALTHY": f"{_PROJECT}-{unhealthy}" if unhealthy else "",
        "STUB_MISSING": f"{_PROJECT}-{missing}" if missing else "",
        "STUB_STARTING": " ".join(f"{_PROJECT}-{s}" for s in starting.split()),
    }
    # GitHub runs a `run:` block as `bash -e {0}`, from the repository root.
    result = subprocess.run(
        ["bash", "-e", str(script)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    polls = len(sleeps.read_text(encoding="utf-8").split())
    return result, seen.read_text(encoding="utf-8").split(), polls


def _show(result: subprocess.CompletedProcess[str]) -> str:
    return f"exit {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"


def test_step_passes_and_checks_every_added_service(tmp_path: Path) -> None:
    result, seen, _ = _run_step(tmp_path)
    assert result.returncode == 0, _show(result)
    for svc in _ADDED:
        assert f"{_PROJECT}-{svc}" in seen, (svc, seen)


def test_step_fails_when_an_added_service_is_unhealthy(tmp_path: Path) -> None:
    result, _, _ = _run_step(tmp_path, unhealthy="projection-api")
    assert result.returncode == 1, _show(result)
    assert "projection-api" in result.stdout


def test_step_fails_when_an_added_service_never_started(tmp_path: Path) -> None:
    result, _, _ = _run_step(tmp_path, missing="omnimarket-projection-llm-cost")
    assert result.returncode == 1, _show(result)
    assert "omnimarket-projection-llm-cost" in result.stdout


def test_step_fails_when_an_added_service_never_leaves_starting(tmp_path: Path) -> None:
    result, _, _ = _run_step(tmp_path, starting="consumer-health-projection")
    assert result.returncode == 1, _show(result)
    assert "consumer-health-projection" in result.stdout


def test_two_stuck_services_share_one_wait_budget(tmp_path: Path) -> None:
    """Hostile review (both models, two passes): each stuck service waited its own
    90 x 10 s, so N stuck services cost N x 15 minutes of shared CI. One budget of
    90 polls now covers the whole step."""
    result, _, polls = _run_step(
        tmp_path, starting="consumer-health-projection projection-api"
    )
    assert result.returncode == 1, _show(result)
    assert "consumer-health-projection" in result.stdout
    assert "projection-api" in result.stdout
    assert polls <= 90, polls


# --- the negative control that runs on the CI runner (OMN-19972 AC2) ----------
#
# The stub tests above prove the step fails when Docker *says* unhealthy. They
# cannot show what a real engine reports for a container that was stopped after
# it reached healthy, and the closer's review (OMN-19972, 2026-10-04) argued from
# the moby source that such a container keeps reading healthy. moby v28.0.4 (the
# engine of the ubuntu-24.04 runner image 20260927.320) says otherwise:
# container/health.go CloseMonitorChannel sets Health.Status to unhealthy when
# the monitor stops, and daemon/monitor.go calls it on container exit. Source and
# a lab daemon are not the runner, so the workflow carries its own step that
# stops each added service on the runner and requires the health step to fail
# naming it. These tests run that step's text, unmodified, against a stateful
# stub Docker: one that models the engine as moby does, one that models the
# closer's claim, and one that never recovers.

_NEG_STEP = (
    "Stopped laptop service fails the health step and restoring it passes "
    "(negative control)"
)
_NEG_STUB = """#!/usr/bin/env bash
# Stateful stub of the docker verbs the health step and its negative control use.
set -u
state="$STUB_STATE"
verb="$1"; shift
container="${!#}"
echo "$verb $container" >> "$STUB_LOG"
case "$verb" in
  stop)
    echo exited > "$state/$container.run"
    if [ "${STUB_STOPPED_KEEPS_HEALTHY:-0}" = 1 ]; then echo healthy > "$state/$container.health";
    else echo unhealthy > "$state/$container.health"; fi
    # a different service going red at the same moment: the step then fails, but
    # not because of the service that was stopped
    if [ -n "${STUB_STOP_BREAKS_OTHER:-}" ]; then
      echo unhealthy > "$state/$STUB_STOP_BREAKS_OTHER.health"
      echo healthy > "$state/$container.health"
    fi ;;
  start)
    echo running > "$state/$container.run"
    if [ "${STUB_NEVER_RECOVERS:-0}" = 1 ]; then echo unhealthy > "$state/$container.health";
    else echo healthy > "$state/$container.health"; fi ;;
  inspect)
    [ -f "$state/$container.run" ] || { echo "Error: No such object: $container" >&2; exit 1; }
    case "$*" in
      *RestartCount*) echo 0 ;;
      *State.Status*) echo "$(cat "$state/$container.run") health=$(cat "$state/$container.health")" ;;
      *Health.Status*) cat "$state/$container.health" ;;
      *) echo "stub docker: unexpected inspect $*" >&2; exit 2 ;;
    esac ;;
  *) echo "stub docker: unexpected $verb" >&2; exit 2 ;;
esac
"""


def _neg_script() -> str:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][_JOB]["steps"]
    matches = [s for s in steps if s.get("name") == _NEG_STEP]
    assert len(matches) == 1, f"expected exactly one step named {_NEG_STEP!r}"
    run = matches[0]["run"]
    assert isinstance(run, str)
    return run


def _services_of_the_local_render() -> list[str]:
    from omnibase_infra.docker.catalog.generator import generate_compose
    from omnibase_infra.docker.catalog.resolver import CatalogResolver

    compose = generate_compose(
        CatalogResolver(catalog_dir=str(_REPO / "docker" / "catalog")).resolve(
            ["local"]
        )
    )
    services = cast("dict[str, dict[str, Any]]", compose["services"])
    return [
        name
        for name, svc in services.items()
        if "healthcheck" in svc and svc.get("restart") != "no"
    ]


def _run_negative_control(
    tmp_path: Path, **stub_env: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "docker", _NEG_STUB)
    _write_executable(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    state = tmp_path / "state"
    state.mkdir()
    for svc in _services_of_the_local_render():
        (state / f"{_PROJECT}-{svc}.run").write_text("running\n", encoding="utf-8")
        (state / f"{_PROJECT}-{svc}.health").write_text("healthy\n", encoding="utf-8")
    log = tmp_path / "docker.log"
    log.write_text("", encoding="utf-8")
    script = tmp_path / "neg.sh"
    script.write_text(_neg_script(), encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LOCAL_PROJECT": _PROJECT,
        "STUB_STATE": str(state),
        "STUB_LOG": str(log),
        **stub_env,
    }
    result = subprocess.run(
        ["bash", "-e", str(script)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return result, log.read_text(encoding="utf-8").splitlines()


def test_negative_control_step_follows_the_health_step() -> None:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    names = [s.get("name") for s in workflow["jobs"][_JOB]["steps"]]
    assert names.index(_NEG_STEP) == names.index(_STEP) + 1


def test_negative_control_stops_each_added_service_and_restores_it(
    tmp_path: Path,
) -> None:
    result, log = _run_negative_control(tmp_path)
    assert result.returncode == 0, _show(result)
    for svc in _ADDED:
        container = f"{_PROJECT}-{svc}"
        assert f"stop {container}" in log, (svc, log)
        assert f"start {container}" in log, (svc, log)
        assert log.index(f"stop {container}") < log.index(f"start {container}")


def test_negative_control_goes_red_if_a_stopped_service_still_reads_healthy(
    tmp_path: Path,
) -> None:
    """The closer's claim, modelled: an engine that leaves a stopped container
    reading healthy makes the health step pass with a service stopped, and the
    control must say so rather than pass."""
    result, log = _run_negative_control(tmp_path, STUB_STOPPED_KEEPS_HEALTHY="1")
    assert result.returncode == 1, _show(result)
    assert "the health step passed with projection-api stopped" in result.stdout
    assert f"start {_PROJECT}-projection-api" in log, "the service must be restored"


def test_negative_control_goes_red_if_a_restored_service_never_recovers(
    tmp_path: Path,
) -> None:
    result, _ = _run_negative_control(tmp_path, STUB_NEVER_RECOVERS="1")
    assert result.returncode == 1, _show(result)
    assert "laptop services not healthy" in result.stdout


def test_negative_control_goes_red_if_the_step_fails_for_another_service(
    tmp_path: Path,
) -> None:
    """A red health step proves nothing about the stopped service when some other
    service is the one it names."""
    result, _ = _run_negative_control(
        tmp_path, STUB_STOP_BREAKS_OTHER=f"{_PROJECT}-postgres"
    )
    assert result.returncode == 1, _show(result)
    assert "but not because of projection-api" in result.stdout


def test_negative_control_goes_red_when_the_health_step_cannot_be_read(
    tmp_path: Path,
) -> None:
    """A renamed health step must break the control, not let it read nothing."""
    script = _neg_script().replace(
        "'Every health-checked laptop service reports healthy'",
        "'A step that does not exist'",
        1,
    )
    assert "A step that does not exist" in script
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "docker", _NEG_STUB)
    (tmp_path / "state").mkdir()
    (tmp_path / "docker.log").write_text("", encoding="utf-8")
    runner = tmp_path / "neg.sh"
    runner.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-e", str(runner)],
        cwd=_REPO,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "LOCAL_PROJECT": _PROJECT,
            "STUB_STATE": str(tmp_path / "state"),
            "STUB_LOG": str(tmp_path / "docker.log"),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0, _show(result)
    assert "expected exactly one step named" in (result.stderr + result.stdout)
