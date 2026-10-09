# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-merge runtime boot also boots the dev lane's configuration class (OMN-17427).

Two changes crash-looped the .201 dev lane on 2026-10-09, and every CI runtime
boot passed both:

* omnibase_infra#4750 made the runtime refuse to start whenever
  ``INFISICAL_ADDR`` is set and no store overlay is built. The dev lane binds
  ``INFISICAL_ADDR`` (``x-runtime-env`` in docker-compose.infra.yml); the
  laptop render blanks it and the compose-mode boot never sets it, and the
  PR's own landing fixes taught both CI boots to seed the overlay the PR
  demanded, which the dev lane does not have.
* omnimarket 0.4.308 added two nodes whose auto-wired route ids exceeded
  ``ModelDispatchRoute.route_id``. The dev lane binds
  ``ONEX_WIRING_STRICT_MODE=1`` (``x-dev-lane-strict-wiring-env`` in
  docker-compose.dev-lane.yml), so the ``ValidationError`` killed boot; no CI
  boot sets strict mode, so the failure was quarantined and /health stayed
  healthy.

The laptop boot job (``boot-catalog-local``) now ends with a dev-lane class
phase: it rebuilds the runtime image on the current omnimarket release,
recreates both kernels with the environment DERIVED from the dev lane's own
compose files, and fails unless both stay healthy and running with no restart.
These tests run the phase's own ``run:`` text from the workflow, unmodified:

* the derivation step, for real, against the repo's compose files and against
  copies with the dev lane's binding removed (it must refuse, never default);
* the crash-loop gate, against a stub Docker that reports restarts, an exited
  kernel, and a healthy one.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "reusable-runtime-boot.yml"
_CI = _REPO / ".github" / "workflows" / "ci.yml"
_JOB = "boot-catalog-local"
_DERIVE = (
    "Dev-lane class -- derive the kernel environment from the dev lane's compose files"
)
_RELEASE = "Dev-lane class -- resolve the current omnimarket release"
_REBUILD = "Dev-lane class -- rebuild the runtime image on that release"
_RECREATE = "Dev-lane class -- recreate both kernels with that environment"
_GATE = "Dev-lane class -- both kernels stay healthy and running with no restart (crash-loop gate)"
_KERNELS = ("omninode-runtime", "runtime-effects")


# Compose's merge tags (``!override``, ``!reset``) are not YAML that SafeLoader
# accepts; they only change how compose merges files, so they are dropped here.
_COMPOSE_TAG = re.compile(r"(?<=\s)!(override|reset)(?=\s)")


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return cast("list[dict[str, Any]]", workflow["jobs"][_JOB]["steps"])


def _step(name: str) -> dict[str, Any]:
    hits = [s for s in _steps() if s.get("name") == name]
    assert len(hits) == 1, f"expected exactly one step named {name!r} in {_JOB}"
    return hits[0]


def _run_step(
    name: str, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    step = _step(name)
    step_env = {str(k): str(v) for k, v in (step.get("env") or {}).items()}
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=cwd,
        env={**os.environ, **step_env, **env},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _compose(path: Path) -> dict[str, Any]:
    text = _COMPOSE_TAG.sub("", path.read_text(encoding="utf-8"))
    return cast("dict[str, Any]", yaml.safe_load(text))


# --------------------------------------------------------------------------
# Wiring: the phase exists, in order, inside the job CI Summary sweeps.
# --------------------------------------------------------------------------


def test_the_phase_runs_in_order_after_the_laptop_checks_and_before_the_log_tail() -> (
    None
):
    names = [s.get("name") for s in _steps()]
    for name in (_DERIVE, _RELEASE, _REBUILD, _RECREATE, _GATE):
        assert name in names, f"{_JOB} lacks the step {name!r}"
    order = [names.index(n) for n in (_DERIVE, _RELEASE, _REBUILD, _RECREATE, _GATE)]
    assert order == sorted(order)
    assert (
        names.index("Delegate-skill command topic has a live consumer group") < order[0]
    )
    assert order[-1] < names.index("Container state and log tail (always)")
    for name in (_DERIVE, _RELEASE, _REBUILD, _RECREATE, _GATE):
        assert "if" not in _step(name), f"{name!r} must not be conditional"
        assert not _step(name).get("continue-on-error"), (
            f"{name!r} must not be advisory"
        )


def test_the_laptop_job_runs_when_a_dev_lane_class_input_changes() -> None:
    ci = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    run = ci["jobs"]["laptop-profile-paths"]["steps"][1]["run"]
    pattern = re.search(r"pattern='([^']+)'", run)
    assert pattern is not None
    for path in (
        "docker/docker-compose.dev-lane.yml",
        "docker/docker-compose.infra.yml",
    ):
        assert re.search(pattern.group(1), path), (
            f"{path} does not trigger the laptop boot"
        )


def test_no_literal_dev_lane_value_is_restated_in_the_workflow() -> None:
    derive = _step(_DERIVE)
    text = derive["run"] + yaml.safe_dump(derive.get("env") or {})
    assert "ONEX_WIRING_STRICT_MODE" not in text, (
        "the strict-wiring binding must be read from docker-compose.dev-lane.yml, "
        "never restated in the workflow"
    )


# --------------------------------------------------------------------------
# The derivation step, run for real.
# --------------------------------------------------------------------------


def _derive(
    tmp_path: Path, dev_lane: Path, infra: Path
) -> subprocess.CompletedProcess[str]:
    return _run_step(
        _DERIVE,
        {
            "DEV_LANE_COMPOSE_FILE": str(dev_lane),
            "INFRA_COMPOSE_FILE": str(infra),
            "DEV_LANE_CLASS_OVERRIDE": str(tmp_path / "override.yml"),
            "RUNNER_TEMP": str(tmp_path),
        },
        _REPO,
    )


def test_the_override_carries_the_dev_lane_bindings_on_both_kernels(
    tmp_path: Path,
) -> None:
    proc = _derive(
        tmp_path,
        _REPO / "docker" / "docker-compose.dev-lane.yml",
        _REPO / "docker" / "docker-compose.infra.yml",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    override = yaml.safe_load((tmp_path / "override.yml").read_text(encoding="utf-8"))
    strict = _compose(_REPO / "docker" / "docker-compose.dev-lane.yml")[
        "x-dev-lane-strict-wiring-env"
    ]
    assert strict, "the dev lane overlay no longer binds strict wiring"
    for kernel in _KERNELS:
        env = override["services"][kernel]["environment"]
        for key, value in strict.items():
            assert env[key] == str(value), f"{kernel}: {key} differs from the dev lane"
        assert env["INFISICAL_ADDR"].strip(), (
            f"{kernel}: the secrets-store address is unset"
        )
    effects_env = override["services"]["runtime-effects"]["environment"]
    infra_effects = _compose(_REPO / "docker" / "docker-compose.infra.yml")["services"][
        "runtime-effects"
    ]["environment"]
    for key in ("ONEX_SECRET_RESOLVER_CONFIG_PATH", "ONEX_SECRET_RESOLVER_CONFIG_JSON"):
        assert effects_env[key] == str(infra_effects[key]), (
            f"runtime-effects: {key} differs from the infra base"
        )


def test_a_dev_lane_without_the_strict_binding_refuses_rather_than_defaults(
    tmp_path: Path,
) -> None:
    dev_lane = _compose(_REPO / "docker" / "docker-compose.dev-lane.yml")
    dev_lane.pop("x-dev-lane-strict-wiring-env")
    mutated = tmp_path / "dev-lane.yml"
    mutated.write_text(yaml.safe_dump(dev_lane), encoding="utf-8")
    proc = _derive(tmp_path, mutated, _REPO / "docker" / "docker-compose.infra.yml")
    assert proc.returncode != 0
    assert "x-dev-lane-strict-wiring-env" in proc.stdout + proc.stderr
    assert not (tmp_path / "override.yml").exists()


def test_an_infra_base_without_the_store_binding_refuses_rather_than_defaults(
    tmp_path: Path,
) -> None:
    infra = _compose(_REPO / "docker" / "docker-compose.infra.yml")
    infra["x-runtime-env"].pop("INFISICAL_ADDR")
    mutated = tmp_path / "infra.yml"
    mutated.write_text(yaml.safe_dump(infra), encoding="utf-8")
    proc = _derive(tmp_path, _REPO / "docker" / "docker-compose.dev-lane.yml", mutated)
    assert proc.returncode != 0
    assert "INFISICAL_ADDR" in proc.stdout + proc.stderr
    assert not (tmp_path / "override.yml").exists()


# --------------------------------------------------------------------------
# The crash-loop gate, against a stub Docker.
# --------------------------------------------------------------------------

_DOCKER_STUB = """#!/usr/bin/env bash
set -u
case "$1" in
  inspect)
    container="${!#}"
    case "$*" in
      *RestartCount*)
        if [ "$container" = "$STUB_LOOPING" ]; then echo 3; else echo 0; fi ;;
      *State.Status*)
        if [ "$container" = "$STUB_EXITED" ]; then echo exited; else echo running; fi ;;
      *Health.Status*) echo healthy ;;
      *) echo "stub docker: unexpected inspect $*" >&2; exit 2 ;;
    esac ;;
  exec)
    echo '{"status": "healthy", "details": {"is_running": true}}' ;;
  logs) echo "stub log line" ;;
  *) echo "stub docker: unexpected $*" >&2; exit 2 ;;
esac
"""


def _stub_path(tmp_path: Path) -> str:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(_DOCKER_STUB, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    sleep = bindir / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    sleep.chmod(sleep.stat().st_mode | stat.S_IEXEC)
    jq = shutil.which("jq")
    assert jq, "jq is required by the gate step"
    return f"{bindir}:{os.environ['PATH']}"


def _gate(
    tmp_path: Path, looping: str = "", exited: str = ""
) -> subprocess.CompletedProcess[str]:
    return _run_step(
        _GATE,
        {
            "PATH": _stub_path(tmp_path),
            "LOCAL_PROJECT": "omnibase-infra-local",
            "STUB_LOOPING": looping,
            "STUB_EXITED": exited,
        },
        _REPO,
    )


def test_the_gate_passes_two_steady_kernels(tmp_path: Path) -> None:
    proc = _gate(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("kernel", _KERNELS)
def test_the_gate_fails_a_restarting_kernel(tmp_path: Path, kernel: str) -> None:
    proc = _gate(tmp_path, looping=f"omnibase-infra-local-{kernel}")
    assert proc.returncode != 0
    assert kernel in proc.stdout + proc.stderr
    assert "restart" in (proc.stdout + proc.stderr).lower()


@pytest.mark.parametrize("kernel", _KERNELS)
def test_the_gate_fails_an_exited_kernel(tmp_path: Path, kernel: str) -> None:
    proc = _gate(tmp_path, exited=f"omnibase-infra-local-{kernel}")
    assert proc.returncode != 0
    assert kernel in proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# The release step, replayed against a recorded live `git ls-remote`.
# --------------------------------------------------------------------------

_GIT_STUB = """#!/usr/bin/env bash
set -u
[ "$1" = "ls-remote" ] || { echo "stub git: unexpected $*" >&2; exit 2; }
printf '%s' "$STUB_LS_REMOTE"
"""


def _release(
    tmp_path: Path, ls_remote: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    git = bindir / "git"
    git.write_text(_GIT_STUB, encoding="utf-8")
    git.chmod(git.stat().st_mode | stat.S_IEXEC)
    github_env = tmp_path / "github_env"
    github_env.write_text("", encoding="utf-8")
    proc = _run_step(
        _RELEASE,
        {
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "STUB_LS_REMOTE": ls_remote,
            "GITHUB_ENV": str(github_env),
        },
        _REPO,
    )
    return proc, github_env


def _semver(tag: str) -> tuple[int, ...]:
    return tuple(int(part) for part in tag.removeprefix("v").split("."))


@pytest.mark.live_contact("tests/ci/fixtures/omnimarket_release_tags_omn17427.json")
def test_the_release_step_picks_the_highest_semver_tag_from_the_recorded_listing(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    stdout = str(recorded_response["stdout"])
    tags = [
        line.split("refs/tags/", 1)[1]
        for line in stdout.splitlines()
        if "refs/tags/" in line
        and re.fullmatch(r"v\d+\.\d+\.\d+", line.split("refs/tags/", 1)[1])
    ]
    expected = max(tags, key=_semver)
    # The recorded listing is in lexical order, where v0.4.99 sorts last: a
    # plain sort would boot a release two hundred versions old.
    assert sorted(tags)[-1] != expected
    proc, github_env = _release(tmp_path, stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert github_env.read_text(encoding="utf-8").strip() == (
        f"OMNIMARKET_RELEASE_REF={expected}"
    )


def test_the_release_step_fails_when_no_release_tag_resolves(tmp_path: Path) -> None:
    proc, github_env = _release(tmp_path, "")
    assert proc.returncode != 0
    assert github_env.read_text(encoding="utf-8") == ""
