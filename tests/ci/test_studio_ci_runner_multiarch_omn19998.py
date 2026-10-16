# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19998 -- the Mac Studio takes arch-neutral private-repo CI, opt-in.

Private-repository CI starved on the amd64 fleet while the arm64 Mac Studio
had headroom. An arm64 runner cannot simply join ``omnibase-ci``: every job
whose ``runs-on`` is ``[self-hosted, omnibase-ci]`` was written assuming amd64,
so it would be free to land there. The fleet therefore grows an OPT-IN label,
``omnibase-ci-multiarch``:

* the Mac Studio pool (``omninode-studio-ci-runner``) carries it, with
  ``arch-arm64`` and WITHOUT ``omnibase-ci``, so no existing job can be placed
  on it;
* both amd64 action pools carry it beside ``omnibase-ci`` and ``arch-amd64``,
  so a job that moves to ``[self-hosted, omnibase-ci-multiarch]`` keeps the
  whole amd64 fleet and gains the Mac Studio, rather than trading one for the
  other;
* a job that needs amd64 or a docker daemon stays on ``omnibase-ci``.

The Mac Studio pool is capped and credential-free: the host also serves the
always-on planner and a compose lane, so a job there gets no docker socket and
no lab credentials (nothing is copied to that host).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
FLEET_CONFIG = ROOT / "config" / "runner_fleet.yaml"
STUDIO_POOL = "omninode-studio-ci-runner"
STUDIO_COMPOSE = DOCKER / f"docker-compose.runners-{STUDIO_POOL}.yml"
MULTIARCH = "omnibase-ci-multiarch"
MULTIARCH_CLASS = "action-multiarch"
# The two amd64 action pools: the primary anchor and the second host's pool file.
AMD64_ACTION_COMPOSE = (
    DOCKER / "docker-compose.runners.yml",
    DOCKER / "docker-compose.runners-omnipc2-ci-runner.yml",
)


def _services(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("services") or {}


def _labels(service: dict[str, Any]) -> set[str]:
    env = service.get("environment") or {}
    return set(str(env.get("RUNNER_LABELS", "")).split(",")) - {""}


def _studio_runners() -> dict[str, dict[str, Any]]:
    return {
        name: service
        for name, service in _services(STUDIO_COMPOSE).items()
        if name.startswith(f"{STUDIO_POOL}-")
    }


def test_studio_host_declares_the_multiarch_pool() -> None:
    config = load_runner_fleet_config(FLEET_CONFIG)
    studio = config.host_of_runner(f"{STUDIO_POOL}-4")
    assert studio is not None
    # The pool sits on the Mac Studio row: the one whose verify runner is
    # omninode-studio-runner, and which is arm64.
    assert studio.runner_name_prefix == "omninode-studio-runner"
    assert studio.arch.value == "arm64"
    pool = next(p for p in studio.pools if p.runner_name_prefix == STUDIO_POOL)
    assert pool.classes == (MULTIARCH_CLASS,)
    assert pool.expected_count == 4


def test_no_arm64_row_or_pool_carries_the_action_class() -> None:
    """`action` is the amd64-assuming `omnibase-ci` fleet; arm64 never joins it."""
    config = load_runner_fleet_config(FLEET_CONFIG)
    for host in config.hosts_for_arch("arm64"):
        assert "action" not in host.classes, host.host
        for pool in host.pools:
            assert "action" not in pool.classes, pool.runner_name_prefix


def test_declared_totals_keep_action_and_count_multiarch_on_both_arches() -> None:
    config = load_runner_fleet_config(FLEET_CONFIG)
    assert config.declared_total("action") == 26
    # 10 on the primary host + 16 on the second amd64 host + 4 on the Mac Studio.
    assert config.declared_total(MULTIARCH_CLASS) == 30


def test_studio_pool_compose_matches_the_declared_count() -> None:
    runners = _studio_runners()
    assert sorted(runners) == [f"{STUDIO_POOL}-{n}" for n in range(1, 5)]


def test_studio_pool_is_opt_in_arm64_and_never_omnibase_ci() -> None:
    for name, runner in _studio_runners().items():
        labels = _labels(runner)
        assert {"self-hosted", MULTIARCH, "arm64", "arch-arm64"} <= labels, name
        assert "omnibase-ci" not in labels, name
        assert not labels & {"x64", "arch-amd64"}, (name, labels)
        assert not any(label.startswith("host-") for label in labels), name


def test_studio_pool_has_no_docker_socket_and_no_credentials() -> None:
    for name, runner in _studio_runners().items():
        joined = "\n".join(str(v) for v in runner.get("volumes") or [])
        for absent in (
            "docker.sock",
            "lab-credentials",
            "operator.env",
            "OMNI_HOME",
            ".omnibase/.env",
        ):
            assert absent not in joined, (name, absent)
        env = runner.get("environment") or {}
        for absent_env in ("KUBECONFIG", "LOCAL_LLM_SHARED_SECRET"):
            assert absent_env not in env, (name, absent_env)


def test_studio_pool_is_capped_for_the_planner_and_dev_200_lane() -> None:
    for name, runner in _studio_runners().items():
        assert runner.get("cpus") == "1.5", name
        assert runner.get("mem_limit") == "3g", name
        assert runner.get("memswap_limit") == "3g", name


def test_amd64_action_pools_also_carry_the_multiarch_label() -> None:
    """Without this, a job moved to the opt-in label would lose the amd64 fleet."""
    for path in AMD64_ACTION_COMPOSE:
        ci_runners = {
            name: service
            for name, service in _services(path).items()
            if "omnibase-ci" in _labels(service or {})
        }
        assert ci_runners, path.name
        for name, service in ci_runners.items():
            labels = _labels(service)
            assert {MULTIARCH, "arch-amd64"} <= labels, (path.name, name, labels)


def _fleet_event() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "runner_fleet_event_omn19998", ROOT / "scripts" / "runner_fleet_event.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fleet_panel_classifies_both_pools() -> None:
    module = _fleet_event()
    amd64 = ["self-hosted", "omnibase-ci", MULTIARCH, "linux", "x64", "arch-amd64"]
    arm64 = ["self-hosted", MULTIARCH, "linux", "arm64", "arch-arm64"]
    assert module.classify(amd64) == "omnibase-ci"
    assert module.classify(arm64) == MULTIARCH
