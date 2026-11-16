# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent's code-ref pin cannot send the build a service the compose lacks (OMN-23368).

THE FAILURE
-----------
Twice on 2026-11-09 a dev deploy died inside ``docker compose`` with ``no such service``
(``intelligence-api``, then ``infra-routing-decisions-consumer``). The agent had followed a
code-ref pin on a branch cut before a dev merge that removed those services, so its own
service lists (``deploy_agent.events``) named services the compose it built no longer
declared. Each failure surfaced as a docker error after the agent had started work.

WHAT THESE TESTS PIN
--------------------
AC1: before any build, ``rebuild_scope`` reads the services the lane's compose files
declare and refuses, naming the pin and the missing service, when the service list names
one the compose lacks. The docker build is never reached.

AC2: a pin behind the dev head is reported stale before the build starts, with the number
of dev commits it lacks.

Each refusal test has a control on the same fixture that names only declared services and
reaches the build, so a green run cannot mean the guard never looks.
"""

from __future__ import annotations

import logging
import subprocess
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.compose_budget import _read_compose_services
from deploy_agent.events import (
    DEV_LANE_BUILD_SERVICES,
    EnumRuntimeLane,
    Scope,
    services_for_scope,
    without_gateway_services,
)
from deploy_agent.executor import DeployExecutor, load_dev_boot_guard

pytestmark = [pytest.mark.unit, pytest.mark.service_pin_guard]

code_ref_pin = load_dev_boot_guard("handler_dev_lane_code_ref_pin")
guard = load_dev_boot_guard("handler_dev_lane_pin_compose_guard")
ComposeServiceMissingError = guard.ComposeServiceMissingError
assert_services_declared = guard.assert_services_declared
_PIN_FILE = code_ref_pin.PIN_FILE_NAME
_PINNED_BRANCH = "omn-22930-dev-effects-fold"

_COMPOSE_WITHOUT_REMOVED = """\
services:
  omninode-runtime:
    image: runtime
  runtime-effects:
    image: runtime
"""


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(cwd),
        },
    )
    return result.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "--quiet", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


class _World:
    """A bare origin whose ``dev`` moved past the pin's branch point, and the agent's clone."""

    def __init__(self, root: Path) -> None:
        self.origin = root / "origin.git"
        seed = root / "seed"
        self.clone = root / "agent-clone"
        self.state = root / "state"
        self.state.mkdir()
        self.compose = root / "docker-compose.infra.yml"
        self.compose.write_text(_COMPOSE_WITHOUT_REMOVED, encoding="utf-8")
        subprocess.run(
            ["git", "init", "--quiet", "--bare", "-b", "dev", str(self.origin)],
            check=True,
        )
        subprocess.run(["git", "init", "--quiet", "-b", "dev", str(seed)], check=True)
        _git(seed, "remote", "add", "origin", str(self.origin))
        self.base_sha = _commit(seed, "base.txt")
        _git(seed, "push", "--quiet", "origin", "dev")
        _git(seed, "checkout", "--quiet", "-b", _PINNED_BRANCH)
        self.pin_sha = _commit(seed, "fix.txt")
        _git(seed, "push", "--quiet", "origin", _PINNED_BRANCH)
        # dev moves on: the merge that removed the services lands after the pin's cut.
        _git(seed, "checkout", "--quiet", "dev")
        _commit(seed, "dev-merge-1.txt")
        self.dev_sha = _commit(seed, "dev-merge-2.txt")
        _git(seed, "push", "--quiet", "origin", "dev")
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "-b",
                _PINNED_BRANCH,
                str(self.origin),
                str(self.clone),
            ],
            check=True,
        )

    def write_pin(self) -> None:
        record = code_ref_pin.ModelDevLaneCodeRefPin(
            branch=_PINNED_BRANCH,
            head=self.pin_sha,
            pr=5772,
            ticket="OMN-23039",
            registered_by="lane-under-test",
            registered_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=48),
        )
        (self.state / _PIN_FILE).write_text(record.model_dump_json(), encoding="utf-8")

    def move_clone_to_dev(self) -> None:
        _git(self.clone, "checkout", "--quiet", "-B", "dev", "origin/dev")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _World:
    built = _World(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(built.clone))
    monkeypatch.setenv(code_ref_pin.ENV_STATE_DIR, str(built.state))
    return built


class _Calls:
    """Order of the docker-reaching steps a rebuild took."""

    def __init__(self) -> None:
        self.order: list[str] = []


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    seen = _Calls()
    monkeypatch.setattr(
        DeployExecutor,
        "_compose_build",
        lambda self, *a, **kw: seen.order.append("compose_build"),
    )
    monkeypatch.setattr(
        DeployExecutor,
        "_compose_up",
        lambda self, *a, **kw: seen.order.append("compose_up"),
    )
    monkeypatch.setattr(
        DeployExecutor,
        "_build_dev_lane_only_services",
        lambda self, *a, **kw: {},
    )
    monkeypatch.setattr(
        DeployExecutor, "_deploy_gateway_lane", lambda self, *a, **kw: None
    )
    return seen


def _rebuild(executor: DeployExecutor, services: list[str], world: _World) -> object:
    config = executor_mod.lane_config_for(EnumRuntimeLane.DEV).model_copy(
        update={
            "compose_files": (str(world.compose),),
            "disabled_services": frozenset(),
        }
    )
    with (
        patch.object(executor_mod, "lane_config_for", return_value=config),
        patch(
            "deploy_agent.executor._hold_build_context",
            side_effect=lambda purpose: nullcontext(),
        ),
    ):
        return executor.rebuild_scope(
            Scope.RUNTIME,
            services,
            lambda *a, **kw: None,
            git_sha="a" * 40,
            lane=EnumRuntimeLane.DEV,
        )


class TestAC1ServiceListAgainstCompose:
    def test_a_service_the_compose_lacks_is_refused_naming_pin_and_service_before_build(
        self, world: _World, calls: _Calls
    ) -> None:
        world.write_pin()

        with pytest.raises(ComposeServiceMissingError) as refused:
            _rebuild(DeployExecutor(), ["omninode-runtime", "intelligence-api"], world)

        message = str(refused.value)
        assert "intelligence-api" in message
        assert _PINNED_BRANCH in message
        assert calls.order == [], "docker was reached before the refusal"

    def test_control_services_the_compose_declares_reach_the_build(
        self, world: _World, calls: _Calls
    ) -> None:
        world.write_pin()

        _rebuild(DeployExecutor(), ["omninode-runtime", "runtime-effects"], world)

        assert calls.order[0] == "compose_build"

    def test_without_a_pin_the_refusal_names_the_tracking_ref(
        self, world: _World, calls: _Calls
    ) -> None:
        with pytest.raises(ComposeServiceMissingError) as refused:
            _rebuild(DeployExecutor(), ["infra-routing-decisions-consumer"], world)

        assert "infra-routing-decisions-consumer" in str(refused.value)
        assert "tracking ref dev" in str(refused.value)
        assert calls.order == []

    def test_every_missing_service_is_named(self) -> None:
        with pytest.raises(ComposeServiceMissingError) as refused:
            assert_services_declared(
                ["a", "b", "omninode-runtime"],
                declared=frozenset({"omninode-runtime"}),
                compose_files=("x.yml",),
                label="code-ref pin p",
            )
        assert refused.value.missing == ("a", "b")


class TestAC1AgainstTheRealDevCompose:
    """Positive control: the dev lane's own service lists are declared by its own compose."""

    @pytest.mark.parametrize(
        "lane", [EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST]
    )
    @pytest.mark.parametrize("scope", [Scope.CORE, Scope.RUNTIME, Scope.FULL])
    def test_the_repository_service_lists_are_declared_by_the_lane_compose(
        self, scope: Scope, lane: EnumRuntimeLane
    ) -> None:
        repo = Path(__file__).resolve().parents[4]
        files = tuple(
            str(repo / "docker" / Path(path).name)
            for path in executor_mod.lane_config_for(lane).compose_files
        )
        declared = frozenset(_read_compose_services(files))
        wanted = executor_mod._without_disabled_services(
            without_gateway_services(services_for_scope(scope, lane=lane)), lane
        )
        if lane is EnumRuntimeLane.DEV and scope is not Scope.CORE:
            wanted = [*wanted, *DEV_LANE_BUILD_SERVICES]
        assert wanted, "the scope names no service: the check below would be empty"
        assert set(wanted) <= declared, sorted(set(wanted) - declared)

    def test_the_real_reader_sees_a_service_the_fixture_lacks(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        files = (str(repo / "docker" / "docker-compose.infra.yml"),)
        assert "intelligence-api" not in frozenset(_read_compose_services(files))
        assert frozenset(_read_compose_services(files))


class TestAC2PinBehindDev:
    def test_a_pin_behind_dev_is_reported_stale_before_the_build(
        self, world: _World, calls: _Calls, caplog: pytest.LogCaptureFixture
    ) -> None:
        world.write_pin()
        executor = DeployExecutor()

        with caplog.at_level(logging.WARNING):
            _rebuild(executor, ["omninode-runtime"], world)

        stale = [r for r in caplog.records if "code_ref_pin_stale" in r.getMessage()]
        assert len(stale) == 1
        text = stale[0].getMessage()
        assert _PINNED_BRANCH in text
        assert "2 commit" in text and "origin/dev" in text
        assert executor.pin_staleness_report == text
        assert calls.order[0] == "compose_build"

    def test_the_report_precedes_the_build(
        self, world: _World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world.write_pin()
        order: list[str] = []
        real = guard.guard_before_build

        def _recording(*args: object, **kwargs: object) -> str:
            report = real(*args, **kwargs)
            order.append("report" if report else "silent")
            return report

        monkeypatch.setattr(guard, "guard_before_build", _recording)
        monkeypatch.setattr(
            DeployExecutor,
            "_compose_build",
            lambda self, *a, **kw: order.append("build"),
        )
        monkeypatch.setattr(DeployExecutor, "_compose_up", lambda self, *a, **kw: None)
        monkeypatch.setattr(
            DeployExecutor, "_build_dev_lane_only_services", lambda self, *a, **kw: {}
        )
        monkeypatch.setattr(
            DeployExecutor, "_deploy_gateway_lane", lambda self, *a, **kw: None
        )

        _rebuild(DeployExecutor(), ["omninode-runtime"], world)

        assert order == ["report", "build"]

    def test_control_a_pin_at_the_dev_head_reports_nothing(
        self, world: _World, calls: _Calls, caplog: pytest.LogCaptureFixture
    ) -> None:
        world.write_pin()
        world.move_clone_to_dev()
        executor = DeployExecutor()

        with caplog.at_level(logging.WARNING):
            _rebuild(executor, ["omninode-runtime"], world)

        assert not [r for r in caplog.records if "code_ref_pin_stale" in r.getMessage()]
        assert executor.pin_staleness_report == ""

    def test_the_staleness_is_named_in_the_missing_service_refusal(
        self, world: _World, calls: _Calls
    ) -> None:
        world.write_pin()

        with pytest.raises(ComposeServiceMissingError) as refused:
            _rebuild(DeployExecutor(), ["intelligence-api"], world)

        assert "2 commit" in str(refused.value)
