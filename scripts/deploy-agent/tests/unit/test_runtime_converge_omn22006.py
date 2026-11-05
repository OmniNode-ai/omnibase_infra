# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime phase recreates only what changed, one runtime at a time (OMN-22006).

Dev deploy job acd585d0 (2026-11-02) finished ``success`` while ``/ready`` on
8085 and 8086 read 000 or 503 for 3.5 minutes, because the runtime phase's
``up`` carried ``--force-recreate`` for all 19 application containers. The job
record (``tests/unit/fixtures/runtime_phase_job_acd585d0_omn22006.json``) shows
``verify_recreate: []`` and ``oneshot_reruns: []``: no other recreate site ran.

What is pinned here, over a fake compose that recreates a container exactly when
compose does -- its config hash or its image id differs from what is rendered
now, or ``--force-recreate`` is on the command line:

1. AC1 -- driven from that fixture's project state, the real ``_compose_up``
   issues no ``--force-recreate`` and recreates nothing.
2. AC2 -- a service whose config hash and image id are unchanged is not
   recreated; only the one that moved is.
3. AC3 -- services that must be recreated are recreated one at a time, each
   running again before the next is touched; a failure ends the roll.
4. The auto-rollback restore keeps its single forced ``up``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).parent / "fixtures" / "runtime_phase_job_acd585d0_omn22006.json"
)
OLD_IMAGE = "sha256:" + "a" * 64
NEW_IMAGE = "sha256:" + "b" * 64


def _noop(phase: Phase, status: PhaseStatus) -> None:
    return None


def _ok(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        [], returncode, stdout, "" if not returncode else "up failed"
    )


class _Lane:
    """A fake dev lane: the project state docker reads and compose acts on."""

    def __init__(self, services: list[str]) -> None:
        self.running = {s: {"hash": f"hash-{s}", "image": OLD_IMAGE} for s in services}
        self.rendered = {s: f"hash-{s}" for s in services}
        self.current_image = OLD_IMAGE
        self.ups: list[list[str]] = []
        self.recreated: list[str] = []
        self.starting: set[str] = set()
        self.max_starting = 0
        self.largest_recreate = 0
        self.fail_up_for = ""
        self.not_running = ""
        self.not_healthy = ""
        self.timed_out = ""
        self.events: list[str] = []
        self.phases: list[tuple[Phase, PhaseStatus]] = []

    def run(self, cmd: list[str], timeout: int = 0, **_: Any) -> Any:
        if "config" in cmd and "--hash" in cmd:
            names = cmd[cmd.index("--hash") + 1].split(",")
            wanted = list(self.rendered) if names == ["*"] else names
            return _ok(
                "".join(f"{n} {self.rendered.get(n, 'h-' + n)}\n" for n in wanted)
            )
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            return self._compose_up(cmd)
        return _ok()

    def _compose_up(self, cmd: list[str]) -> Any:
        self.ups.append(list(cmd))
        services = cmd[cmd.index("--no-deps") + 1 :]
        if self.timed_out in services:
            raise subprocess.TimeoutExpired(cmd, 900)
        if self.fail_up_for in services:
            return _ok(returncode=1)
        if "--wait" in cmd and self.not_healthy in services:
            return _ok(returncode=1)
        forced = "--force-recreate" in cmd
        replaced = [
            s
            for s in services
            if s in self.running
            and (
                forced
                or self.running[s]["hash"] != self.rendered[s]
                or self.running[s]["image"] != self.current_image
            )
        ]
        self.largest_recreate = max(self.largest_recreate, len(replaced))
        for service in replaced:
            self.starting.add(service)
            self.max_starting = max(self.max_starting, len(self.starting))
            self.recreated.append(service)
            self.events.append(f"recreate:{service}")
            self.running[service] = {
                "hash": self.rendered[service],
                "image": self.current_image,
            }
        return _ok()

    def verify(self, services: list[str], **_: Any) -> tuple[bool, list[str]]:
        if len(services) == 1:
            if services[0] == self.not_running:
                return False, services
            if services[0] in self.starting:
                self.starting.discard(services[0])
                self.events.append(f"running:{services[0]}")
        return True, []


def _run_runtime_phase(lane: _Lane, **kwargs: Any) -> None:
    budget = SimpleNamespace(timeout_seconds=600, describe=lambda: "test budget")
    executor = DeployExecutor()
    with (
        patch.object(executor_mod, "_run", lane.run),
        patch.object(executor_mod, "verify_containers_up", lane.verify),
        patch.object(executor_mod, "_compose_env", lambda *a, **k: {}),
        patch.object(executor_mod, "finish_recreate_renames", lambda lane: []),
        patch.object(executor_mod, "preflight_dev_images", lambda *a, **k: None),
        patch.object(executor_mod, "runtime_compose_up_budget", lambda *a: budget),
        patch.object(executor_mod, "runtime_health_probe_budget", lambda *a: 900),
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots", return_value=[]),
        patch.object(executor, "_retire_folded_dev_services"),
    ):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            [],
            lambda phase, status: lane.phases.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
            **kwargs,
        )


def _fixture_lane() -> _Lane:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    services = [c["service"] for c in fixture["containers"]]
    assert len(services) == 19
    return _Lane(services)


class TestRuntimePhaseRecreateOfJobAcd585d0:
    def test_runtime_phase_recreate_forces_nothing_when_nothing_changed(self) -> None:
        lane = _fixture_lane()
        _run_runtime_phase(lane)
        assert lane.ups, "the runtime phase issued no compose up"
        forced = [a for a in lane.ups if "--force-recreate" in a]
        assert not forced, (
            "job acd585d0's runtime up carried --force-recreate and took all 19 "
            f"application containers down at once: {forced}"
        )
        assert lane.recreated == []

    def test_runtime_phase_recreate_keeps_every_up_without_deps_and_unpulled(
        self,
    ) -> None:
        lane = _fixture_lane()
        _run_runtime_phase(lane)
        for argv in lane.ups:
            assert "--no-deps" in argv
            assert argv[argv.index("--pull") + 1] == "never"


class TestRuntimePhaseUnchangedNotRecreated:
    def test_runtime_phase_unchanged_not_recreated_when_only_one_hash_moved(
        self,
    ) -> None:
        lane = _fixture_lane()
        lane.rendered["projection-api"] = "hash-projection-api-new"
        _run_runtime_phase(lane)
        assert lane.recreated == ["projection-api"]
        assert all("--force-recreate" not in a for a in lane.ups)

    def test_runtime_phase_unchanged_not_recreated_when_hash_and_image_match(
        self,
    ) -> None:
        lane = _fixture_lane()
        _run_runtime_phase(lane)
        assert lane.recreated == []

    def test_runtime_phase_unchanged_not_recreated_yet_a_moved_image_is(
        self,
    ) -> None:
        """The same tag rebuilt: the config hash is equal, the image id is not."""
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        for service, state in lane.running.items():
            if service != "onex-api":
                state["image"] = NEW_IMAGE
        _run_runtime_phase(lane)
        assert lane.recreated == ["onex-api"]


class TestRuntimePhaseRolling:
    def test_runtime_phase_rolling_recreates_changed_runtimes_one_at_a_time(
        self,
    ) -> None:
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        _run_runtime_phase(lane)
        # 19 containers in the recorded job; the hook-events writer is among them
        # and OMN-21288 folded it into main, so the runtime scope no longer names it.
        assert len(lane.recreated) == 18
        assert "projection-claude-hook-events-writer" not in lane.recreated
        assert lane.max_starting == 1, "two runtimes were down at once"
        assert lane.largest_recreate == 1
        pairs = list(zip(lane.events[0::2], lane.events[1::2], strict=True))
        for recreated, running in pairs:
            assert recreated.startswith("recreate:")
            assert running == recreated.replace("recreate:", "running:")

    def test_runtime_phase_rolling_names_exactly_one_service_per_up(self) -> None:
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        _run_runtime_phase(lane)
        assert len(lane.ups) == len({a[-1] for a in lane.ups})
        for argv in lane.ups:
            assert argv[argv.index("--no-deps") + 1 :] == [argv[-1]]

    def test_runtime_phase_rolling_stops_at_the_first_up_that_fails(self) -> None:
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        lane.fail_up_for = "runtime-worker"
        with pytest.raises(RuntimeError):
            _run_runtime_phase(lane)
        issued = [a[-1] for a in lane.ups]
        assert issued[-1] == "runtime-worker"
        assert lane.recreated == issued[:-1]
        assert len(issued) < 19

    @pytest.mark.parametrize("failed_service", ["omninode-runtime", "runtime-effects"])
    def test_runtime_phase_rolling_refuses_a_service_that_is_not_running(
        self,
        failed_service: str,
    ) -> None:
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        lane.not_running = failed_service
        with pytest.raises(RuntimeError, match=failed_service):
            _run_runtime_phase(lane)
        issued = [a[-1] for a in lane.ups]
        assert issued[-1] == failed_service
        assert len(issued) < 19
        assert (Phase.RUNTIME, PhaseStatus.SUCCESS) not in lane.phases

    def test_runtime_phase_rolling_does_not_replace_a_peer_while_boot_is_unhealthy(
        self,
    ) -> None:
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        lane.not_healthy = "omninode-runtime"
        with pytest.raises(RuntimeError, match="omninode-runtime"):
            _run_runtime_phase(lane)
        assert [a[-1] for a in lane.ups] == ["omninode-runtime"]
        assert lane.running["runtime-effects"]["image"] == OLD_IMAGE
        assert (Phase.RUNTIME, PhaseStatus.SUCCESS) not in lane.phases

    def test_runtime_phase_rolling_timeout_does_not_report_a_partial_roll_successful(
        self,
    ) -> None:
        lane = _fixture_lane()
        lane.current_image = NEW_IMAGE
        lane.timed_out = "runtime-effects"
        with pytest.raises(RuntimeError, match="runtime-effects: compose up timed out"):
            _run_runtime_phase(lane)
        assert [a[-1] for a in lane.ups] == ["omninode-runtime", "runtime-effects"]
        assert lane.running["runtime-worker"]["image"] == OLD_IMAGE
        assert (Phase.RUNTIME, PhaseStatus.SUCCESS) not in lane.phases

    def test_runtime_phase_rolling_restore_path_still_forces_one_up(self) -> None:
        """The auto-rollback restore passes ``build=False`` and keeps its force."""
        lane = _fixture_lane()
        _run_runtime_phase(lane, force_recreate=True, build=False)
        (batch,) = lane.ups
        assert "--force-recreate" in batch
        assert len(batch[batch.index("--no-deps") + 1 :]) > 1
