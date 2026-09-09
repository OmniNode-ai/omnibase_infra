# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime compose-up ceiling is derived from the compose model (OMN-18057).

``PHASE_TIMEOUTS[Phase.RUNTIME] = 300`` bounded a command that cannot finish
until ``omninode-runtime`` reports healthy, and the compose file gives that
service ``start_period: 1800s`` -- a ceiling six times smaller than the contract
it was bounding. These tests pin the derivation, not a number: the ceiling moves
when the compose file moves, and a zero (no gating service) is proved with a
positive control rather than assumed.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.compose_budget import (
    ComposeBudgetError,
    ComposeDurationError,
    derive_runtime_phase_budget,
    parse_compose_duration,
)
from deploy_agent.events import SCOPE_SERVICES, EnumRuntimeLane, Phase, Scope
from deploy_agent.executor import (
    RUNTIME_COMPOSE_UP_FLOOR_SECONDS,
    RUNTIME_COMPOSE_UP_MARGIN_SECONDS,
    DeployExecutor,
)

pytestmark = pytest.mark.unit

_DOCKER_DIR = Path(__file__).resolve().parents[4] / "docker"
_LIVE_COMPOSE_FILES = (
    str(_DOCKER_DIR / "docker-compose.infra.yml"),
    str(_DOCKER_DIR / "docker-compose.dev-lane.yml"),
)


def _derive(compose_files: tuple[str, ...], services: list[str]) -> Any:
    return derive_runtime_phase_budget(
        compose_files,
        services,
        margin_seconds=RUNTIME_COMPOSE_UP_MARGIN_SECONDS,
        floor_seconds=RUNTIME_COMPOSE_UP_FLOOR_SECONDS,
    )


class TestComposeDuration:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1800s", 1800),
            ("30m", 1800),
            ("1h30m", 5400),
            ("1m30s", 90),
            ("500ms", 1),  # rounds UP -- a ceiling never lands below its source
        ],
    )
    def test_units(self, text: str, expected: int) -> None:
        assert parse_compose_duration(text) == expected

    @pytest.mark.parametrize("bad", ["1800", "", "later", 1800, None, True])
    def test_unsuffixed_or_non_string_is_refused(self, bad: object) -> None:
        """Docker reads an unsuffixed healthcheck duration as NANOSECONDS.

        Guessing seconds there would turn 1800 into a ceiling three orders of
        magnitude off, so the parser refuses rather than picking a unit.
        """
        with pytest.raises(ComposeDurationError):
            parse_compose_duration(bad)


class TestDerivationAgainstTheLiveComposeModel:
    def test_ceiling_comes_from_the_gating_runtime_start_period(self) -> None:
        budget = _derive(_LIVE_COMPOSE_FILES, SCOPE_SERVICES[Scope.RUNTIME])
        assert budget.source_service == "omninode-runtime"
        assert budget.source_start_period_seconds == 1800
        assert budget.timeout_seconds == 1800 + RUNTIME_COMPOSE_UP_MARGIN_SECONDS

    def test_ceiling_exceeds_the_measured_minimum_viable_budget(self) -> None:
        """MEASURED 2026-09-08 (ledger :5076): 336s was the minimum that works.

        The old 300s ceiling sat below it. Any derived ceiling must clear it.
        """
        budget = _derive(_LIVE_COMPOSE_FILES, SCOPE_SERVICES[Scope.RUNTIME])
        assert budget.timeout_seconds > 336

    def test_ceiling_covers_the_unhealthy_detection_tail(self) -> None:
        """compose stops waiting on healthy OR unhealthy; unhealthy is later.

        The compose file states the tail itself: start_period 1800s + interval
        30s * retries 5 = 1950s. A ceiling below that kills the command while
        compose is still legitimately waiting.
        """
        budget = _derive(_LIVE_COMPOSE_FILES, SCOPE_SERVICES[Scope.RUNTIME])
        assert budget.timeout_seconds >= 1800 + 30 * 5


class TestGatingRule:
    """Positive/negative control pair on the ``service_healthy`` condition."""

    @staticmethod
    def _write(tmp_path: Path, body: str) -> tuple[str, ...]:
        path = tmp_path / "compose.yml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return (str(path),)

    def test_service_healthy_dependency_selects_the_service(
        self, tmp_path: Path
    ) -> None:
        files = self._write(
            tmp_path,
            """
            services:
              omninode-runtime:
                healthcheck:
                  start_period: 900s
              runtime-effects:
                depends_on:
                  omninode-runtime:
                    condition: service_healthy
            """,
        )
        budget = _derive(files, ["omninode-runtime", "runtime-effects"])
        assert budget.source_service == "omninode-runtime"
        assert budget.timeout_seconds == 900 + RUNTIME_COMPOSE_UP_MARGIN_SECONDS

    def test_service_started_dependency_does_not(self, tmp_path: Path) -> None:
        """Same file, one word changed: compose does not wait on health here.

        Without this control, the test above would pass on a derivation that
        merely took the largest start_period and never read the condition.
        """
        files = self._write(
            tmp_path,
            """
            services:
              omninode-runtime:
                healthcheck:
                  start_period: 900s
              runtime-effects:
                depends_on:
                  omninode-runtime:
                    condition: service_started
            """,
        )
        budget = _derive(files, ["omninode-runtime", "runtime-effects"])
        assert budget.source_service is None
        assert budget.timeout_seconds == RUNTIME_COMPOSE_UP_FLOOR_SECONDS

    def test_out_of_scope_service_cannot_inflate_the_ceiling(
        self, tmp_path: Path
    ) -> None:
        files = self._write(
            tmp_path,
            """
            services:
              unrelated:
                healthcheck:
                  start_period: 9000s
              something-else:
                depends_on:
                  unrelated:
                    condition: service_healthy
            """,
        )
        budget = _derive(files, ["omninode-runtime"])
        assert budget.source_service is None
        assert budget.timeout_seconds == RUNTIME_COMPOSE_UP_FLOOR_SECONDS

    def test_compose_merge_tags_are_understood(self, tmp_path: Path) -> None:
        """``!override`` appears in the live dev-lane overlay.

        ``yaml.safe_load`` refuses an unknown tag, which would make the model
        unreadable on the one lane this ceiling matters most for.
        """
        files = self._write(
            tmp_path,
            """
            services:
              omninode-runtime:
                labels: !override
                  - a=b
                healthcheck:
                  start_period: 120s
              runtime-effects:
                depends_on:
                  omninode-runtime:
                    condition: service_healthy
            """,
        )
        budget = _derive(files, ["omninode-runtime"])
        assert budget.source_start_period_seconds == 120


class TestFailsClosed:
    def test_unreadable_compose_file_refuses_rather_than_falling_to_the_floor(
        self, tmp_path: Path
    ) -> None:
        """A silent revert to the floor is the bare constant coming back.

        The deploy is doomed anyway when its own compose file is unreadable, so
        the refusal costs nothing and names the file.
        """
        missing = tmp_path / "absent.yml"
        with pytest.raises(ComposeBudgetError) as excinfo:
            _derive((str(missing),), ["omninode-runtime"])
        assert str(missing) in str(excinfo.value)

    def test_unparseable_compose_file_refuses(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.yml"
        broken.write_text("services: [unclosed\n", encoding="utf-8")
        with pytest.raises(ComposeBudgetError):
            _derive((str(broken),), ["omninode-runtime"])


class TestComposeUpUsesTheDerivedCeiling:
    def test_runtime_compose_up_is_given_the_derived_ceiling_not_300(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression guard: the old value was the bare ``300``."""
        from deploy_agent import executor as executor_mod

        seen: list[int] = []

        def _fake_run(cmd: list[str], timeout: int, **kwargs: Any) -> Any:
            if "up" in cmd and "--force-recreate" in cmd:
                seen.append(timeout)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        def _all_running(
            lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        ) -> dict[str, tuple[str, int | None]]:
            return dict.fromkeys(SCOPE_SERVICES[Scope.RUNTIME], ("running", None))

        monkeypatch.setattr(executor_mod, "_run", _fake_run)
        monkeypatch.setattr(executor_mod, "_compose_service_states", _all_running)
        monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})
        monkeypatch.setattr(
            executor_mod.DeployExecutor,
            "_ensure_runtime_migrations_ready",
            lambda self, **kwargs: None,
        )

        DeployExecutor()._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            [],
            lambda phase, status: None,
            lane=EnumRuntimeLane.DEV,
        )

        expected = _derive(
            _LIVE_COMPOSE_FILES, SCOPE_SERVICES[Scope.RUNTIME]
        ).timeout_seconds
        assert seen == [expected]
        assert seen != [300]

    def test_core_compose_up_keeps_its_flat_phase_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Core services are not gated on a runtime healthcheck; nothing changed there."""
        from deploy_agent import executor as executor_mod

        seen: list[int] = []

        def _fake_run(cmd: list[str], timeout: int, **kwargs: Any) -> Any:
            if "up" in cmd and "--force-recreate" in cmd:
                seen.append(timeout)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps({}), stderr=""
            )

        def _all_running(
            lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        ) -> dict[str, tuple[str, int | None]]:
            return dict.fromkeys(SCOPE_SERVICES[Scope.CORE], ("running", None))

        monkeypatch.setattr(executor_mod, "_run", _fake_run)
        monkeypatch.setattr(executor_mod, "_compose_service_states", _all_running)
        monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})

        DeployExecutor()._compose_up(
            Phase.CORE,
            Scope.CORE,
            [],
            lambda phase, status: None,
            lane=EnumRuntimeLane.DEV,
        )

        assert seen == [executor_mod.PHASE_TIMEOUTS[Phase.CORE]]
