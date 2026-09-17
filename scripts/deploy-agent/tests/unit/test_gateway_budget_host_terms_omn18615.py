# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The GATEWAY deploy ceiling reads the machine too (OMN-18615, second pass).

WHAT THIS FIXES, AND HOW IT WAS FOUND
-------------------------------------

OMN-18615's first pass taught ``derive_image_build_budget`` to read the machine
and wired it at ``executor.runtime_image_build_budget``. It MISSED the second
caller. ``gateway_budget.derive_gateway_deploy_budget`` reuses the same build
derivation over the gateway compose file and did not pass ``host``, so the
gateway deploy's build half stayed blind to the machine -- the exact defect the
ticket exists to remove, surviving on a call site nobody looked at.

MEASURED on the ``.201`` dev lane, 2026-09-17. Job ``a0b496ed`` FAILED at
22:33:27Z, on a host whose load1 had peaked at 73.73 minutes earlier:

    GATEWAY_DEPLOY_TIMED_OUT: bash .../deploy-gateway.sh --execute exceeded its
    1180s ceiling and was killed. Ceiling derivation: 1180s = build 940s (940s =
    model 940s (60 work steps of '.../Dockerfile.runtime' x 15s/step (one shared
    BuildKit solve) + 2 buildable service(s) in profile 'gateway' x 20s/image
    (export)), NO HOST CONDITIONS WERE READ (model terms only -- this is the
    OMN-18072 derivation, blind to the machine), floor 600s, hard upper bound
    3600s) + recreate 240s (...)

The capitalised clause is the one the first pass added precisely so an
unadjusted derivation could not pass silently. It reported this defect in its
own words, on a live failure, two hours after it shipped. That is the clause
doing its job, and this file is the repair.

THE GUARD THAT MATTERS MOST HERE is not either arithmetic test below. It is
``TestNoThirdCallSiteCanBeBlind``, which parses the package and fails if ANY
call to ``derive_image_build_budget`` omits ``host``. Two call sites existed and
one was missed; a third would be missed the same way, and a reviewer cannot be
the mechanism.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from deploy_agent.gateway_budget import derive_gateway_deploy_budget
from deploy_agent.host_conditions import (
    EnumBuildCacheState,
    ModelHostConditions,
    probe_host_conditions,
)

# The derivation the 2026-09-17T22:33:27Z kill logged, term by term.
RECORDED_BUILD_STEPS = 60
RECORDED_GATEWAY_SERVICES = 2
RECORDED_PER_STEP_SECONDS = 15
RECORDED_PER_IMAGE_SECONDS = 20
RECORDED_BUILD_FLOOR_SECONDS = 600
RECORDED_BUILD_SECONDS = 940  # 60 x 15 + 2 x 20
RECORDED_RECREATE_SECONDS = 240
RECORDED_CEILING_SECONDS = 1180  # 940 + 240

# The `.201` dev lane host, and the load1 it had peaked at before the kill.
LAB_HOST_CPU_COUNT = 32
RECORDED_PEAK_LOAD1 = 73.73


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _recorded_model(tmp_path: Path) -> tuple[tuple[str, ...], str]:
    """The gateway build model that derives to exactly the recorded 940s."""
    _write(
        tmp_path,
        "docker/Dockerfile.runtime",
        "FROM scratch\n"
        + "".join(f"RUN echo {i}\n" for i in range(RECORDED_BUILD_STEPS)),
    )
    body = ["services:"]
    for index in range(RECORDED_GATEWAY_SERVICES):
        body.append(f"  gw{index}:")
        body.append("    profiles: [gateway]")
        body.append("    build:")
        body.append("      context: ..")
        body.append("      dockerfile: docker/Dockerfile.runtime")
    compose = _write(
        tmp_path, "docker/docker-compose.gateway.yml", "\n".join(body) + "\n"
    )
    unit = _write(
        tmp_path,
        "docker/gateway/onex-gateway-forwarder.service",
        "[Service]\n"
        "ExecReload=/usr/bin/docker compose up -d --force-recreate --wait "
        "--wait-timeout 120 gateway-forwarder\n",
    )
    return (str(compose),), str(unit)


def _derive(tmp_path: Path, host: ModelHostConditions | None) -> object:
    compose_files, unit = _recorded_model(tmp_path)
    return derive_gateway_deploy_budget(
        compose_files,
        "gateway",
        service_unit_path=unit,
        per_step_seconds=RECORDED_PER_STEP_SECONDS,
        per_image_seconds=RECORDED_PER_IMAGE_SECONDS,
        build_floor_seconds=RECORDED_BUILD_FLOOR_SECONDS,
        reload_margin_seconds=120,
        reload_floor_seconds=180,
        host=host,
    )


def _idle_warm() -> ModelHostConditions:
    return probe_host_conditions(
        loadavg_reader=lambda: (1.2, 1.1, 1.0),
        cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
        builder_cache_reader=lambda: EnumBuildCacheState.WARM,
    )


def _contended_cold(load1: float) -> ModelHostConditions:
    return probe_host_conditions(
        loadavg_reader=lambda: (load1, load1, load1),
        cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
        builder_cache_reader=lambda: EnumBuildCacheState.COLD,
    )


@pytest.mark.unit
class TestTheRecordedGatewayKillIsAFixture:
    def test_the_host_blind_derivation_reproduces_the_kill(
        self, tmp_path: Path
    ) -> None:
        """Pins the number that killed a live gateway deploy at 22:33:27Z."""
        budget = _derive(tmp_path, None)
        assert budget.timeout_seconds == RECORDED_CEILING_SECONDS
        assert budget.build.timeout_seconds == RECORDED_BUILD_SECONDS

    def test_the_host_blind_derivation_says_it_is_blind(self, tmp_path: Path) -> None:
        """The clause that reported this defect on the live failure."""
        assert "no host conditions" in _derive(tmp_path, None).describe()


@pytest.mark.unit
class TestTheGatewayCeilingMovesWithTheMachine:
    def test_idle_warm_and_contended_cold_differ(self, tmp_path: Path) -> None:
        idle = _derive(tmp_path, _idle_warm())
        contended = _derive(tmp_path, _contended_cold(RECORDED_PEAK_LOAD1))
        assert contended.timeout_seconds != idle.timeout_seconds
        assert contended.timeout_seconds > idle.timeout_seconds

    def test_the_recorded_kill_would_have_been_granted_more_room(
        self, tmp_path: Path
    ) -> None:
        """A fix that still kills the deploy it was written for is not a fix."""
        contended = _derive(tmp_path, _contended_cold(RECORDED_PEAK_LOAD1))
        assert contended.timeout_seconds > RECORDED_CEILING_SECONDS

    def test_only_the_build_half_is_widened(self, tmp_path: Path) -> None:
        """The recreate half is the unit's OWN --wait-timeout and is not ours.

        ``systemctl reload`` enforces that number itself, so inflating it here
        would describe a bound nothing honours.
        """
        idle = _derive(tmp_path, _idle_warm())
        contended = _derive(tmp_path, _contended_cold(RECORDED_PEAK_LOAD1))
        assert contended.recreate_seconds == idle.recreate_seconds
        assert contended.build.timeout_seconds > idle.build.timeout_seconds

    def test_the_gateway_ceiling_reports_its_host_terms(self, tmp_path: Path) -> None:
        text = _derive(tmp_path, _contended_cold(RECORDED_PEAK_LOAD1)).describe()
        assert "load1" in text
        assert "cache" in text

    def test_the_gateway_ceiling_is_still_bounded(self, tmp_path: Path) -> None:
        """AC5's bound applies here too: adapting is not unbounded."""
        from deploy_agent.build_budget import HARD_UPPER_BOUND_SECONDS

        absurd = probe_host_conditions(
            loadavg_reader=lambda: (1_000_000.0, 1.0, 1.0),
            cpu_count_reader=lambda: 1,
            builder_cache_reader=lambda: EnumBuildCacheState.COLD,
        )
        budget = _derive(tmp_path, absurd)
        assert budget.build.timeout_seconds <= HARD_UPPER_BOUND_SECONDS


@pytest.mark.unit
class TestTheLiveGatewayPathReadsTheMachine:
    def test_executor_gateway_budget_carries_host_conditions(self) -> None:
        """The production entry point must probe, never default to blind."""
        from deploy_agent import executor as executor_mod

        budget = executor_mod.gateway_deploy_budget()
        assert budget.build.host is not None, (
            "executor.gateway_deploy_budget() derived a host-blind ceiling; "
            "this is the exact defect that killed job a0b496ed at 22:33:27Z"
        )

    def test_the_live_gateway_description_is_not_blind(self) -> None:
        from deploy_agent import executor as executor_mod

        assert (
            "no host conditions" not in executor_mod.gateway_deploy_budget().describe()
        )


@pytest.mark.unit
class TestNoThirdCallSiteCanBeBlind:
    """The guard that actually prevents recurrence.

    Two call sites existed, one was wired and one was missed, and the miss was
    found by a production failure rather than by review. A third call site
    would be missed the same way. This parses the package and fails on any call
    to ``derive_image_build_budget`` that omits ``host``.
    """

    def _package_root(self) -> Path:
        import deploy_agent

        return Path(deploy_agent.__file__).parent

    def test_every_call_passes_host(self) -> None:
        offenders: list[str] = []
        for source_path in sorted(self._package_root().rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else ""
                )
                if name not in {
                    "derive_image_build_budget",
                    "derive_gateway_deploy_budget",
                }:
                    continue
                if not any(kw.arg == "host" for kw in node.keywords):
                    offenders.append(f"{source_path.name}:{node.lineno}")
        assert not offenders, (
            "these budget derivations do not pass host, so they "
            f"derive a ceiling blind to the machine (OMN-18615): {offenders}"
        )

    def test_the_guard_can_actually_fail(self, tmp_path: Path) -> None:
        """A positive control: an empty result must mean 'none', not 'not run'."""
        blind = tmp_path / "blind.py"
        blind.write_text(
            "derive_image_build_budget(files, profile, floor_seconds=1)\n",
            encoding="utf-8",
        )
        found = [
            node
            for node in ast.walk(ast.parse(blind.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "derive_image_build_budget"
            and not any(kw.arg == "host" for kw in node.keywords)
        ]
        assert len(found) == 1
