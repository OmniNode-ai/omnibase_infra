# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The idle CPU, saturated disk on dev-202 must widen the build ceiling."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from deploy_agent import host_conditions
from deploy_agent.build_budget import (
    HARD_UPPER_BOUND_SECONDS,
    ModelBuildBudget,
    derive_image_build_budget,
)
from deploy_agent.host_conditions import (
    EnumBuildCacheState,
    ModelHostConditions,
    probe_host_conditions,
)

pytestmark = pytest.mark.unit

RECORDED_BUILD_STEPS = 61
RECORDED_BUILDABLE_SERVICES = 9
RECORDED_CEILING_SECONDS = 1095
RECORDED_IO_PRESSURE_PERCENT = 50.99


def _host(reader: Callable[[], float | None]) -> ModelHostConditions:
    return probe_host_conditions(
        loadavg_reader=lambda: (14.0, 14.0, 14.0),
        cpu_count_reader=lambda: 32,
        builder_cache_reader=lambda: EnumBuildCacheState.WARM,
        io_pressure_reader=reader,
    )


def _unreadable() -> float | None:
    raise PermissionError("IO PSI denied")


def _derive(tmp_path: Path, host: ModelHostConditions) -> ModelBuildBudget:
    """Use the host-terms fixture's model, with dev-202's 61 work steps."""
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    (docker_dir / "Dockerfile.runtime").write_text(
        "FROM scratch\n"
        + "".join(f"RUN echo {i}\n" for i in range(RECORDED_BUILD_STEPS)),
        encoding="utf-8",
    )
    compose = docker_dir / "docker-compose.yml"
    compose.write_text(
        "services:\n"
        + "".join(
            f"  svc{i}:\n"
            "    profiles: [runtime, full]\n"
            "    build:\n"
            "      context: ..\n"
            "      dockerfile: docker/Dockerfile.runtime\n"
            for i in range(RECORDED_BUILDABLE_SERVICES)
        ),
        encoding="utf-8",
    )
    return derive_image_build_budget(
        (str(compose),),
        "runtime",
        per_step_seconds=15,
        per_image_seconds=20,
        floor_seconds=600,
        host=host,
    )


def test_recorded_io_stalls_widen_the_idle_warm_ceiling(tmp_path: Path) -> None:
    host = _host(lambda: RECORDED_IO_PRESSURE_PERCENT)
    budget = _derive(tmp_path, host)
    assert host.saturation == pytest.approx(14 / 32)
    assert host.contention_multiplier == host.cache_multiplier == 1.0
    assert host.io_pressure_state is host_conditions.EnumIoPressureState.READ
    assert host.io_pressure_some_percent == RECORDED_IO_PRESSURE_PERCENT
    assert host.io_pressure_multiplier == pytest.approx(1.8198)
    assert host.multiplier == pytest.approx(1.8198)
    assert budget.model_seconds == RECORDED_CEILING_SECONDS
    assert budget.timeout_seconds > RECORDED_CEILING_SECONDS
    assert budget.timeout_seconds >= 1.5 * RECORDED_CEILING_SECONDS
    assert budget.timeout_seconds == round(RECORDED_CEILING_SECONDS * 1.8198)
    assert budget.timeout_seconds <= HARD_UPPER_BOUND_SECONDS


def test_io_not_exposed_preserves_the_idle_warm_ceiling(tmp_path: Path) -> None:
    host = _host(lambda: None)
    assert host.io_pressure_some_percent is None
    assert host.io_pressure_state is host_conditions.EnumIoPressureState.NOT_EXPOSED
    assert host.io_pressure_multiplier == 1.0
    assert _derive(tmp_path, host).timeout_seconds == RECORDED_CEILING_SECONDS


def test_unreadable_io_is_unknown_and_widens(tmp_path: Path) -> None:
    host = _host(_unreadable)
    assert host.io_pressure_some_percent is None
    assert host.io_pressure_state is host_conditions.EnumIoPressureState.UNKNOWN
    assert host.io_pressure_multiplier == 1.5
    assert host.multiplier == 1.5
    assert _derive(tmp_path, host).timeout_seconds == round(
        RECORDED_CEILING_SECONDS * 1.5
    )


def test_full_io_pressure_is_capped() -> None:
    host = _host(lambda: 100.0)
    assert host.io_pressure_multiplier == host_conditions.MAX_IO_PRESSURE_MULTIPLIER
    assert host.io_pressure_multiplier == 2.0


@pytest.mark.parametrize("percent", [0.0, 9.99, 10.0])
def test_io_at_or_below_threshold_is_unchanged(percent: float) -> None:
    assert _host(lambda: percent).io_pressure_multiplier == 1.0


def test_all_host_terms_still_obey_the_hard_bound(tmp_path: Path) -> None:
    host = probe_host_conditions(
        loadavg_reader=lambda: (1_000_000.0, 1.0, 1.0),
        cpu_count_reader=lambda: 1,
        builder_cache_reader=lambda: EnumBuildCacheState.COLD,
        io_pressure_reader=lambda: 100.0,
    )
    assert host.multiplier == 12.0
    assert _derive(tmp_path, host).timeout_seconds == HARD_UPPER_BOUND_SECONDS


@pytest.mark.parametrize(("avg60", "avg300"), [(40.95, 50.99), (50.99, 40.95)])
def test_default_reader_uses_the_larger_sustained_some_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, avg60: float, avg300: float
) -> None:
    path = tmp_path / "io"
    path.write_text(
        f"some avg10=99.00 avg60={avg60} avg300={avg300} total=12345\n"
        "full avg10=45.96 avg60=39.13 avg300=48.83 total=6789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(host_conditions, "IO_PRESSURE_PATH", path)
    assert host_conditions._default_io_pressure_reader() == 50.99


def test_default_reader_missing_path_is_not_exposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_conditions, "IO_PRESSURE_PATH", tmp_path / "missing")
    assert host_conditions._default_io_pressure_reader() is None


@pytest.mark.parametrize(
    "body",
    [
        "garbage\n",
        "full avg60=40.95 avg300=50.99\n",
        "some avg60=oops avg300=50.99\n",
        "some avg60=40.95\n",
    ],
)
def test_default_reader_garbage_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    path = tmp_path / "io"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(host_conditions, "IO_PRESSURE_PATH", path)
    with pytest.raises(ValueError):
        host_conditions._default_io_pressure_reader()
    assert _host(host_conditions._default_io_pressure_reader).io_pressure_state is (
        host_conditions.EnumIoPressureState.UNKNOWN
    )


@pytest.mark.parametrize(
    ("reader", "term"),
    [
        (lambda: 50.99, "io psi some 50.99% (x1.82)"),
        (lambda: None, "io psi not exposed (x1.00)"),
        (_unreadable, "io psi unreadable (x1.50)"),
    ],
)
def test_describe_names_the_io_reading(
    reader: Callable[[], float | None], term: str
) -> None:
    assert term in _host(reader).describe()
