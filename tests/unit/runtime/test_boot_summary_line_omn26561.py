# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev boot logs one ``boot summary:`` line at the first ``/health`` 200 (OMN-26561).

Three boot-performance merges (each contract parsed once, the topic scan and route
discovery off the loop, boot step 0 holds removed) could not be read on a running boot
because it reported nothing about them. These tests drive the paths a boot takes: boot
step 0 through ``run_boot_catalog``, then the health handler through ``ServiceHealth``,
and read the line from the log the way ``docker logs | grep`` reads it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from omnibase_infra.contracts.contract_read_cache import ContractReadCache
from omnibase_infra.handlers import handler_package_announcer as announcer
from omnibase_infra.handlers.handler_prebuilt_manifest import run_boot_catalog
from omnibase_infra.models.event_bus.model_contract_topic_scan import (
    ModelContractTopicScan,
)
from omnibase_infra.services.health_checker import ServiceHealth

_EP = "omnibase_infra.handlers.handler_package_announcer.entry_points"
_EP_STAMP = "omnibase_infra.handlers.handler_prebuilt_manifest.entry_points"
_LINE = re.compile(
    r"boot summary: parse_runs=(?P<parse_runs>\d+) topic_scans=(?P<topic_scans>\d+) "
    r"seconds_since_kernel_start=(?P<since>[\d.]+) "
    r"gap_boot_step0=(?P<boot_step0>[\d.]+|none) "
    r"gap_provisioner=(?P<provisioner>[\d.]+|none) "
    r"gap_wiring_pass=(?P<wiring_pass>[\d.]+|none) "
    r"gap_local_ingress_start=(?P<local_ingress_start>[\d.]+|none) "
    r"gap_other=(?P<other>[\d.]+|none)"
)


@pytest.fixture(autouse=True)
def _fresh_clock() -> Iterator[None]:
    """Reset the boot clock when the module exists; a base without it has nothing to reset."""
    try:
        from omnibase_infra.models.node_announce.model_boot_clock import ModelBootClock
    except ImportError:
        yield
        return
    ModelBootClock.reset()
    yield
    ModelBootClock.reset()


def _runtime(*, healthy: bool = True) -> MagicMock:
    runtime = MagicMock()
    runtime.health_check = AsyncMock(
        return_value={
            "healthy": healthy,
            "degraded": False,
            "is_running": healthy,
            "runtime_attached": True,
            "event_bus_healthy": healthy,
        }
    )
    return runtime


async def _health(server: ServiceHealth) -> int:
    response = await server._handle_health(MagicMock(spec=web.Request))
    assert response.text is not None
    json.loads(response.text)
    return response.status


def _summary_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "boot summary:" in r.getMessage()]


async def _boot_step0(tmp_path: Path, *, hold_seconds: float) -> None:
    """Boot step 0 with a blocking hold of ``hold_seconds`` injected in the announcer."""

    async def held_announce(*, workload: str) -> object:
        if hold_seconds:
            time.sleep(hold_seconds)  # the injected hold blocks the loop on purpose
        return await announcer.run_boot_step0(workload=workload)

    with patch(_EP, return_value=[]), patch(_EP_STAMP, return_value=[]):
        await run_boot_catalog(
            workload="main", announce=held_announce, path=tmp_path / "absent.json"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_health_200_logs_one_boot_summary_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    await _boot_step0(tmp_path, hold_seconds=0.0)

    assert await _health(ServiceHealth(runtime=_runtime(), version="1.0.0")) == 200

    lines = _summary_lines(caplog)
    assert len(lines) == 1, f"expected one 'boot summary:' line, got {lines!r}"
    match = _LINE.search(lines[0])
    assert match is not None, lines[0]
    assert int(match["parse_runs"]) == ContractReadCache.parse_runs()
    assert int(match["topic_scans"]) == ModelContractTopicScan.runs()
    assert float(match["since"]) >= 0.0
    assert match["boot_step0"] != "none"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_two_second_hold_in_step0_reads_as_a_gap_over_one_second(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    await _boot_step0(tmp_path, hold_seconds=2.0)

    await _health(ServiceHealth(runtime=_runtime(), version="1.0.0"))

    lines = _summary_lines(caplog)
    assert len(lines) == 1, f"expected one 'boot summary:' line, got {lines!r}"
    match = _LINE.search(lines[0])
    assert match is not None, lines[0]
    assert float(match["boot_step0"]) >= 1.9
    assert match["provisioner"] == "none"
    assert match["wiring_pass"] == "none"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_line_is_written_once_and_only_for_a_200(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    await _boot_step0(tmp_path, hold_seconds=0.0)

    assert (
        await _health(ServiceHealth(runtime=_runtime(healthy=False), version="1.0.0"))
        == 503
    )
    assert _summary_lines(caplog) == []

    server = ServiceHealth(runtime=_runtime(), version="1.0.0")
    assert await _health(server) == 200
    assert await _health(server) == 200

    assert len(_summary_lines(caplog)) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_each_phase_reads_its_own_longest_gap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from omnibase_infra.handlers.handler_boot_summary import emit_boot_summary_once
    from omnibase_infra.models.node_announce.model_boot_clock import ModelBootClock

    caplog.set_level(logging.INFO)

    async def held(seconds: float) -> None:
        time.sleep(seconds)  # the injected hold blocks the loop on purpose

    started = ModelBootClock.begin("wiring_pass")
    await held(0.6)
    ModelBootClock.end("wiring_pass", started)
    await ModelBootClock.run("local_ingress_start", held(0.3))

    emit_boot_summary_once(200)

    lines = _summary_lines(caplog)
    assert len(lines) == 1, lines
    match = _LINE.search(lines[0])
    assert match is not None, lines[0]
    assert float(match["wiring_pass"]) >= 0.55
    assert 0.25 <= float(match["local_ingress_start"]) < 0.55
    assert match["boot_step0"] == "none"
