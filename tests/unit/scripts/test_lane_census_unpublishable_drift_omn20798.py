# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A drift the census cannot publish is its own failure (OMN-20798).

On 2026-10-09 the second lab host's hourly census found a critical drift every
hour and dropped it: ``KAFKA_BOOTSTRAP_SERVERS unset — cannot publish. Drift
event logged above for manual replay.`` The run then exited 30, the same code a
drift that WAS published exits with, so nothing told a reader that the alert
never reached the bus. The branch could not have worked in any case: ``rpk`` is
not on a lab host's PATH (it lives inside the broker container), so naming a
broker address in the unit would only have moved the failure to
``rpk not found``.

Properties proven here, each by driving the real script against a ``docker``
shim:

  1. Drift with no broker container named exits 8, publishes nothing, and says
     so.
  2. Drift whose produce fails exits 8.
  3. ``KAFKA_BOOTSTRAP_SERVERS`` is not a publish route: with it set and an
     ``rpk`` on PATH, a run with no broker container still exits 8 and never
     calls ``rpk``.
  4. Drift that IS published exits 30 (drift, delivered), through
     ``docker exec -i <broker> sh -c ... rpk topic produce <drift topic>`` with
     the SASL pair named by variable, never by value, on this host's argv.
  5. ``--dry-run`` still publishes nothing and exits 30.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "lane-census-check.sh"

_DRIFT_TOPIC = "onex.evt.infra.lane-census-drift.v1"
_BROKER = "test-broker-container"
_LANE = "judge"
_LANE_OUTAGE_PS = (
    "omnibase-infra-judge-postgres\trunning\tUp 2 hours\tpostgres:16"
    "\tcom.omninode.lane=judge\n"
)
_NO_NETWORKS = "bridge\nhost\n"


def _shim(bin_dir: Path, name: str, body: str, calllog: Path) -> None:
    shim = bin_dir / name
    shim.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{calllog}"\n{body}\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    args: list[str],
    tmp_path: Path,
    *,
    broker_container: str | None,
    produce_rc: int = 0,
    bootstrap: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    """Run the census against a drifting lane; return (proc, calls, produced stdin)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calllog = tmp_path / "calls.log"
    calllog.write_text("")
    produced = tmp_path / "produced.json"
    ps_file = tmp_path / "ps.ndjson"
    ps_file.write_text(_LANE_OUTAGE_PS)
    net_file = tmp_path / "networks.txt"
    net_file.write_text(_NO_NETWORKS)

    docker_body = (
        'case "$*" in\n'
        f'  *"ps -a"*) cat "{ps_file}" ;;\n'
        f'  *"network ls"*) cat "{net_file}" ;;\n'
        '  "inspect --format "*) shift 3; for n in "$@"; do printf '
        '\'{"Names":"/%s","State":{"Health":null},'
        '"Config":{"Healthcheck":null},"RestartCount":0}\\n\' "$n"; done ;;\n'
        f'  "exec -i "*) cat > "{produced}"; exit {produce_rc} ;;\n'
        "  *) : ;;\n"
        "esac\n"
        "exit 0"
    )
    _shim(bin_dir, "docker", docker_body, calllog)
    _shim(bin_dir, "rpk", "exit 0", calllog)
    _shim(bin_dir, "hostname", 'echo "omninode-pc"', calllog)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/lane-census-test.sock"
    env["HOME"] = str(tmp_path)
    for name in ("KAFKA_BOOTSTRAP_SERVERS", "LANE_MEMORY_BROKER_CONTAINER"):
        env.pop(name, None)
    if bootstrap:
        env["KAFKA_BOOTSTRAP_SERVERS"] = bootstrap
    if broker_container:
        env["LANE_MEMORY_BROKER_CONTAINER"] = broker_container

    proc = subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    sent = produced.read_text() if produced.exists() else ""
    return proc, calllog.read_text(), sent


def test_lane_census_unpublishable_drift_fails_without_a_broker_container(
    tmp_path: Path,
) -> None:
    proc, calls, sent = _run(["--lane", _LANE], tmp_path, broker_container=None)
    assert proc.returncode == 8, proc.stderr
    assert "produce" not in calls and not sent, calls
    assert "NOT published" in proc.stderr


def test_lane_census_unpublishable_drift_fails_when_the_produce_fails(
    tmp_path: Path,
) -> None:
    proc, calls, _ = _run(
        ["--lane", _LANE], tmp_path, broker_container=_BROKER, produce_rc=1
    )
    assert proc.returncode == 8, proc.stderr
    assert f"exec -i {_BROKER}" in calls
    assert "NOT published" in proc.stderr


def test_lane_census_unpublishable_drift_fails_despite_a_bootstrap_address(
    tmp_path: Path,
) -> None:
    """KAFKA_BOOTSTRAP_SERVERS is not a route: a host-side rpk is never called."""
    proc, calls, _ = _run(
        ["--lane", _LANE],
        tmp_path,
        broker_container=None,
        bootstrap="redpanda:9092",
    )
    assert proc.returncode == 8, proc.stderr
    assert "rpk" not in calls, calls


def test_lane_census_published_drift_exits_30_through_the_broker_container(
    tmp_path: Path,
) -> None:
    proc, calls, sent = _run(["--lane", _LANE], tmp_path, broker_container=_BROKER)
    assert proc.returncode == 30, proc.stderr
    exec_lines = [c for c in calls.splitlines() if c.startswith("docker exec -i")]
    assert len(exec_lines) == 1, calls
    line = exec_lines[0]
    assert f"exec -i {_BROKER} sh -c" in line
    assert f'rpk topic produce "{_DRIFT_TOPIC}"' in line
    # The SASL pair is named by variable and expanded inside the container.
    assert "${DEV_KAFKA_SASL_USERNAME}" in line
    assert "${DEV_KAFKA_SASL_PASSWORD}" in line
    assert '"event_type"' in sent or '"findings"' in sent, sent


def test_lane_census_dry_run_still_publishes_nothing(tmp_path: Path) -> None:
    proc, calls, sent = _run(
        ["--lane", _LANE, "--dry-run"], tmp_path, broker_container=_BROKER
    )
    assert proc.returncode == 30, proc.stderr
    assert "exec -i" not in calls and not sent, calls
