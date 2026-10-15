# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dry-run + fail-fast safety tests for lane-census-check.sh (OMN-13011).

Properties proven here:
  1. --dry-run NEVER publishes to the bus (no `rpk produce`).
  2. On drift that reaches the bus, the script exits 30 (fail-fast, no
     warn-only mode).
  3. With no broker container configured, a live run does NOT publish
     (fail-fast, no localhost default) and exits 8, not 30 (OMN-20798).
  4. A clean lane (no drift) exits 0.

We shim `docker`, `rpk`, and `hostname` on PATH with a recorder so the test is
hermetic and asserts on the exact subcommands the script issued.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from tests.helpers.lane_census_fixtures import (
    NEUTRAL_CONFIG_ROOT,
    NEUTRAL_HOSTNAME,
    NEUTRAL_LANE,
    NEUTRAL_LANE_NETWORK,
    NEUTRAL_MANIFEST,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "lane-census-check.sh"


def _shim(bin_dir: Path, name: str, body: str, calllog: Path) -> None:
    shim = bin_dir / name
    shim.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{calllog}"\n{body}\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    args: list[str],
    tmp_path: Path,
    *,
    ps_rows: str,
    networks: str,
    broker: str | None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calllog = tmp_path / "calls.log"
    calllog.write_text("")

    # Write the canned inventory to files the shim cats — avoids any inline
    # quoting/printf interpretation of the JSON rows.
    ps_file = tmp_path / "ps.ndjson"
    ps_file.write_text(ps_rows)
    net_file = tmp_path / "networks.txt"
    net_file.write_text(networks)

    # docker shim: ps -a -> rows; network ls -> networks; inspect -> one
    # no-healthcheck, zero-restart reading per name (OMN-19416); else empty.
    docker_body = (
        'case "$*" in\n'
        f'  *"ps -a"*) cat "{ps_file}" ;;\n'
        f'  *"network ls"*) cat "{net_file}" ;;\n'
        '  "inspect --format "*) shift 3; for n in "$@"; do printf '
        '\'{"Names":"/%s","State":{"Health":null},'
        '"Config":{"Healthcheck":null},"RestartCount":0}\\n\' "$n"; done ;;\n'
        "  *) : ;;\n"
        "esac\n"
        "exit 0"
    )
    _shim(bin_dir, "docker", docker_body, calllog)
    _shim(bin_dir, "rpk", "exit 0", calllog)
    _shim(bin_dir, "hostname", f'echo "{NEUTRAL_HOSTNAME}"', calllog)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    # OMN-15466: the collector prefers the Docker Engine API. Point it at a
    # socket that cannot exist so these tests keep exercising the docker-CLI
    # fallback against the shim, hermetically and on any host.
    env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/lane-census-test.sock"
    env["HOME"] = str(tmp_path)  # log dir under tmp, never ~
    # OMN-20934: the census reads the neutral manifest, never a lab's.
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
    env.pop("LANE_MANIFEST", None)
    env.pop("LANE_MEMORY_BROKER_CONTAINER", None)
    env.pop("LANE_CENSUS_DESIRED_STATE", None)
    if broker:
        env["LANE_MEMORY_BROKER_CONTAINER"] = broker
    env.update(extra_env or {})

    proc = subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return proc, calllog.read_text()


# A lane with its runtime containers absent + network detached, so every live run
# here MUST detect drift.
# Tab-delimited rows: Names \t State \t Status \t Image \t Labels. OMN-15466
# replaced the `{{json .}}` inventory format, which silently requested per-
# container size (90.363 s vs 0.150 s on .201's 111 containers).
#
# OMN-20934: the lane is the neutral fixture's required lane. These tests drive
# the real script as a subprocess, so the lane must be one the manifest it reads
# declares; the assertions are about CLI behaviour (exit codes and whether
# anything is published), never about which lane it is.
_LANE = NEUTRAL_LANE
_LANE_OUTAGE_PS = (
    "example-judge-postgres\trunning\tUp 2 hours\tpostgres:16"
    f"\tcom.omninode.lane={_LANE}\n"
)
_NO_NETWORKS = "bridge\nhost\n"


def test_dry_run_does_not_publish(tmp_path: Path) -> None:
    proc, calls = _run(
        ["--lane", _LANE, "--dry-run"],
        tmp_path,
        ps_rows=_LANE_OUTAGE_PS,
        networks=_NO_NETWORKS,
        broker="redpanda",
    )
    assert proc.returncode == 30, proc.stderr
    assert "produce" not in calls, f"dry-run published to bus:\n{calls}"


def test_drift_exits_30(tmp_path: Path) -> None:
    proc, _ = _run(
        ["--lane", _LANE],
        tmp_path,
        ps_rows=_LANE_OUTAGE_PS,
        networks=_NO_NETWORKS,
        broker="redpanda",
    )
    assert proc.returncode == 30, proc.stderr


def test_live_without_broker_does_not_publish(tmp_path: Path) -> None:
    """No broker container => no publish (fail-fast, no default), exit 8."""
    proc, calls = _run(
        ["--lane", _LANE],
        tmp_path,
        ps_rows=_LANE_OUTAGE_PS,
        networks=_NO_NETWORKS,
        broker=None,
    )
    assert "produce" not in calls, f"published with no broker configured:\n{calls}"
    assert proc.returncode == 8


def _clean_lane_inventory() -> tuple[str, str]:
    """A fully-running lane with its network present: (ps rows, networks)."""
    # Build a healthy inventory for the lane from the manifest.
    import yaml

    manifest = yaml.safe_load(NEUTRAL_MANIFEST.read_text())
    rows = []
    for svc in manifest["lanes"][_LANE]["services"]:
        name = svc["name"]
        if svc.get("kind") == "profile_gated":
            # OMN-16803: the lane's compose file disables these via a profile
            # override, so ABSENT is their healthy state. Modelling them as
            # Running would make the "clean lane" fixture assert the exact
            # condition the census now reports as profile_gated_present drift.
            continue
        if svc.get("kind") == "oneshot":
            rows.append(
                f"{name}\texited\tExited (0) 1 hour ago\tx:1\tcom.omninode.lane={_LANE}"
            )
        else:
            rows.append(f"{name}\trunning\tUp 1 hour\tx:1\tcom.omninode.lane={_LANE}")
    return "\n".join(rows) + "\n", f"{NEUTRAL_LANE_NETWORK}\nbridge\n"


def test_clean_lane_exits_zero(tmp_path: Path) -> None:
    """A fully-running lane with its network present exits 0 (no drift)."""
    ps, networks = _clean_lane_inventory()
    proc, calls = _run(
        ["--lane", _LANE],
        tmp_path,
        ps_rows=ps,
        networks=networks,
        broker="redpanda",
    )
    assert proc.returncode == 0, proc.stderr
    assert "produce" not in calls, "clean lane must not publish"


def test_clean_lane_with_the_desired_state_removed_is_drift(tmp_path: Path) -> None:
    """OMN-19414 AC2: the lane is clean, the state it is graded against is gone.

    The same inventory exits 0 above; naming a desired-state file that does not
    exist makes the census exit 30 and carry desired_state_unreadable, never 0.
    """
    ps, networks = _clean_lane_inventory()
    proc, _ = _run(
        ["--lane", _LANE, "--dry-run", "--json"],
        tmp_path,
        ps_rows=ps,
        networks=networks,
        broker="redpanda",
        extra_env={"LANE_CENSUS_DESIRED_STATE": str(tmp_path / "removed.json")},
    )
    assert proc.returncode == 30, proc.stderr
    assert "desired_state_unreadable" in proc.stdout
    assert "removed.json" in proc.stdout
