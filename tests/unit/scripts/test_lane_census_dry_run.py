# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dry-run + fail-fast safety tests for lane-census-check.sh (OMN-13011).

Properties proven here:
  1. --dry-run NEVER publishes to the bus (no `rpk produce`).
  2. On drift, the script exits 30 (fail-fast, no warn-only mode).
  3. With no broker configured, a live run does NOT publish (fail-fast, no
     localhost default) — it logs for manual replay and still exits 30 on drift.
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

    # docker shim: ps -a -> rows; network ls -> networks; everything else empty.
    docker_body = (
        'case "$*" in\n'
        f'  *"ps -a"*) cat "{ps_file}" ;;\n'
        f'  *"network ls"*) cat "{net_file}" ;;\n'
        "  *) : ;;\n"
        "esac\n"
        "exit 0"
    )
    _shim(bin_dir, "docker", docker_body, calllog)
    _shim(bin_dir, "rpk", "exit 0", calllog)
    _shim(bin_dir, "hostname", 'echo "omninode-pc"', calllog)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)  # log dir under tmp, never ~
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    if broker:
        env["KAFKA_BOOTSTRAP_SERVERS"] = broker

    proc = subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return proc, calllog.read_text()


# A prod lane with the runtime containers absent + network detached (tonight's
# outage), so every live run here MUST detect drift.
_PROD_OUTAGE_PS = (
    '{"Names":"omnibase-infra-prod-postgres","State":"running",'
    '"Status":"Up 2 hours","Image":"postgres:16","Labels":"com.omninode.lane=prod"}\n'
)
_NO_NETWORKS = "bridge\nhost\n"


def test_dry_run_does_not_publish(tmp_path: Path) -> None:
    proc, calls = _run(
        ["--lane", "prod", "--dry-run"],
        tmp_path,
        ps_rows=_PROD_OUTAGE_PS,
        networks=_NO_NETWORKS,
        broker="redpanda:9092",
    )
    assert proc.returncode == 30, proc.stderr
    assert "produce" not in calls, f"dry-run published to bus:\n{calls}"


def test_drift_exits_30(tmp_path: Path) -> None:
    proc, _ = _run(
        ["--lane", "prod"],
        tmp_path,
        ps_rows=_PROD_OUTAGE_PS,
        networks=_NO_NETWORKS,
        broker="redpanda:9092",
    )
    assert proc.returncode == 30, proc.stderr


def test_live_without_broker_does_not_publish(tmp_path: Path) -> None:
    """No KAFKA_BOOTSTRAP_SERVERS => no publish (fail-fast, no localhost default)."""
    proc, calls = _run(
        ["--lane", "prod"],
        tmp_path,
        ps_rows=_PROD_OUTAGE_PS,
        networks=_NO_NETWORKS,
        broker=None,
    )
    assert "produce" not in calls, f"published with no broker configured:\n{calls}"
    assert proc.returncode == 30


def test_clean_lane_exits_zero(tmp_path: Path) -> None:
    """A fully-running prod lane with its network present exits 0 (no drift)."""
    # Build a healthy prod inventory from the manifest.
    import yaml

    manifest = yaml.safe_load(
        (_REPO / "deploy" / "lane-census" / "lane-manifest.yaml").read_text()
    )
    rows = []
    for svc in manifest["lanes"]["prod"]["services"]:
        name = svc["name"]
        if svc.get("kind") == "oneshot":
            rows.append(
                f'{{"Names":"{name}","State":"exited",'
                '"Status":"Exited (0) 1 hour ago",'
                '"Image":"x:1","Labels":"com.omninode.lane=prod"}'
            )
        else:
            rows.append(
                f'{{"Names":"{name}","State":"running","Status":"Up 1 hour",'
                '"Image":"x:1","Labels":"com.omninode.lane=prod"}'
            )
    ps = "\n".join(rows) + "\n"
    networks = "omnibase-infra-prod-network\nbridge\n"
    proc, calls = _run(
        ["--lane", "prod"],
        tmp_path,
        ps_rows=ps,
        networks=networks,
        broker="redpanda:9092",
    )
    assert proc.returncode == 0, proc.stderr
    assert "produce" not in calls, "clean lane must not publish"
