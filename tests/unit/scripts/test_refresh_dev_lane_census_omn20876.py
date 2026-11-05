# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A warm refresh of the dev lane ends by reading the lane census (OMN-20876).

On 2026-10-09 the lane census alarm reported the dev lane degraded -- a required
service container absent after a recreate -- and a temp-hash-named leftover
container (``b4320b97cc9a_omnibase-infra-forward-migration``) within hours of a
warm refresh. A recreate that is interrupted leaves the old container renamed to
``<12 hex>_<name>``; nothing repaired it, so the drift sat until a lane noticed.

``refresh_dev_lane.sh`` now closes with the census reader that already exists
(``lane-census-check.sh`` over ``lane_census_plan.py``): it removes the leftover
recreate containers, reads the census, and exits with the census's own code when
the lane has drifted. ``--census-gate`` runs that closing step alone, which is
also how a lane that already carries the drift is repaired without a refresh.
Recovery runs the same closing step while preserving the failed refresh's
result and exit code; remaining census findings are recorded separately.

Every case drives the real script against a ``docker`` shim that holds the lane
inventory in a file, so a ``docker rm`` changes what the second census read sees.
The shim never reaches a daemon: the trusted CI pool is self-hosted on the lab
hosts, and a test that reached the real one would inventory real lanes.

Falsifiers (OMN-20876): ``-k refresh_dev_lane_fails_on_census_drift`` and
``-k refresh_removes_leftover_lane_containers``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.helpers.lane_census_fixtures import (
    NEUTRAL_CONFIG_ROOT,
    NEUTRAL_HOSTNAME,
    NEUTRAL_MANIFEST,
    NEUTRAL_REGISTRY,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
_MANIFEST = NEUTRAL_MANIFEST

_EXIT_DRIFT = 30
_EXIT_INVENTORY_UNOBSERVABLE = 4

_MANIFEST_DATA = yaml.safe_load(_MANIFEST.read_text())
_DEV_SERVICE = next(
    s["name"]
    for s in _MANIFEST_DATA["lanes"]["dev"]["services"]
    if s["kind"] == "service"
)
_LEFTOVER = f"b4320b97cc9a_{_DEV_SERVICE}"

_DOCKER_SHIM = r"""#!/usr/bin/env python3
import json, os, sys

state = os.environ["SHIM_INVENTORY"]
with open(os.environ["SHIM_CALLS"], "a") as fh:
    fh.write("docker " + " ".join(sys.argv[1:]) + "\n")
if os.environ.get("SHIM_FAIL_PS") and sys.argv[1:3] == ["ps", "-a"]:
    sys.stderr.write("Cannot connect to the Docker daemon\n")
    sys.exit(1)
rows = json.load(open(state))
args = sys.argv[1:]
if args[:2] == ["ps", "-a"]:
    for r in rows:
        print("\t".join([r["name"], r["state"], r["status"], r["image"], r["labels"]]))
elif args[:2] == ["network", "ls"]:
    print(os.environ["SHIM_NETWORK"])
    print("bridge")
elif args[0] == "inspect":
    fmt = args[args.index("--format") + 1]
    names = args[args.index("--format") + 2:]
    by_name = {r["name"]: r for r in rows}
    for n in names:
        if "index .Config.Labels" in fmt:
            if n not in by_name:
                sys.stderr.write("Error: No such object: " + n + "\n")
                sys.exit(1)
            labels = dict(kv.split("=", 1) for kv in by_name[n]["labels"].split(","))
            print(labels.get("com.omninode.lane", ""))
        else:
            print(json.dumps({"Names": "/" + n, "State": {"Health": None},
                              "Config": {"Healthcheck": None}, "RestartCount": 0}))
elif args[:2] == ["rm", "-f"]:
    target = args[2]
    if target not in {r["name"] for r in rows}:
        sys.stderr.write("Error: No such container: " + target + "\n")
        sys.exit(1)
    json.dump([r for r in rows if r["name"] != target], open(state, "w"))
    print(target)
sys.exit(0)
"""


def _row(name: str, lane: str, *, oneshot: bool = False) -> dict[str, str]:
    return {
        "name": name,
        "state": "exited" if oneshot else "running",
        "status": "Exited (0) 2 hours ago" if oneshot else "Up 2 hours",
        "image": "omnibase-infra:test",
        "labels": f"com.omninode.lane={lane},com.docker.compose.project=omnibase-infra",
    }


def _clean_dev_lane() -> list[dict[str, str]]:
    """Every container the manifest declares for the dev lane, in its steady state."""
    services = _MANIFEST_DATA["lanes"]["dev"]["services"]
    return [_row(s["name"], "dev", oneshot=s["kind"] == "oneshot") for s in services]


def _run(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *args: str,
    fail_ps: bool = False,
    completion_result: str | None = None,
    manifest: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[dict[str, str]]]:
    """Run the census-only command or the refresh's completion block over rows.

    Returns (process, docker call log lines, inventory after the run).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "docker"
    shim.write_text(
        _DOCKER_SHIM.replace("#!/usr/bin/env python3", f"#!{sys.executable}")
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(rows))
    calls = tmp_path / "calls.log"
    calls.write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["OMNI_HOME"] = str(tmp_path)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
    env["LANE_CENSUS_REGISTRY"] = str(NEUTRAL_REGISTRY)
    env.pop("LANE_MANIFEST", None)
    if manifest is not None:
        env["LANE_MANIFEST"] = str(manifest)
    env["SHIM_INVENTORY"] = str(inventory)
    env["SHIM_CALLS"] = str(calls)
    env["SHIM_NETWORK"] = _MANIFEST_DATA["lanes"]["dev"]["network"]
    env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/refresh-dev-lane-test.sock"
    env["LANE_CENSUS_HOST"] = NEUTRAL_HOSTNAME
    env["LANE_CENSUS_PYTHON"] = sys.executable
    env["ONEX_LANE_LOCK_DIR"] = str(tmp_path / "locks")
    for name in ("OMNIBASE_INFRA_COMPOSE_PROJECT", "LANE_MEMORY_BROKER_CONTAINER"):
        env.pop(name, None)
    if fail_ps:
        env["SHIM_FAIL_PS"] = "1"

    command = ["bash", str(_SCRIPT), "--census-gate", *args]
    if completion_result is not None:
        # Run the actual completion block, census functions and final exit
        # mapping. Upstream build/rollback results are inputs; the census still
        # inventories and removes through the same Docker shim as --census-gate.
        text = _SCRIPT.read_text()
        functions_start = text.index('CENSUS_SCRIPT="')
        functions = text[
            functions_start : text.index('if [[ "${CENSUS_ONLY}"', functions_start)
        ]
        completion = text[
            text.index("# --- close with the lane census") : text.index(
                "# --- emit receipt"
            )
        ]
        exit_case = text[text.index('case "${RESULT}" in') :]
        shell = "\n".join(
            [
                "set -euo pipefail",
                'REPO_ROOT="$1"; RESULT="$2"; MODE="$3"; LANE=dev',
                'log() { echo "LOG:$*" >&2; }',
                'err() { echo "ERR:$*" >&2; }',
                functions,
                completion,
                'jq -n --arg result "${RESULT}" '
                '--argjson ran "${CENSUS_RAN}" '
                '--argjson exit "${CENSUS_EXIT}" '
                '--argjson removed "${CENSUS_REMOVED_JSON}" '
                "'{result: $result, lane_census: "
                "{ran: $ran, exit: $exit, leftovers_removed: $removed}}'",
                exit_case,
            ]
        )
        command = [
            "bash",
            "-c",
            shell,
            "refresh-completion",
            str(_REPO),
            completion_result,
            "execute" if "--execute" in args else "plan",
        ]

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO),
        timeout=120,
        check=False,
    )
    return proc, calls.read_text().splitlines(), json.loads(inventory.read_text())


def _rm_calls(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("docker rm ")]


# --- AC1: the refresh ends by reading the census and fails on drift ---------


def test_refresh_dev_lane_census_gate_passes_on_a_clean_lane(tmp_path: Path) -> None:
    """Positive control: the fixture is a clean lane, so a drift verdict below is earned."""
    proc, calls, _ = _run(tmp_path, _clean_dev_lane(), "--execute")
    assert proc.returncode == 0, proc.stderr
    assert "No lane drift" in proc.stderr
    assert _rm_calls(calls) == []


def test_refresh_dev_lane_fails_on_census_drift_when_a_required_container_is_absent(
    tmp_path: Path,
) -> None:
    # The optional lane stays partially up, so its missing service is drift.
    rows = [r for r in _clean_dev_lane() if r["name"] != _DEV_SERVICE]
    proc, calls, _ = _run(tmp_path, rows, "--execute")
    assert proc.returncode == _EXIT_DRIFT, proc.stderr
    assert f"container_absent {_DEV_SERVICE} (dev)" in proc.stderr
    assert _rm_calls(calls) == []


def test_refresh_dev_lane_fails_on_census_drift_when_an_unexpected_container_carries_the_lane_label(
    tmp_path: Path,
) -> None:
    rows = [*_clean_dev_lane(), _row("rogue-sidecar", "dev")]
    proc, calls, _ = _run(tmp_path, rows, "--execute")
    assert proc.returncode == _EXIT_DRIFT, proc.stderr
    assert "unexpected_container rogue-sidecar (dev)" in proc.stderr
    # Not a recreate leftover (no hash prefix): reported, never removed.
    assert _rm_calls(calls) == []


def test_refresh_dev_lane_fails_on_census_drift_in_plan_mode_too(
    tmp_path: Path,
) -> None:
    rows = [r for r in _clean_dev_lane() if r["name"] != _DEV_SERVICE]
    proc, _, _ = _run(tmp_path, rows)
    assert proc.returncode == _EXIT_DRIFT, proc.stderr


def test_refresh_dev_lane_census_gate_fails_loud_when_the_inventory_is_unobservable(
    tmp_path: Path,
) -> None:
    """A host the census cannot see is not a clean host and not a drifted one."""
    proc, _, _ = _run(tmp_path, _clean_dev_lane(), "--execute", fail_ps=True)
    assert proc.returncode == _EXIT_INVENTORY_UNOBSERVABLE, proc.stderr


def test_refresh_dev_lane_main_flow_ends_with_the_census_gate() -> None:
    """The closing step is wired into the refresh itself, after the health gate.

    ``--census-gate`` is the same function; this pins that the refresh calls it
    and lets its exit code replace a SUCCESS, which a standalone mode cannot show.
    """
    text = _SCRIPT.read_text()
    health = text.index('log "=== Health-gate ==="')
    close = text.index("lane_census_close ||", health)
    receipt = text.index('RECEIPT_PATH="${HISTORY_DIR}', close)
    assert health < close < receipt
    assert re.search(
        r'RESULT="SUCCESS".*?CENSUS_EXIT.*?FAILED_CENSUS_DRIFT',
        text[health:receipt],
        re.S,
    ), "a census failure must replace a SUCCESS result"
    assert re.search(r'FAILED_CENSUS_DRIFT\)\s*exit "\$\{CENSUS_EXIT\}"', text), (
        "the refresh must exit with the census's code"
    )


# --- AC2: leftover recreate containers are removed, only this lane's --------


def test_refresh_removes_leftover_lane_containers(tmp_path: Path) -> None:
    rows = [*_clean_dev_lane(), _row(_LEFTOVER, "dev", oneshot=True)]
    proc, calls, after = _run(tmp_path, rows, "--execute")
    assert _rm_calls(calls) == [f"docker rm -f {_LEFTOVER}"], calls
    assert _LEFTOVER not in {r["name"] for r in after}
    # With the leftover gone the second census read is clean.
    assert proc.returncode == 0, proc.stderr
    assert f"removed leftover recreate container {_LEFTOVER}" in proc.stderr


def test_refresh_removes_leftover_lane_containers_never_touches_another_lane(
    tmp_path: Path,
) -> None:
    # Every other lane the manifest declares: a hash-prefixed container under each
    # lane's own first service, labelled for that lane, plus one of the dev lane's
    # declared names under a hash, labelled for another lane.
    other_lanes = [lane for lane in _MANIFEST_DATA["lanes"] if lane != "dev"]
    assert len(other_lanes) >= 2, other_lanes
    foreign = [
        _row(f"{i:012x}_{_MANIFEST_DATA['lanes'][lane]['services'][0]['name']}", lane)
        for i, lane in enumerate(other_lanes, start=0xAAAAAAAAAAAA)
    ]
    foreign.append(_row(f"0123456789ab_{_DEV_SERVICE}", other_lanes[0]))
    undeclared = _row("dddddddddddd_not-a-dev-service", "dev")
    rows = [
        *_clean_dev_lane(),
        _row(_LEFTOVER, "dev", oneshot=True),
        undeclared,
        *foreign,
    ]
    proc, calls, after = _run(tmp_path, rows, "--execute")

    assert _rm_calls(calls) == [f"docker rm -f {_LEFTOVER}"], calls
    remaining = {r["name"] for r in after}
    for row in foreign:
        assert row["name"] in remaining, f"{row['name']} belongs to another lane"
    # A dev-labelled hash name that is not a prefix of a declared service stays,
    # and stays drift: it is reported, not guessed at.
    assert undeclared["name"] in remaining
    assert proc.returncode == _EXIT_DRIFT, proc.stderr
    assert "unexpected_container dddddddddddd_not-a-dev-service (dev)" in proc.stderr


def test_refresh_removes_leftover_lane_containers_only_in_execute_mode(
    tmp_path: Path,
) -> None:
    rows = [*_clean_dev_lane(), _row(_LEFTOVER, "dev", oneshot=True)]
    proc, calls, after = _run(tmp_path, rows)
    assert _rm_calls(calls) == []
    assert _LEFTOVER in {r["name"] for r in after}
    assert proc.returncode == _EXIT_DRIFT, proc.stderr
    assert f"would remove leftover recreate container {_LEFTOVER}" in proc.stderr


@pytest.fixture
def recorded_refresh(tmp_path: Path) -> tuple[Path, list[dict[str, str]], str]:
    """Replay the 2026-10-09 dev leftover against an isolated manifest.

    The recorded alarm supplies the name and lane; steady-state companion rows
    are fixture data, not a claim about the historical host's inventory.
    """
    declared = "omnibase-infra-forward-migration"
    leftover = f"b4320b97cc9a_{declared}"
    data = yaml.safe_load(_MANIFEST.read_text())
    data["lanes"]["dev"]["services"].append({"name": declared, "kind": "oneshot"})
    manifest = tmp_path / "lane-manifest.yaml"
    manifest.write_text(yaml.safe_dump(data))
    rows = [
        *_clean_dev_lane(),
        _row(declared, "dev", oneshot=True),
        _row(leftover, "dev", oneshot=True),
    ]
    return manifest, rows, leftover


@pytest.mark.parametrize(
    ("result", "exit_code"),
    [
        ("SUCCESS", 0),
        ("FAILED_ROLLED_BACK", 1),
        ("FAILED_BUILD_PROVENANCE", 1),
        ("FAILED_CLOUD_MIGRATION", 1),
        ("FAILED", 2),
    ],
)
@pytest.mark.parametrize("drift", [False, True])
def test_refresh_completion_reconciles_recorded_leftover_without_hiding_failures(
    tmp_path: Path,
    recorded_refresh: tuple[Path, list[dict[str, str]], str],
    result: str,
    exit_code: int,
    drift: bool,
) -> None:
    manifest, rows, leftover = recorded_refresh
    foreign = "aaaaaaaaaaaa_example-dogfood-runtime"
    rows.append(_row(foreign, "dogfood"))
    if drift:
        rows.append(_row("dddddddddddd_not-a-dev-service", "dev"))
        rows = [r for r in rows if r["name"] != _DEV_SERVICE]
    proc, calls, after = _run(
        tmp_path,
        rows,
        "--execute",
        completion_result=result,
        manifest=manifest,
    )
    assert _rm_calls(calls) == [f"docker rm -f {leftover}"], proc.stderr
    assert sum(c.startswith("docker ps -a ") for c in calls) == 2
    assert leftover not in {r["name"] for r in after}
    assert foreign in {r["name"] for r in after}
    census_exit = _EXIT_DRIFT if drift else 0
    expected_result = "FAILED_CENSUS_DRIFT" if drift and result == "SUCCESS" else result
    expected_exit = census_exit if result == "SUCCESS" else exit_code
    assert proc.returncode == expected_exit, proc.stderr
    assert json.loads(proc.stdout) == {
        "result": expected_result,
        "lane_census": {
            "ran": True,
            "exit": census_exit,
            "leftovers_removed": [leftover],
        },
    }
    if drift:
        assert (
            "unexpected_container dddddddddddd_not-a-dev-service (dev)" in proc.stderr
        )
        assert f"container_absent {_DEV_SERVICE} (dev)" in proc.stderr


@pytest.mark.parametrize(
    ("result", "exit_code"),
    [("SUCCESS", 4), ("FAILED_ROLLED_BACK", 1), ("FAILED", 2)],
)
def test_refresh_completion_records_unobservable_census_without_replacing_failure(
    tmp_path: Path, result: str, exit_code: int
) -> None:
    proc, calls, _ = _run(
        tmp_path,
        _clean_dev_lane(),
        "--execute",
        fail_ps=True,
        completion_result=result,
    )
    assert proc.returncode == exit_code, proc.stderr
    assert _rm_calls(calls) == []
    receipt = json.loads(proc.stdout)
    assert receipt["lane_census"] == {
        "ran": True,
        "exit": _EXIT_INVENTORY_UNOBSERVABLE,
        "leftovers_removed": [],
    }
    assert receipt["result"] == (
        "FAILED_CENSUS_DRIFT" if result == "SUCCESS" else result
    )
