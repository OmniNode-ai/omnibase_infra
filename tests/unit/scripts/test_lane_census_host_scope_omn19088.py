# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lane census is host-aware (OMN-19088).

Until this change the manifest had no notion of WHERE a lane runs, so the
census evaluated every lane on whatever host it was run on. Measured on a
proof surface on 2026-09-21: one lane genuinely present and clean, and 25
critical ``container_absent`` / ``network_detached`` findings for lanes that
live on another host. Those lanes were not missing, they were elsewhere, and the
one real signal was a line in the noise.

What these tests pin:

* every lane declares the host(s) it runs on, from a registry in the manifest;
* the planner evaluates only the lanes declared for the host it runs on and
  reports the rest ``lanes_not_applicable`` (AC-1), while a declared lane with a
  stopped container is still critical (AC-1's negative control);
* a lane found running on a host it is not declared for is a finding, and the
  finding clears when the declaration is corrected (AC-2);
* a host the manifest does not declare is refused loudly, never evaluated as if
  it were every host (the pre-fix behaviour) and never as if it were none (a
  silent clean).

OMN-20934: the lab's manifest is private config, so these run over the neutral
manifest (two hosts, three lanes). Which lab hosts carry which lanes is a fact
about the lab, checked where its manifest lives.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.lane_census_fixtures import (
    NEUTRAL_CONFIG_ROOT,
    NEUTRAL_HOST_ID,
    NEUTRAL_HOSTNAME,
    NEUTRAL_LANE,
    NEUTRAL_MANIFEST,
    NEUTRAL_OPTIONAL_LANE,
    NEUTRAL_OTHER_HOST_LANE,
    NEUTRAL_OTHER_HOSTNAME,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PLAN_PATH = _REPO / "scripts" / "lane_census_plan.py"
_SCRIPT = _REPO / "scripts" / "lane-census-check.sh"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lane_census_plan_omn19088", _PLAN_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load_planner()
MANIFEST = PLAN.load_manifest(NEUTRAL_MANIFEST)

_HOST_A = NEUTRAL_HOSTNAME
_HOST_B = NEUTRAL_OTHER_HOSTNAME
_HOST_B_ID = "host-b"
_LANES_A = {NEUTRAL_LANE, NEUTRAL_OPTIONAL_LANE}
_LANES_B = {NEUTRAL_OTHER_HOST_LANE}


def _row(
    name: str,
    *,
    running: bool = True,
    exit_code: int = 0,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "Names": name,
        "State": "running" if running else "exited",
        "Status": "Up 2 hours" if running else f"Exited ({exit_code}) 1 hour ago",
        "Image": "example-runtime:1",
        "Labels": labels or {},
    }


def _healthy_lane_rows(lane: str) -> list[dict[str, Any]]:
    """Every declared container of ``lane`` in its healthy steady state."""
    rows: list[dict[str, Any]] = []
    project = MANIFEST["lanes"][lane]["compose_project"]
    for svc in MANIFEST["lanes"][lane]["services"]:
        kind = svc.get("kind", "service")
        if kind == "profile_gated":
            continue
        labels = {"com.docker.compose.project": project}
        if kind == "oneshot":
            rows.append(_row(svc["name"], running=False, labels=labels))
        else:
            rows.append(_row(svc["name"], labels=labels))
    return rows


def _envelope(
    host: str | None,
    containers: list[dict[str, Any]],
    networks: list[str],
    *,
    lane: str | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "lane": lane,
        "containers": containers,
        "networks": networks,
        "runtime_tag": None,
    }
    if host is not None:
        env["host"] = host
    return env


def _lane_on(lane: str, host: str) -> dict[str, Any]:
    return _envelope(
        host, _healthy_lane_rows(lane), [MANIFEST["lanes"][lane]["network"]]
    )


def _runtime(lane: str) -> str:
    return next(
        s["name"] for s in MANIFEST["lanes"][lane]["services"] if "runtime" in s["name"]
    )


# --- the manifest declares where every lane runs ---------------------------------


def test_every_lane_declares_at_least_one_registered_host() -> None:
    registry = MANIFEST["hosts"]
    assert registry, "the manifest declares no hosts"
    for lane, spec in MANIFEST["lanes"].items():
        hosts = spec.get("hosts")
        assert isinstance(hosts, list), f"lane {lane!r} declares no hosts list"
        assert hosts, f"lane {lane!r} declares an empty hosts list"
        for host_id in hosts:
            assert host_id in registry, (
                f"lane {lane!r} names host {host_id!r}, which the hosts registry "
                "does not declare"
            )


def test_the_manifest_validates() -> None:
    PLAN.validate_hosts(MANIFEST)


def test_an_alias_claimed_by_two_hosts_is_refused() -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["hosts"][_HOST_B_ID]["aliases"].append(_HOST_A)
    with pytest.raises(PLAN.HostDeclarationError, match=_HOST_A):
        PLAN.validate_hosts(manifest)


def test_the_docker_desktop_daemon_name_is_refused_as_an_alias() -> None:
    """Every Docker Desktop host reports the daemon name ``docker-desktop``.

    That name identifies no host; declaring it would attribute one host's
    census to another.
    """
    manifest = copy.deepcopy(MANIFEST)
    manifest["hosts"][_HOST_B_ID]["aliases"].append("docker-desktop")
    with pytest.raises(PLAN.HostDeclarationError, match="docker-desktop"):
        PLAN.validate_hosts(manifest)


def test_a_lane_without_hosts_is_refused() -> None:
    manifest = copy.deepcopy(MANIFEST)
    del manifest["lanes"][NEUTRAL_OTHER_HOST_LANE]["hosts"]
    with pytest.raises(PLAN.HostDeclarationError, match=NEUTRAL_OTHER_HOST_LANE):
        PLAN.validate_hosts(manifest)


def test_a_lane_naming_an_unregistered_host_is_refused() -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["lanes"][NEUTRAL_OTHER_HOST_LANE]["hosts"] = ["host-999"]
    with pytest.raises(PLAN.HostDeclarationError, match="host-999"):
        PLAN.validate_hosts(manifest)


# --- host resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_HOST_A, NEUTRAL_HOST_ID),
        (f"{_HOST_A}.tailnet.ts.net", NEUTRAL_HOST_ID),
        (f"{_HOST_B}.local", _HOST_B_ID),
        (_HOST_B_ID, _HOST_B_ID),
        (f"  {_HOST_B.upper()}  ", _HOST_B_ID),
    ],
)
def test_a_declared_host_resolves(raw: str, expected: str) -> None:
    assert PLAN.resolve_host(raw, MANIFEST) == expected


@pytest.mark.parametrize("raw", ["", "docker-desktop", "some-new-box", "localhost"])
def test_an_undeclared_host_is_refused(raw: str) -> None:
    with pytest.raises(PLAN.HostDeclarationError):
        PLAN.resolve_host(raw, MANIFEST)


def test_an_envelope_without_a_host_is_refused() -> None:
    with pytest.raises(PLAN.HostDeclarationError, match="host"):
        PLAN.build_plan(_envelope(None, [], []), MANIFEST)


# --- AC-1: evaluate only the lanes declared for this host -----------------------


def test_a_host_reports_only_its_own_lane() -> None:
    """The ticket's measurement, replayed: the other host's lanes are not absent."""
    plan = PLAN.build_plan(_lane_on(NEUTRAL_OTHER_HOST_LANE, _HOST_B), MANIFEST)
    assert plan["host"] == _HOST_B_ID
    assert plan["lanes_checked"] == [NEUTRAL_OTHER_HOST_LANE]
    assert set(plan["lanes_not_applicable"]) == _LANES_A
    assert plan["findings"] == []
    assert plan["has_drift"] is False


def test_checked_and_not_applicable_partition_the_manifest() -> None:
    plan = PLAN.build_plan(_lane_on(NEUTRAL_LANE, _HOST_A), MANIFEST)
    checked = set(plan["lanes_checked"])
    not_applicable = set(plan["lanes_not_applicable"])
    assert checked.isdisjoint(not_applicable)
    assert checked | not_applicable == set(MANIFEST["lanes"])


def test_two_hosts_each_return_findings_only_for_their_own_lanes() -> None:
    """AC-1's falsifier: two hosts, and each answers only for itself.

    Both inventories are EMPTY, the harshest input: every declared service is
    absent. Each run files its findings against its own lanes only.
    """
    plan_a = PLAN.build_plan(_envelope(_HOST_A, [], []), MANIFEST)
    plan_b = PLAN.build_plan(_envelope(_HOST_B, [], []), MANIFEST)

    lanes_a = {f["lane"] for f in plan_a["findings"]}
    assert lanes_a, "positive control: an empty inventory on host A must be drift"
    assert lanes_a <= _LANES_A
    assert NEUTRAL_OTHER_HOST_LANE in plan_a["lanes_not_applicable"]

    assert {f["lane"] for f in plan_b["findings"]} <= _LANES_B
    assert set(plan_b["lanes_not_applicable"]) == _LANES_A


def test_a_stopped_container_in_a_declared_lane_is_still_critical() -> None:
    """AC-1's negative control: scoping must not hide a real outage."""
    runtime = _runtime(NEUTRAL_LANE)
    rows = [r for r in _healthy_lane_rows(NEUTRAL_LANE) if r["Names"] != runtime]
    rows.append(_row(runtime, running=False, exit_code=137))
    plan = PLAN.build_plan(
        _envelope(_HOST_A, rows, [MANIFEST["lanes"][NEUTRAL_LANE]["network"]]),
        MANIFEST,
    )
    absent = [
        f
        for f in plan["findings"]
        if f["kind"] == "container_absent" and f["container"] == runtime
    ]
    assert len(absent) == 1, plan["findings"]
    assert absent[0]["severity"] == "critical"
    assert absent[0]["lane"] == NEUTRAL_LANE


def test_a_requested_lane_off_this_host_is_not_applicable() -> None:
    plan = PLAN.build_plan(_envelope(_HOST_B, [], [], lane=NEUTRAL_LANE), MANIFEST)
    assert plan["lanes_checked"] == []
    assert plan["lanes_not_applicable"] == [NEUTRAL_LANE]
    assert plan["findings"] == []


# --- AC-2: a lane on a host it is not declared for is a finding -----------------


def test_a_lane_running_on_an_undeclared_host_is_named() -> None:
    """A lane declared for host B, running on host A, is a finding."""
    lane = NEUTRAL_OTHER_HOST_LANE
    plan = PLAN.build_plan(
        _envelope(
            _HOST_A,
            _healthy_lane_rows(lane),
            [MANIFEST["lanes"][lane]["network"]],
            lane=lane,
        ),
        MANIFEST,
    )
    assert plan["lanes_not_applicable"] == [lane]
    undeclared = [f for f in plan["findings"] if f["kind"] == "lane_on_undeclared_host"]
    assert len(undeclared) == 1, plan["findings"]
    finding = undeclared[0]
    assert finding["lane"] == lane
    assert finding["severity"] == "critical"
    assert NEUTRAL_HOST_ID in finding["detail"]
    assert _runtime(lane) in finding["detail"]
    assert plan["has_drift"] is True


def test_the_finding_clears_when_the_declaration_is_corrected() -> None:
    lane = NEUTRAL_OTHER_HOST_LANE
    manifest = copy.deepcopy(MANIFEST)
    manifest["lanes"][lane]["hosts"].append(NEUTRAL_HOST_ID)
    plan = PLAN.build_plan(
        _envelope(
            _HOST_A,
            _healthy_lane_rows(lane),
            [manifest["lanes"][lane]["network"]],
            lane=lane,
        ),
        manifest,
    )
    assert plan["lanes_checked"] == [lane]
    assert plan["lanes_not_applicable"] == []
    assert plan["findings"] == []


def test_an_undeclared_lane_is_recognised_by_its_compose_project_alone() -> None:
    """A renamed container still carries the project it was started under."""
    row = _row(
        "some-renamed-runtime",
        labels={
            "com.docker.compose.project": MANIFEST["lanes"][NEUTRAL_LANE][
                "compose_project"
            ]
        },
    )
    plan = PLAN.build_plan(_envelope(_HOST_B, [row], []), MANIFEST)
    found = [
        f
        for f in plan["findings"]
        if f["kind"] == "lane_on_undeclared_host" and f["lane"] == NEUTRAL_LANE
    ]
    assert len(found) == 1, plan["findings"]
    assert "some-renamed-runtime" in found[0]["detail"]


def test_an_exited_leftover_is_not_a_lane_running_there() -> None:
    lane = NEUTRAL_OTHER_HOST_LANE
    rows = [
        _row(
            _runtime(lane),
            running=False,
            labels={
                "com.docker.compose.project": MANIFEST["lanes"][lane]["compose_project"]
            },
        )
    ]
    plan = PLAN.build_plan(_envelope(_HOST_A, rows, [], lane=lane), MANIFEST)
    assert plan["findings"] == []


# --- the shell driver refuses an undeclared host --------------------------------


def _run_driver(tmp_path: Path, hostname: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("docker", 'case "$*" in *"ps -a"*) : ;; *"network ls"*) : ;; esac; exit 0'),
        ("rpk", "exit 0"),
        ("hostname", f'echo "{hostname}"'),
    ):
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/lane-census-test.sock"
    env["HOME"] = str(tmp_path)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
    env.pop("LANE_MANIFEST", None)
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    env.pop("LANE_CENSUS_HOST", None)
    return subprocess.run(
        ["bash", str(_SCRIPT), "--json", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_the_driver_exits_5_on_a_host_the_manifest_does_not_declare(
    tmp_path: Path,
) -> None:
    proc = _run_driver(tmp_path, "docker-desktop")
    assert proc.returncode == 5, (proc.returncode, proc.stderr)
    assert "docker-desktop" in proc.stderr
    assert proc.stdout.strip() == "", "no plan may be emitted for an undeclared host"


def test_the_driver_scopes_to_the_host_it_runs_on(tmp_path: Path) -> None:
    """Positive control for the refusal above: a declared host yields a plan."""
    proc = _run_driver(tmp_path, _HOST_B)
    assert proc.returncode in (0, 30), (proc.returncode, proc.stderr)
    plan = json.loads(proc.stdout.splitlines()[0])
    assert plan["host"] == _HOST_B_ID
    assert sorted(plan["lanes_checked"]) == sorted(_LANES_B)


# --- the planner run on its own, as the manifest's hand recipe runs it ----------


def _run_planner(
    host: str, *, config_root: Path | None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["LANE_CENSUS_HOST"] = host
    env.pop("LANE_MANIFEST", None)
    env.pop("OMNI_HOME", None)
    if config_root is None:
        env.pop("ONEX_WORKSPACE_CONFIG_ROOT", None)
    else:
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(config_root)
    return subprocess.run(
        [sys.executable, str(_PLAN_PATH)],
        input=json.dumps({"lane": None, "containers": [], "networks": []}),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_the_planner_resolves_the_host_from_the_environment() -> None:
    proc = _run_planner(_HOST_B, config_root=NEUTRAL_CONFIG_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["host"] == _HOST_B_ID


def test_the_planner_exits_5_on_an_undeclared_host() -> None:
    proc = _run_planner("docker-desktop", config_root=NEUTRAL_CONFIG_ROOT)
    assert proc.returncode == PLAN.EXIT_HOST_UNDECLARED == 5
    assert "docker-desktop" in proc.stderr
    assert proc.stdout == ""


def test_the_planner_exits_3_with_no_manifest_to_read(tmp_path: Path) -> None:
    """OMN-20934: no config root, no manifest, no plan, and a loud reason."""
    proc = _run_planner(_HOST_B, config_root=None)
    assert proc.returncode == PLAN.EXIT_NOT_CONFIGURED == 3
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in proc.stderr
    assert proc.stdout == ""
    empty = _run_planner(_HOST_B, config_root=tmp_path)
    assert empty.returncode == 3
    assert "does not exist" in empty.stderr
