# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lane census is host-aware (OMN-19088).

Until this change the manifest had no notion of WHERE a lane runs, so the
census evaluated every lane on whatever host it was run on. Measured on the
.101 proof surface on 2026-09-21: one lane genuinely present and clean, and 25
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

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PLAN_PATH = _REPO / "scripts" / "lane_census_plan.py"
_MANIFEST_PATH = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"
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
MANIFEST = PLAN.load_manifest(_MANIFEST_PATH)

# The hostnames the three measured hosts report (read 2026-09-23 over read-only
# ssh: `hostname`). The .101 and .200 hosts are macOS and report a `.local`
# suffix; the runner container on .201 reports the daemon name instead, which
# is the same short name.
_HOST_201 = "omninode-pc"
_HOST_101 = "Stickybeatz.local"
_HOST_105 = "omnibook"

_LANES_201 = {
    "stability-test",
    "judge",
    "dev",
    "lakshman",
    "prepr-1",
    "prepr-2",
    "ci-bus",
}


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
        "Image": "omninode-runtime:dogfood",
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


def _dogfood_on(host: str) -> dict[str, Any]:
    return _envelope(
        host,
        _healthy_lane_rows("dogfood"),
        [MANIFEST["lanes"]["dogfood"]["network"]],
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


def test_the_201_lanes_and_the_dogfood_surfaces_are_declared_where_they_run() -> None:
    lanes = MANIFEST["lanes"]
    assert {lane for lane, s in lanes.items() if s["hosts"] == ["lab-201"]} == (
        _LANES_201
    )
    assert sorted(lanes["dogfood"]["hosts"]) == ["lab-101", "lab-105", "lab-200"]


def test_the_manifest_validates() -> None:
    PLAN.validate_hosts(MANIFEST)


def test_an_alias_claimed_by_two_hosts_is_refused() -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["hosts"]["lab-105"]["aliases"].append("omninode-pc")
    with pytest.raises(PLAN.HostDeclarationError, match="omninode-pc"):
        PLAN.validate_hosts(manifest)


def test_the_docker_desktop_daemon_name_is_refused_as_an_alias() -> None:
    """Every Docker Desktop host reports the daemon name ``docker-desktop``.

    Three of the four hosts run Docker Desktop, so that name identifies none of
    them; declaring it would attribute one host's census to another.
    """
    manifest = copy.deepcopy(MANIFEST)
    manifest["hosts"]["lab-101"]["aliases"].append("docker-desktop")
    with pytest.raises(PLAN.HostDeclarationError, match="docker-desktop"):
        PLAN.validate_hosts(manifest)


def test_a_lane_without_hosts_is_refused() -> None:
    manifest = copy.deepcopy(MANIFEST)
    del manifest["lanes"]["dogfood"]["hosts"]
    with pytest.raises(PLAN.HostDeclarationError, match="dogfood"):
        PLAN.validate_hosts(manifest)


def test_a_lane_naming_an_unregistered_host_is_refused() -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["lanes"]["dogfood"]["hosts"] = ["lab-999"]
    with pytest.raises(PLAN.HostDeclarationError, match="lab-999"):
        PLAN.validate_hosts(manifest)


# --- host resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_HOST_201, "lab-201"),
        ("omninode-pc.tail75df5e.ts.net", "lab-201"),
        (_HOST_101, "lab-101"),
        (_HOST_105, "lab-105"),
        ("Stickybeatz-Studio.local", "lab-200"),
        ("lab-105", "lab-105"),
        ("  OMNIBOOK  ", "lab-105"),
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


def test_the_101_replay_reports_only_its_own_lane() -> None:
    """The ticket's measurement, replayed: 25 criticals become zero.

    The inventory is the dogfood surface healthy and nothing else, which is what
    the .101 host carries. Before this change the same envelope produced a
    ``container_absent`` for every .201 service and a ``network_detached`` for
    every .201 network.
    """
    plan = PLAN.build_plan(_dogfood_on(_HOST_101), MANIFEST)
    assert plan["host"] == "lab-101"
    assert plan["lanes_checked"] == ["dogfood"]
    assert set(plan["lanes_not_applicable"]) == _LANES_201
    assert plan["findings"] == []
    assert plan["has_drift"] is False


def test_checked_and_not_applicable_partition_the_manifest() -> None:
    plan = PLAN.build_plan(_dogfood_on(_HOST_105), MANIFEST)
    checked = set(plan["lanes_checked"])
    not_applicable = set(plan["lanes_not_applicable"])
    assert checked.isdisjoint(not_applicable)
    assert checked | not_applicable == set(MANIFEST["lanes"])


def test_two_hosts_each_return_findings_only_for_their_own_lanes() -> None:
    """AC-1's falsifier: two hosts, and each answers only for itself.

    Both inventories are EMPTY, the harshest input: every declared service is
    absent. The .201 run must file its criticals against .201 lanes only, and
    the .105 run against no .201 lane at all.
    """
    plan_201 = PLAN.build_plan(_envelope(_HOST_201, [], []), MANIFEST)
    plan_105 = PLAN.build_plan(_envelope(_HOST_105, [], []), MANIFEST)

    lanes_201 = {f["lane"] for f in plan_201["findings"]}
    assert lanes_201, "positive control: an empty .201 inventory must be drift"
    assert lanes_201 <= _LANES_201
    assert "dogfood" in plan_201["lanes_not_applicable"]

    assert {f["lane"] for f in plan_105["findings"]} <= {"dogfood"}
    assert set(plan_105["lanes_not_applicable"]) == _LANES_201


def test_a_stopped_container_in_a_declared_lane_is_still_critical() -> None:
    """AC-1's negative control: scoping must not hide a real outage."""
    rows = [
        r
        for r in _healthy_lane_rows("dogfood")
        if r["Names"] != "omninode-dogfood-runtime"
    ]
    rows.append(_row("omninode-dogfood-runtime", running=False, exit_code=137))
    plan = PLAN.build_plan(
        _envelope(_HOST_101, rows, [MANIFEST["lanes"]["dogfood"]["network"]]),
        MANIFEST,
    )
    absent = [
        f
        for f in plan["findings"]
        if f["kind"] == "container_absent"
        and f["container"] == "omninode-dogfood-runtime"
    ]
    assert len(absent) == 1, plan["findings"]
    assert absent[0]["severity"] == "critical"
    assert absent[0]["lane"] == "dogfood"


def test_a_requested_lane_off_this_host_is_not_applicable() -> None:
    plan = PLAN.build_plan(
        _envelope(_HOST_101, [], [], lane="stability-test"), MANIFEST
    )
    assert plan["lanes_checked"] == []
    assert plan["lanes_not_applicable"] == ["stability-test"]
    assert plan["findings"] == []


# --- AC-2: a lane on a host it is not declared for is a finding -----------------


def test_a_lane_running_on_an_undeclared_host_is_named() -> None:
    """dogfood is not declared for .201; a dogfood stack there is a finding."""
    plan = PLAN.build_plan(
        _envelope(
            _HOST_201,
            _healthy_lane_rows("dogfood"),
            [MANIFEST["lanes"]["dogfood"]["network"]],
            lane="dogfood",
        ),
        MANIFEST,
    )
    assert plan["lanes_not_applicable"] == ["dogfood"]
    undeclared = [f for f in plan["findings"] if f["kind"] == "lane_on_undeclared_host"]
    assert len(undeclared) == 1, plan["findings"]
    finding = undeclared[0]
    assert finding["lane"] == "dogfood"
    assert finding["severity"] == "critical"
    assert "lab-201" in finding["detail"]
    assert "omninode-dogfood-runtime" in finding["detail"]
    assert plan["has_drift"] is True


def test_the_finding_clears_when_the_declaration_is_corrected() -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["lanes"]["dogfood"]["hosts"].append("lab-201")
    plan = PLAN.build_plan(
        _envelope(
            _HOST_201,
            _healthy_lane_rows("dogfood"),
            [manifest["lanes"]["dogfood"]["network"]],
            lane="dogfood",
        ),
        manifest,
    )
    assert plan["lanes_checked"] == ["dogfood"]
    assert plan["lanes_not_applicable"] == []
    assert plan["findings"] == []


def test_an_undeclared_lane_is_recognised_by_its_compose_project_alone() -> None:
    """A renamed container still carries the project it was started under."""
    row = _row(
        "some-renamed-runtime",
        labels={"com.docker.compose.project": "omnibase-infra-judge"},
    )
    plan = PLAN.build_plan(_envelope(_HOST_105, [row], []), MANIFEST)
    judge = [
        f
        for f in plan["findings"]
        if f["kind"] == "lane_on_undeclared_host" and f["lane"] == "judge"
    ]
    assert len(judge) == 1, plan["findings"]
    assert "some-renamed-runtime" in judge[0]["detail"]


def test_an_exited_leftover_is_not_a_lane_running_there() -> None:
    rows = [
        _row(
            "omninode-dogfood-runtime",
            running=False,
            labels={"com.docker.compose.project": "omnibase-infra-dogfood"},
        )
    ]
    plan = PLAN.build_plan(_envelope(_HOST_201, rows, [], lane="dogfood"), MANIFEST)
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
    proc = _run_driver(tmp_path, _HOST_105)
    assert proc.returncode in (0, 30), (proc.returncode, proc.stderr)
    plan = json.loads(proc.stdout.splitlines()[0])
    assert plan["host"] == "lab-105"
    assert plan["lanes_checked"] == ["dogfood"]


# --- the planner run on its own, as the manifest's hand recipe runs it ----------


def _run_planner(host: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["LANE_CENSUS_HOST"] = host
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
    proc = _run_planner(_HOST_105)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["host"] == "lab-105"


def test_the_planner_exits_5_on_an_undeclared_host() -> None:
    proc = _run_planner("docker-desktop")
    assert proc.returncode == PLAN.EXIT_HOST_UNDECLARED == 5
    assert "docker-desktop" in proc.stderr
    assert proc.stdout == ""
