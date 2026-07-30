# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Inventory-probe tests for the lane census (OMN-15466).

These pin the two defects reproduced live on ``.201`` on 2026-07-30 against the
census's own inventory line, plus the repair's contract.

D1 — SIZE ON THE CRITICAL PATH. ``docker ps -a --no-trunc --format '{{json .}}'``
     carries a ``Size`` field, so the CLI sends ``size=1`` and the daemon runs
     ``snapshotter.Usage`` per container. Measured with 111 containers:

       docker ps -a --no-trunc --format '{{json .}}'   90.363 s   <- the census
       docker ps -a --format '{{.ID}}'                  0.128 s
       docker ps -a --size --format '{{.ID}}'          75.509 s  (rc=1, hard fail)
       GET /containers/json?all=1                       0.150 s   <- the repair
       GET /containers/json?all=1&size=1               57.265 s

     Transport is not the variable; ``size`` is. So the tests assert on the
     ABSENCE of size-triggering forms on BOTH paths, not on "uses the API".

D2 — FAIL-OPEN. ``2>/dev/null || : >ps.ndjson`` turned any docker failure into a
     fabricated empty inventory, which the planner renders as 32 critical
     findings across all four lanes and publishes as genuine drift. A probe that
     cannot see must not be reported as a probe that saw nothing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_INVENTORY_PATH = _REPO / "scripts" / "lane_census_inventory.py"
_PLAN_PATH = _REPO / "scripts" / "lane_census_plan.py"
_SCRIPT = _REPO / "scripts" / "lane-census-check.sh"
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lane_census"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory() -> Any:
    return _load("lane_census_inventory", _INVENTORY_PATH)


@pytest.fixture(scope="module")
def planner() -> Any:
    return _load("lane_census_plan", _PLAN_PATH)


# ---------------------------------------------------------------------------
# D1 — no size on either path (acceptance criteria 1 + 2)
# ---------------------------------------------------------------------------


def test_engine_api_path_never_requests_size(inventory: Any) -> None:
    """The Engine API inventory URLs must carry no `size` parameter."""
    for path in (inventory.API_CONTAINERS_PATH, inventory.API_NETWORKS_PATH):
        assert "size=1" not in path
        assert "size=true" not in path
        assert "size" not in path.lower(), (
            f"{path!r} requests container size — that forces daemon-side "
            "snapshotter.Usage per container (90 s vs 0.15 s on .201)"
        )


def test_cli_fallback_format_is_not_json_dot(inventory: Any) -> None:
    """`{{json .}}` emits a Size field and silently opts into size=1."""
    fmt = inventory.CLI_CONTAINER_FORMAT
    assert "{{json .}}" not in fmt, (
        "the CLI fallback re-introduced the size-triggering format; enumerate "
        "the consumed fields explicitly instead"
    )
    assert ".Size" not in fmt
    # It must still carry every field the planner reads.
    for field in ("Names", "State", "Status", "Image", "Labels"):
        assert f"{{{{.{field}}}}}" in fmt


def _executable_lines(path: Path) -> list[str]:
    """Shell lines with comments stripped.

    The driver deliberately *documents* the removed size-triggering command in a
    comment so nobody reinstates it; only executable lines are asserted on.
    """
    return [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_census_driver_no_longer_uses_the_size_triggering_command() -> None:
    """RED before OMN-15466: the driver's inventory line used `{{json .}}`."""
    executable = "\n".join(_executable_lines(_SCRIPT))
    assert "{{json .}}" not in executable, (
        "lane-census-check.sh still gathers the inventory with a size-triggering "
        "format — measured 90.363 s vs 0.150 s on .201"
    )
    assert "docker ps" not in executable, (
        "the driver still shells out to `docker ps` directly; the inventory must "
        "go through the fail-loud collector"
    )


def test_census_driver_bounds_the_cli_fallback(inventory: Any) -> None:
    """The CLI fallback must be bounded; an unbounded docker call can hang forever."""
    source = _INVENTORY_PATH.read_text()
    # The bound must be the portable one: coreutils `timeout(1)` is absent on
    # macOS and the gate/push host is a Mac, so a `timeout`-only bound would be
    # no bound at all there.
    assert "timeout=timeout_s" in source, "the CLI fallback has no portable bound"
    assert "subprocess.TimeoutExpired" in source, "a fallback timeout is not handled"
    assert inventory.DEFAULT_CLI_TIMEOUT_S > 0
    assert inventory.DEFAULT_API_TIMEOUT_S > 0


# ---------------------------------------------------------------------------
# D2 — fail loud, never fabricate an empty inventory (criteria 3 + 4)
# ---------------------------------------------------------------------------


def test_driver_no_longer_truncates_inventory_on_error() -> None:
    """RED before OMN-15466: `|| : >"$SCRATCH/ps.ndjson"` fabricated an empty host."""
    executable = "\n".join(_executable_lines(_SCRIPT))
    assert ': >"$SCRATCH/ps.ndjson"' not in executable
    assert ': >"$SCRATCH/networks.txt"' not in executable
    assert "2>/dev/null || :" not in executable, (
        "a docker failure is still being converted into an empty inventory"
    )


def test_probe_failure_exits_distinctly_from_drift(
    inventory: Any, tmp_path: Path
) -> None:
    """Both paths unavailable => exit 4, no envelope on stdout."""
    env = dict(os.environ)
    # Point at a socket that does not exist and strip docker from PATH so the
    # CLI fallback cannot succeed either.
    env["LANE_CENSUS_DOCKER_SOCKET"] = str(tmp_path / "absent.sock")
    env["PATH"] = str(tmp_path / "empty-bin")
    (tmp_path / "empty-bin").mkdir()

    proc = subprocess.run(
        [sys.executable, str(_INVENTORY_PATH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == inventory.EXIT_PROBE_FAILED == 4
    assert proc.returncode != 30, "probe failure must not be reported as drift"
    assert proc.returncode != 0
    assert proc.stdout.strip() == "", (
        "an unobservable host emitted an envelope; the planner would read it as "
        "a total outage"
    )
    assert "FAILED" in proc.stderr


def test_empty_inventory_is_the_total_outage_signal(planner: Any) -> None:
    """Why D2 mattered: an empty envelope is indistinguishable from a dead host.

    This pins the blast radius the fail-open produced, so nobody reintroduces a
    "just default to empty" shortcut.
    """
    manifest = planner.load_manifest()
    plan = planner.build_plan(
        {"lane": None, "containers": [], "networks": [], "runtime_tag": None},
        manifest,
    )
    assert plan["has_drift"] is True
    assert len(plan["findings"]) >= 30
    assert {f["severity"] for f in plan["findings"]} == {"critical"}


# ---------------------------------------------------------------------------
# Normalization equivalence against RECORDED REAL responses (criterion 5)
# ---------------------------------------------------------------------------


def test_api_and_cli_normalize_to_identical_envelopes(inventory: Any) -> None:
    """Recorded real `.201` responses for the SAME four containers must agree.

    Fixtures were captured live from `omninode-pc` on 2026-07-30 — an Engine API
    `GET /containers/json?all=1` response and the tab-delimited CLI rows for the
    same containers (two running stability-test services, two exited dev
    one-shots).
    """
    api_rows = json.loads((_FIXTURES / "engine_api_containers.json").read_text())
    cli_text = (_FIXTURES / "docker_cli_containers.tsv").read_text()

    from_api = inventory.normalize_api_containers(api_rows)
    from_cli = inventory.normalize_cli_containers(cli_text)

    assert len(from_api) == 4
    assert [r["Names"] for r in from_api] == [r["Names"] for r in from_cli]
    for api_row, cli_row in zip(from_api, from_cli, strict=True):
        for field in ("Names", "State", "Status", "Image"):
            assert api_row[field] == cli_row[field], field
        # The CLI carries strictly more labels (compose internals); every label
        # the API reports must match the CLI's value exactly.
        for key, value in api_row["Labels"].items():
            assert cli_row["Labels"][key] == value, key


def test_cli_label_parser_preserves_commas_inside_values(inventory: Any) -> None:
    """`com.docker.compose.project.config_files` routinely contains commas.

    A flat `split(",")` corrupts it. The Engine API path avoids the ambiguity
    entirely; the CLI parser must reconstruct it.
    """
    raw = (
        "com.omninode.lane=dev,"
        "com.docker.compose.project.config_files=/a/docker-compose.infra.yml,"
        "/a/docker-compose.dev-lane.yml,"
        "com.omninode.service=forward-migration"
    )
    labels = inventory.parse_cli_labels(raw)
    assert labels["com.omninode.lane"] == "dev"
    assert labels["com.omninode.service"] == "forward-migration"
    assert labels["com.docker.compose.project.config_files"] == (
        "/a/docker-compose.infra.yml,/a/docker-compose.dev-lane.yml"
    )


def test_recorded_api_envelope_drives_the_planner(inventory: Any, planner: Any) -> None:
    """End-to-end: recorded API rows -> normalized envelope -> planner, no crash.

    Guards the seam directly: the planner must accept the mapping-typed Labels
    the API path produces (it previously assumed a comma-joined string).
    """
    api_rows = json.loads((_FIXTURES / "engine_api_containers.json").read_text())
    envelope = inventory.build_envelope(
        lane=None,
        runtime_tag=None,
        containers=inventory.normalize_api_containers(api_rows),
        networks=["omnibase-infra-stability-test-network"],
        source="engine_api",
    )
    plan = planner.build_plan(envelope, planner.load_manifest())
    assert plan["schema_version"]
    # The lane label on the recorded rows must be read through the mapping form.
    labeled = [
        c for c in envelope["containers"] if c["Labels"].get("com.omninode.lane")
    ]
    assert labeled, "fixture lost its com.omninode.lane labels"


def test_labels_mapping_and_string_forms_agree(planner: Any) -> None:
    """The planner's label coercion accepts both the API and CLI forms."""
    as_map = planner._labels_to_dict({"com.omninode.lane": "prod", "a": "b"})
    as_str = planner._labels_to_dict("com.omninode.lane=prod,a=b")
    assert as_map == as_str == {"com.omninode.lane": "prod", "a": "b"}


# ---------------------------------------------------------------------------
# Engine API is genuinely preferred when the socket answers
# ---------------------------------------------------------------------------


def _fake_docker_socket(tmp_path: Path, containers: list[dict[str, Any]]) -> Path:
    """Serve one canned /containers/json and /networks response over AF_UNIX.

    The socket lives in a short mkdtemp path, not pytest's tmp_path: macOS caps
    AF_UNIX paths near 104 bytes and pytest's fixture paths exceed that, which
    would silently push every assertion onto the CLI fallback.
    """
    sock_dir = Path(tempfile.mkdtemp(prefix="lc-"))
    sock_path = sock_dir / "d.sock"
    server = tmp_path / "server.py"
    server.write_text(
        "import json, socket, sys\n"
        "sock_path = sys.argv[1]\n"
        "payload = json.load(open(sys.argv[2]))\n"
        "srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "srv.bind(sock_path)\n"
        "srv.listen(8)\n"
        "for _ in range(2):\n"
        "    conn, _addr = srv.accept()\n"
        "    req = conn.recv(65536).decode('utf-8', 'replace')\n"
        "    body = json.dumps(\n"
        "        payload['containers'] if '/containers/json' in req else payload['networks']\n"
        "    ).encode()\n"
        "    conn.sendall(\n"
        "        b'HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\n'\n"
        "        b'Content-Length: ' + str(len(body)).encode() + b'\\r\\n\\r\\n' + body\n"
        "    )\n"
        "    conn.close()\n"
    )
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "containers": containers,
                "networks": [{"Name": "omnibase-infra-prod-network"}],
            }
        )
    )
    proc = subprocess.Popen([sys.executable, str(server), str(sock_path), str(payload)])
    for _ in range(200):
        if sock_path.exists():
            break
        time.sleep(0.05)
    assert sock_path.exists(), "fake docker socket never came up"
    return sock_path, proc  # type: ignore[return-value]


def test_engine_api_is_used_when_the_socket_answers(
    inventory: Any, tmp_path: Path
) -> None:
    """With a live socket the CLI is never invoked — and no size is requested."""
    sock_path, proc = _fake_docker_socket(  # type: ignore[misc]
        tmp_path,
        [
            {
                "Names": ["/omnibase-infra-prod-postgres"],
                "State": "running",
                "Status": "Up 2 hours",
                "Image": "postgres:16-alpine",
                "Labels": {"com.omninode.lane": "prod"},
            }
        ],
    )
    try:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        calllog = tmp_path / "calls.log"
        shim = bin_dir / "docker"
        shim.write_text(
            f'#!/usr/bin/env bash\necho "docker $*" >> "{calllog}"\nexit 0\n'
        )
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        env = dict(os.environ)
        env["LANE_CENSUS_DOCKER_SOCKET"] = str(sock_path)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"

        result = subprocess.run(
            [sys.executable, str(_INVENTORY_PATH)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        envelope = json.loads(result.stdout)
        assert envelope["inventory_source"] == "engine_api"
        assert envelope["containers"][0]["Names"] == "omnibase-infra-prod-postgres"
        assert envelope["networks"] == ["omnibase-infra-prod-network"]
        assert not calllog.exists(), "docker CLI was invoked despite a live socket"
    finally:
        proc.kill()
