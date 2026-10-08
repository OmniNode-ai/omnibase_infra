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


def test_collector_inspects_health_and_restarts(
    inventory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def read(socket: str, path: str, timeout: float) -> Any:
        if path == inventory.API_CONTAINERS_PATH:
            return [
                {"Names": ["/runtime"], "State": "running", "Status": "Up (unhealthy)"}
            ]
        if path == inventory.API_NETWORKS_PATH:
            return []
        assert path == "/containers/runtime/json"
        return {
            "State": {
                "Health": {"Status": "unhealthy", "FailingStreak": 61, "Log": []}
            },
            "Config": {
                "Healthcheck": {"Interval": 30_000_000_000},
                "Env": ["SECRET=private"],
            },
            "RestartCount": 6,
        }

    monkeypatch.setattr(inventory, "api_get", read)
    rows, _, source, _ = inventory.collect_inventory(
        socket_path="unused", api_timeout_s=1, cli_timeout_s=1
    )
    assert source == "engine_api"
    assert rows[0]["Health"] == {"Status": "unhealthy", "FailingStreak": 61}
    assert rows[0]["HealthcheckIntervalSeconds"] == 30
    assert rows[0]["RestartCount"] == 6
    assert "SECRET" not in json.dumps(rows)


def test_collector_cli_fallback_reads_same_health_fields(
    inventory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> Any:
        raise inventory.InventoryProbeError("socket unreadable")

    def cli(argv: list[str], timeout: float) -> str:
        if argv[1] == "ps":
            return "runtime\trunning\tUp (unhealthy)\timage:tag\t\n"
        if argv[1] == "network":
            return "network\n"
        assert argv[1] == "inspect"
        assert argv[-1] == "runtime"
        assert ".Config.Env" not in argv[3]
        return json.dumps(
            {
                "Names": "/runtime",
                "State": {"Health": {"Status": "unhealthy", "FailingStreak": 60}},
                "Config": {"Healthcheck": {"Interval": 30_000_000_000}},
                "RestartCount": 3,
            }
        )

    monkeypatch.setattr(inventory, "api_get", fail)
    monkeypatch.setattr(inventory, "_run_cli", cli)
    rows, networks, source, warnings = inventory.collect_inventory(
        socket_path="unused", api_timeout_s=1, cli_timeout_s=1
    )
    assert source == "docker_cli"
    assert networks == ["network"]
    assert warnings
    assert rows[0]["HealthcheckIntervalSeconds"] == 30
    assert rows[0]["Health"]["FailingStreak"] == 60
    assert rows[0]["RestartCount"] == 3


def test_missing_cli_inspect_reading_is_not_a_clean_inventory(
    inventory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> Any:
        raise inventory.InventoryProbeError("socket unreadable")

    def cli(argv: list[str], timeout: float) -> str:
        return "runtime\trunning\tUp\timage:tag\t\n" if argv[1] == "ps" else ""

    monkeypatch.setattr(inventory, "api_get", fail)
    monkeypatch.setattr(inventory, "_run_cli", cli)
    with pytest.raises(inventory.InventoryProbeError, match="no reading"):
        inventory.collect_inventory(
            socket_path="unused", api_timeout_s=1, cli_timeout_s=1
        )


def test_inspect_uses_docker_default_interval_when_zero(inventory: Any) -> None:
    readings = inventory._inspect_readings(
        {
            "State": {"Health": {"Status": "unhealthy", "FailingStreak": 60}},
            "Config": {"Healthcheck": {"Interval": 0}},
            "RestartCount": 0,
        }
    )
    assert readings["HealthcheckIntervalSeconds"] == 30


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
        {
            "lane": None,
            # OMN-19088: the planner scopes to the lanes declared for this host.
            "host": "omninode-pc",
            "containers": [],
            "networks": [],
            "runtime_tag": None,
        },
        manifest,
    )
    assert plan["has_drift"] is True
    # OMN-16803 moved this floor from 30 to 20. Eight declared `kind: service`
    # entries (four on stability-test, four on prod) were reclassified
    # `profile_gated` because the lane compose files disable them via profile
    # overrides — they can never run there, so their eight "absent" findings were
    # permanent FALSE criticals padding this count. The signal being pinned here
    # is unchanged: an empty envelope must still produce a large, uniformly
    # critical, multi-lane finding set, never a quiet zero.
    assert len(plan["findings"]) >= 20
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

    Equality (not subset) is the right bar, and it holds at host scale: running
    both paths back-to-back on `.201` at 2026-07-30T05:45Z, over all 91
    containers then on the host, produced zero mismatches in `Labels`, `State`,
    `Image`, or `Status`.
    """
    api_rows = json.loads((_FIXTURES / "engine_api_containers.json").read_text())
    cli_text = (_FIXTURES / "docker_cli_containers.tsv").read_text()

    from_api = inventory.normalize_api_containers(api_rows)
    from_cli = inventory.normalize_cli_containers(cli_text)

    assert len(from_api) == 4
    assert [r["Names"] for r in from_api] == [r["Names"] for r in from_cli]
    for api_row, cli_row in zip(from_api, from_cli, strict=True):
        # Full equality on every field the planner reads, Labels included.
        # A subset check (`api Labels ⊆ cli Labels`) would let the CLI path
        # emit spurious extra keys unnoticed; the two paths must be
        # interchangeable, so nothing weaker than equality is correct here.
        for field in inventory._CLI_FIELDS:
            assert api_row[field] == cli_row[field], field
        assert api_row == cli_row


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
    # OMN-19088: the collector does not name the host; the driver does.
    plan = planner.build_plan(
        {**envelope, "host": "omninode-pc"}, planner.load_manifest()
    )
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
        "for _ in range(2 + len(payload['containers'])):\n"
        "    conn, _addr = srv.accept()\n"
        "    req = conn.recv(65536).decode('utf-8', 'replace')\n"
        "    if '/containers/json' in req:\n"
        "        result = payload['containers']\n"
        "    elif '/networks' in req:\n"
        "        result = payload['networks']\n"
        "    else:\n"
        "        result = {'State': {'Health': None}, 'Config': {}, 'RestartCount': 0}\n"
        "    body = json.dumps(result).encode()\n"
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


# ---------------------------------------------------------------------------
# OMN-19959 — the memory observation, read in the same pass
# ---------------------------------------------------------------------------

_MEMORY_FIXTURES = (
    Path(__file__).resolve().parent / "fixtures" / "lane_container_memory"
)


def _fake_engine() -> Any:
    name = "fake_docker_engine_omn19959"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / f"{name}.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        # Registered before exec: a dataclass resolves its module by name.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_runner_prefixes_come_from_every_host_in_the_fleet_config(
    inventory: Any,
) -> None:
    """The prefix set runner-monitor.sh reads, nested role pools included (OMN-19842)."""
    import yaml

    config = yaml.safe_load((_MEMORY_FIXTURES / "runner_fleet.yaml").read_text())
    assert inventory.runner_prefixes_from_fleet_config(config) == [
        "omninode-runner",
        "omnipc2-ci-runner",
        "omnipc2-customer-plane-runner",
        "omnipc2-verify-runner",
    ]


def test_lane_projects_map_compose_projects_to_lane_names(inventory: Any) -> None:
    import yaml

    manifest = yaml.safe_load((_MEMORY_FIXTURES / "lane-manifest.yaml").read_text())
    assert inventory.lane_projects_from_manifest(manifest) == {
        "omnibase-infra-sim-202": "sim-202"
    }


def _memory_run(
    tmp_path: Path, host: Any, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    out = tmp_path / "memory.json"
    bin_dir = tmp_path / "journal-bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        journal = tmp_path / "journal.txt"
        if not journal.exists():
            journal.write_text("")
        _fake_engine().write_journalctl_stub(bin_dir, journal)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.update(host.env())
    env["LANE_MANIFEST"] = str(_MEMORY_FIXTURES / "lane-manifest.yaml")
    env["LANE_MEMORY_RUNNER_FLEET_CONFIG"] = str(_MEMORY_FIXTURES / "runner_fleet.yaml")
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(_INVENTORY_PATH), "--memory-out", str(out)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def _memory_host(tmp_path: Path, *, diag_readable: bool = True) -> Any:
    fake = _fake_engine()
    containers = [
        fake.FakeContainer(
            cid="a" * 64,
            name="omnibase-infra-sim-202-redpanda",
            pid=5101,
            project=fake.SIM_PROJECT,
        ),
        fake.FakeContainer(
            cid="b" * 64,
            name="some-unrelated-container",
            pid=5102,
            project="not-a-lane",
        ),
        fake.FakeContainer(
            cid="c" * 64,
            name="omnipc2-ci-runner-13",
            pid=5103,
            project="omnipc2-ci-runner",
            diag_readable=diag_readable,
            worker_logs={
                # Written after boot: read.
                "Worker_20260928-185104-utc.log": (
                    fake.worker_log(
                        repo="OmniNode-ai/omnimarket",
                        run_id="36454760449",
                        started="2026-09-28 18:51:04Z",
                        completed="2026-09-28 18:51:33Z",
                    ),
                    fake.BOOT_EPOCH + 3600,
                ),
                # Last written before this boot: not read.
                "Worker_20260927-100000-utc.log": ("old\n", fake.BOOT_EPOCH - 3600),
            },
        ),
    ]
    return fake.FakeHost(tmp_path / "host", containers)


def test_memory_observation_reads_lane_counters_and_runner_worker_logs(
    tmp_path: Path,
) -> None:
    host = _memory_host(tmp_path)
    try:
        result = _memory_run(tmp_path, host)
    finally:
        host.close()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["inventory_source"] == "engine_api", (
        "the census envelope must still be printed unchanged"
    )
    observation = json.loads((tmp_path / "memory.json").read_text())
    assert observation["host_boot_id"] == _fake_engine().BOOT_ID
    assert observation["boot_time"] == "2026-09-28T12:10:00Z"
    names = [c["container_name"] for c in observation["containers"]]
    assert names == ["omnibase-infra-sim-202-redpanda"], (
        "only lane containers carry a record; runners and unrelated containers do not"
    )
    redpanda = observation["containers"][0]
    assert redpanda["lane"] == "sim-202"
    assert redpanda["memory_max"].strip() == "max"
    assert "oom_kill 0" in redpanda["memory_events"]
    # The pre-boot log is not read: one run, parsed as the archive streamed.
    assert observation["worker_runs"] == [
        {
            "repo": "OmniNode-ai/omnimarket",
            "run_id": "36454760449",
            "runner_name": "omnipc2-ci-runner-13",
            "job_started_at": "2026-09-28T18:51:04.000000Z",
            "job_completed_at": "2026-09-28T18:51:33.000000Z",
        }
    ]


def test_an_unreadable_counter_exits_7_and_keeps_the_census_envelope(
    tmp_path: Path, inventory: Any
) -> None:
    host = _memory_host(tmp_path)
    try:
        (host.sysfs_root / f"system.slice/docker-{'a' * 64}.scope/memory.peak").unlink()
        result = _memory_run(tmp_path, host)
    finally:
        host.close()
    assert result.returncode == inventory.EXIT_MEMORY_UNOBSERVABLE == 7
    assert json.loads(result.stdout)["containers"], "the census envelope was lost"
    assert "memory.peak" in result.stderr


def test_an_unreadable_runner_diag_is_an_error_never_an_empty_list(
    tmp_path: Path, inventory: Any
) -> None:
    host = _memory_host(tmp_path, diag_readable=False)
    try:
        result = _memory_run(tmp_path, host)
    finally:
        host.close()
    assert result.returncode == inventory.EXIT_MEMORY_UNOBSERVABLE, result.stderr
    assert "omnipc2-ci-runner-13" in result.stderr


def test_an_absent_fleet_config_is_an_error_unless_declared_empty(
    tmp_path: Path, inventory: Any
) -> None:
    host = _memory_host(tmp_path)
    try:
        absent = _memory_run(
            tmp_path,
            host,
            {"LANE_MEMORY_RUNNER_FLEET_CONFIG": str(tmp_path / "no-such.yaml")},
        )
        declared = _memory_run(tmp_path, host, {"LANE_MEMORY_RUNNER_FLEET_CONFIG": ""})
    finally:
        host.close()
    assert absent.returncode == inventory.EXIT_MEMORY_UNOBSERVABLE, absent.stderr
    assert declared.returncode == 0, declared.stderr
    observation = json.loads((tmp_path / "memory.json").read_text())
    assert observation["worker_runs"] == []


def test_journal_oom_kills_are_attributed_to_lane_containers_only(
    tmp_path: Path,
) -> None:
    """A lane container's kill is named; a runner's belongs to the runner monitor."""
    fake = _fake_engine()
    host = _memory_host(tmp_path)
    (tmp_path / "journal.txt").write_text(
        fake.oom_kill_line("a" * 64, fake.BOOT_EPOCH + 60)
        + fake.oom_kill_line("a" * 64, fake.BOOT_EPOCH + 61)
        + fake.oom_kill_line("c" * 64, fake.BOOT_EPOCH + 62, "omnirunners.slice")
    )
    try:
        result = _memory_run(tmp_path, host)
    finally:
        host.close()
    assert result.returncode == 0, result.stderr
    observation = json.loads((tmp_path / "memory.json").read_text())
    assert observation["journal_oom_kills"] == [
        {
            "container_id": "a" * 64,
            "container_name": "omnibase-infra-sim-202-redpanda",
            "lane": "sim-202",
            "count": 2,
        }
    ]


def test_an_unreadable_kernel_journal_is_an_error_never_zero_kills(
    tmp_path: Path, inventory: Any
) -> None:
    """A user without journal access reads nothing and exit 0; that is refused."""
    host = _memory_host(tmp_path)
    bin_dir = tmp_path / "journal-bin"
    bin_dir.mkdir()
    blind = bin_dir / "journalctl"
    blind.write_text("#!/usr/bin/env bash\nexit 0\n")
    blind.chmod(0o755)
    try:
        result = _memory_run(tmp_path, host)
    finally:
        host.close()
    assert result.returncode == inventory.EXIT_MEMORY_UNOBSERVABLE, result.stderr
    assert "kernel journal reads empty" in result.stderr


def test_the_journal_names_a_container_under_either_cgroup_driver(
    inventory: Any,
) -> None:
    systemd = "task_memcg=/system.slice/docker-" + "a" * 64 + ".scope,task=python"
    cgroupfs = "task_memcg=/docker/" + "b" * 64 + ",task=python"
    assert inventory._OOM_KILL_MEMCG.search(systemd).group(1) == "a" * 64
    assert inventory._OOM_KILL_MEMCG.search(cgroupfs).group(1) == "b" * 64


def test_a_completed_log_naming_no_job_is_reported_not_fatal(tmp_path: Path) -> None:
    """Failing the pass would hold the window open and re-read the log forever."""
    fake = _fake_engine()
    host = _memory_host(tmp_path)
    host.containers[2].worker_logs["Worker_20260928-183000-utc.log"] = (
        "[2026-09-28 18:30:00Z INFO HostContext] start\n"
        "[2026-09-28 18:31:00Z INFO Worker] Job completed.\n",
        fake.BOOT_EPOCH + 7200,
    )
    try:
        result = _memory_run(tmp_path, host)
    finally:
        host.close()
    assert result.returncode == 0, result.stderr
    observation = json.loads((tmp_path / "memory.json").read_text())
    assert observation["unnamed_worker_logs"] == [
        {
            "runner_name": "omnipc2-ci-runner-13",
            "log_name": "Worker_20260928-183000-utc.log",
        }
    ]
    assert len(observation["worker_runs"]) == 1
