# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Parsing and aggregation of the lab-tenant benchmark harness (OMN-20213).

The harness's I/O runs on lab hosts and inside runtime containers; what these
tests pin is everything between the raw tool output and the numbers that end
up in the before/after table: every parser against output captured from the
h105 satellite and the .201 dependency host on 2026-09-30, the percentile
arithmetic, the median/spread aggregation, the per-arm names, and host
resolution failing fast instead of guessing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "bench_lab_tenant.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_lab_tenant", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load()

pytestmark = pytest.mark.unit

# Captured from h105 (omnibook), 2026-09-30T23:47Z.
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    78446.
Pages active:                                 664043.
Pages inactive:                               666942.
Pages speculative:                             13077.
Pages throttled:                                   0.
Pages wired down:                             140886.
Pages purgeable:                                1695.
"Translation faults":                     1565215115.
Pages stored in compressor:                  1403831.
Pages occupied by compressor:                 493433.
Swapouts:                                     461826.
"""


def test_vm_stat_uses_header_page_size_and_placement_available() -> None:
    out = bench.parse_vm_stat(VM_STAT)
    page = 16384
    assert out["page_size"] == page
    assert out["free_bytes"] == 78446 * page
    assert out["compressed_bytes"] == 493433 * page
    assert out["wired_bytes"] == 140886 * page
    assert out["stored_in_compressor_bytes"] == 1403831 * page
    # landing_placement's mem_avail is free + inactive + speculative.
    assert out["available_bytes"] == (78446 + 666942 + 13077) * page


def test_vm_stat_other_page_size() -> None:
    text = VM_STAT.replace("16384", "4096")
    assert bench.parse_vm_stat(text)["free_bytes"] == 78446 * 4096


def test_swapusage() -> None:
    out = bench.parse_swapusage(
        "vm.swapusage: total = 4096.00M  used = 3136.56M  free = 959.44M  (encrypted)"
    )
    assert out == {"total_mib": 4096.0, "used_mib": 3136.56, "free_mib": 959.44}
    assert bench.parse_swapusage("vm.swapusage: total = 1.50G  used = 0.00M")[
        "total_mib"
    ] == pytest.approx(1536.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("658.4MiB", int(658.4 * 1024**2)),
        ("15.6GiB", int(15.6 * 1024**3)),
        ("16G", 16 * 1024**3),
        ("512M+", 512 * 1024**2),
        ("274MB", 274 * 1000**2),
        ("0B", 0),
        ("12", 12),
        ("n/a", None),
        ("", None),
        ("3XB", None),
    ],
)
def test_parse_size(text: str, expected: int | None) -> None:
    assert bench.parse_size(text) == expected


def test_docker_mem_usage_and_percent() -> None:
    used, limit = bench.parse_docker_mem_usage("478.5MiB / 1.5GiB")
    assert used == int(478.5 * 1024**2)
    assert limit == int(1.5 * 1024**3)
    assert bench.parse_docker_mem_usage("--") == (None, None)
    assert bench.parse_percent("31.15%") == pytest.approx(31.15)
    assert bench.parse_percent("--") is None


def test_top_mem_and_ps_rss() -> None:
    top = "Processes: 612 total\n\nPID   MEM\n1798  16G\n"
    assert bench.parse_top_mem(top) == 16 * 1024**3
    ps = (
        " 1798 10648592 /System/Library/Frameworks/Virtualization.framework/Versions/A/"
        "XPCServices/com.apple.Virtualization.VirtualMachine.xpc/Contents/MacOS/"
        "com.apple.Virtualization.VirtualMachine\n"
        "  412  20480 /usr/libexec/something\n"
    )
    rows = bench.parse_ps_rss(ps, bench.DOCKER_VM_PROCESS)
    assert rows == [{"pid": 1798, "rss_bytes": 10648592 * 1024}]


def test_linux_meminfo_loadavg_and_busy_cores() -> None:
    meminfo = "MemTotal:       96423936 kB\nMemFree:         6740000 kB\nMemAvailable:   69435392 kB\nSwapTotal:      209714176 kB\nSwapFree:       164683776 kB\nDirty: 12 kB\n"
    out = bench.parse_meminfo(meminfo)
    assert out["MemAvailable"] == 69435392 * 1024
    assert "Dirty" not in out
    assert bench.parse_loadavg("16.13 12.27 12.78 4/9305 4163709") == {
        "load1": 16.13,
        "load5": 12.27,
        "load15": 12.78,
    }
    a = "cpu  100 0 100 700 100 0 0 0 0 0\n"
    b = "cpu  200 0 200 1300 200 0 0 0 0 0\n"
    # delta total 900, delta idle+iowait 700 -> 2/9 busy of 9 cores = 2.0
    assert bench.cpu_busy_cores(a, b, 9) == pytest.approx(2.0)
    assert bench.cpu_busy_cores("", b, 9) is None


# Shape captured from .201's Redpanda /public_metrics, 2026-09-30.
PROM_A = """# HELP redpanda_kafka_records_produced_total x
# TYPE redpanda_kafka_records_produced_total counter
redpanda_kafka_records_produced_total{redpanda_namespace="kafka",redpanda_topic="onex.evt.platform.node-heartbeat.v1"} 100
redpanda_kafka_records_produced_total{redpanda_namespace="kafka",redpanda_topic="tenant-lab-h105.onex.evt.platform.node-heartbeat.v1"} 10
redpanda_kafka_records_produced_total{redpanda_namespace="redpanda",redpanda_topic="controller"} 999
redpanda_kafka_records_fetched_total{redpanda_namespace="kafka",redpanda_topic="tenant-lab-h105.x"} 5
redpanda_kafka_request_bytes_total{redpanda_request="produce"} 1000
redpanda_kafka_request_bytes_total{redpanda_request="consume"} 2000
"""
PROM_B = (
    PROM_A.replace("} 100\n", "} 130\n")
    .replace("} 10\n", "} 20\n")
    .replace("} 1000\n", "} 1600\n")
)


def test_redpanda_counters_and_rates() -> None:
    a = bench.redpanda_counters(PROM_A)
    assert a["records_produced"] == 110  # the internal controller namespace is excluded
    assert a["records_produced.tenant-lab-h105"] == 10
    assert a["records_fetched.tenant-lab-h105"] == 5
    assert a["bytes_produce"] == 1000
    b = bench.redpanda_counters(PROM_B)
    rates = bench.counter_rates(a, b, 10.0)
    assert rates["records_produced"] == pytest.approx(4.0)
    assert rates["records_produced.tenant-lab-h105"] == pytest.approx(1.0)
    assert rates["bytes_produce"] == pytest.approx(60.0)
    assert rates["bytes_consume"] == 0.0
    assert bench.counter_rates(a, b, 0) == {}


def test_percentile_matches_linear_interpolation() -> None:
    values = [float(v) for v in range(1, 101)]
    assert bench.percentile(values, 50) == pytest.approx(50.5)
    assert bench.percentile(values, 95) == pytest.approx(95.05)
    assert bench.percentile(values, 99) == pytest.approx(99.01)
    assert bench.percentile([7.0], 99) == 7.0
    assert bench.percentile([], 50) is None
    stats = bench.latency_stats([3.0, 1.0, 2.0])
    assert stats["n"] == 3
    assert stats["p50_ms"] == 2.0
    assert stats["min_ms"] == 1.0
    assert bench.latency_stats([]) == {"n": 0}


def test_ntp_offset_math() -> None:
    # Server 50 ms ahead, symmetric 10 ms each way, 1 ms server processing.
    t0 = 1000.000
    t1 = 1000.060
    t2 = 1000.061
    t3 = 1000.021
    offset, delay = bench.ntp_offset(t0, t1, t2, t3)
    assert offset == pytest.approx(0.050)
    assert delay == pytest.approx(0.020)


def test_clock_tool_parsers() -> None:
    assert bench.parse_sntp_cli(
        "+0.024490 +/- 0.016059 time.apple.com 17.253.2.43\n"
    ) == pytest.approx(24.49)
    assert bench.parse_sntp_cli("-0.5 +/- 0.1 x") == pytest.approx(-500.0)
    assert bench.parse_sntp_cli("sntp: timed out") is None
    assert bench.parse_timesync_offset("       Offset: -6.081ms\n") == pytest.approx(
        -6.081
    )
    assert bench.parse_timesync_offset("Offset: +250us") == pytest.approx(0.25)
    assert bench.parse_timesync_offset("no offset here") is None


LAB_TABLE = """# comment
hosts:
  - name: h200
    target: local
    local: true
  - name: h105
    target: 192.168.86.105  # onex-allow-internal-ip
  - name: h201
    target: 192.168.86.201  # onex-allow-internal-ip
    penalty: 0.25
bus_lane: dogfood
"""


def test_lab_table_and_target_resolution(tmp_path: Path) -> None:
    table = tmp_path / "lab_run_hosts.yaml"
    table.write_text(LAB_TABLE)
    assert (
        bench.parse_lab_table(LAB_TABLE)["h105"] == "192.168.86.105"
    )  # onex-allow-internal-ip
    env = {"ONEX_LAB_RUN_HOSTS": str(table)}
    assert (
        bench.resolve_target("h201", None, env) == "192.168.86.201"
    )  # onex-allow-internal-ip
    assert bench.resolve_target("h105", "other-host", env) == "other-host"
    with pytest.raises(SystemExit, match="not in"):
        bench.resolve_target("h999", None, env)
    # Rule 8: no table and no OMNI_HOME fails fast rather than guessing a path.
    with pytest.raises(KeyError):
        bench.resolve_target("h105", None, {})


def test_bench_names_stay_inside_the_runtime_namespace() -> None:
    local = bench.bench_names({"KAFKA_ENVIRONMENT": "dogfood"}, "r1")
    assert local["req_topic"] == "bench.lab-tenant.r1.req"
    assert local["echo_group"] == "dogfood.bench-lab-tenant.r1.echo"
    tenant_env = {
        "KAFKA_TOPIC_NAMESPACE": "tenant-lab-h105",
        "KAFKA_ENVIRONMENT": "tenant-lab-h105",
    }
    tenant = bench.bench_names(tenant_env, "r1")
    # Both halves carry the tenant ACL prefix (topics and groups).
    assert tenant["req_topic"].startswith("tenant-lab-h105.")
    assert tenant["ack_topic"].startswith("tenant-lab-h105.")
    assert tenant["echo_group"].startswith("tenant-lab-h105.")
    assert bench.topic_namespace_prefix({"KAFKA_TOPIC_NAMESPACE": " x. "}) == "x."
    assert bench.topic_namespace_prefix({}) == ""


def test_arm_containers_match_the_dogfood_compose_file() -> None:
    compose = (REPO / "docker" / "docker-compose.dogfood.yml").read_text()
    local = bench.ARM_CONTAINERS["local-stack"]
    assert f"name: {local['project']}" in compose
    assert f"container_name: {local['runtime']}" in compose
    assert f"container_name: {local['effects']}" in compose
    # The lab-tenant names are OMN-20207's overlay (docker/docker-compose.lab-tenant.yml).
    tenant = bench.ARM_CONTAINERS["lab-tenant"]
    assert tenant == {
        "project": "omnibase-infra-lab-tenant",
        "runtime": "omninode-lab-tenant-runtime",
        "effects": "omninode-lab-tenant-runtime-effects",
    }
    assert set(bench.ARMS) == set(bench.ARM_CONTAINERS)


def _rep(avail_gb: float, rtt_p50: float, refusal: str | None) -> dict[str, object]:
    gib = 1024**3
    return {
        "remote": {
            "memory": {
                "vm_stat": {
                    "available_bytes": int(avail_gb * gib),
                    "free_bytes": gib,
                },
                "swapusage": {"used_mib": 100.0},
                "pressure_level": 1,
                "docker_vm": {"rss_bytes": 10 * gib, "footprint_bytes": 16 * gib},
                "docker_settings": {"MemoryMiB": 16384},
                "docker_stats": [
                    {"name": "a", "mem_used_bytes": gib},
                    {"name": "b", "mem_used_bytes": gib // 2},
                ],
            },
            "probe": {
                "bus_roundtrip": {"p50_ms": rtt_p50, "p95_ms": 1.0, "p99_ms": 2.0},
                "projection": {"p50_ms": 6.0, "timeouts": 0},
                "health": {"p50_ms": 0.5},
            },
            "ntp": {"offset_ms": 3.0},
        },
        "placement": {
            "plugin_cache": {
                "slots": 3,
                "mem_avail_gb": avail_gb,
                "load1": 2.0,
                "admission_refusal": refusal,
            },
            "controller": {"error": "unreadable"},
        },
    }


def test_rep_metrics_and_aggregate() -> None:
    reps = [_rep(11.0, 0.3, None), _rep(12.0, 0.5, None), _rep(9.0, 0.4, "mem")]
    metrics = [bench.rep_metrics(r) for r in reps]
    m0 = metrics[0]
    assert m0["mem.host.available_gb"] == pytest.approx(11.0)
    assert m0["mem.docker_vm.configured_gb"] == 16.0
    assert m0["mem.docker_vm.footprint_gb"] == 16.0
    assert m0["mem.containers.total_gb"] == 1.5
    assert m0["mem.container.a_mib"] == 1024.0
    assert m0["projection.timeouts"] == 0.0
    assert m0["placement.plugin_cache.admitted"] == 1.0
    # An unreadable placement copy contributes nothing rather than a zero.
    assert not any(k.startswith("placement.controller.") for k in m0)
    agg = bench.aggregate(metrics)
    assert agg["bus_roundtrip.p50_ms"] == {
        "n": 3,
        "median": 0.4,
        "min": 0.3,
        "max": 0.5,
        "spread": 0.2,
    }
    assert agg["placement.plugin_cache.admitted"]["min"] == 0.0
    assert agg["mem.host.available_gb"]["median"] == pytest.approx(11.0)


def test_render_summary_has_one_column_per_host_arm() -> None:
    metrics = bench.aggregate([bench.rep_metrics(_rep(11.0, 0.3, None))])
    results = [
        {
            "kind": "satellite",
            "host": "h105",
            "arm": "local-stack",
            "reps": [{}],
            "summary": metrics,
        },
        {
            "kind": "dependency",
            "host": "h201",
            "arm": "local-stack",
            "started_at": "t",
            "summary": bench.aggregate([{"pg.connections_total": 90.0}]),
        },
    ]
    text = bench.render_summary(results)
    assert "| metric | h105 local-stack (n=1) |" in text
    assert "| bus round trip p50 ms | 0.3 |" in text
    assert "placement.plugin_cache.slots" in text
    assert "| pg.connections_total | 90 |" in text


def test_render_summary_puts_before_and_after_side_by_side() -> None:
    before = bench.aggregate([bench.rep_metrics(_rep(7.0, 0.5, "mem"))])
    after_rep = _rep(15.0, 4.0, None)
    after_rep["remote"]["cold_start"] = {"pair_healthy_s": 42.0}  # type: ignore[index]
    after = bench.aggregate([bench.rep_metrics(after_rep)])
    results = [
        {"kind": "satellite", "host": "h101", "arm": "lab-tenant", "reps": [{}], "summary": after},
        {"kind": "satellite", "host": "h101", "arm": "local-stack", "reps": [{}], "summary": before},
        {"kind": "dependency", "host": "h201", "arm": "lab-tenant", "started_at": "2026-10-02T00:00:00Z",
         "reps": [{}], "summary": bench.aggregate([{"pg.connections_total": 60.0}])},
        {"kind": "dependency", "host": "h201", "arm": "local-stack", "started_at": "2026-10-01T00:15:00Z",
         "reps": [{}], "summary": bench.aggregate([{"pg.connections_total": 45.0}])},
    ]  # fmt: skip
    text = bench.render_summary(results)
    # local-stack (BEFORE) is the left column of each host, lab-tenant (AFTER) the right.
    assert "| h101 local-stack (n=1) | h101 lab-tenant (n=1) |" in text
    assert "| bus round trip p50 ms | 0.5 | 4 |" in text
    assert "| cold start, runtime pair to healthy s | - | 42 |" in text
    assert "| pg.connections_total | 45 | 60 |" in text


def test_dependency_metrics_flattening() -> None:
    dep = {
        "cpu": {"ncpu": 32, "load1": 10.5, "busy_cores": 6.4},
        "memory": {"MemAvailable": 61.6, "psi_some": "some avg10=0.26"},
        "postgres": {"connections_per_db": {"omnibase_infra": 57}, "total": 90},
        "redpanda": {"rates_per_s": {"records_produced": 5.7}},
        "dev_lane_containers": {"mem_total_gb": 8.2, "count": 26},
    }
    out = bench.dependency_metrics(dep)
    assert out["cpu.load1"] == 10.5
    assert out["mem.MemAvailable_gb"] == 61.6
    assert "mem.psi_some_gb" not in out
    assert out["pg.connections.omnibase_infra"] == 57.0
    assert out["pg.connections_total"] == 90.0
    assert out["redpanda.records_produced_per_s"] == 5.7
    assert out["dev_lane.count"] == 26.0


def test_placement_default_modules(tmp_path: Path) -> None:
    home = tmp_path
    plugin = home / "cache" / "omni" / "9.9.9"
    script = plugin / "skills" / "merge-drain" / "scripts" / "landing_placement.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    (home / ".claude" / "plugins").mkdir(parents=True)
    (home / ".claude" / "plugins" / "installed_plugins.json").write_text(
        json.dumps(
            {"plugins": {"omni@omninode-internal": [{"installPath": str(plugin)}]}}
        )
    )
    found = bench.default_placement_modules(home)
    assert found == {"plugin_cache": script}
    ctl = (
        home
        / ".omninode"
        / "landing-controller"
        / "skills"
        / "merge-drain"
        / "scripts"
        / "landing_placement.py"
    )
    ctl.parent.mkdir(parents=True)
    ctl.write_text("")
    assert bench.default_placement_modules(home)["controller"] == ctl


def test_controller_refuses_without_host_or_arm() -> None:
    with pytest.raises(SystemExit):
        bench.main(["--reps", "1"])


def test_urls_are_derived_from_the_container_not_spelled() -> None:
    # The dogfood runtime's own healthcheck, read by docker inspect on h105.
    test = ["CMD", "curl", "-sf", "http://localhost:8085/health"]
    assert bench.healthcheck_url(test) == "http://localhost:8085/health"
    assert bench.healthcheck_url(["CMD-SHELL", "pg_isready"]) is None
    assert bench.healthcheck_url(None) is None
    # `docker port omnibase-infra-redpanda 9644/tcp` on .201.
    assert (
        bench.published_url("0.0.0.0:9644\n[::]:9644\n", "/public_metrics")
        == "http://0.0.0.0:9644/public_metrics"
    )
    assert bench.published_url("[::]:9644\n", "/x") is None
    assert bench.published_url("", "/x") is None


def test_min_uptime_marks_a_freshly_redeployed_dependency_lane() -> None:
    from datetime import UTC, datetime

    now = datetime(2026, 10, 1, 0, 11, 4, tzinfo=UTC)
    stamps = [
        "2026-09-30T20:55:00.123456789Z",  # infra, up hours
        "2026-09-30T23:56:04.5Z",  # app tier, redeployed 15 minutes ago
        "garbage",
    ]
    assert bench.min_uptime_s(stamps, now) == 900.0
    assert bench.min_uptime_s([], now) is None


def test_home_relative_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert bench.home_relative(tmp_path / ".omninode" / "x.py") == "~/.omninode/x.py"
    assert bench.home_relative(Path("/opt/elsewhere/x.py")) == "/opt/elsewhere/x.py"
