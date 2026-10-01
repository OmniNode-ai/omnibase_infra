# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Repeatable before/after benchmark for the lab-tenant runtime configuration (OMN-20213).

WHAT IT COMPARES
  A lab satellite (h101, h105) either runs its whole local "dogfood" stack (arm
  ``local-stack``: Postgres, Redpanda, Valkey, migrations and the projection API
  beside the runtime pair) or only the runtime pair as a tenant of the .201 dev
  lane's shared servers (arm ``lab-tenant``, the overlay OMN-20207 adds). This
  harness measures the same things in both arms so the switch can be judged on
  numbers: host memory, the Docker VM, bus and projection latency as the runtime
  sees them, the placement admission the landing controller computes, and the
  load that moves onto .201.

ONE FILE, FOUR ROLES
  controller (default)   runs on the operator Mac. Copies this file to the
                         target host, runs the remote role over ssh, reads the
                         placement admission locally, repeats ``--reps`` times
                         and writes ONE JSON result with a median/spread summary.
  --remote-collect       runs on the satellite. Reads memory (vm_stat, sysctl,
                         the Docker VM process, docker stats, Docker Desktop's
                         configured VM size), the runtime pair's start and
                         health state, the host clock offset, then pipes this
                         file into the runtime container for the probe role.
  --container-probe      runs INSIDE the runtime container, with the runtime's
                         own bus and database configuration from its env, so
                         both arms are measured from the same vantage point.
                         Bus round trip (publish -> consume -> commit -> ack
                         event -> receive), projection latency (publish on a
                         topic only the event-ledger projection consumes ->
                         row visible in ``event_ledger``) and health latency.
  --remote-dependency    runs on .201 and reads the dependency lane's side:
                         Postgres connections per database, CPU/load, Redpanda
                         produce/fetch rates, memory.
  --summarize DIR        renders a markdown table from the result JSONs in DIR.

WHAT IT WRITES (and removes)
  It never starts, stops or reconfigures a container, except the timed restart
  of the runtime pair, which runs only with ``--allow-restart`` AND
  ``--arm lab-tenant``. It writes: two per-run benchmark topics and their
  consumer groups (deleted at the end), and N projection events whose
  ``event_ledger`` rows are deleted by correlation id at the end. The projection
  events themselves stay in the projection topic's log unless the log held
  nothing else, in which case it is trimmed back (DeleteRecords). Every cleanup
  outcome is recorded in the result.

USAGE
  uv run python scripts/bench_lab_tenant.py --host h105 --arm local-stack \\
      --reps 3 --out-dir benchmarks/lab-tenant/results/<UTC>
  uv run python scripts/bench_lab_tenant.py --dependency --host h201 \\
      --arm local-stack --out-dir benchmarks/lab-tenant/results/<UTC>
  uv run python scripts/bench_lab_tenant.py --summarize benchmarks/lab-tenant/results/<UTC>

Host names resolve through the lab host table (``ONEX_LAB_RUN_HOSTS``, else
``$OMNI_HOME/../omnibase_internal/src/omnibase_internal/lab_run_hosts.yaml``);
``--target`` overrides it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import statistics
import struct
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "bench-lab-tenant.v1"
TICKET = "OMN-20213"
ARMS = ("local-stack", "lab-tenant")

# The runtime pair each arm runs, and the compose project that owns the arm's
# containers. local-stack is the deployed dogfood bundle
# (docker/docker-compose.dogfood.yml); lab-tenant is OMN-20207's overlay
# (docker/docker-compose.lab-tenant.yml).
ARM_CONTAINERS: dict[str, dict[str, str]] = {
    "local-stack": {
        "project": "omnibase-infra-dogfood",
        "runtime": "omninode-dogfood-runtime",
        "effects": "omninode-dogfood-runtime-effects",
    },
    "lab-tenant": {
        "project": "omnibase-infra-lab-tenant",
        "runtime": "omninode-lab-tenant-runtime",
        "effects": "omninode-lab-tenant-runtime-effects",
    },
}

LAB_TABLE_REL = (
    Path("omnibase_internal") / "src" / "omnibase_internal" / "lab_run_hosts.yaml"
)
DOCKER_SETTINGS_REL = (
    Path("Library") / "Group Containers" / "group.com.docker" / "settings-store.json"
)
DOCKER_VM_PROCESS = "com.apple.Virtualization.VirtualMachine"
REMOTE_PATH = "$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH:/usr/sbin:/sbin"
REMOTE_DIR = ".cache/omni-bench-lab-tenant"
BREW_PYTHON = "/opt/homebrew/bin/python3.13"
NTP_SERVER = "time.apple.com"
NTP_SAMPLES = 4
NTP_EPOCH_DELTA = 2208988800  # seconds from 1900-01-01 to 1970-01-01
SSH_OPTS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=30",
)
REDPANDA_ADMIN_PORT = 9644
GIB = 1024**3
MIB = 1024**2

# ---------------------------------------------------------------------------
# Pure parsing helpers (unit tested; no I/O)
# ---------------------------------------------------------------------------

_SIZE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
    # top(1) on macOS prints binary units with a single letter.
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}


def parse_size(text: str) -> int | None:
    """``"658.4MiB"`` / ``"15.6GiB"`` / ``"16G"`` / ``"512M+"`` -> bytes. None when unparseable."""
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)[+-]?\s*", text or "")
    if not m:
        return None
    unit = (m.group(2) or "B").upper()
    if unit not in _SIZE_UNITS:
        return None
    return int(float(m.group(1)) * _SIZE_UNITS[unit])


def parse_docker_mem_usage(text: str) -> tuple[int | None, int | None]:
    """docker stats ``MemUsage`` (``"478.5MiB / 1.5GiB"``) -> (used bytes, limit bytes)."""
    parts = [p.strip() for p in (text or "").split("/")]
    if len(parts) != 2:
        return None, None
    return parse_size(parts[0]), parse_size(parts[1])


def parse_percent(text: str) -> float | None:
    try:
        return float((text or "").strip().rstrip("%"))
    except ValueError:
        return None


def parse_vm_stat(text: str) -> dict[str, int]:
    """macOS ``vm_stat`` -> bytes per category, using the page size from its header line."""
    page = 16384
    header = re.search(r"page size of (\d+) bytes", text)
    if header:
        page = int(header.group(1))
    pages: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r'^"?([A-Za-z][A-Za-z \-]+?)"?:\s+(\d+)\.?\s*$', line.strip())
        if m:
            pages[m.group(1).strip().lower()] = int(m.group(2))
    wanted = {
        "free": "pages free",
        "active": "pages active",
        "inactive": "pages inactive",
        "speculative": "pages speculative",
        "wired": "pages wired down",
        "purgeable": "pages purgeable",
        "compressed": "pages occupied by compressor",
        "stored_in_compressor": "pages stored in compressor",
    }
    out = {"page_size": page}
    for key, label in wanted.items():
        if label in pages:
            out[f"{key}_bytes"] = pages[label] * page
    if all(f"{k}_bytes" in out for k in ("free", "inactive", "speculative")):
        # The landing placement probe's mem_avail (free + inactive + speculative).
        out["available_bytes"] = (
            out["free_bytes"] + out["inactive_bytes"] + out["speculative_bytes"]
        )
    return out


def parse_swapusage(text: str) -> dict[str, float]:
    """``vm.swapusage: total = 4096.00M  used = 3136.56M  free = 959.44M`` -> MiB values."""
    out: dict[str, float] = {}
    for key in ("total", "used", "free"):
        m = re.search(rf"{key}\s*=\s*([0-9.]+)([KMG])", text)
        if m:
            scale = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[m.group(2)]
            out[f"{key}_mib"] = round(float(m.group(1)) * scale, 2)
    return out


def parse_top_mem(text: str) -> int | None:
    """Last data line of ``top -l 1 -pid N -stats pid,mem`` -> the MEM column (phys_footprint) in bytes."""
    for line in reversed(text.strip().splitlines()):
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            return parse_size(parts[1])
    return None


def parse_ps_rss(text: str, needle: str) -> list[dict[str, int]]:
    """``ps -axo pid=,rss=,command=`` lines whose command contains ``needle`` -> [{pid, rss_bytes}]."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(None, 2)
        if (
            len(parts) == 3
            and needle in parts[2]
            and parts[0].isdigit()
            and parts[1].isdigit()
        ):
            rows.append({"pid": int(parts[0]), "rss_bytes": int(parts[1]) * 1024})
    return rows


def parse_meminfo(text: str) -> dict[str, int]:
    """Linux ``/proc/meminfo`` -> bytes for the fields the dependency side reports."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*kB", line)
        if m and m.group(1) in (
            "MemTotal",
            "MemAvailable",
            "MemFree",
            "Cached",
            "SwapTotal",
            "SwapFree",
        ):
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def parse_loadavg(text: str) -> dict[str, float]:
    parts = text.split()
    if len(parts) < 3:
        return {}
    return {
        "load1": float(parts[0]),
        "load5": float(parts[1]),
        "load15": float(parts[2]),
    }


def cpu_busy_cores(stat_a: str, stat_b: str, ncpu: int) -> float | None:
    """Two ``/proc/stat`` ``cpu`` lines -> busy cores over the interval (idle + iowait counted idle)."""

    def fields(text: str) -> list[int] | None:
        for line in text.splitlines():
            if line.startswith("cpu "):
                return [int(x) for x in line.split()[1:9]]
        return None

    a, b = fields(stat_a), fields(stat_b)
    if a is None or b is None:
        return None
    idle = (b[3] + b[4]) - (a[3] + a[4])
    total = sum(b) - sum(a)
    if total <= 0:
        return None
    return round(ncpu * (1 - idle / total), 2)


def parse_prom_text(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Prometheus exposition text -> [(metric, labels, value)]; comments and junk skipped."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{(.*)\})?\s+(\S+)", line)
        if not m:
            continue
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(3) or ""))
        try:
            out.append((m.group(1), labels, float(m.group(4))))
        except ValueError:
            continue
    return out


def redpanda_counters(text: str) -> dict[str, float]:
    """Sum the Redpanda public counters the dependency side rates: records produced/fetched
    (total, and per ``tenant-<slug>.`` topic prefix) and Kafka request bytes by request type."""
    out: dict[str, float] = {}
    for metric, labels, value in parse_prom_text(text):
        if metric in (
            "redpanda_kafka_records_produced_total",
            "redpanda_kafka_records_fetched_total",
        ):
            if labels.get("redpanda_namespace") != "kafka":
                continue
            kind = "produced" if "produced" in metric else "fetched"
            out[f"records_{kind}"] = out.get(f"records_{kind}", 0.0) + value
            topic = labels.get("redpanda_topic", "")
            if topic.startswith("tenant-") and "." in topic:
                key = f"records_{kind}.{topic.split('.', 1)[0]}"
                out[key] = out.get(key, 0.0) + value
        elif metric == "redpanda_kafka_request_bytes_total":
            req = labels.get("redpanda_request", "other")
            out[f"bytes_{req}"] = out.get(f"bytes_{req}", 0.0) + value
    return out


def counter_rates(
    a: Mapping[str, float], b: Mapping[str, float], seconds: float
) -> dict[str, float]:
    if seconds <= 0:
        return {}
    return {
        k: round((b[k] - a.get(k, 0.0)) / seconds, 3)
        for k in b
        if b[k] >= a.get(k, 0.0)
    }


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolation percentile (numpy's default), no numpy."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def latency_stats(values_ms: Sequence[float]) -> dict[str, Any]:
    if not values_ms:
        return {"n": 0}
    return {
        "n": len(values_ms),
        "p50_ms": round(percentile(values_ms, 50) or 0.0, 3),
        "p95_ms": round(percentile(values_ms, 95) or 0.0, 3),
        "p99_ms": round(percentile(values_ms, 99) or 0.0, 3),
        "min_ms": round(min(values_ms), 3),
        "max_ms": round(max(values_ms), 3),
        "mean_ms": round(statistics.fmean(values_ms), 3),
    }


def ntp_offset(t0: float, t1: float, t2: float, t3: float) -> tuple[float, float]:
    """RFC 4330 clock offset and round-trip delay (seconds) from the four SNTP timestamps."""
    return ((t1 - t0) + (t2 - t3)) / 2.0, (t3 - t0) - (t2 - t1)


def parse_lab_table(text: str) -> dict[str, str]:
    """The lab host table (``hosts: - name: X / target: Y``) -> {name: target}. A YAML subset
    reader, so the controller needs no third-party package."""
    hosts: dict[str, str] = {}
    name: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        m = re.match(r"^\s*-?\s*(name|target):\s*(\S+)\s*$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip("'\"")
        if key == "name":
            name = value
        elif name is not None:
            hosts[name] = value
    return hosts


def healthcheck_url(test: Sequence[str] | None) -> str | None:
    """The URL a container's own Docker healthcheck probes (``Config.Healthcheck.Test``), so the
    benchmark times the same request Docker does instead of spelling an address of its own."""
    for token in test or ():
        if re.match(r"^https?://", token):
            return token
    return None


def published_url(docker_port_output: str, path: str) -> str | None:
    """``docker port <c> <port>/tcp`` (``0.0.0.0:9644`` / ``[::]:9644``) -> an http URL on the
    first IPv4 binding."""
    for line in docker_port_output.splitlines():
        m = re.match(r"^(\d+\.\d+\.\d+\.\d+):(\d+)$", line.strip())
        if m:
            return f"http://{m.group(1)}:{m.group(2)}{path}"
    return None


def min_uptime_s(started_at: Iterable[str], now: datetime) -> float | None:
    """Seconds since the most recently (re)started container: a dependency-side sample taken
    minutes after a redeploy reads warm-up memory, and this is how the table says so."""
    ages = []
    for stamp in started_at:
        m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", stamp or "")
        if m:
            when = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=UTC
            )
            ages.append((now - when).total_seconds())
    return round(min(ages), 1) if ages else None


def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float]:
    """Nested dict -> {dotted.key: number} for every int/float leaf (bools skipped)."""
    out: dict[str, float] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(flatten_numeric(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def aggregate(
    metrics_per_rep: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Per metric across repetitions: median, min, max and spread (max - min)."""
    keys = sorted({k for rep in metrics_per_rep for k in rep})
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [rep[key] for rep in metrics_per_rep if key in rep]
        out[key] = {
            "n": len(vals),
            "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "spread": round(max(vals) - min(vals), 4),
        }
    return out


def rep_metrics(rep: Mapping[str, Any]) -> dict[str, float]:
    """The headline numbers of one satellite repetition, flat, in human units."""
    remote = rep.get("remote") or {}
    mem = remote.get("memory") or {}
    vm = mem.get("vm_stat") or {}
    out: dict[str, float] = {}
    for key in (
        "free",
        "inactive",
        "speculative",
        "compressed",
        "wired",
        "active",
        "available",
    ):
        if f"{key}_bytes" in vm:
            out[f"mem.host.{key}_gb"] = round(vm[f"{key}_bytes"] / GIB, 3)
    swap = mem.get("swapusage") or {}
    if "used_mib" in swap:
        out["mem.host.swap_used_mib"] = swap["used_mib"]
    if isinstance(mem.get("pressure_level"), int):
        out["mem.host.pressure_level"] = float(mem["pressure_level"])
    dvm = mem.get("docker_vm") or {}
    if dvm.get("rss_bytes") is not None:
        out["mem.docker_vm.rss_gb"] = round(dvm["rss_bytes"] / GIB, 3)
    if dvm.get("footprint_bytes") is not None:
        out["mem.docker_vm.footprint_gb"] = round(dvm["footprint_bytes"] / GIB, 3)
    settings = mem.get("docker_settings") or {}
    if isinstance(settings.get("MemoryMiB"), (int, float)):
        out["mem.docker_vm.configured_gb"] = round(settings["MemoryMiB"] / 1024, 3)
    stats = mem.get("docker_stats") or []
    if stats:
        out["mem.containers.total_gb"] = round(
            sum(s.get("mem_used_bytes") or 0 for s in stats) / GIB, 3
        )
        for s in stats:
            if s.get("mem_used_bytes") is not None:
                out[f"mem.container.{s['name']}_mib"] = round(
                    s["mem_used_bytes"] / MIB, 1
                )
    probe = remote.get("probe") or {}
    for section in ("bus_roundtrip", "bus_oneway", "projection", "health"):
        stats_ = probe.get(section) or {}
        for pct in ("p50_ms", "p95_ms", "p99_ms"):
            if pct in stats_:
                out[f"{section}.{pct}"] = stats_[pct]
    if (probe.get("projection") or {}).get("timeouts") is not None:
        out["projection.timeouts"] = float(probe["projection"]["timeouts"])
    ntp = remote.get("ntp") or {}
    if ntp.get("offset_ms") is not None:
        out["ntp.host_offset_ms"] = ntp["offset_ms"]
    for label, reading in (rep.get("placement") or {}).items():
        if not isinstance(reading, Mapping) or reading.get("error"):
            continue
        for key in ("slots", "mem_avail_gb", "load1"):
            if isinstance(reading.get(key), (int, float)):
                out[f"placement.{label}.{key}"] = float(reading[key])
        out[f"placement.{label}.admitted"] = (
            0.0 if reading.get("admission_refusal") else 1.0
        )
    return out


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run(
    argv: Sequence[str], timeout: float = 60.0, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(argv), input=stdin, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            list(argv), 124, exc.stdout or b"", f"timed out after {timeout}s".encode()
        )
    except OSError as exc:
        return subprocess.CompletedProcess(list(argv), 127, b"", str(exc).encode())


def text_of(argv: Sequence[str], timeout: float = 60.0) -> tuple[int, str]:
    done = run(argv, timeout=timeout)
    return done.returncode, done.stdout.decode(errors="replace")


def parse_sntp_cli(text: str) -> float | None:
    """macOS ``sntp`` output (``+0.024490 +/- 0.016059 time.apple.com 17.253.2.43``) -> offset ms."""
    for line in reversed(text.strip().splitlines()):
        m = re.match(r"^\s*([+-]?[0-9]+\.[0-9]+)\s+\+/-", line)
        if m:
            return round(float(m.group(1)) * 1000, 3)
    return None


def parse_timesync_offset(text: str) -> float | None:
    """``timedatectl timesync-status`` ``Offset: -6.081ms`` (or ``us``/``s``) -> ms."""
    m = re.search(r"Offset:\s*([+-]?[0-9.]+)\s*(us|ms|s)\b", text)
    if not m:
        return None
    scale = {"us": 0.001, "ms": 1.0, "s": 1000.0}[m.group(2)]
    return round(float(m.group(1)) * scale, 3)


def clock_offset(server: str = NTP_SERVER) -> dict[str, Any]:
    """This host's clock offset: an in-process SNTP query, else the OS's own tool (``sntp`` on
    macOS, ``timedatectl`` on systemd Linux). A satellite's firewall can drop the reply to an
    unsigned interpreter, which is why the fallback exists."""
    samples = [sntp_offset(server) for _ in range(NTP_SAMPLES)]
    good = [x for x in samples if x.get("offset_ms") is not None]
    first = samples[0]
    if good:
        # The sample with the least network delay carries the least asymmetry error.
        best = min(good, key=lambda x: float(x["delay_ms"]))
        return {
            **best,
            "method": "sntp-inprocess",
            "samples": len(good),
            "offsets_ms": [x["offset_ms"] for x in good],
        }
    rc, text = text_of(["sntp", "-t", "3", server], timeout=15)
    parsed = parse_sntp_cli(text) if rc == 0 else None
    if parsed is not None:
        return {"server": server, "offset_ms": parsed, "method": "sntp-cli"}
    rc, text = text_of(["timedatectl", "timesync-status"], timeout=15)
    parsed = parse_timesync_offset(text) if rc == 0 else None
    if parsed is not None:
        return {"offset_ms": parsed, "method": "timedatectl"}
    return {"server": server, "error": first.get("error"), "method": "none"}


def sntp_offset(server: str = NTP_SERVER, timeout: float = 3.0) -> dict[str, Any]:
    """One SNTP v4 query (RFC 4330). Offset > 0 means this host's clock is behind the server."""
    packet = b"\x23" + 47 * b"\0"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            t0 = time.time()
            sock.sendto(packet, (server, 123))
            data, _ = sock.recvfrom(512)
            t3 = time.time()
    except OSError as exc:
        return {"server": server, "error": str(exc)}
    if len(data) < 48:
        return {"server": server, "error": f"short reply ({len(data)} bytes)"}

    def ts(offset: int) -> float:
        secs, frac = struct.unpack("!II", data[offset : offset + 8])
        return float(secs - NTP_EPOCH_DELTA + frac / 2**32)

    offset, delay = ntp_offset(t0, ts(32), ts(40), t3)
    return {
        "server": server,
        "offset_ms": round(offset * 1000, 3),
        "delay_ms": round(delay * 1000, 3),
    }


# ---------------------------------------------------------------------------
# Role: container probe (runs inside the runtime container)
# ---------------------------------------------------------------------------


def _kafka_base_config(env: Mapping[str, str]) -> dict[str, Any]:
    conf: dict[str, Any] = {
        "bootstrap.servers": env["KAFKA_BOOTSTRAP_SERVERS"],
        "client.id": "bench-lab-tenant",
    }
    protocol = env.get("KAFKA_SECURITY_PROTOCOL", "").strip()
    if protocol:
        conf["security.protocol"] = protocol
    if protocol.upper().startswith("SASL"):
        conf["sasl.mechanism"] = env.get("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
        conf["sasl.username"] = env["KAFKA_SASL_USERNAME"]
        conf["sasl.password"] = env["KAFKA_SASL_PASSWORD"]
    return conf


def topic_namespace_prefix(env: Mapping[str, str]) -> str:
    """Mirror of omnibase_infra.topics.topic_namespace.resolve_topic_namespace (OMN-18891)."""
    token = env.get("KAFKA_TOPIC_NAMESPACE", "").strip().rstrip(".")
    return f"{token}." if token else ""


def bench_names(env: Mapping[str, str], run_id: str) -> dict[str, str]:
    """Per-run topic and group names inside the runtime's own namespace (tenant prefix or none)."""
    prefix = topic_namespace_prefix(env)
    group_root = (env.get("KAFKA_ENVIRONMENT") or "").strip() or "bench"
    return {
        "req_topic": f"{prefix}bench.lab-tenant.{run_id}.req",
        "ack_topic": f"{prefix}bench.lab-tenant.{run_id}.ack",
        "echo_group": f"{group_root}.bench-lab-tenant.{run_id}.echo",
        "origin_group": f"{group_root}.bench-lab-tenant.{run_id}.origin",
    }


def _poll_one(consumer: Any, deadline: float, producer: Any) -> Any:
    while time.perf_counter() < deadline:
        producer.poll(0)
        msg = consumer.poll(0.05)
        if msg is None:
            continue
        if msg.error():
            continue
        return msg
    return None


def probe_bus_roundtrip(
    env: Mapping[str, str], run_id: str, n: int, warmup: int
) -> dict[str, Any]:
    from confluent_kafka import (
        Consumer,
        Producer,
        TopicPartition,
    )
    from confluent_kafka.admin import (  # type: ignore[attr-defined]
        AdminClient,
        NewTopic,
    )

    base = _kafka_base_config(env)
    names = bench_names(env, run_id)
    admin = AdminClient(base)
    result: dict[str, Any] = {
        "topics": [names["req_topic"], names["ack_topic"]],
        "groups": [names["echo_group"]],
    }
    futures = admin.create_topics(
        [NewTopic(names["req_topic"], 1, 1), NewTopic(names["ack_topic"], 1, 1)],
        operation_timeout=15,
    )
    for topic, fut in futures.items():
        fut.result(timeout=30)
        result.setdefault("created", []).append(topic)
    producer = Producer({**base, "linger.ms": 0, "acks": "all"})
    common = {**base, "enable.auto.commit": False, "auto.offset.reset": "earliest"}
    echo = Consumer({**common, "group.id": names["echo_group"]})
    origin = Consumer({**common, "group.id": names["origin_group"]})
    echo.assign([TopicPartition(names["req_topic"], 0, 0)])
    origin.assign([TopicPartition(names["ack_topic"], 0, 0)])
    rtt: list[float] = []
    oneway: list[float] = []
    commit_ms: list[float] = []
    failures = 0
    try:
        for i in range(warmup + n):
            key = f"{run_id}-{i}".encode()
            t0 = time.perf_counter()
            producer.produce(
                names["req_topic"], value=b'{"bench":"rtt","seq":%d}' % i, key=key
            )
            producer.poll(0)
            got = _poll_one(echo, t0 + 15, producer)
            if got is None:
                failures += 1
                continue
            t1 = time.perf_counter()
            echo.commit(message=got, asynchronous=False)
            t2 = time.perf_counter()
            producer.produce(
                names["ack_topic"], value=b'{"bench":"ack","seq":%d}' % i, key=key
            )
            producer.poll(0)
            ack = _poll_one(origin, t2 + 15, producer)
            if ack is None:
                failures += 1
                continue
            t3 = time.perf_counter()
            if i >= warmup:
                oneway.append((t1 - t0) * 1000)
                commit_ms.append((t2 - t1) * 1000)
                rtt.append((t3 - t0) * 1000)
        producer.flush(10)
    finally:
        echo.close()
        origin.close()
    result["roundtrip"] = latency_stats(rtt)
    result["oneway"] = latency_stats(oneway)
    result["commit"] = latency_stats(commit_ms)
    result["failures"] = failures
    result["warmup"] = warmup
    cleanup: dict[str, Any] = {}
    for topic, fut in admin.delete_topics(
        [names["req_topic"], names["ack_topic"]], operation_timeout=15
    ).items():
        try:
            fut.result(timeout=30)
            cleanup[f"topic:{topic}"] = "deleted"
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            cleanup[f"topic:{topic}"] = f"error: {exc}"
    try:
        for group, fut in admin.delete_consumer_groups(
            [names["echo_group"]], request_timeout=15
        ).items():
            try:
                fut.result(timeout=30)
                cleanup[f"group:{group}"] = "deleted"
            except Exception as exc:  # noqa: BLE001
                # A group that only ever committed through assign() may already be gone.
                cleanup[f"group:{group}"] = (
                    "absent" if "GROUP_ID_NOT_FOUND" in str(exc) else f"error: {exc}"
                )
    except Exception as exc:  # noqa: BLE001
        cleanup["groups"] = f"error: {exc}"
    result["cleanup"] = cleanup
    return result


def _envelope(correlation_id: str, message_id: str, run_id: str, seq: int) -> bytes:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return json.dumps(
        {
            "envelope_id": message_id,
            "envelope_timestamp": now,
            "correlation_id": correlation_id,
            "payload": {
                "bench": "lab-tenant-projection",
                "ticket": TICKET,
                "run_id": run_id,
                "seq": seq,
            },
        }
    ).encode()


def probe_projection(
    env: Mapping[str, str], run_id: str, n: int, warmup: int, timeout_s: float
) -> dict[str, Any]:
    """Publish on the one topic only node_ledger_projection_compute consumes and time until the
    event_ledger row with the event's correlation id is visible; then delete those rows."""
    import psycopg2
    from confluent_kafka import Consumer, Producer, TopicPartition
    from confluent_kafka.admin import AdminClient

    from omnibase_infra.enums.generated.enum_steel_onslaught_topic import (
        EnumSteelOnslaughtTopic,
    )

    canonical = EnumSteelOnslaughtTopic.EVT_MATCH_TERMINAL_V1.value
    topic = f"{topic_namespace_prefix(env)}{canonical}"
    base = _kafka_base_config(env)
    result: dict[str, Any] = {
        "topic": topic,
        "table": "event_ledger",
        "database_env": "OMNIBASE_INFRA_DB_URL",
    }
    watermark_consumer = Consumer({**base, "group.id": f"bench-wm-{run_id}"})
    low0, high0 = watermark_consumer.get_watermark_offsets(
        TopicPartition(topic, 0), timeout=10
    )
    result["log_before"] = {"low": low0, "high": high0}
    producer = Producer({**base, "linger.ms": 0, "acks": "all"})
    conn = psycopg2.connect(env["OMNIBASE_INFRA_DB_URL"], connect_timeout=10)
    conn.autocommit = True
    cur = conn.cursor()
    latencies: list[float] = []
    ack_ms: list[float] = []
    corr_ids: list[str] = []
    delivered = 0
    timeouts = 0
    for i in range(warmup + n):
        corr = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())
        corr_ids.append(corr)
        headers: list[tuple[str, str | bytes | None]] = [
            ("content_type", b"application/json"),
            ("correlation_id", corr.encode()),
            ("message_id", msg_id.encode()),
            ("timestamp", datetime.now(UTC).isoformat().encode()),
            ("source", b"bench-lab-tenant"),
            ("event_type", canonical.encode()),
            ("schema_version", b"1.0.0"),
            ("priority", b"normal"),
            ("retry_count", b"0"),
            ("max_retries", b"3"),
        ]
        acked: list[float] = []

        def on_delivery(err: Any, _msg: Any, sink: list[float] = acked) -> None:
            if err is None:
                sink.append(time.perf_counter())

        t0 = time.perf_counter()
        producer.produce(
            topic,
            value=_envelope(corr, msg_id, run_id, i),
            headers=headers,
            on_delivery=on_delivery,
        )
        producer.flush(10)
        if acked:
            delivered += 1
        deadline = t0 + timeout_s
        seen = None
        while time.perf_counter() < deadline:
            cur.execute(
                "SELECT 1 FROM event_ledger WHERE correlation_id = %s::uuid LIMIT 1",
                (corr,),
            )
            if cur.fetchone():
                seen = time.perf_counter()
                break
            time.sleep(0.005)
        if seen is None:
            timeouts += 1
            continue
        if i >= warmup:
            latencies.append((seen - t0) * 1000)
            if acked:
                ack_ms.append((acked[0] - t0) * 1000)
    stats = latency_stats(latencies)
    result.update(stats)
    result["publish_ack"] = latency_stats(ack_ms)
    result["timeouts"] = timeouts
    result["warmup"] = warmup
    result["delivered"] = delivered
    # Cleanup 1: the event_ledger rows, by correlation id.
    cleanup: dict[str, Any] = {}
    try:
        time.sleep(
            1.0
        )  # rows of events that timed out may still land; delete after a beat
        cur.execute(
            "DELETE FROM event_ledger WHERE correlation_id = ANY(%s::uuid[])",
            (corr_ids,),
        )
        cleanup["event_ledger_rows_deleted"] = cur.rowcount
    except Exception as exc:  # noqa: BLE001 - recorded, never raised
        cleanup["event_ledger_rows_deleted"] = f"error: {exc}"
        cleanup["correlation_ids"] = corr_ids
    conn.close()
    # Cleanup 2: trim the topic back only when its retained log held nothing but our events.
    low1, high1 = watermark_consumer.get_watermark_offsets(
        TopicPartition(topic, 0), timeout=10
    )
    watermark_consumer.close()
    result["log_after"] = {"low": low1, "high": high1}
    if low0 == high0 and high1 - high0 == delivered:
        admin = AdminClient(base)
        try:
            fut = admin.delete_records(
                [TopicPartition(topic, 0, high1)], request_timeout=15
            )
            for f in fut.values() if isinstance(fut, dict) else [fut]:
                f.result(timeout=30)
            cleanup["topic_trimmed_to"] = high1
        except Exception as exc:  # noqa: BLE001
            cleanup["topic_trimmed_to"] = f"error: {exc}"
    else:
        cleanup["topic_trimmed_to"] = "skipped: the log held other events"
    result["cleanup"] = cleanup
    return result


def probe_health(n: int, url: str) -> dict[str, Any]:
    import urllib.request

    lat: list[float] = []
    codes: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - loopback health URL
                resp.read()
                codes.append(resp.status)
        except Exception:  # noqa: BLE001
            codes.append(0)
            continue
        lat.append((time.perf_counter() - t0) * 1000)
    out = latency_stats(lat)
    out["first_ms"] = round(lat[0], 3) if lat else None
    out["status_codes"] = codes
    return out


def container_probe(args: argparse.Namespace) -> dict[str, Any]:
    env = dict(os.environ)
    out: dict[str, Any] = {
        "role": "container-probe",
        "started_at": utc_now(),
        "bus_config": {
            "bootstrap": env.get("KAFKA_BOOTSTRAP_SERVERS"),
            "security_protocol": env.get("KAFKA_SECURITY_PROTOCOL") or "PLAINTEXT",
            "topic_namespace": topic_namespace_prefix(env),
            "environment": env.get("KAFKA_ENVIRONMENT"),
        },
        "postgres_host": env.get("POSTGRES_HOST"),
    }

    def health() -> dict[str, Any]:
        if not args.health_url:
            return {
                "error": "no health URL: the container declares no http healthcheck"
            }
        return probe_health(args.n_health, args.health_url)

    def bus_rtt() -> dict[str, Any]:
        return probe_bus_roundtrip(env, args.run_id, args.n_rtt, args.warmup)

    def projection() -> dict[str, Any]:
        return probe_projection(
            env, args.run_id, args.n_proj, args.proj_warmup, args.proj_timeout
        )

    sections: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("health", health),
        ("bus", bus_rtt),
        ("projection", projection),
    )
    for name, fn in sections:
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a failed section is data, not a crash
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    bus_raw = out.get("bus")
    bus: dict[str, Any] = bus_raw if isinstance(bus_raw, dict) else {}
    out["bus_roundtrip"] = bus.get("roundtrip", {})
    out["bus_oneway"] = bus.get("oneway", {})
    out["finished_at"] = utc_now()
    return out


# ---------------------------------------------------------------------------
# Role: remote collect (runs on the satellite)
# ---------------------------------------------------------------------------


def docker_bin() -> str:
    for cand in ("/usr/local/bin/docker", "/opt/homebrew/bin/docker"):
        if Path(cand).exists():
            return cand
    return "docker"


def collect_memory(docker: str) -> dict[str, Any]:
    mem: dict[str, Any] = {}
    _, vm = text_of(["vm_stat"])
    mem["vm_stat"] = parse_vm_stat(vm)
    _, swap = text_of(["sysctl", "vm.swapusage"])
    mem["swapusage"] = parse_swapusage(swap)
    rc, level = text_of(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    mem["pressure_level"] = (
        int(level.strip()) if rc == 0 and level.strip().isdigit() else None
    )
    _, memsize = text_of(["sysctl", "-n", "hw.memsize"])
    mem["hw_memsize_bytes"] = (
        int(memsize.strip()) if memsize.strip().isdigit() else None
    )
    _, ps = text_of(["ps", "-axo", "pid=,rss=,command="])
    procs: list[dict[str, Any]] = [dict(v) for v in parse_ps_rss(ps, DOCKER_VM_PROCESS)]
    footprint = 0
    for v in procs:
        _, top = text_of(
            ["top", "-l", "1", "-pid", str(v["pid"]), "-stats", "pid,mem"], timeout=20
        )
        size = parse_top_mem(top)
        v["footprint_bytes"] = size
        footprint += size or 0
    dvm: dict[str, Any] = {
        "processes": procs,
        "rss_bytes": sum(v["rss_bytes"] for v in procs) if procs else None,
        "footprint_bytes": footprint if procs else None,
    }
    mem["docker_vm"] = dvm
    settings_path = Path.home() / DOCKER_SETTINGS_REL
    try:
        settings = json.loads(settings_path.read_text())
        mem["docker_settings"] = {
            k: settings.get(k) for k in ("MemoryMiB", "Cpus", "SwapMiB", "DiskSizeMiB")
        }
    except (OSError, ValueError) as exc:
        mem["docker_settings"] = {"error": str(exc)}
    rc, info = text_of([docker, "info", "--format", "{{json .}}"], timeout=30)
    if rc == 0:
        try:
            d = json.loads(info)
            mem["docker_info"] = {
                "MemTotal": d.get("MemTotal"),
                "NCPU": d.get("NCPU"),
                "ServerVersion": d.get("ServerVersion"),
            }
        except ValueError:
            pass
    rc, stats = text_of(
        [docker, "stats", "--no-stream", "--format", "{{json .}}"], timeout=90
    )
    rows = []
    for line in stats.splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue
        used, limit = parse_docker_mem_usage(s.get("MemUsage", ""))
        rows.append(
            {
                "name": s.get("Name"),
                "mem_used_bytes": used,
                "mem_limit_bytes": limit,
                "cpu_percent": parse_percent(s.get("CPUPerc", "")),
                "pids": int(s["PIDs"]) if str(s.get("PIDs", "")).isdigit() else None,
            }
        )
    mem["docker_stats"] = rows
    return mem


def inspect_runtime(docker: str, names: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        rc, text = text_of([docker, "inspect", name], timeout=30)
        if rc != 0:
            out[name] = {"present": False}
            continue
        d = json.loads(text)[0]
        state = d.get("State") or {}
        health = state.get("Health") or {}
        log = health.get("Log") or []
        out[name] = {
            "present": True,
            "image": (d.get("Config") or {}).get("Image"),
            "status": state.get("Status"),
            "started_at": state.get("StartedAt"),
            "health": health.get("Status"),
            "last_health_probe": log[-1].get("End") if log else None,
            "restart_count": d.get("RestartCount"),
            "healthcheck_url": healthcheck_url(
                ((d.get("Config") or {}).get("Healthcheck") or {}).get("Test")
            ),
            "compose_project": ((d.get("Config") or {}).get("Labels") or {}).get(
                "com.docker.compose.project"
            ),
        }
    return out


def timed_restart(
    docker: str, names: Sequence[str], timeout_s: float
) -> dict[str, Any]:
    """Restart the pair and time until both report healthy. Only reached with --allow-restart
    on the lab-tenant arm."""
    t0 = time.monotonic()
    done = run([docker, "restart", *names], timeout=300)
    out: dict[str, Any] = {"restart_rc": done.returncode, "healthy_after_s": {}}
    pending = set(names)
    while pending and time.monotonic() - t0 < timeout_s:
        for name in list(pending):
            _, h = text_of(
                [docker, "inspect", "--format", "{{.State.Health.Status}}", name],
                timeout=15,
            )
            if h.strip() == "healthy":
                out["healthy_after_s"][name] = round(time.monotonic() - t0, 2)
                pending.discard(name)
        time.sleep(1.0)
    out["timed_out"] = sorted(pending)
    out["pair_healthy_s"] = (
        max(out["healthy_after_s"].values())
        if not pending and out["healthy_after_s"]
        else None
    )
    return out


def remote_collect(args: argparse.Namespace) -> dict[str, Any]:
    docker = docker_bin()
    arm = ARM_CONTAINERS[args.arm]
    out: dict[str, Any] = {
        "role": "remote-collect",
        "hostname": socket.gethostname(),
        "arm": args.arm,
        "started_at": utc_now(),
    }
    _, ncpu = text_of(["sysctl", "-n", "hw.ncpu"])
    _, load = text_of(["sysctl", "-n", "vm.loadavg"])
    out["cpu"] = {
        "ncpu": int(ncpu.strip()) if ncpu.strip().isdigit() else None,
        **parse_loadavg(load.strip("{} \n")),
    }
    out["ntp"] = clock_offset()
    out["memory"] = collect_memory(docker)
    pair = [arm["runtime"], arm["effects"]]
    out["runtime"] = inspect_runtime(docker, pair)
    if args.allow_restart and args.arm == "lab-tenant":
        out["cold_start"] = timed_restart(docker, pair, args.restart_timeout)
    else:
        out["cold_start"] = {
            "skipped": "timed restart runs only with --allow-restart on the lab-tenant arm",
            "last_started_at": {
                n: (out["runtime"].get(n) or {}).get("started_at") for n in pair
            },
        }
    if not (out["runtime"].get(arm["runtime"]) or {}).get("present"):
        out["probe"] = {"error": f"runtime container {arm['runtime']} is not present"}
        return out
    source = Path(__file__).read_bytes()
    probe_argv = [
        docker, "exec", "-i", arm["runtime"], "python", "-", "--container-probe",
        "--run-id", args.run_id, "--n-rtt", str(args.n_rtt), "--warmup", str(args.warmup),
        "--n-proj", str(args.n_proj), "--proj-warmup", str(args.proj_warmup),
        "--proj-timeout", str(args.proj_timeout), "--n-health", str(args.n_health),
    ]  # fmt: skip
    health_url = args.health_url or (out["runtime"].get(arm["runtime"]) or {}).get(
        "healthcheck_url"
    )
    if health_url:
        probe_argv += ["--health-url", health_url]
    done = run(probe_argv, timeout=args.probe_timeout, stdin=source)
    try:
        out["probe"] = json.loads(done.stdout.decode())
    except ValueError:
        out["probe"] = {
            "error": f"rc={done.returncode}",
            "stderr": done.stderr.decode(errors="replace")[-2000:],
        }
    out["finished_at"] = utc_now()
    return out


# ---------------------------------------------------------------------------
# Role: dependency side (runs on .201)
# ---------------------------------------------------------------------------


def remote_dependency(args: argparse.Namespace) -> dict[str, Any]:
    docker = docker_bin()
    out: dict[str, Any] = {
        "role": "remote-dependency",
        "hostname": socket.gethostname(),
        "started_at": utc_now(),
    }
    ncpu = os.cpu_count() or 0
    stat_a = Path("/proc/stat").read_text()
    time.sleep(1.0)
    stat_b = Path("/proc/stat").read_text()
    out["cpu"] = {
        "ncpu": ncpu,
        **parse_loadavg(Path("/proc/loadavg").read_text()),
        "busy_cores": cpu_busy_cores(stat_a, stat_b, ncpu),
    }
    out["memory"] = {
        k: round(v / GIB, 3)
        for k, v in parse_meminfo(Path("/proc/meminfo").read_text()).items()
    }
    try:
        psi = Path("/proc/pressure/memory").read_text().splitlines()[0]
        out["memory"]["psi_some"] = psi
    except (OSError, IndexError):
        pass
    out["ntp"] = clock_offset()
    pg = args.dep_postgres_container
    sql = "select coalesce(datname,'<none>'), coalesce(usename,'<none>'), state, count(*) from pg_stat_activity group by 1,2,3 order by 4 desc"
    rc, text = text_of(
        [
            docker,
            "exec",
            pg,
            "sh",
            "-c",
            f'psql -U "$POSTGRES_USER" -d postgres -AtF"|" -c {shlex.quote(sql)}',
        ],
        timeout=30,
    )
    per_db: dict[str, int] = {}
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) == 4 and parts[3].isdigit():
            rows.append(
                {
                    "datname": parts[0],
                    "usename": parts[1],
                    "state": parts[2],
                    "count": int(parts[3]),
                }
            )
            per_db[parts[0]] = per_db.get(parts[0], 0) + int(parts[3])
    out["postgres"] = {
        "container": pg,
        "rc": rc,
        "connections_per_db": per_db,
        "total": sum(per_db.values()),
        "rows": rows,
    }
    import urllib.request

    metrics_url = args.dep_redpanda_metrics_url
    if not metrics_url:
        _, ports = text_of(
            [docker, "port", args.dep_redpanda_container, f"{REDPANDA_ADMIN_PORT}/tcp"]
        )
        metrics_url = published_url(ports, "/public_metrics")
    out["redpanda_metrics_url"] = metrics_url

    def scrape() -> dict[str, float]:
        if not metrics_url:
            raise RuntimeError(
                f"{args.dep_redpanda_container} publishes no admin port {REDPANDA_ADMIN_PORT}"
            )
        with urllib.request.urlopen(metrics_url, timeout=15) as resp:  # noqa: S310
            return redpanda_counters(resp.read().decode())

    try:
        # Each sample is stamped when its scrape returns, so scrape time is not rated.
        a = scrape()
        ta = time.monotonic()
        time.sleep(args.dep_rate_seconds)
        b = scrape()
        interval = time.monotonic() - ta
        out["redpanda"] = {
            "interval_s": round(interval, 2),
            "rates_per_s": counter_rates(a, b, interval),
        }
    except Exception as exc:  # noqa: BLE001
        out["redpanda"] = {"error": str(exc)}
    rc, stats = text_of(
        [docker, "stats", "--no-stream", "--format", "{{json .}}"], timeout=120
    )
    _rc, names = text_of(
        [
            docker,
            "ps",
            "--filter",
            f"label=com.docker.compose.project={args.dep_compose_project}",
            "--format",
            "{{.Names}}",
        ]
    )
    lane = set(names.split())
    lane_rows: list[dict[str, Any]] = []
    for line in stats.splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue
        if s.get("Name") not in lane:
            continue
        used, _limit = parse_docker_mem_usage(s.get("MemUsage", ""))
        lane_rows.append(
            {
                "name": s.get("Name"),
                "mem_used_bytes": used,
                "cpu_percent": parse_percent(s.get("CPUPerc", "")),
            }
        )
    started: dict[str, str] = {}
    if lane:
        _, inspect = text_of(
            [
                docker,
                "inspect",
                "--format",
                "{{.Name}} {{.State.StartedAt}}",
                *sorted(lane),
            ],
            timeout=60,
        )
        for line in inspect.splitlines():
            parts = line.split()
            if len(parts) == 2:
                started[parts[0].lstrip("/")] = parts[1]
    for row in lane_rows:
        row["started_at"] = started.get(str(row["name"]))
    out["dev_lane_containers"] = {
        "project": args.dep_compose_project,
        "count": len(lane_rows),
        "min_uptime_s": min_uptime_s(started.values(), datetime.now(UTC)),
        "mem_total_gb": round(
            sum(r["mem_used_bytes"] or 0 for r in lane_rows) / GIB, 3
        ),
        "cpu_percent_total": round(sum(r["cpu_percent"] or 0 for r in lane_rows), 2),
        "rows": lane_rows,
    }
    out["finished_at"] = utc_now()
    return out


def dependency_metrics(dep: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (dep.get("cpu") or {}).items():
        if isinstance(v, (int, float)):
            out[f"cpu.{k}"] = float(v)
    for k, v in (dep.get("memory") or {}).items():
        if isinstance(v, (int, float)):
            out[f"mem.{k}_gb"] = float(v)
    pg = dep.get("postgres") or {}
    for db, count in (pg.get("connections_per_db") or {}).items():
        out[f"pg.connections.{db}"] = float(count)
    if isinstance(pg.get("total"), int):
        out["pg.connections_total"] = float(pg["total"])
    for k, v in ((dep.get("redpanda") or {}).get("rates_per_s") or {}).items():
        out[f"redpanda.{k}_per_s"] = float(v)
    lane = dep.get("dev_lane_containers") or {}
    for k in ("mem_total_gb", "cpu_percent_total", "count", "min_uptime_s"):
        if isinstance(lane.get(k), (int, float)):
            out[f"dev_lane.{k}"] = float(lane[k])
    return out


# ---------------------------------------------------------------------------
# Role: controller (runs on the operator Mac)
# ---------------------------------------------------------------------------


def resolve_target(host: str, explicit: str | None, env: Mapping[str, str]) -> str:
    if explicit:
        return explicit
    if env.get("ONEX_LAB_RUN_HOSTS"):
        table = Path(env["ONEX_LAB_RUN_HOSTS"])
    else:
        # Fail fast (Operating Rule 8): no default path when OMNI_HOME is unset.
        table = Path(env["OMNI_HOME"]).parent / LAB_TABLE_REL
    hosts = parse_lab_table(table.read_text())
    if host not in hosts:
        raise SystemExit(
            f"host {host!r} is not in {table}; known: {sorted(hosts)}; pass --target"
        )
    return hosts[host]


def ssh(
    target: str,
    lane: str,
    command: str,
    stdin: bytes | None = None,
    timeout: float = 900,
) -> subprocess.CompletedProcess[bytes]:
    return run(
        ["ssh", *SSH_OPTS, target, f"# lane={lane}\n{command}"],
        timeout=timeout,
        stdin=stdin,
    )


def stage_self(target: str, lane: str) -> tuple[str, str]:
    """Copy this file to the target once per content hash; return (remote path, remote python)."""
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:12]
    rel = f"{REMOTE_DIR}/bench_lab_tenant.{digest}.py"
    cmd = (
        f'export PATH="{REMOTE_PATH}"; mkdir -p "$HOME/{REMOTE_DIR}" && cat > "$HOME/{rel}" && '
        f"if [ -x {BREW_PYTHON} ]; then echo {BREW_PYTHON}; else command -v python3; fi"
    )
    done = ssh(target, lane, cmd, stdin=source, timeout=60)
    if done.returncode != 0:
        raise SystemExit(
            f"staging to {target} failed rc={done.returncode}: {done.stderr.decode()[-500:]}"
        )
    return rel, done.stdout.decode().strip().splitlines()[-1]


def remote_json(
    target: str,
    lane: str,
    rel: str,
    python: str,
    role_args: Sequence[str],
    timeout: float,
) -> dict[str, Any]:
    cmd = f'export PATH="{REMOTE_PATH}"; {python} "$HOME/{rel}" {shlex.join(role_args)}'
    done = ssh(target, lane, cmd, timeout=timeout)
    try:
        parsed: dict[str, Any] = json.loads(done.stdout.decode())
        return parsed
    except ValueError:
        return {
            "error": f"rc={done.returncode}",
            "stderr": done.stderr.decode(errors="replace")[-2000:],
        }


def default_placement_modules(home: Path) -> dict[str, Path]:
    """The two copies of landing_placement.py the landing controller's admission comes from:
    the installed omni plugin (cache) and the controller's own deployed copy."""
    found: dict[str, Path] = {}
    try:
        installed = json.loads(
            (home / ".claude" / "plugins" / "installed_plugins.json").read_text()
        )
        for key, entries in (installed.get("plugins") or {}).items():
            if key.startswith("omni@") and entries:
                path = (
                    Path(entries[0]["installPath"])
                    / "skills"
                    / "merge-drain"
                    / "scripts"
                    / "landing_placement.py"
                )
                if path.exists():
                    found["plugin_cache"] = path
    except (OSError, ValueError, KeyError, IndexError):
        pass
    controller = (
        home
        / ".omninode"
        / "landing-controller"
        / "skills"
        / "merge-drain"
        / "scripts"
        / "landing_placement.py"
    )
    if controller.exists():
        found["controller"] = controller
    return found


def home_relative(path: Path) -> str:
    """``~/...`` instead of an absolute home path, so a committed result names no machine path."""
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text.startswith(home + os.sep) else text


def read_placement(host: str, modules: Mapping[str, Path], lane: str) -> dict[str, Any]:
    """Run each landing_placement copy's own read for ``host`` exactly as the controller does."""
    import importlib.util

    out: dict[str, Any] = {}
    for label, path in modules.items():
        try:
            spec = importlib.util.spec_from_file_location(
                f"_bench_placement_{label}", path
            )
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            pool = [h for h in mod.load_pool() if h.name == host]
            if not pool:
                out[label] = {
                    "error": f"{host} is not in this copy's pool",
                    "path": home_relative(path),
                }
                continue
            placed = (
                mod.placed_counts(mod.placement_dir())
                if hasattr(mod, "placed_counts")
                else {}
            )
            reading = mod.read_pool(
                pool, placed=placed, lane=f"{lane} bench-placement"
            )[0]
            out[label] = {
                "path": home_relative(path),
                "describe": reading.describe(),
                "slots": reading.slots,
                "cap": reading.cap,
                "mem_avail_gb": round(reading.mem_avail_gb, 3),
                "load1": reading.load1,
                "cores": reading.cores,
                "mem_pressure": getattr(reading, "mem_pressure", None),
                "admission_refusal": getattr(reading, "admission_refusal", None),
                "admission_min_free_gb": getattr(mod, "ADMIT_MIN_FREE_GB", None),
                "error": reading.error,
            }
        except Exception as exc:  # noqa: BLE001 - a failed read is data
            out[label] = {
                "error": f"{type(exc).__name__}: {exc}",
                "path": home_relative(path),
            }
    return out


def harness_sha() -> str | None:
    rc, sha = text_of(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"]
    )
    return sha.strip() if rc == 0 else None


def controller(args: argparse.Namespace) -> dict[str, Any]:
    target = resolve_target(args.host, args.target, os.environ)
    rel, python = stage_self(target, args.lane)
    started = utc_now()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "kind": "dependency" if args.dependency else "satellite",
        "host": args.host,
        "target": target,
        "arm": args.arm,
        "started_at": started,
        "harness_sha": harness_sha(),
        "remote_python": python,
        "controller_ntp": clock_offset(),
        "params": {k: v for k, v in vars(args).items() if k not in ("func",)},
        "reps": [],
    }
    try:
        if args.dependency:
            dep_args = [
                "--remote-dependency", "--dep-postgres-container", args.dep_postgres_container,
                "--dep-redpanda-container", args.dep_redpanda_container,
                "--dep-compose-project", args.dep_compose_project,
                "--dep-rate-seconds", str(args.dep_rate_seconds),
            ]  # fmt: skip
            if args.dep_redpanda_metrics_url:
                dep_args += [
                    "--dep-redpanda-metrics-url",
                    args.dep_redpanda_metrics_url,
                ]
            for i in range(1, args.reps + 1):
                dep = remote_json(target, args.lane, rel, python, dep_args, timeout=300)
                result["reps"].append({"rep": i, "dependency": dep})
                if i < args.reps:
                    time.sleep(args.rep_gap)
            result["summary"] = aggregate(
                [dependency_metrics(r["dependency"]) for r in result["reps"]]
            )
            return result
        modules = (
            {k: Path(v) for k, v in (m.split("=", 1) for m in args.placement_module)}
            if args.placement_module
            else default_placement_modules(Path.home())
        )
        for i in range(1, args.reps + 1):
            run_id = f"{time.strftime('%Y%m%dt%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
            role_args = [
                "--remote-collect", "--arm", args.arm, "--run-id", run_id,
                "--n-rtt", str(args.n_rtt), "--warmup", str(args.warmup),
                "--n-proj", str(args.n_proj), "--proj-warmup", str(args.proj_warmup),
                "--proj-timeout", str(args.proj_timeout), "--n-health", str(args.n_health),
                "--probe-timeout", str(args.probe_timeout),
                "--restart-timeout", str(args.restart_timeout),
            ]  # fmt: skip
            if args.health_url:
                role_args += ["--health-url", args.health_url]
            if args.allow_restart:
                role_args.append("--allow-restart")
            remote = remote_json(
                target,
                args.lane,
                rel,
                python,
                role_args,
                timeout=args.probe_timeout + 300,
            )
            placement = (
                {}
                if args.skip_placement
                else read_placement(args.host, modules, args.lane)
            )
            rep = {"rep": i, "run_id": run_id, "remote": remote, "placement": placement}
            result["reps"].append(rep)
            print(
                f"rep {i}/{args.reps} {args.host} {args.arm}: {json.dumps(rep_metrics(rep), sort_keys=True)[:400]}",
                file=sys.stderr,
            )
            if i < args.reps:
                time.sleep(args.rep_gap)
        result["summary"] = aggregate([rep_metrics(r) for r in result["reps"]])
        return result
    finally:
        ssh(target, args.lane, f'rm -f "$HOME/{rel}"', timeout=30)
        result["finished_at"] = utc_now()


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

SUMMARY_ROWS: tuple[tuple[str, str], ...] = (
    ("mem.host.available_gb", "host available (free+inactive+spec) GB"),
    ("mem.host.free_gb", "host free GB"),
    ("mem.host.inactive_gb", "host inactive GB"),
    ("mem.host.speculative_gb", "host speculative GB"),
    ("mem.host.compressed_gb", "host compressor GB"),
    ("mem.host.wired_gb", "host wired GB"),
    ("mem.host.swap_used_mib", "swap used MiB"),
    ("mem.host.pressure_level", "memory pressure level"),
    ("mem.docker_vm.configured_gb", "Docker VM configured GB"),
    ("mem.docker_vm.rss_gb", "Docker VM process RSS GB"),
    ("mem.docker_vm.footprint_gb", "Docker VM process footprint GB"),
    ("mem.containers.total_gb", "containers total (docker stats) GB"),
    ("bus_roundtrip.p50_ms", "bus round trip p50 ms"),
    ("bus_roundtrip.p95_ms", "bus round trip p95 ms"),
    ("bus_roundtrip.p99_ms", "bus round trip p99 ms"),
    ("bus_oneway.p50_ms", "bus publish->consume p50 ms"),
    ("projection.p50_ms", "projection publish->row p50 ms"),
    ("projection.p95_ms", "projection publish->row p95 ms"),
    ("projection.p99_ms", "projection publish->row p99 ms"),
    ("health.p50_ms", "health probe p50 ms"),
    ("ntp.host_offset_ms", "host NTP offset ms"),
)


def _fmt(cell: Mapping[str, float] | None) -> str:
    if not cell:
        return "-"
    if cell["n"] <= 1:
        return f"{cell['median']:g}"
    return f"{cell['median']:g} ({cell['min']:g}-{cell['max']:g})"


def render_summary(results: Sequence[Mapping[str, Any]]) -> str:
    sats = sorted(
        (r for r in results if r.get("kind") == "satellite"),
        key=lambda r: (r["arm"], r["host"]),
    )
    deps = [r for r in results if r.get("kind") == "dependency"]
    lines: list[str] = []
    if sats:
        lines.append("Satellites: median (min-max) over repetitions")
        lines.append("")
        lines.append(
            "| metric | "
            + " | ".join(f"{r['host']} {r['arm']} (n={len(r['reps'])})" for r in sats)
            + " |"
        )
        lines.append("|---|" + "---|" * len(sats))
        placement_keys = sorted(
            {
                k
                for r in sats
                for k in r.get("summary", {})
                if k.startswith("placement.")
            }
        )
        container_keys = sorted(
            {
                k
                for r in sats
                for k in r.get("summary", {})
                if k.startswith("mem.container.")
            }
        )
        rows = [
            *SUMMARY_ROWS,
            *((k, k) for k in placement_keys),
            *((k, k) for k in container_keys),
        ]
        for key, label in rows:
            if not any(key in r.get("summary", {}) for r in sats):
                continue
            lines.append(
                f"| {label} | "
                + " | ".join(_fmt(r.get("summary", {}).get(key)) for r in sats)
                + " |"
            )
        lines.append("")
    for r in deps:
        lines.append(
            f"Dependency side: {r['host']} while satellites are {r['arm']} "
            f"({r['started_at']}, n={len(r.get('reps') or [])})"
        )
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, cell in sorted(r.get("summary", {}).items()):
            lines.append(f"| {key} | {_fmt(cell)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    role = p.add_mutually_exclusive_group()
    role.add_argument("--remote-collect", action="store_true", help=argparse.SUPPRESS)
    role.add_argument("--container-probe", action="store_true", help=argparse.SUPPRESS)
    role.add_argument(
        "--remote-dependency", action="store_true", help=argparse.SUPPRESS
    )
    role.add_argument(
        "--summarize",
        metavar="DIR",
        help="render a markdown summary of the JSON results in DIR",
    )
    role.add_argument(
        "--dependency",
        action="store_true",
        help="measure the dependency lane host (.201) instead of a satellite",
    )
    p.add_argument(
        "--host", help="lab host name from the lab host table (h105, h101, h201)"
    )
    p.add_argument("--target", help="ssh destination; overrides the lab host table")
    p.add_argument("--arm", choices=ARMS, default=None)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--rep-gap", type=float, default=20.0, help="seconds between repetitions"
    )
    p.add_argument("--out-dir", help="directory for the JSON result (default: stdout)")
    p.add_argument(
        "--lane", default=os.environ.get("ONEX_LAB_RUN_LANE", "lab-tenant-bench")
    )
    p.add_argument("--run-id", default="")
    p.add_argument("--n-rtt", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--n-proj", type=int, default=50)
    p.add_argument("--proj-warmup", type=int, default=3)
    p.add_argument("--proj-timeout", type=float, default=30.0)
    p.add_argument("--n-health", type=int, default=5)
    p.add_argument(
        "--health-url",
        default=None,
        help="override; default is the URL the runtime container's own healthcheck probes",
    )
    p.add_argument("--probe-timeout", type=float, default=900.0)
    p.add_argument(
        "--allow-restart",
        action="store_true",
        help="lab-tenant arm only: time a restart of the runtime pair to healthy",
    )
    p.add_argument("--restart-timeout", type=float, default=900.0)
    p.add_argument(
        "--placement-module", action="append", default=[], metavar="LABEL=PATH"
    )
    p.add_argument("--skip-placement", action="store_true")
    p.add_argument("--dep-postgres-container", default="omnibase-infra-postgres")
    p.add_argument(
        "--dep-redpanda-metrics-url",
        default=None,
        help="override; default is the dependency Redpanda's published admin port",
    )
    p.add_argument("--dep-redpanda-container", default="omnibase-infra-redpanda")
    p.add_argument("--dep-compose-project", default="omnibase-infra")
    p.add_argument("--dep-rate-seconds", type=float, default=30.0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def emit(obj: Any) -> None:
        print(json.dumps(obj, sort_keys=True, default=str))

    if args.container_probe:
        emit(container_probe(args))
        return 0
    if args.remote_collect:
        emit(remote_collect(args))
        return 0
    if args.remote_dependency:
        emit(remote_dependency(args))
        return 0
    if args.summarize:
        results = [
            json.loads(p.read_text())
            for p in sorted(Path(args.summarize).glob("*.json"))
        ]
        print(render_summary(results), end="")
        return 0
    if not args.host or not args.arm:
        build_parser().error("--host and --arm are required")
    result = controller(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{result['kind']}-{args.host}-{args.arm}.json"
        path.write_text(text + "\n")
        print(str(path))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
