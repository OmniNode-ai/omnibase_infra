# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""lane_census_inventory.py — fail-loud docker inventory collector (OMN-15466).

Collection counterpart to the pure planner (``lane_census_plan.py``). The planner
performs NO I/O; this module performs ALL of the census's docker I/O and emits the
planner's stdin envelope on stdout.

WHY THIS EXISTS — two defects in the single line it replaces
------------------------------------------------------------
``lane-census-check.sh`` previously gathered the inventory with::

    docker ps -a --no-trunc --format '{{json .}}' >ps.ndjson 2>/dev/null || : >ps.ndjson

D1 — ``{{json .}}`` silently opts into per-container SIZE computation.
    The Docker CLI's ``{{json .}}`` context carries a ``Size`` field, so the CLI
    sets ``size=1`` on ``GET /containers/json``. The daemon then runs
    ``snapshotter.Usage`` for EVERY container. Measured on ``.201`` (111
    containers, 2026-07-30): **90.363 s** for that exact command versus
    **0.150 s** for ``GET /containers/json?all=1`` without ``size``. The census
    reads none of the size data — the planner consumes only Names/State/Status/
    Image/Labels — so the cost is pure waste on the critical path.

    Transport is NOT the variable: the Engine API *with* ``size=1`` is equally
    slow (57.265 s) and the CLI *without* size is equally fast (0.128 s). That is
    why the fallback below pins an explicit field list and never ``{{json .}}``.

D2 — ``2>/dev/null || : >ps.ndjson`` converted any docker failure into a
    fabricated EMPTY inventory. ``2>/dev/null`` discarded the error, ``|| :``
    defeated ``set -e``, and the truncation handed the planner a zero-container
    envelope indistinguishable from a genuine total outage: 32 findings, all
    ``critical``, across all four lanes, published as a real drift event. A probe
    that cannot see is not a probe that saw nothing. Both paths here fail LOUD.

Ordering: Engine API first (authoritative, cheapest, unambiguous label typing),
Docker CLI second under an explicit ``timeout``, then hard failure. Exit code
``4`` is reserved for "the inventory could not be observed" and is deliberately
distinct from the driver's drift code ``30`` so an unobservable host can never be
reported as a drifted host.

Label typing note: the Engine API returns ``Labels`` as a real mapping. The CLI
returns a comma-joined ``k=v`` string in which a VALUE may itself contain commas
(``com.docker.compose.project.config_files`` routinely does), so the CLI form is
ambiguous by construction. The CLI parser here rejoins continuation segments so
both paths yield an identical envelope; the API path avoids the ambiguity
entirely.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.parse
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

# Exit code for "inventory could not be observed". Distinct from the driver's
# drift code (30) and its bad-args (2) / missing-deps (3) codes.
EXIT_PROBE_FAILED = 4

DEFAULT_DOCKER_SOCKET = "/var/run/docker.sock"
DEFAULT_API_TIMEOUT_S = 15.0
DEFAULT_CLI_TIMEOUT_S = 30.0

# Engine API inventory paths. NEITHER carries a `size` parameter — see D1.
API_CONTAINERS_PATH = "/containers/json?all=1"
API_NETWORKS_PATH = "/networks"

# Docker CLI fallback format. Enumerates exactly the fields the planner reads.
# MUST NOT be '{{json .}}': that emits a Size field and triggers size=1 (D1).
CLI_CONTAINER_FORMAT = "{{.Names}}\t{{.State}}\t{{.Status}}\t{{.Image}}\t{{.Labels}}"
_CLI_FIELDS = ("Names", "State", "Status", "Image", "Labels")


class InventoryProbeError(RuntimeError):
    """Raised when the container/network inventory could not be observed."""


class _UnixHTTPConnection(HTTPConnection):
    """HTTPConnection over an AF_UNIX socket (the Docker Engine API socket)."""

    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path
        self._timeout = timeout

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def api_get(socket_path: str, path: str, timeout: float) -> Any:
    """GET a Docker Engine API path over the unix socket and decode the JSON body."""
    conn = _UnixHTTPConnection(socket_path, timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise InventoryProbeError(
                f"Docker Engine API GET {path} returned HTTP {response.status}: "
                f"{body[:400].decode('utf-8', 'replace')}"
            )
        return json.loads(body)
    finally:
        conn.close()


def parse_cli_labels(raw: str) -> dict[str, str]:
    """Parse the Docker CLI's comma-joined ``k=v`` label string into a mapping.

    A label VALUE may contain commas (``...config_files=/a.yml,/b.yml``), so a
    naive ``split(",")`` corrupts such values. Segments without ``=`` are treated
    as continuations of the preceding value.
    """
    labels: dict[str, str] = {}
    current: str | None = None
    for segment in raw.split(","):
        if "=" in segment:
            key, value = segment.split("=", 1)
            current = key.strip()
            labels[current] = value.strip()
        elif current is not None and segment:
            labels[current] = f"{labels[current]},{segment}"
    return labels


def normalize_api_containers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Engine API container rows to the planner's envelope shape."""
    out: list[dict[str, Any]] = []
    for row in rows:
        names = row.get("Names") or []
        name = (names[0] if names else row.get("Name") or "").lstrip("/").strip()
        if not name:
            continue
        labels = row.get("Labels") or {}
        out.append(
            {
                "Names": name,
                "State": str(row.get("State") or ""),
                "Status": str(row.get("Status") or ""),
                "Image": str(row.get("Image") or ""),
                "Labels": {str(k): str(v) for k, v in labels.items()},
            }
        )
    return sorted(out, key=lambda r: str(r["Names"]))


def normalize_cli_containers(text: str) -> list[dict[str, Any]]:
    """Normalize tab-delimited Docker CLI rows to the planner's envelope shape."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(_CLI_FIELDS):
            parts = parts + [""] * (len(_CLI_FIELDS) - len(parts))
        name = parts[0].lstrip("/").strip()
        if not name:
            continue
        out.append(
            {
                "Names": name,
                "State": parts[1].strip(),
                "Status": parts[2].strip(),
                "Image": parts[3].strip(),
                "Labels": parse_cli_labels(parts[4]),
            }
        )
    return sorted(out, key=lambda r: str(r["Names"]))


def normalize_api_networks(rows: list[dict[str, Any]]) -> list[str]:
    """Extract network names from an Engine API ``GET /networks`` response."""
    return sorted({str(r.get("Name") or "") for r in rows if r.get("Name")})


def normalize_cli_networks(text: str) -> list[str]:
    """Extract network names from ``docker network ls --format '{{.Name}}'``."""
    return sorted({line.strip() for line in text.splitlines() if line.strip()})


def _run_cli(args: list[str], timeout_s: float) -> str:
    """Run a docker CLI command under an explicit bound. Raises on any failure.

    The bound is ``subprocess.run(timeout=...)``, which is portable — coreutils
    ``timeout(1)`` does not exist on macOS, and the gate/push host is a Mac. When
    ``timeout(1)`` IS present it is layered underneath as defence in depth so the
    docker client is reaped even if this process is itself wedged.
    """
    if shutil.which("docker") is None:
        raise InventoryProbeError("docker CLI not found on PATH")

    command = list(args)
    timeout_bin = shutil.which("timeout")
    if timeout_bin is not None:
        command = [timeout_bin, str(int(timeout_s)), *args]

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise InventoryProbeError(
            f"docker CLI fallback exceeded {timeout_s}s: {' '.join(command)}"
        ) from exc
    if proc.returncode == 124:
        raise InventoryProbeError(
            f"docker CLI fallback timed out after {timeout_s}s: {' '.join(command)}"
        )
    if proc.returncode != 0:
        raise InventoryProbeError(
            f"docker CLI fallback failed (exit {proc.returncode}): "
            f"{' '.join(command)}: {proc.stderr.strip()[:400]}"
        )
    return proc.stdout


def collect_inventory(
    *,
    socket_path: str,
    api_timeout_s: float,
    cli_timeout_s: float,
) -> tuple[list[dict[str, Any]], list[str], str, list[str]]:
    """Collect containers + networks, Engine API first, bounded CLI fallback.

    Returns ``(containers, networks, source, warnings)``. Raises
    :class:`InventoryProbeError` when BOTH paths fail — never returns an empty
    inventory to signal a failed probe (D2).
    """
    warnings: list[str] = []

    try:
        containers = normalize_api_containers(
            api_get(socket_path, API_CONTAINERS_PATH, api_timeout_s)
        )
        networks = normalize_api_networks(
            api_get(socket_path, API_NETWORKS_PATH, api_timeout_s)
        )
        return containers, networks, "engine_api", warnings
    except (OSError, InventoryProbeError, ValueError) as exc:
        warnings.append(
            f"engine_api path failed ({exc}); falling back to bounded docker CLI"
        )

    containers = normalize_cli_containers(
        _run_cli(
            ["docker", "ps", "-a", "--no-trunc", "--format", CLI_CONTAINER_FORMAT],
            cli_timeout_s,
        )
    )
    networks = normalize_cli_networks(
        _run_cli(["docker", "network", "ls", "--format", "{{.Name}}"], cli_timeout_s)
    )
    return containers, networks, "docker_cli", warnings


def build_envelope(
    *,
    lane: str | None,
    runtime_tag: str | None,
    containers: list[dict[str, Any]],
    networks: list[str],
    source: str,
) -> dict[str, Any]:
    """Assemble the planner's stdin envelope."""
    return {
        "lane": lane or None,
        "containers": containers,
        "networks": networks,
        "runtime_tag": runtime_tag or None,
        "inventory_source": source,
    }


# ---------------------------------------------------------------------------
# Memory observation (OMN-19959)
#
# Read in the same pass as the inventory, over the same Engine API socket, and
# written to its own file so the planner's stdin envelope is unchanged. Every
# read fails LOUD: an unreadable counter is an error, never a zero, because a
# zero is exactly what a healthy container reports.
# ---------------------------------------------------------------------------

#: The memory observation could not be read. The inventory envelope on stdout is
#: still valid, so the driver keeps its census; only the memory pass fails.
EXIT_MEMORY_UNOBSERVABLE = 7

DEFAULT_SYSFS_CGROUP_ROOT = "/sys/fs/cgroup"
DEFAULT_PROC_ROOT = "/proc"
#: The runner image's RUNNER_HOME is /home/runner/actions-runner
#: (docker/runners/entrypoint.sh); the worker logs sit in its _diag directory.
DEFAULT_RUNNER_DIAG_PATH = "/home/runner/actions-runner/_diag"
DEFAULT_RUNNER_FLEET_CONFIG = "~/.omnibase/runners/config/runner_fleet.yaml"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
_MEMORY_FILES = ("memory.max", "memory.peak", "memory.events")


class MemoryProbeError(RuntimeError):
    """Raised when a memory counter or a runner worker log could not be read."""


def api_get_raw(socket_path: str, path: str, timeout: float) -> bytes:
    """GET a Docker Engine API path and return the raw body (for archives)."""
    conn = _UnixHTTPConnection(socket_path, timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise MemoryProbeError(
                f"Docker Engine API GET {path} returned HTTP {response.status}: "
                f"{body[:400].decode('utf-8', 'replace')}"
            )
        return body
    finally:
        conn.close()


def lane_projects_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    """Map each lane's compose project to its lane name."""
    projects: dict[str, str] = {}
    for lane, spec in (manifest.get("lanes") or {}).items():
        project = (spec or {}).get("compose_project")
        if project:
            projects[str(project)] = str(lane)
    return projects


def runner_prefixes_from_fleet_config(config: Any) -> list[str]:
    """Every ``runner_name_prefix`` under the fleet config's ``hosts:`` block.

    The same set ``runner-monitor.sh`` ``config_host_prefixes`` reads, including
    the role pools nested under a host. Never a literal (OMN-19842).
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "runner_name_prefix" and isinstance(value, str) and value:
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    if isinstance(config, dict):
        walk(config.get("hosts") or {})
    return sorted(found)


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryProbeError(f"cannot read {path}: {exc}") from exc


def cgroup_dir_for_pid(pid: int, *, proc_root: str, sysfs_root: str) -> str:
    """Resolve a process's cgroup v2 directory from ``/proc/<pid>/cgroup``.

    Read from the process rather than assumed from the container id, so the
    runner pool's ``CgroupParent=omnirunners.slice`` and the systemd and cgroupfs
    drivers all resolve the same way.
    """
    text = _read_text(Path(proc_root) / str(pid) / "cgroup")
    for line in text.splitlines():
        if line.startswith("0::"):
            return str(Path(sysfs_root) / line[3:].lstrip("/"))
    raise MemoryProbeError(f"pid {pid} has no cgroup v2 entry: {text!r}")


def boot_time_iso(proc_root: str) -> str:
    """The host's boot time from ``/proc/stat`` ``btime``, as RFC 3339 UTC."""
    for line in _read_text(Path(proc_root) / "stat").splitlines():
        if line.startswith("btime "):
            stamp = datetime.fromtimestamp(int(line.split()[1]), tz=UTC)
            return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    raise MemoryProbeError(f"{proc_root}/stat carries no btime line")


def worker_logs_from_archive(
    archive: bytes, *, runner_name: str, since_epoch: float
) -> list[dict[str, str]]:
    """Worker logs in a ``_diag`` archive last written at or after ``since_epoch``.

    A log's mtime is its last write, so a log older than the window's start
    belongs to a job that finished before the window and is not read.
    """
    logs: list[dict[str, str]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            for member in tar:
                base = Path(member.name).name
                if not member.isfile() or not base.startswith("Worker_"):
                    continue
                if member.mtime < int(since_epoch):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    raise MemoryProbeError(f"{runner_name}: cannot read {member.name}")
                logs.append(
                    {
                        "runner_name": runner_name,
                        "log_name": base,
                        "text": handle.read().decode("utf-8", "replace"),
                    }
                )
    except tarfile.TarError as exc:
        raise MemoryProbeError(
            f"{runner_name}: unreadable _diag archive: {exc}"
        ) from exc
    return sorted(logs, key=lambda log: log["log_name"])


def _state_lower_bound(state_path: str | None, boot_id: str, boot_time: str) -> float:
    """Epoch lower bound for worker logs: the last published window's end, else boot."""
    fallback = datetime.fromisoformat(boot_time.replace("Z", "+00:00")).timestamp()
    if not state_path or not Path(state_path).exists():
        return fallback
    try:
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError) as exc:
        raise MemoryProbeError(f"unreadable memory state {state_path}: {exc}") from exc
    if state.get("host_boot_id") != boot_id or not state.get("window_end"):
        return fallback
    end = str(state["window_end"]).replace("Z", "+00:00")
    # Microsecond fractions are all fromisoformat needs here; the builder wrote it.
    return datetime.fromisoformat(end).timestamp()


def collect_memory_observation(
    *,
    socket_path: str,
    api_timeout_s: float,
    lane_projects: dict[str, str],
    runner_prefixes: list[str],
    proc_root: str,
    sysfs_root: str,
    runner_diag_path: str,
    state_path: str | None,
) -> dict[str, Any]:
    """Read every lane container's memory counters and the runners' worker logs."""
    boot_id = _read_text(Path(proc_root) / "sys/kernel/random/boot_id").strip()
    if not boot_id:
        raise MemoryProbeError("empty host boot id")
    boot_time = boot_time_iso(proc_root)
    since_epoch = _state_lower_bound(state_path, boot_id, boot_time)
    read_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    try:
        rows = api_get(socket_path, API_CONTAINERS_PATH, api_timeout_s)
    except (OSError, InventoryProbeError, ValueError) as exc:
        raise MemoryProbeError(f"container list unreadable: {exc}") from exc

    containers: list[dict[str, Any]] = []
    worker_logs: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("State") or "") != "running":
            continue
        cid = str(row.get("Id") or "")
        names = row.get("Names") or []
        name = (names[0] if names else "").lstrip("/").strip()
        if not cid or not name:
            continue
        labels = row.get("Labels") or {}
        lane = lane_projects.get(str(labels.get(COMPOSE_PROJECT_LABEL) or ""))
        is_runner = any(name.startswith(prefix) for prefix in runner_prefixes)
        if lane is None and not is_runner:
            continue

        try:
            detail = api_get(socket_path, f"/containers/{cid}/json", api_timeout_s)
        except (OSError, InventoryProbeError, ValueError) as exc:
            raise MemoryProbeError(f"{name}: inspect failed: {exc}") from exc
        state = detail.get("State") or {}
        pid = int(state.get("Pid") or 0)
        if not state.get("Running") or pid <= 0:
            # Stopped between the list and the inspect: no cgroup to read.
            continue

        if lane is not None:
            cgroup_dir = cgroup_dir_for_pid(
                pid, proc_root=proc_root, sysfs_root=sysfs_root
            )
            files = {f: _read_text(Path(cgroup_dir) / f) for f in _MEMORY_FILES}
            containers.append(
                {
                    "container_id": cid,
                    "container_name": name,
                    "lane": lane,
                    "started_at": str(state.get("StartedAt") or ""),
                    "memory_max": files["memory.max"],
                    "memory_peak": files["memory.peak"],
                    "memory_events": files["memory.events"],
                }
            )
        if is_runner:
            query = urllib.parse.urlencode({"path": runner_diag_path})
            try:
                archive = api_get_raw(
                    socket_path, f"/containers/{cid}/archive?{query}", api_timeout_s
                )
            except (OSError, MemoryProbeError) as exc:
                raise MemoryProbeError(f"{name}: _diag archive failed: {exc}") from exc
            worker_logs.extend(
                worker_logs_from_archive(
                    archive, runner_name=name, since_epoch=since_epoch
                )
            )

    return {
        "host_boot_id": boot_id,
        "boot_time": boot_time,
        "read_at": read_at,
        "containers": sorted(containers, key=lambda c: str(c["container_name"])),
        "worker_logs": worker_logs,
    }


def _memory_inputs() -> tuple[dict[str, str], list[str]]:
    """Lane projects from the lane manifest and runner prefixes from the fleet config.

    The fleet config is required: an absent file would read as "no runners here"
    and silently drop every CI job. A host with no runners says so explicitly
    with ``LANE_MEMORY_RUNNER_FLEET_CONFIG=""``.
    """
    import yaml  # lazy: the inventory path itself needs no third-party import

    manifest_path = os.environ.get("LANE_MANIFEST") or str(
        Path(__file__).resolve().parent.parent
        / "deploy"
        / "lane-census"
        / "lane-manifest.yaml"
    )
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise MemoryProbeError(f"lane manifest unreadable: {exc}") from exc

    fleet_path = os.environ.get("LANE_MEMORY_RUNNER_FLEET_CONFIG")
    if fleet_path is None:
        fleet_path = str(Path(DEFAULT_RUNNER_FLEET_CONFIG).expanduser())
    if fleet_path == "":
        return lane_projects_from_manifest(manifest), []
    try:
        with open(fleet_path, encoding="utf-8") as fh:
            fleet = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise MemoryProbeError(
            f"runner fleet config unreadable ({fleet_path}): {exc}. Set "
            'LANE_MEMORY_RUNNER_FLEET_CONFIG="" on a host that runs no runners.'
        ) from exc
    return lane_projects_from_manifest(manifest), runner_prefixes_from_fleet_config(
        fleet
    )


def write_memory_observation(
    path: str, *, socket_path: str, api_timeout_s: float, state_path: str | None
) -> None:
    lane_projects, runner_prefixes = _memory_inputs()
    observation = collect_memory_observation(
        socket_path=socket_path,
        api_timeout_s=api_timeout_s,
        lane_projects=lane_projects,
        runner_prefixes=runner_prefixes,
        proc_root=os.environ.get("LANE_MEMORY_PROC_ROOT", DEFAULT_PROC_ROOT),
        sysfs_root=os.environ.get("LANE_MEMORY_SYSFS_ROOT", DEFAULT_SYSFS_CGROUP_ROOT),
        runner_diag_path=os.environ.get(
            "LANE_MEMORY_RUNNER_DIAG_PATH", DEFAULT_RUNNER_DIAG_PATH
        ),
        state_path=state_path,
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(observation, fh, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default=os.environ.get("LANE") or None)
    parser.add_argument("--runtime-tag", default=os.environ.get("RUNTIME_TAG") or None)
    parser.add_argument(
        "--memory-out",
        default=None,
        help="also write the lane container memory observation here (OMN-19959)",
    )
    parser.add_argument(
        "--memory-state",
        default=None,
        help="the memory pass's state file, read only for the worker-log lower bound",
    )
    args = parser.parse_args(argv)

    socket_path = os.environ.get("LANE_CENSUS_DOCKER_SOCKET", DEFAULT_DOCKER_SOCKET)
    api_timeout_s = float(
        os.environ.get("LANE_CENSUS_API_TIMEOUT_S", DEFAULT_API_TIMEOUT_S)
    )
    cli_timeout_s = float(
        os.environ.get("LANE_CENSUS_CLI_TIMEOUT_S", DEFAULT_CLI_TIMEOUT_S)
    )

    try:
        containers, networks, source, warnings = collect_inventory(
            socket_path=socket_path,
            api_timeout_s=api_timeout_s,
            cli_timeout_s=cli_timeout_s,
        )
    except (OSError, InventoryProbeError, ValueError) as exc:
        # FAIL LOUD. Never emit an envelope — an unobservable host must not be
        # reported to the planner as an empty (i.e. totally-down) host.
        print(
            f"lane-census inventory probe FAILED (both Engine API and docker CLI): {exc}",
            file=sys.stderr,
        )
        return EXIT_PROBE_FAILED

    for warning in warnings:
        print(f"lane-census inventory: {warning}", file=sys.stderr)

    json.dump(
        build_envelope(
            lane=args.lane,
            runtime_tag=args.runtime_tag,
            containers=containers,
            networks=networks,
            source=source,
        ),
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()

    if args.memory_out:
        try:
            write_memory_observation(
                args.memory_out,
                socket_path=socket_path,
                api_timeout_s=api_timeout_s,
                state_path=args.memory_state,
            )
        except (OSError, MemoryProbeError, ValueError) as exc:
            print(
                f"lane container memory probe FAILED: {exc}",
                file=sys.stderr,
            )
            return EXIT_MEMORY_UNOBSERVABLE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
