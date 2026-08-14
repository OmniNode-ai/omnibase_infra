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
import json
import os
import shutil
import socket
import subprocess
import sys
from http.client import HTTPConnection
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default=os.environ.get("LANE") or None)
    parser.add_argument("--runtime-tag", default=os.environ.get("RUNTIME_TAG") or None)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
