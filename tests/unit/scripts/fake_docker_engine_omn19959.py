# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A fake Docker Engine API and host tree for the lane memory pass tests (OMN-19959).

The memory pass reads three things a unit test must not touch on a real host:
the Docker Engine API socket, cgroup v2 files under ``/sys/fs/cgroup`` and
``/proc``. This module serves the first over AF_UNIX from a thread and writes
the other two as plain files, so the collector and ``lane-census-check.sh`` run
end to end against fixtures. The trusted CI pool is self-hosted on the lab
hosts, so a test that reached the real daemon would inventory real lanes.
"""

from __future__ import annotations

import io
import json
import os
import socketserver
import tarfile
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BOOT_ID = "be799359-43ff-492e-9c8b-67d7f0fb9c51"
# 2026-09-28T12:10:00Z
BOOT_EPOCH = 1790597400
SIM_PROJECT = "omnibase-infra-sim-202"


@dataclass
class FakeContainer:
    cid: str
    name: str
    pid: int
    project: str | None
    started_at: str = "2026-09-28T12:11:26.55947631Z"
    memory_max: str = "max\n"
    memory_peak: str = "2114498560\n"
    memory_events: str = "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
    image: str = "redpandadata/redpanda:v24"
    # For a runner: {"Worker_...log": (text, mtime_epoch)}
    worker_logs: dict[str, tuple[str, int]] = field(default_factory=dict)
    diag_readable: bool = True


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    containers: list[FakeContainer]


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def address_string(self) -> str:
        return "unix"

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        by_id = {c.cid: c for c in self.server.containers}
        if path == "/containers/json":
            rows = [
                {
                    "Id": c.cid,
                    "Names": [f"/{c.name}"],
                    "State": "running",
                    "Status": "Up 1 hour",
                    "Image": c.image,
                    "Labels": (
                        {"com.docker.compose.project": c.project} if c.project else {}
                    ),
                }
                for c in self.server.containers
            ]
            self._send(200, json.dumps(rows).encode())
            return
        if path == "/networks":
            self._send(200, json.dumps([{"Name": f"{SIM_PROJECT}-network"}]).encode())
            return
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "containers" and parts[1] in by_id:
            container = by_id[parts[1]]
            if parts[2] == "json":
                detail = {
                    "State": {
                        "Running": True,
                        "Pid": container.pid,
                        "StartedAt": container.started_at,
                    }
                }
                self._send(200, json.dumps(detail).encode())
                return
            if parts[2] == "archive":
                if not container.worker_logs or not container.diag_readable:
                    self._send(404, b'{"message":"Could not find the file"}')
                    return
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w") as tar:
                    for log_name, (text, mtime) in container.worker_logs.items():
                        data = text.encode()
                        info = tarfile.TarInfo(name=f"_diag/{log_name}")
                        info.size = len(data)
                        info.mtime = mtime
                        tar.addfile(info, io.BytesIO(data))
                self._send(200, buf.getvalue(), "application/x-tar")
                return
        self._send(404, b'{"message":"not found"}')


class FakeHost:
    """A running fake Engine API plus proc and cgroup trees for its containers."""

    def __init__(self, root: Path, containers: list[FakeContainer]) -> None:
        # macOS caps AF_UNIX paths near 104 bytes; pytest's tmp_path exceeds it.
        self._sock_dir = Path(tempfile.mkdtemp(prefix="mem-"))
        self.socket_path = self._sock_dir / "d.sock"
        self.proc_root = root / "proc"
        self.sysfs_root = root / "cgroup"
        self.containers = containers
        self._server = _Server(str(self.socket_path), _Handler)
        self._server.containers = containers
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.write_tree()

    def write_tree(self) -> None:
        (self.proc_root / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
        (self.proc_root / "sys/kernel/random/boot_id").write_text(f"{BOOT_ID}\n")
        (self.proc_root / "stat").write_text(f"cpu 1 2 3\nbtime {BOOT_EPOCH}\n")
        for c in self.containers:
            rel = f"system.slice/docker-{c.cid}.scope"
            (self.proc_root / str(c.pid)).mkdir(parents=True, exist_ok=True)
            (self.proc_root / str(c.pid) / "cgroup").write_text(f"0::/{rel}\n")
            cg = self.sysfs_root / rel
            cg.mkdir(parents=True, exist_ok=True)
            (cg / "memory.max").write_text(c.memory_max)
            (cg / "memory.peak").write_text(c.memory_peak)
            (cg / "memory.events").write_text(c.memory_events)

    def env(self) -> dict[str, str]:
        return {
            "LANE_CENSUS_DOCKER_SOCKET": str(self.socket_path),
            "LANE_MEMORY_PROC_ROOT": str(self.proc_root),
            "LANE_MEMORY_SYSFS_ROOT": str(self.sysfs_root),
        }

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.socket_path.unlink(missing_ok=True)


def worker_log(*, repo: str, run_id: str, started: str, completed: str | None) -> str:
    """A runner worker log in the shape read on omnipc2-ci-runner-13 (2026-09-28)."""
    lines = [
        f"[{started} INFO HostContext] No proxy settings were found",
        f"[{started} INFO Worker] Waiting to receive the job message from the channel.",
        "        {",
        '          "k": "repository",',
        f'          "v": "{repo}"',
        "        },",
        "        {",
        '          "k": "run_id",',
        f'          "v": "{run_id}"',
        "        },",
    ]
    if completed:
        lines.append(
            f"[{completed} INFO JobRunner] Raising job completed against run service"
        )
        lines.append(f"[{completed} INFO Worker] Job completed.")
    return "\n".join(lines) + "\n"
