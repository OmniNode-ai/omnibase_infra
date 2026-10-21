# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read Docker inspect over HTTP and grade its restart window through the CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import UnixStreamServer
from threading import Thread

import pytest

from scripts import lane_census_inventory as inventory
from tests.helpers.lane_census_fixtures import NEUTRAL_HOSTNAME, NEUTRAL_MANIFEST

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
_OBSERVED_AT = "2026-10-20T18:49:18.941827+00:00"


@pytest.mark.parametrize(
    ("started_at", "state", "observed_at", "is_loop"),
    [
        ("2026-10-20T17:45:04.123456789Z", "running", _OBSERVED_AT, False),
        ("2026-10-20T18:19:18.941827Z", "running", _OBSERVED_AT, False),
        ("2026-10-20T18:19:19.941827Z", "running", _OBSERVED_AT, True),
        ("2026-10-20T18:48:48Z", "running", _OBSERVED_AT, True),
        ("0001-01-01T00:00:00Z", "running", _OBSERVED_AT, True),
        ("2026-10-20T17:45:04Z", "exited", _OBSERVED_AT, True),
        ("2026-10-20T17:45:04Z", "running", "", True),
    ],
    ids=[
        "historical",
        "boundary",
        "inside-window",
        "recent",
        "zero",
        "exited",
        "no-census-time",
    ],
)
def test_inspect_start_time_reaches_planner_cli(
    tmp_path: Path, started_at: str, state: str, observed_at: str, is_loop: bool
) -> None:
    """Neither the collector nor the JSON/CLI seam may lose the clock fields."""
    requests: list[str] = []
    names = ("example-dev-redpanda", "example-dev-runtime")

    class DockerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            payload: object
            if self.path == "/containers/json?all=1":
                payload = [
                    {
                        "Names": [f"/{name}"],
                        "State": state if name == names[0] else "running",
                        "Status": "Exited (1)"
                        if name == names[0] and state == "exited"
                        else "Up (healthy)",
                        "Image": "example:current",
                        "Labels": {},
                    }
                    for name in names
                ]
            elif self.path == "/networks":
                payload = [{"Name": "example-dev-network"}]
            else:
                assert self.path in {f"/containers/{name}/json" for name in names}
                payload = {
                    "State": {
                        "StartedAt": started_at,
                        "Health": {"Status": "healthy", "FailingStreak": 0},
                    },
                    "Config": {"Healthcheck": {"Interval": 30_000_000_000}},
                    "RestartCount": 201 if names[0] in self.path else 0,
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    socket_path = str(tmp_path / "docker.sock")
    with UnixStreamServer(socket_path, DockerHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            containers, networks, source, warnings = inventory.collect_inventory(
                socket_path=socket_path, api_timeout_s=2, cli_timeout_s=2
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert source == "engine_api"
    assert warnings == []
    assert requests == [
        "/containers/json?all=1",
        "/networks",
        *(f"/containers/{name}/json" for name in names),
    ]
    envelope = inventory.build_envelope(
        lane="dev",
        runtime_tag=None,
        containers=containers,
        networks=networks,
        source=source,
        observed_at=observed_at,
    )
    envelope["host"] = NEUTRAL_HOSTNAME
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "scripts/lane_census_plan.py")],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "LANE_MANIFEST": str(NEUTRAL_MANIFEST),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        cwd=_ROOT,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    loops = [row for row in plan["findings"] if row["kind"] == "container_restart_loop"]
    assert [row["container"] for row in loops] == ([names[0]] if is_loop else [])
    assert plan["has_drift"] is is_loop
