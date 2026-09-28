# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19894: lab probes pick their lane through the overlay's chain of responders.

Operator rulings 2026-09-28T01:58:06Z ("it's a problem if checks are reliant on
a specific machine") and 01:58:17Z ("chain of responders right?"). These tests
pin both halves of the fix:

* the resolver (``.github/actions/resolve-lab-lane``) grades the FIRST lane of
  the overlay's ordered list whose required surfaces answer, refuses a
  host-local address, and is RED -- never a skip, never a default -- when no
  lane answers;
* the five M4 chain canaries (C11, C12, C15, C16, C28) name no host label, no
  Docker host-gateway alias and no address, and take their runner pool and
  lane from the overlay.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "resolve-lab-lane"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

POOL_RUNS_ON = "${{ fromJSON(vars.LAB_PROBE_RUNS_ON_JSON) }}"
LANE_SIDE_RUNS_ON = "${{ fromJSON(needs.resolve-lane.outputs.docker_runs_on) }}"

# (workflow, probe job, where the probe job runs)
CANARIES = (
    ("chain-canary.yml", "chain-canary", POOL_RUNS_ON),
    ("chain-canary-c11-negative-paths.yml", "c11-negative-paths", POOL_RUNS_ON),
    (
        "chain-canary-c12-provider-catalogue.yml",
        "c12-provider-catalogue",
        LANE_SIDE_RUNS_ON,
    ),
    ("chain-canary-c16-receipt-identity.yml", "c16-receipt-identity", POOL_RUNS_ON),
    ("chain-canary-c28-consumer-flow.yml", "c28-consumer-flow", LANE_SIDE_RUNS_ON),
)


def _load_resolver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "resolve_lab_lane", ACTION_DIR / "resolve_lab_lane.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()

LANES = [
    {
        "name": "lane-a",
        "gateway_url": "http://10.0.0.1:8090",
        "projection_url": "http://10.0.0.1:3002",
        "docker_runs_on": ["self-hosted", "pool", "beside-a"],
    },
    {
        "name": "lane-b",
        "gateway_url": "http://10.0.0.2:8090",
        "projection_url": "http://10.0.0.2:3002",
        "docker_runs_on": ["self-hosted", "pool", "beside-b"],
    },
]


def _run_main(
    tmp_path: Path,
    lanes: Any,
    require: str,
    answering: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, dict[str, str], dict[str, str]]:
    probed: list[str] = []

    def fake_health(url: str, timeout: float) -> str | None:
        probed.append(url)
        return None if url in answering else "no answer (stub refuses)"

    monkeypatch.setattr(resolver, "http_health", fake_health)
    env_file = tmp_path / "env"
    out_file = tmp_path / "out"
    env_file.write_text("")
    out_file.write_text("")
    code = resolver.main(
        {
            "LAB_LANES_JSON": lanes if isinstance(lanes, str) else json.dumps(lanes),
            "LANE_REQUIRE": require,
            "GITHUB_ENV": str(env_file),
            "GITHUB_OUTPUT": str(out_file),
        }
    )

    def parse(path: Path) -> dict[str, str]:
        return dict(
            line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
        )

    return code, parse(env_file), parse(out_file)


def test_canary_lane_resolution_takes_the_first_lane_that_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, env, out = _run_main(
        tmp_path, LANES, "gateway_url", {"http://10.0.0.1:8090"}, monkeypatch
    )
    assert code == 0
    assert env["LANE_NAME"] == "lane-a"
    assert env["LANE_GATEWAY_URL"] == "http://10.0.0.1:8090"
    assert json.loads(out["docker_runs_on"]) == ["self-hosted", "pool", "beside-a"]


def test_canary_lane_resolution_falls_through_to_the_next_responder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain of responders: a silent first lane is skipped, not graded."""
    code, env, _ = _run_main(
        tmp_path, LANES, "gateway_url", {"http://10.0.0.2:8090"}, monkeypatch
    )
    assert code == 0
    assert env["LANE_NAME"] == "lane-b"


def test_canary_lane_resolution_needs_every_required_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, env, _ = _run_main(
        tmp_path,
        LANES,
        "gateway_url projection_url",
        {"http://10.0.0.1:8090", "http://10.0.0.2:8090", "http://10.0.0.2:3002"},
        monkeypatch,
    )
    assert code == 0
    assert env["LANE_NAME"] == "lane-b"


def test_canary_lane_resolution_skips_a_lane_missing_a_required_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lanes = [{"name": "no-docker", "gateway_url": "http://10.0.0.3:8090"}, *LANES]
    code, env, _ = _run_main(
        tmp_path,
        lanes,
        "gateway_url docker_runs_on",
        {"http://10.0.0.3:8090", "http://10.0.0.1:8090"},
        monkeypatch,
    )
    assert code == 0
    assert env["LANE_NAME"] == "lane-a"


def test_canary_no_responder_is_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, env, _ = _run_main(tmp_path, LANES, "gateway_url", set(), monkeypatch)
    assert code == 1
    assert env == {}
    err = capsys.readouterr().out
    assert "::error::" in err and "RED, not a skip" in err
    assert "lane-a" in err and "lane-b" in err


@pytest.mark.parametrize("raw", ["", "   ", "[]", "{}", "not json"])
def test_canary_no_responder_is_red_when_the_overlay_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    code, env, _ = _run_main(tmp_path, raw, "gateway_url", set(), monkeypatch)
    assert code == 1
    assert env == {}


@pytest.mark.parametrize(
    "url",
    [
        "http://host.docker.internal:8090",
        "http://localhost:8090",
        "http://127.0.0.1:8090",
        "http://10.0.0.1",
        "ftp://10.0.0.1:21",
    ],
)
def test_canary_lane_resolution_refuses_a_host_local_or_malformed_address(
    url: str,
) -> None:
    with pytest.raises(resolver.OverlayError):
        resolver.parse_lanes(json.dumps([{"name": "x", "gateway_url": url}]))


def test_canary_lane_resolution_refuses_a_duplicate_lane() -> None:
    with pytest.raises(resolver.OverlayError):
        resolver.parse_lanes(json.dumps([LANES[0], LANES[0]]))


@pytest.fixture
def health_server() -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_canary_lane_resolution_health_reads_a_real_answer(health_server: str) -> None:
    assert resolver.http_health(health_server, 5) is None
    closed = resolver.http_health("http://127.0.0.1:9", 2)
    assert closed is not None and "no answer" in closed


# --- the five canaries ------------------------------------------------------


def _job(workflow: str, job: str) -> dict[str, Any]:
    loaded = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    return loaded["jobs"][job]


@pytest.mark.parametrize(("workflow", "job", "runs_on"), CANARIES)
def test_canary_lane_resolution_wired_into_each_canary(
    workflow: str, job: str, runs_on: str
) -> None:
    text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
    assert _job(workflow, job)["runs-on"] == runs_on
    assert "./.github/actions/resolve-lab-lane" in text
    assert "vars.LAB_LANES_JSON" in text
    if runs_on == LANE_SIDE_RUNS_ON:
        assert _job(workflow, "resolve-lane")["runs-on"] == POOL_RUNS_ON


@pytest.mark.parametrize(("workflow", "job", "runs_on"), CANARIES)
def test_canary_names_no_machine(workflow: str, job: str, runs_on: str) -> None:
    text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
    for pin in ("host-201", "host-202", "host.docker.internal", "192.168.", "omnipc2"):
        assert pin not in text, f"{workflow} names a machine: {pin!r}"


def test_chain_canary_reads_the_lane_over_the_network_only() -> None:
    text = (WORKFLOWS / "chain-canary.yml").read_text(encoding="utf-8")
    for pin in ("docker logs", "docker.sock", "host.docker.internal"):
        assert pin not in text
