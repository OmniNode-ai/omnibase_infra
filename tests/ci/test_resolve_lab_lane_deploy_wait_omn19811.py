# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19811: lane resolution waits out a redeploy instead of going RED on it.

Scheduled chain-canary run 37125721313 (2026-10-03T13:18:24Z, head cbf0020dc52e)
failed at "Resolve the lab lane" with ``dev-201 ingress_url
http://192.168.86.201:8085 no answer (Connection refused)``. A redeploy-start
for that very sha had been published at 13:04:56Z (correlation caa1fd34), and
the deploy agent runs a deploy for about 987 s (its own mean service time), so
the runtime was being recreated when the resolver asked once and gave up. The
deploy-aware logic of OMN-19811 (pre-fire wait, one retry on terminal_missing)
lives in the chain-canary node, which only runs AFTER this step succeeds, so it
never saw that red.

These tests pin the shape of the fix:

* when no lane answers and the caller granted a wait budget
  (``LANE_DEPLOY_WAIT_SECONDS``, opt-in, default 0), the resolver reads the deploy
  agent each lane declares (``deploy_agent_url``) and keeps asking while the agent
  reports a deploy in progress (the same busy test as
  ``ModelDeployAgentSnapshot.busy``: not idle, settling, or commands queued);
* an idle agent, an unreadable agent, a lane that declares no agent, and a budget
  that runs out are all still RED, naming why. An unreadable agent is never
  evidence of a deploy, and a refusing ingress behind an idle agent is a lane
  that is down, not one that is redeploying;
* the seconds spent waiting are exported as ``LANE_WAITED_SECONDS`` so the
  workflow can charge them to the one ``deploy_wait_seconds`` budget the job's
  timeout was sized for.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "resolve-lab-lane"
CHAIN_CANARY = REPO_ROOT / ".github" / "workflows" / "chain-canary.yml"


def _load_resolver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "resolve_lab_lane_omn19811", ACTION_DIR / "resolve_lab_lane.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()

LANE_A = {
    "name": "lane-a",
    "gateway_url": "http://10.0.0.1:8090",
    "ingress_url": "http://10.0.0.1:8085",
    "deploy_agent_url": "http://10.0.0.1:8098",
}
LANE_B = {
    "name": "lane-b",
    "gateway_url": "http://10.0.0.2:8090",
    "ingress_url": "http://10.0.0.2:8085",
    "deploy_agent_url": "http://10.0.0.2:8098",
}
LANE_A_NO_AGENT = {k: v for k, v in LANE_A.items() if k != "deploy_agent_url"}

POLL_SECONDS = 5
BUDGET_SECONDS = 100


class World:
    """A fake clock, a lane whose ingress answers from ``ready_at`` and a deploy
    agent that reports ``busy_payload`` until ``busy_until``.

    ``time.sleep`` advances the clock and nothing else, so a test is instant and
    exact about how long the resolver waited.
    """

    def __init__(
        self,
        *,
        ready_at: float,
        busy_until: float,
        busy_payload: dict[str, Any] | None = None,
        agent_readable: bool = True,
    ) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.agent_reads: list[str] = []
        self.ready_at = ready_at
        self.busy_until = busy_until
        self.busy_payload = busy_payload or DEPLOYING
        self.agent_readable = agent_readable

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def health(self, url: str, timeout: float) -> str | None:
        if url.startswith("http://10.0.0.1:") and self.now >= self.ready_at:
            return None
        return "no answer (URLError: [Errno 111] Connection refused)"

    def read_agent(self, url: str, timeout: float) -> dict[str, Any] | None:
        self.agent_reads.append(url)
        if not self.agent_readable:
            return None
        if self.now < self.busy_until:
            return self.busy_payload
        return IDLE


IDLE: dict[str, Any] = {
    "health": {"state": "idle", "active_job": None, "last_result": {"settling": False}},
    "queue": {"commands_ahead": 0, "control_topic_lag_age_seconds": 1.0},
}
DEPLOYING: dict[str, Any] = {
    "health": {
        "state": "deploying",
        "active_job": {"correlation_id": "c7530761-6009-45cb-9e36-d8174755ac3b"},
        "last_result": {"settling": False},
    },
    "queue": {"commands_ahead": 0, "control_topic_lag_age_seconds": 1.0},
}
SETTLING: dict[str, Any] = {
    "health": {
        "state": "settling",
        "active_job": None,
        "last_result": {"settling": True, "settling_stage": "lab_overlay"},
    },
    "queue": {"commands_ahead": 0, "control_topic_lag_age_seconds": 1.0},
}
IDLE_BUT_LAST_RESULT_SETTLING: dict[str, Any] = {
    "health": {
        "state": "idle",
        "active_job": None,
        "last_result": {"settling": True},
    },
    "queue": {"commands_ahead": 0, "control_topic_lag_age_seconds": 1.0},
}
IDLE_BUT_COMMAND_QUEUED: dict[str, Any] = {
    "health": {"state": "idle", "active_job": None, "last_result": {"settling": False}},
    "queue": {"commands_ahead": 2, "control_topic_lag_age_seconds": 1.0},
}
BUSY_SHAPES = (
    pytest.param(DEPLOYING, id="deploying"),
    pytest.param(SETTLING, id="settling"),
    pytest.param(IDLE_BUT_LAST_RESULT_SETTLING, id="last-result-settling"),
    pytest.param(IDLE_BUT_COMMAND_QUEUED, id="command-queued"),
)


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    lanes: list[dict[str, Any]],
    *,
    wait: str | None = str(BUDGET_SECONDS),
) -> tuple[int, dict[str, str]]:
    monkeypatch.setattr(resolver, "http_health", world.health)
    monkeypatch.setattr(resolver, "read_deploy_agent", world.read_agent, raising=False)
    monkeypatch.setattr(time, "sleep", world.sleep)
    monkeypatch.setattr(time, "monotonic", world.monotonic)
    env_file = tmp_path / "env"
    env_file.write_text("")
    environ = {
        "LAB_LANES_JSON": json.dumps(lanes),
        "LANE_REQUIRE": "ingress_url gateway_url",
        "LANE_DEPLOY_POLL_SECONDS": str(POLL_SECONDS),
        "GITHUB_ENV": str(env_file),
    }
    if wait is not None:
        environ["LANE_DEPLOY_WAIT_SECONDS"] = wait
    code = resolver.main(environ)
    written = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
    )
    return code, written


@pytest.mark.parametrize("busy", BUSY_SHAPES)
def test_ingress_refusing_during_a_redeploy_is_waited_out(
    busy: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 13:18Z red: the agent is mid-deploy and the ingress refuses. The
    resolver waits the deploy out, then grades the lane that came back."""
    world = World(ready_at=40, busy_until=40, busy_payload=busy)
    code, written = _run(tmp_path, monkeypatch, world, [LANE_A])
    assert code == 0
    assert written["LANE_NAME"] == "lane-a"
    waited = float(written["LANE_WAITED_SECONDS"])
    assert 40 <= waited <= 40 + POLL_SECONDS
    assert world.agent_reads == [LANE_A["deploy_agent_url"]] * len(world.agent_reads)


def test_a_refusing_ingress_behind_an_idle_agent_is_down_not_redeploying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=10_000, busy_until=0)
    code, _ = _run(tmp_path, monkeypatch, world, [LANE_A])
    assert code == 1
    assert world.sleeps == []


def test_an_unreadable_agent_is_never_evidence_of_a_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=10_000, busy_until=10_000, agent_readable=False)
    code, _ = _run(tmp_path, monkeypatch, world, [LANE_A])
    assert code == 1
    assert world.sleeps == []


def test_a_lane_that_declares_no_agent_is_never_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=10_000, busy_until=10_000)
    code, _ = _run(tmp_path, monkeypatch, world, [LANE_A_NO_AGENT])
    assert code == 1
    assert world.sleeps == []
    assert world.agent_reads == []


def test_a_deploy_that_outlasts_the_budget_is_red_and_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    world = World(ready_at=10_000, busy_until=10_000)
    code, written = _run(tmp_path, monkeypatch, world, [LANE_A])
    out = capsys.readouterr().out
    assert code == 1
    assert 0 < world.now <= BUDGET_SECONDS
    assert "deploying" in out
    assert str(BUDGET_SECONDS) in out
    assert "LANE_NAME" not in written


def test_a_deploy_that_ends_with_the_ingress_still_down_is_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent going idle ends the wait: a lane whose redeploy finished and
    whose ingress still refuses is down, and waiting longer would hide that."""
    world = World(ready_at=10_000, busy_until=40)
    code, _ = _run(tmp_path, monkeypatch, world, [LANE_A])
    assert code == 1
    assert world.now <= 40 + POLL_SECONDS


@pytest.mark.parametrize("wait", [None, "0"])
def test_waiting_is_opt_in(
    wait: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other probe that uses this action keeps its single-shot answer."""
    world = World(ready_at=40, busy_until=40)
    code, _ = _run(tmp_path, monkeypatch, world, [LANE_A], wait=wait)
    assert code == 1
    assert world.sleeps == []


def test_a_lane_that_answers_is_chosen_without_any_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain of responders is unchanged: the next lane that answers wins and
    the redeploying lane before it costs nothing."""
    world = World(ready_at=10_000, busy_until=10_000)
    monkeypatch.setattr(
        world,
        "health",
        lambda url, timeout: (
            None if url.startswith("http://10.0.0.2:") else "no answer (refused)"
        ),
    )
    code, written = _run(tmp_path, monkeypatch, world, [LANE_A, LANE_B])
    assert code == 0
    assert written["LANE_NAME"] == "lane-b"
    assert world.sleeps == []
    assert "LANE_WAITED_SECONDS" not in written


# --- the agent reader, over a real socket ----------------------------------


def _serve(routes: dict[str, tuple[int, bytes]]) -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, body = routes.get(self.path, (404, b"{}"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_the_agent_reader_takes_a_503_body_and_the_queue() -> None:
    """The agent answers /health 503 with a full body when its accept backlog is
    unhealthy (deploy_agent_window.read_deploy_agent_via_httpx says so); that
    body is still the agent's state."""
    health = json.dumps({"state": "deploying", "active_job": None}).encode()
    queue = json.dumps({"commands_ahead": 1}).encode()
    for base in _serve({"/health": (503, health), "/queue": (200, queue)}):
        snapshot = resolver.read_deploy_agent(base, 5)
    assert snapshot == {
        "health": {"state": "deploying", "active_job": None},
        "queue": {"commands_ahead": 1},
    }
    assert resolver.deploy_agent_busy_reason(snapshot) == "state=deploying"


def test_the_agent_reader_keeps_health_when_the_queue_is_unreadable() -> None:
    health = json.dumps({"state": "idle", "active_job": None}).encode()
    for base in _serve({"/health": (200, health), "/queue": (200, b"not json")}):
        snapshot = resolver.read_deploy_agent(base, 5)
    assert snapshot == {"health": {"state": "idle", "active_job": None}, "queue": None}
    assert resolver.deploy_agent_busy_reason(snapshot) is None


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b""])
def test_the_agent_reader_returns_none_for_a_health_that_is_not_an_object(
    body: bytes,
) -> None:
    for base in _serve({"/health": (200, body)}):
        assert resolver.read_deploy_agent(base, 5) is None


def test_the_agent_reader_returns_none_for_a_closed_port_and_a_bad_scheme() -> None:
    assert resolver.read_deploy_agent("http://127.0.0.1:9", 2) is None
    assert resolver.read_deploy_agent("file:///etc/passwd", 2) is None


def test_a_stale_queue_sample_is_not_a_queued_command() -> None:
    """Same bound as queued_commands_from_payload: the agent samples lag only
    when it polls, so an old sample describes the queue before the rebuild."""
    idle_health = {"state": "idle", "active_job": None, "last_result": {}}
    stale = {"commands_ahead": 3, "control_topic_lag_age_seconds": 500}
    fresh = {"commands_ahead": 3, "control_topic_lag_age_seconds": 5}
    assert (
        resolver.deploy_agent_busy_reason({"health": idle_health, "queue": stale})
        is None
    )
    assert (
        resolver.deploy_agent_busy_reason({"health": idle_health, "queue": fresh})
        == "3 command(s) queued"
    )


# --- the workflow half ------------------------------------------------------


def _chain_canary() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(CHAIN_CANARY.read_text(encoding="utf-8"))
    return loaded


def _resolve_step() -> dict[str, Any]:
    steps = _chain_canary()["jobs"]["chain-canary"]["steps"]
    found = [s for s in steps if s.get("uses") == "./.github/actions/resolve-lab-lane"]
    assert len(found) == 1
    step: dict[str, Any] = found[0]
    return step


def test_the_action_declares_an_opt_in_wait_input() -> None:
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text(encoding="utf-8"))
    declared = action["inputs"]["deploy-wait-seconds"]
    assert declared["default"] == "0"
    assert declared["required"] is False
    resolve = action["runs"]["steps"][0]
    assert "inputs.deploy-wait-seconds" in resolve["env"]["LANE_DEPLOY_WAIT_SECONDS"]


def test_chain_canary_gives_lane_resolution_the_same_deploy_budget() -> None:
    with_block = _resolve_step()["with"]
    assert "inputs.deploy_wait_seconds" in with_block["deploy-wait-seconds"]
    assert "'2400'" in with_block["deploy-wait-seconds"]


def test_chain_canary_charges_the_lane_wait_to_the_probes_deploy_budget() -> None:
    """The job's 65-minute timeout was sized for ONE 2400 s deploy wait shared by
    the pre-fire wait and the retry. Resolving the lane may not add a second one:
    the probe gets what the resolver did not spend."""
    steps = _chain_canary()["jobs"]["chain-canary"]["steps"]
    fire = next(s for s in steps if s.get("name", "").startswith("Fire one live"))
    assert "LANE_WAITED_SECONDS" in fire["run"]
