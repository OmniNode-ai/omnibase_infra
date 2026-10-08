# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker, HTTP and pytest are faked; the collection path itself runs."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    grade_consumer_flow,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.unit


class FakeIO:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.sample = 0
        self.hwm = 10
        self.envelope: dict[str, Any] = {}
        self.urls: list[str] = []

    def http(self, url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == 30
        self.urls.append(url)
        obs = json.loads(
            (Path(__file__).parent / "fixtures/consumer_flow_recorded.json").read_text()
        )
        rows = obs["kinds"]["samples"][self.sample % 4]
        self.sample += 1
        return io.BytesIO(
            json.dumps(
                {"rows": rows, "row_count": 3, "row_limit": 500, "next_cursor": None}
            ).encode()
        )

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs["timeout"]))
        out = ""
        if argv[0] == "fake-pytest":
            xml = Path(
                next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml="))
            )
            branch = next((b for b in script.BRANCHES if b in xml.name), None)
            names = [
                script.AST_GATE_TEST,
                "counter[event_bus]",
                "counter[raw_event_projection]",
            ]
            cases = "".join(
                f'<testcase name="{n}">'
                + (
                    "<failure/>"
                    if branch and (n == script.AST_GATE_TEST or f"[{branch}]" in n)
                    else ""
                )
                + "</testcase>"
                for n in names
            )
            xml.write_text(f"<testsuite>{cases}</testsuite>")
            return subprocess.CompletedProcess(argv, int(branch is not None), "", "")
        assert argv[0] == "fake-docker"
        if argv[1] == "inspect":
            out = json.dumps(
                [
                    {
                        "Id": argv[-1],
                        "State": {
                            "Status": "running",
                            "StartedAt": "2026-09-26T00:00:00Z",
                            "Health": {"Status": "healthy"},
                        },
                    }
                ]
            )
        elif argv[1] == "logs":
            out = "runtime alive\n"
            if "--since" in argv:
                out += f"validation error for ModelConsumerFlowStallAlertTrigger\nmetric_name=boundary_swallow_prevented dlq_routed=true x topic={script.APPLIED_TOPIC} correlation_id={self.envelope['correlation_id']}\n"
        elif "psql" in argv[-1]:
            out = "local.omnimarket.projection_consumer_flow.consume.1.0.1\n"
        elif "describe" in argv:
            self.hwm += 1
            out = f"PARTITION HIGH-WATERMARK\n0 {self.hwm}\n"
        elif "produce" in argv:
            self.envelope = json.loads(kwargs["input"])
            out = "Produced to partition 0 at offset 42"
        elif "consume" in argv:
            if script.GENERIC_DLQ in argv:
                out = json.dumps(self.envelope)
            else:
                out = json.dumps({"value": json.dumps({"payload": {}}), "offset": 10})
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_full_collection_uses_bounded_io_and_restores_mutations(tmp_path: Path) -> None:
    target = tmp_path / script.WIRING_MODULE
    target.parent.mkdir(parents=True)
    original = "\n".join(
        f'def {f}():\n    flow_counters.register("group")\n'
        for f in script.BRANCHES.values()
    )
    target.write_text(original)
    io_layer = FakeIO()
    adapter = HandlerDockerConsumerFlowTarget(
        runner=io_layer.run,
        urlopen=io_layer.http,
        sleep=lambda _: None,
        repo_root=tmp_path,
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        base_url="http://projection.test",
        samples=2,
        sample_interval=0,
        settle_seconds=0,
        injection_wait=0,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )
    observed = asyncio.run(adapter.observe(request))
    assert observed.read_ok, observed.read_error
    assert grade_consumer_flow(request, observed).outcome == "PASS"
    assert target.read_text() == original
    assert all(0 < timeout <= 900 for _, timeout in io_layer.calls)
    assert any(timeout == 900 for _, timeout in io_layer.calls)
    assert all(
        url.startswith("http://projection.test/projection/") for url in io_layer.urls
    )
    assert observed.boot["injection"]["offset"] == 42


@pytest.mark.parametrize("failure", ["exit", "timeout", "json", "http"])
def test_unreadable_io_never_raises(failure: str) -> None:
    io_layer = FakeIO()

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 30)
        if failure == "exit":
            return subprocess.CompletedProcess(argv, 1, "", "unreachable docker")
        if failure == "json":
            return subprocess.CompletedProcess(argv, 0, "{}", "")
        return io_layer.run(argv, **kwargs)

    def http(url: str, **kwargs: Any) -> io.BytesIO:
        raise OSError("unreachable HTTP")

    adapter = HandlerDockerConsumerFlowTarget(
        runner=runner, urlopen=http, sleep=lambda _: None
    )
    observation = asyncio.run(
        adapter.observe(
            ModelConsumerFlowRequest(subject_lane="dev", docker_bin="fake-docker")
        )
    )
    assert not observation.read_ok
    assert observation.read_error


def test_cursor_walk_uses_since_and_stops_on_repeated_cursor() -> None:
    from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
        walk,
    )
    from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
        ConsumerFlowLane,
    )

    urls: list[str] = []

    def http(url: str, **kwargs: Any) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(
            json.dumps(
                {
                    "rows": [{"consumer_group": str(len(urls))}],
                    "row_count": 1,
                    "row_limit": 1,
                    "next_cursor": "same",
                }
            ).encode()
        )

    lane = ConsumerFlowLane(
        docker="unused", base_url="http://projection.test", urlopen=http
    )
    observed = walk(lane)
    assert not observed["terminated"]
    assert len(urls) == 2
    assert urls[1].endswith("?since=same")


@pytest.mark.parametrize(
    ("case", "beyond_rows", "reread_cursor", "proven"),
    [
        ("empty_beyond", [], None, True),
        ("silent_truncation", [{"projection_cursor": 10}], None, False),
        ("late_rows", [{"projection_cursor": 10}], "9", True),
        ("missing_cursor", [], None, False),
    ],
)
def test_cursor_walk_measures_the_full_final_page_end(
    case: str,
    beyond_rows: list[dict[str, Any]],
    reread_cursor: str | None,
    proven: bool,
) -> None:
    from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
        walk,
    )
    from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
        ConsumerFlowLane,
    )

    first_rows = [{"projection_cursor": 1}, {"projection_cursor": 2}]
    final_rows = (
        [{"consumer_group": "a"}, {"consumer_group": "b"}]
        if case == "missing_cursor"
        else [{"projection_cursor": "9"}, {"projection_cursor": 3}]
    )
    responses = [
        {"rows": first_rows, "row_count": 2, "row_limit": 2, "next_cursor": "2"},
        {"rows": final_rows, "row_count": 2, "row_limit": 2, "next_cursor": None},
        {"rows": beyond_rows},
        {
            "rows": final_rows,
            "row_count": 2,
            "row_limit": 2,
            "next_cursor": reread_cursor,
        },
    ]
    urls: list[str] = []

    def http(url: str, **kwargs: Any) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(json.dumps(responses[len(urls) - 1]).encode())

    observed = walk(
        ConsumerFlowLane(
            docker="unused", base_url="http://projection.test", urlopen=http
        )
    )
    assert observed["terminated"]
    assert len(observed["pages"]) == 2
    assert observed["rows"] == first_rows + final_rows
    assert "end_proof" not in observed["pages"][0]
    assert observed["pages"][-1]["end_proof"] == {
        "since": None if case == "missing_cursor" else "9",
        "beyond_row_count": None if case == "missing_cursor" else len(beyond_rows),
        "reread_next_cursor": reread_cursor,
        "proven": proven,
    }
    assert urls[1].endswith("?since=2")
    if case == "missing_cursor":
        assert len(urls) == 2
    else:
        assert urls[2].endswith("?since=9")
        assert len(urls) == (4 if beyond_rows else 3)
        if beyond_rows:
            assert urls[3] == urls[1]


@pytest.mark.parametrize("always_changes", [False, True])
def test_boot_change_retries_the_whole_observation(
    tmp_path: Path, always_changes: bool
) -> None:
    target = tmp_path / script.WIRING_MODULE
    target.parent.mkdir(parents=True)
    target.write_text(
        "\n".join(
            f'def {f}():\n    flow_counters.register("group")\n'
            for f in script.BRANCHES.values()
        )
    )
    fake = FakeIO()
    inspections = 0

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal inspections
        proc = fake.run(argv, **kwargs)
        if argv[1] == "inspect":
            inspections += 1
            body = json.loads(proc.stdout)
            # First after-read differs; subsequent reads agree unless requested.
            body[0]["Id"] = str(
                (inspections - 1) // 3 if always_changes else int(inspections > 3)
            )
            proc.stdout = json.dumps(body)
        return proc

    adapter = HandlerDockerConsumerFlowTarget(
        runner=run, urlopen=fake.http, sleep=lambda _: None, repo_root=tmp_path
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        samples=2,
        sample_interval=0,
        injection_wait=0,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )
    observed = asyncio.run(adapter.observe(request))
    assert observed.read_ok is not always_changes
    assert inspections == 12
    if always_changes:
        assert "replaced during the run" in observed.read_error
        assert not any(argv[0] == "fake-pytest" for argv, _ in fake.calls)


@pytest.mark.parametrize("refused_gets", [1, 2])
@pytest.mark.parametrize(
    "identity_change", ["replaced", "restarted", "stopped", "unreadable"]
)
def test_mid_run_redeploy_retries_after_connection_refused(
    tmp_path: Path, refused_gets: int, identity_change: str
) -> None:
    target = tmp_path / script.WIRING_MODULE
    target.parent.mkdir(parents=True)
    target.write_text(
        "\n".join(
            f'def {f}():\n    flow_counters.register("group")\n'
            for f in script.BRANCHES.values()
        )
    )
    fake = FakeIO()
    failures = 0
    identity_reread_pending = False

    def http(url: str, **kwargs: Any) -> io.BytesIO:
        nonlocal failures, identity_reread_pending
        if failures < refused_gets:
            failures += 1
            identity_reread_pending = True
            raise urllib.error.URLError(
                ConnectionRefusedError(111, "Connection refused")
            )
        return fake.http(url, **kwargs)

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal identity_reread_pending
        proc = fake.run(argv, **kwargs)
        if argv[1] == "inspect" and failures:
            body = json.loads(proc.stdout)
            if identity_change in ("replaced", "restarted"):
                body[0]["State"]["StartedAt"] = f"2026-10-08T00:59:{failures:02d}Z"
                if identity_change == "replaced":
                    body[0]["Id"] = f"new-{failures}-{argv[-1]}"
            elif identity_reread_pending:
                identity_reread_pending = False
                if identity_change == "unreadable":
                    return subprocess.CompletedProcess(argv, 1, "", "container missing")
                body[0]["State"]["Status"] = "exited"
            proc.stdout = json.dumps(body)
        return proc

    adapter = HandlerDockerConsumerFlowTarget(
        runner=run, urlopen=http, sleep=lambda _: None, repo_root=tmp_path
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        samples=2,
        sample_interval=0,
        settle_seconds=0,
        injection_wait=0,
        attempts=3,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )
    observed = asyncio.run(adapter.observe(request))
    assert observed.read_ok, observed.read_error
    assert failures == refused_gets
    assert grade_consumer_flow(request, observed).outcome == "PASS"
    assert sum(
        argv[1] == "logs" and "--since" not in argv for argv, _ in fake.calls
    ) == (refused_gets + 1) * len(script.RUNTIME_CONTAINERS)
    assert any(argv[0] == "fake-pytest" for argv, _ in fake.calls)


def test_connection_refused_with_unchanged_running_identity_stays_unreadable(
    tmp_path: Path,
) -> None:
    fake = FakeIO()
    gets = 0

    def http(url: str, **kwargs: Any) -> io.BytesIO:
        nonlocal gets
        gets += 1
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    adapter = HandlerDockerConsumerFlowTarget(
        runner=fake.run, urlopen=http, sleep=lambda _: None, repo_root=tmp_path
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        settle_seconds=0,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )
    observed = asyncio.run(adapter.observe(request))
    assert not observed.read_ok
    assert "Connection refused" in observed.read_error
    assert "replaced during the run" not in observed.read_error
    assert "ConsumerFlowBootChangedError" not in observed.read_error
    assert grade_consumer_flow(request, observed).outcome == "INDETERMINATE"
    assert gets == 1
    assert not any(argv[0] == "fake-pytest" for argv, _ in fake.calls)


@pytest.mark.parametrize("attempts", [1, 3])
def test_repeated_mid_run_redeploy_exhausts_attempts(
    tmp_path: Path, attempts: int
) -> None:
    fake = FakeIO()
    inspections = 0
    gets = 0

    def http(url: str, **kwargs: Any) -> io.BytesIO:
        nonlocal gets
        gets += 1
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal inspections
        proc = fake.run(argv, **kwargs)
        if argv[1] == "inspect":
            inspections += 1
            body = json.loads(proc.stdout)
            revision = (inspections - 1) // len(script.BOOT_CONTAINERS)
            body[0]["Id"] = str(revision)
            body[0]["State"]["StartedAt"] = f"2026-10-08T00:59:{revision:02d}Z"
            proc.stdout = json.dumps(body)
        return proc

    adapter = HandlerDockerConsumerFlowTarget(
        runner=run, urlopen=http, sleep=lambda _: None, repo_root=tmp_path
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        settle_seconds=0,
        attempts=attempts,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )
    observed = asyncio.run(adapter.observe(request))
    assert not observed.read_ok
    assert "replaced during the run" in observed.read_error
    assert "first read error:" in observed.read_error
    assert "Connection refused" in observed.read_error
    assert grade_consumer_flow(request, observed).outcome == "INDETERMINATE"
    assert gets == request.attempts
    assert inspections == 2 * request.attempts * len(script.BOOT_CONTAINERS)
    assert not any(argv[0] == "fake-pytest" for argv, _ in fake.calls)


def test_negative_timeout_restores_the_exact_original_bytes(tmp_path: Path) -> None:
    from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
        run_negative,
    )
    from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
        ConsumerFlowInputError,
    )

    target = tmp_path / script.WIRING_MODULE
    target.parent.mkdir(parents=True)
    original = (
        b'def _make_event_bus_callback():\r\n    flow_counters.register("group")\r\n'
    )
    target.write_bytes(original)
    fake = FakeIO()

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "event_bus.xml" in argv[-1]:
            assert b"flow_counters.register" not in target.read_bytes()
            raise subprocess.TimeoutExpired(argv, 900)
        return fake.run(argv, **kwargs)

    with pytest.raises(ConsumerFlowInputError, match="TimeoutExpired"):
        run_negative(tmp_path, ["fake-pytest"], tmp_path, runner=run)
    assert target.read_bytes() == original


@pytest.mark.parametrize("heals_at", [400.0, None])
def test_unhealthy_lane_waits_for_convergence_and_retries_once(
    tmp_path: Path, heals_at: float | None
) -> None:
    """OMN-20410: one unhealthy settle window is not a verdict.

    The .201 dev runtime flaps unhealthy for a few minutes at a time (C28 run
    37146175477 went INDETERMINATE on exactly that), so the observation waits a
    second settle window before giving up. A lane that never converges still
    reads unreadable, which grades INDETERMINATE and never PASS.
    """
    target = tmp_path / script.WIRING_MODULE
    target.parent.mkdir(parents=True)
    target.write_text(
        "\n".join(
            f'def {f}():\n    flow_counters.register("group")\n'
            for f in script.BRANCHES.values()
        )
    )
    fake = FakeIO()
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        proc = fake.run(argv, **kwargs)
        if argv[1] == "inspect" and (heals_at is None or clock[0] < heals_at):
            body = json.loads(proc.stdout)
            body[0]["State"]["Health"]["Status"] = "unhealthy"
            proc.stdout = json.dumps(body)
        return proc

    adapter = HandlerDockerConsumerFlowTarget(
        runner=run,
        urlopen=fake.http,
        sleep=sleep,
        monotonic=lambda: clock[0],
        repo_root=tmp_path,
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        samples=2,
        sample_interval=0,
        settle_seconds=300,
        injection_wait=0,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )
    observed = asyncio.run(adapter.observe(request))
    outcome = grade_consumer_flow(request, observed).outcome
    if heals_at is not None:
        assert observed.read_ok, observed.read_error
        assert outcome == "PASS"
        return
    assert not observed.read_ok
    assert "not running and healthy" in str(observed.read_error)
    assert "after 2 settle window(s)" in str(observed.read_error)
    assert clock[0] >= 2 * request.settle_seconds
    assert outcome == "INDETERMINATE"
    assert not any(argv[0] == "fake-pytest" for argv, _ in fake.calls)
