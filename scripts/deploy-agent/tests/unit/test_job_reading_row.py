# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A deploy job's terminal outcome is a ledger reading row (OMN-24617).

A failed job on the lab dev host left no reading anyone acted on: the stability follow
failed from 09:58Z and no ledger row named it for about four hours. These tests hold the
row the agent now writes for each terminal job on the dev and stability lanes, the route
it takes (the append command the reading writer already uses, no topic of its own) and the
call site in ``DeployAgent._publish_terminal``.

The handler is loaded the way the agent loads it, by file path from its own checkout.
"""

from __future__ import annotations

import ast
import json
import re
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from deploy_agent import agent as agent_mod
from deploy_agent import events as events_mod
from deploy_agent.agent import JOB_READING
from deploy_agent.events import DEPLOY_PHASE_ORDER, EnumRuntimeLane, Phase, PhaseStatus
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.job_state import JobState

pytestmark = pytest.mark.unit

runtime_mod = load_dev_boot_guard(JOB_READING)
handler_mod = runtime_mod.handler_module
request_mod = runtime_mod.request_module

DEV = EnumRuntimeLane.DEV.value
STABILITY = EnumRuntimeLane.STABILITY_TEST.value
PROD = EnumRuntimeLane.PROD.value

NOW = datetime(2026, 11, 15, 14, 40, 0, tzinfo=UTC)
DONE = datetime(2026, 11, 15, 14, 39, 30, tzinfo=UTC)

# The topics node_reading_writer_effect declares in its contract (omnimarket), spelled here
# so the check runs on every runner whether or not that clone is present.
READING_WRITER_TOPICS = frozenset(
    {
        "onex.cmd.omnimarket.reading-window-requested.v1",
        "onex.evt.omnimarket.reading-piece-written.v1",
        "onex.evt.omnimarket.reading-piece-failed.v1",
        "onex.evt.omnimarket.reading-window-completed.v1",
        "onex.evt.omnimarket.reading-window-failed.v1",
        "onex.cmd.omnimarket.relation-map-reading-requested.v1",
        "onex.evt.omnimarket.relation-map-reading-written.v1",
        "onex.evt.omnimarket.relation-map-reading-failed.v1",
    }
)

# The agent's own topic constants before this change; the change adds none.
AGENT_TOPICS_BEFORE = frozenset(
    {
        "onex.cmd.deploy.rebuild-requested.v1",
        "onex.evt.deploy.rebuild-completed.v1",
        "onex.evt.deploy.rebuild-rejected.v1",
        "onex.evt.deploy.dev-lane-drift.v1",
        "onex.evt.deploy.dev-tag-heartbeat.v1",
        "onex.dlq.omnibase-infra.deploy-command.v1",
    }
)


def _job(
    lane: str,
    status: str,
    *,
    failed_phase: Phase | None = None,
    scope: str = "runtime",
) -> JobState:
    results = dict.fromkeys(DEPLOY_PHASE_ORDER, PhaseStatus.SUCCESS)
    if failed_phase is not None:
        index = DEPLOY_PHASE_ORDER.index(failed_phase)
        for phase in DEPLOY_PHASE_ORDER[index:]:
            results[phase] = PhaseStatus.SKIPPED
        results[failed_phase] = PhaseStatus.FAILED
    return JobState(
        correlation_id=uuid4(),
        command={"runtime_lane": lane, "scope": scope},
        status=status,
        phase_results=results,
        completed_at=DONE,
    )


def _cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    return dict(part.split("=", 1) for part in parts[2:] if "=" in part)


def _request(job: JobState) -> Any:
    """The handler's request for ``job``, as ``publish`` builds it, at a fixed read time."""
    record = json.loads(job.model_dump_json())
    return request_mod.ModelDeployJobReadingRequest(
        correlation_id=record["correlation_id"],
        runtime_lane=record["command"]["runtime_lane"],
        scope=record["command"]["scope"],
        job_outcome=record["status"],
        phase_results=record["phase_results"],
        not_started=record.get("not_started") or "",
        first_error=(record.get("errors") or [""])[0],
        completed_at=record.get("completed_at"),
        published_at=NOW,
    )


def _handler() -> Any:
    return handler_mod.HandlerDeployJobReading(
        settings=runtime_mod.load_settings(agent_mod), environ={}
    )


def _row(job: JobState) -> str | None:
    row: str | None = _handler().build_row(_request(job))
    return row


@pytest.mark.parametrize(
    ("lane", "failed_phase"),
    [(DEV, Phase.COMPOSE_GEN), (STABILITY, Phase.RUNTIME)],
)
def test_failed_job_yields_not_working_row_naming_the_step(
    lane: str, failed_phase: Phase
) -> None:
    job = _job(lane, "failed", failed_phase=failed_phase)
    row = _row(job)
    assert row is not None
    assert row.startswith("2026-11-15T14:40:00Z | STATUS | ")
    cells = _cells(row)
    assert cells["src"] == "reading"
    assert cells["verdict"] == "NOT_WORKING"
    assert cells["reason"] == f"job-failed:{failed_phase.value}"
    assert cells["failing_step"] == failed_phase.value
    assert cells["deploy_lane"] == lane
    assert cells["job"] == str(job.correlation_id)
    assert cells["kind"] == "runtime"
    assert cells["completed_at"] == "2026-11-15T14:39:30Z"
    assert cells["published_at"] == "2026-11-15T14:40:00Z"
    # one row, one line
    assert "\n" not in row


def test_successful_job_yields_working_row_with_no_failing_step() -> None:
    row = _row(_job(DEV, "success"))
    assert row is not None
    cells = _cells(row)
    assert cells["verdict"] == "WORKING"
    assert cells["reason"] == "job-success"
    assert "failing_step" not in cells
    assert "failed" not in row.lower()


def test_failing_step_is_the_first_failed_phase_in_pipeline_order() -> None:
    job = _job(DEV, "failed", failed_phase=Phase.SEED)
    job.phase_results[Phase.VERIFICATION] = PhaseStatus.FAILED
    # the record's own insertion order must not decide it
    job.phase_results = dict(reversed(list(job.phase_results.items())))
    assert _cells(_row(job) or "")["failing_step"] == "seed"


def test_failed_job_with_no_failed_phase_names_why_nothing_ran() -> None:
    job = _job(DEV, "failed")
    job.not_started = events_mod.EnumJobNotStarted.LANE_LOCK_CONTENDED
    assert _cells(_row(job) or "")["failing_step"] == "not-started-lane_lock_contended"
    job.not_started = None
    assert _cells(_row(job) or "")["failing_step"] == "unknown"


def test_contract_phase_order_is_the_agents_pipeline_order() -> None:
    settings = runtime_mod.load_settings(agent_mod)
    assert tuple(settings["phase_order"]) == tuple(p.value for p in DEPLOY_PHASE_ORDER)


@pytest.mark.parametrize("lane", [lane.value for lane in EnumRuntimeLane])
@pytest.mark.parametrize("status", ["success", "failed"])
def test_every_lane_but_prod_has_a_row_and_prod_has_none(
    lane: str, status: str
) -> None:
    row = _row(
        _job(lane, status, failed_phase=Phase.GIT if status == "failed" else None)
    )
    assert (row is None) == (lane == PROD)


@pytest.mark.parametrize("status", ["superseded", "in_progress", "accepted"])
def test_no_row_for_a_job_without_a_terminal_outcome(status: str) -> None:
    job = _job(DEV, "success").model_copy(update={"status": status})
    assert _row(job) is None


def _fake_append_command(tmp_path: Path, exit_code: int = 0) -> tuple[str, Path]:
    """A stand-in for the bus append command: records argv and stdin, prints a receipt."""
    record = tmp_path / "record.json"
    script = tmp_path / "append.py"
    script.write_text(
        "import json, os, sys\n"
        "body = sys.stdin.read()\n"
        # the first invocation is the reading; a failed job's alarm row follows it (OMN-24562)
        f"if not os.path.exists({str(record)!r}):\n"
        f"    open({str(record)!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], "
        "'stdin': body}))\n"
        "print(json.dumps({'status': 'accepted'}))\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return f"{sys.executable} {script}", record


def _route(
    monkeypatch: pytest.MonkeyPatch, command: str, host: str = "lab-host"
) -> None:
    settings = runtime_mod.load_settings(agent_mod)
    monkeypatch.setenv(settings["append_command_env"], command)
    monkeypatch.setenv(settings["host_name_env"], host)


def test_row_is_appended_through_the_reading_writers_append_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, record = _fake_append_command(tmp_path)
    _route(monkeypatch, command)
    job = _job(STABILITY, "failed", failed_phase=Phase.VERIFICATION)

    result = runtime_mod.publish(job, agent_mod)

    assert result is not None and result.outcome == "appended"
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["argv"] == [
        "--lane",
        "deploy-agent",
        "--host",
        "lab-host",
        "--request-id",
        result.append_key,
        "--rows-file",
        "-",
    ]
    # exactly one row on stdin, carrying the request cell the bus append requires
    lines = seen["stdin"].splitlines()
    assert len(lines) == 1
    assert lines[0] == f"{result.row} | req={result.append_key}"
    cells = _cells(lines[0])
    assert cells["verdict"] == "NOT_WORKING"
    assert cells["failing_step"] == "verification"
    assert cells["deploy_lane"] == STABILITY
    # a retry of the same job is the same request
    again = handler_mod._request_id(_request(job))
    assert again == result.append_key


def test_prod_job_appends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, record = _fake_append_command(tmp_path)
    _route(monkeypatch, command)
    result = runtime_mod.publish(
        _job(PROD, "failed", failed_phase=Phase.RUNTIME), agent_mod
    )
    assert result is not None and result.outcome == "skipped"
    assert not record.exists()


def test_append_refusal_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, _ = _fake_append_command(tmp_path, exit_code=65)
    _route(monkeypatch, command)
    result = runtime_mod.publish(_job(DEV, "success"), agent_mod)
    assert result is not None and result.outcome == "unappended"
    assert "exit-65" in result.detail


def test_no_route_configured_appends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = runtime_mod.load_settings(agent_mod)
    monkeypatch.delenv(settings["append_command_env"], raising=False)
    monkeypatch.delenv(settings["host_name_env"], raising=False)
    result = runtime_mod.publish(_job(DEV, "success"), agent_mod)
    assert result is not None and result.outcome == "unappended"
    assert settings["append_command_env"] in result.detail


def test_declares_no_topic_beyond_what_the_reading_writer_and_agent_declare() -> None:
    """AC3: the row goes through the bus append command; this node owns no topic."""
    topic = re.compile(r"onex\.(cmd|evt|dlq)\.")
    handler_source = Path(handler_mod.__file__).read_text(encoding="utf-8")
    literals = [
        node.value
        for node in ast.walk(ast.parse(handler_source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not [text for text in literals if topic.search(text)]
    contract = runtime_mod.contract_path(agent_mod).read_text(encoding="utf-8")
    parsed = yaml.safe_load(contract)
    assert not topic.search(contract)
    assert "event_bus" not in parsed and "topics" not in parsed
    # the agent's own topic constants are unchanged, and none is a reading topic
    declared = {
        value
        for name, value in vars(events_mod).items()
        if name.startswith("TOPIC_") and isinstance(value, str)
    }
    assert declared == AGENT_TOPICS_BEFORE
    assert declared.isdisjoint(READING_WRITER_TOPICS)
    # the row names the ledger, not a topic
    assert not topic.search(_row(_job(DEV, "success")) or "")


def test_publish_terminal_hands_the_settled_job_to_the_reading() -> None:
    """The call site: the agent's terminal publish writes the reading after the event."""
    tree = ast.parse(Path(agent_mod.__file__).read_text(encoding="utf-8"))
    publish_terminal = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_publish_terminal"
    )
    calls = [
        ast.unparse(call.func)
        for call in ast.walk(publish_terminal)
        if isinstance(call, ast.Call)
    ]
    assert "publish_result" in calls
    reading = "self._job_reading"
    assert reading in calls
    assert calls.index(reading) > calls.index("publish_result")


def test_job_refused_between_phases_names_the_phase_that_was_due() -> None:
    """The 14:26Z stability follow: refused with no phase failed, so the step is the first not run."""
    job = _job(STABILITY, "failed")
    for phase in (Phase.CORE, Phase.RUNTIME, Phase.VERIFICATION):
        job.phase_results[phase] = PhaseStatus.SKIPPED
    job.errors = ["Refusing a runtime deploy: no healthy, retained rollback point"]
    cells = _cells(_row(job) or "")
    assert cells["failing_step"] == "core"
    assert cells["reason"] == "job-failed:core"
    assert (
        cells["error"]
        == "Refusing-a-runtime-deploy:-no-healthy-retained-rollback-point"
    )


def test_error_cell_is_bounded_and_a_successful_job_carries_none() -> None:
    job = _job(DEV, "failed", failed_phase=Phase.GIT)
    job.errors = ["x" * 1000]
    assert len(_cells(_row(job) or "")["error"]) == 160
    ok = _job(DEV, "success")
    ok.errors = ["a note"]
    assert "error" not in _cells(_row(ok) or "")


def test_the_agents_prefixed_broker_login_reaches_the_append_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent holds its login as DEV_KAFKA_SASL_*; the append command reads the unprefixed names."""
    record = tmp_path / "login.json"
    script = tmp_path / "append.py"
    script.write_text(
        "import json, os, sys\n"
        "sys.stdin.read()\n"
        f"open({str(record)!r}, 'w').write(json.dumps({{"
        "'user': os.environ.get('KAFKA_SASL_USERNAME') == 'dev-user', "
        "'password': os.environ.get('KAFKA_SASL_PASSWORD') == 'dev-pass'}))\n"
        "print(json.dumps({'status': 'accepted'}))\n",
        encoding="utf-8",
    )
    _route(monkeypatch, f"{sys.executable} {script}")
    monkeypatch.setenv("KAFKA_SASL_ENV_PREFIX", "DEV_")
    monkeypatch.setenv("DEV_KAFKA_SASL_USERNAME", "dev-user")
    monkeypatch.setenv("DEV_KAFKA_SASL_PASSWORD", "dev-pass")
    monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
    monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
    result = runtime_mod.publish(_job(DEV, "success"), agent_mod)
    assert result is not None and result.outcome == "appended"
    assert json.loads(record.read_text(encoding="utf-8")) == {
        "user": True,
        "password": True,
    }
