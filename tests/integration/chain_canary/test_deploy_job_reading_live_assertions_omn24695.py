# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The live deploy-job-reading assertions can fail on a real defect and only then (OMN-24695).

The two live tests of OMN-24674 read the agent's ledger rows and its ``/health``. Review of
omnibase_infra#6029 at head 9c2cff86c628 found they could turn red with no defect (a ledger
roll moved the rows out of the one file they read; a lane-lock skipped job has no failed
phase; a job's row is published after its overlay settle) and could stay green with one (only
the newest row was checked). Each test here feeds the live tests' own helpers a synthetic
ledger or job record and runs hermetically on every runner.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from deploy_agent.events import EnumRuntimeLane

from tests.integration.chain_canary.conftest import (
    ONE_TICK_SECONDS,
    SETTLE_CEILING_SECONDS,
    LiveDeployAgentChain,
    chain_from_environment,
    handler_failing_step,
    reading_problems,
)

pytestmark = pytest.mark.integration

_HERE = Path(__file__).resolve().parent
_CONTRACT = (
    _HERE.parents[2]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_SETTINGS = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["settings"]
_NOW = datetime(2026, 11, 16, 3, 0, 0, tzinfo=UTC)
_STABILITY = EnumRuntimeLane.STABILITY_TEST.value


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(
    *,
    job: str | None = None,
    verdict: str = "NOT_WORKING",
    step: str = "preflight",
    completed: datetime = _NOW - timedelta(minutes=5),
    lag_seconds: float = 3.0,
    host: str = "h201",
    lane: str = "dev",
) -> str:
    job = job or str(uuid.uuid4())
    published = completed + timedelta(seconds=lag_seconds)
    request_id = uuid.uuid5(uuid.NAMESPACE_URL, f"onex:deploy-job-reading:{job}")
    failed = verdict == "NOT_WORKING"
    cells = [
        "lane=deploy-agent",
        "src=reading",
        "actor=deploy-agent:job-reading",
        "model=none",
        f"piece={_SETTINGS['piece']}",
        f"name={_SETTINGS['piece_name']}",
        f"verdict={verdict}",
        "outcome=read",
        f"reason=job-failed:{step}" if failed else "reason=job-success",
        f"deploy_lane={lane}",
        f"job={job}",
        "kind=runtime",
    ]
    if failed:
        cells.append(f"failing_step={step}")
    cells += [
        f"completed_at={_stamp(completed)}",
        f"probed_at={_stamp(published)}",
        f"published_at={_stamp(published)}",
        "max_age=7200",
        f"key={job}",
        f"req={request_id}",
        f"via=bus:{host}",
    ]
    return f"{_stamp(published)} | STATUS | " + " | ".join(cells)


def _chain(
    tmp_path: Path, live: list[str], rolls: dict[str, list[str]], *, host: str = "h201"
) -> LiveDeployAgentChain:
    ledger = tmp_path / "copy.md"
    ledger.write_text("\n".join(live) + "\n", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    for day, rows in rolls.items():
        (archive / f"copy_{day}-split.md").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    return LiveDeployAgentChain(
        ledger_path=ledger,
        agent_url="",
        ledger_is_current=False,
        agent_host=host,
        window_seconds=6 * 3600.0,
    )


def test_deploy_job_reading_archive_roll_rows_are_read(tmp_path: Path) -> None:
    """A NOT_WORKING row that sits only in a roll file is still found, in time order."""
    failed = str(uuid.uuid4())
    chain = _chain(
        tmp_path,
        live=[_row(verdict="WORKING", step="", completed=_NOW)],
        rolls={"2026-11-15": [_row(job=failed, completed=_NOW - timedelta(hours=9))]},
    )
    rows = chain.rows()
    assert [row["job"] for row in rows if row["verdict"] == "NOT_WORKING"] == [failed]
    assert rows[-1]["verdict"] == "WORKING", "the live file's newer row must come last"


def test_deploy_job_reading_every_row_in_the_window_is_checked(tmp_path: Path) -> None:
    """A late row in the middle of the window is named; checking only the newest row misses it."""
    late = str(uuid.uuid4())
    live = [
        _row(completed=_NOW - timedelta(minutes=50)),
        _row(job=late, completed=_NOW - timedelta(minutes=55), lag_seconds=2876),
        _row(completed=_NOW - timedelta(minutes=2)),
    ]
    chain = _chain(tmp_path, live, {})
    in_window = chain.window_rows(now=_NOW)
    assert len(in_window) == 3
    problems = reading_problems(in_window)
    assert len(problems) == 1
    assert late in problems[0]
    assert "2876" in problems[0]
    assert reading_problems(in_window[-1:]) == [], "the newest row alone looks fine"


def test_deploy_job_reading_every_row_window_drops_rows_before_it(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        live=[],
        rolls={
            "2026-11-15": [
                _row(completed=_NOW - timedelta(hours=9), lag_seconds=2876),
            ]
        },
    )
    assert len(chain.rows()) == 1
    assert chain.window_rows(now=_NOW) == []


def test_deploy_job_reading_every_row_accepts_the_handlers_unknown_step() -> None:
    """The handler emits ``unknown`` only when every phase succeeded; that is a correct row.

    A failed job with a phase that did not succeed names that phase, and one with no phase
    record names the first phase (``handler_deploy_job_reading.py`` ``failing_step``).
    """
    row = {
        k.strip(): v.strip()
        for k, v in (
            cell.split("=", 1)
            for cell in _row(step="unknown").split(" | ")[2:]
            if "=" in cell
        )
    }
    assert reading_problems([row]) == []
    row["failing_step"] = "no-such-phase"
    row["reason"] = "job-failed:no-such-phase"
    assert reading_problems([row])


def test_deploy_job_reading_skipped_phases_name_the_first_non_success_phase() -> None:
    """A lane-lock or host-slot job settles every phase skipped; the row names ``preflight``."""
    order = _SETTINGS["phase_order"]
    assert handler_failing_step(_SETTINGS, dict.fromkeys(order, "skipped")) == order[0]
    mixed = {
        "preflight": "success",
        "git": "success",
        "compose_gen": "skipped",
        "seed": "skipped",
    }
    assert handler_failing_step(_SETTINGS, mixed) == "compose_gen"
    failed = {**dict.fromkeys(order, "skipped"), "runtime": "failed"}
    assert handler_failing_step(_SETTINGS, failed) == "runtime"
    assert handler_failing_step(_SETTINGS, {}) == order[0]


def test_deploy_job_reading_settling_wait_holds_until_the_agent_stops_settling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-tick bound starts when ``last_result.settling`` clears, not at ``completed_at``."""
    job_id = str(uuid.uuid4())
    bodies: list[dict[str, object]] = [
        {"last_result": {"correlation_id": job_id, "settling": True}},
        {"last_result": {"correlation_id": job_id, "settling": True}},
        {"last_result": {"correlation_id": job_id, "settling": False}},
    ]
    calls: list[int] = []

    def fake_health(self: LiveDeployAgentChain) -> dict[str, object]:
        calls.append(1)
        return bodies[min(len(calls), len(bodies)) - 1]

    monkeypatch.setattr(LiveDeployAgentChain, "health", fake_health)
    sleeps: list[float] = []
    clock = [_NOW]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += timedelta(seconds=seconds)

    chain = LiveDeployAgentChain(
        ledger_path=None,
        agent_url="http://agent.invalid",
        ledger_is_current=True,
        agent_host="h201",
        window_seconds=3600.0,
    )
    settled_at = chain.wait_while_settling(
        job_id, sleep=fake_sleep, now=lambda: clock[0], ceiling=timedelta(minutes=30)
    )
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert settled_at == clock[0]


def test_deploy_job_reading_settling_wait_gives_up_at_its_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = str(uuid.uuid4())
    monkeypatch.setattr(
        LiveDeployAgentChain,
        "health",
        lambda self: {"last_result": {"correlation_id": job_id, "settling": True}},
    )
    clock = [_NOW]

    def fake_sleep(seconds: float) -> None:
        clock[0] += timedelta(seconds=seconds)

    chain = LiveDeployAgentChain(
        ledger_path=None,
        agent_url="http://agent.invalid",
        ledger_is_current=True,
        agent_host="h201",
        window_seconds=3600.0,
    )
    with pytest.raises(AssertionError, match="still settling"):
        chain.wait_while_settling(
            job_id,
            sleep=fake_sleep,
            now=lambda: clock[0],
            ceiling=timedelta(minutes=10),
        )


def test_deploy_job_reading_host_variable_selects_the_agents_host(
    tmp_path: Path,
) -> None:
    """The host whose rows are read is a field of the chain, taken from the environment."""
    own, other = str(uuid.uuid4()), str(uuid.uuid4())
    live = [
        _row(job=own, host="h053"),
        _row(job=other, host="h201"),
    ]
    assert [r["job"] for r in _chain(tmp_path, live, {}, host="h053").rows()] == [own]


def test_deploy_job_reading_host_variable_fails_closed_without_a_host(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError, match="CHAIN_CANARY_AGENT_LEDGER_HOST"):
        _chain(tmp_path, [_row()], {}, host="").rows()


def _cells(line: str) -> dict[str, str]:
    return {
        k.strip(): v.strip()
        for k, v in (
            cell.split("=", 1) for cell in line.split(" | ")[2:] if "=" in cell
        )
    }


def test_deploy_job_reading_published_after_settle_passes_for_a_dev_row() -> None:
    """A dev row published after the overlay settle and the repair build is a correct row.

    ``completed_at`` is stamped before the settle and the reading is published after it, so a
    well-formed WORKING dev row is 360 s late with nothing wrong (review of omnibase_infra#6029
    at head 4ba6768c6077).
    """
    working = _cells(_row(verdict="WORKING", step="", lag_seconds=360))
    failed = _cells(_row(lag_seconds=360))
    assert reading_problems([working, failed]) == []
    edge = _cells(_row(lag_seconds=SETTLE_CEILING_SECONDS + ONE_TICK_SECONDS))
    assert reading_problems([edge]) == []


def test_deploy_job_reading_published_after_settle_is_bounded_by_the_ceiling() -> None:
    """A dev row later than the settle ceiling plus one tick is still named."""
    late = _cells(_row(lag_seconds=SETTLE_CEILING_SECONDS + ONE_TICK_SECONDS + 1))
    problems = reading_problems([late])
    assert len(problems) == 1
    assert late["job"] in problems[0]
    # A row published before its job completed is wrong whatever the lane.
    early = _cells(_row(lag_seconds=-5))
    assert len(reading_problems([early])) == 1


def test_deploy_job_reading_published_after_settle_does_not_loosen_stability() -> None:
    """The settle runs on the dev lane; a stability row keeps the one-tick bound."""
    late = _cells(_row(lane=_STABILITY, lag_seconds=360))
    assert len(reading_problems([late])) == 1
    on_time = _cells(_row(lane=_STABILITY, lag_seconds=ONE_TICK_SECONDS))
    assert reading_problems([on_time]) == []


def test_deploy_job_reading_window_parse_reads_the_variable_at_test_time() -> None:
    assert chain_from_environment({}).window_seconds == 6 * 3600.0
    assert (
        chain_from_environment({"CHAIN_CANARY_WINDOW_SECONDS": "172800"}).window_seconds
        == 172800.0
    )


@pytest.mark.parametrize("value", ["forty-eight hours", "0", "-60", "nan"])
def test_deploy_job_reading_window_parse_fails_the_test_not_the_collection(
    value: str,
) -> None:
    """A window that is not a positive number fails the test that asks for it, by name."""
    with pytest.raises(AssertionError, match="CHAIN_CANARY_WINDOW_SECONDS"):
        chain_from_environment({"CHAIN_CANARY_WINDOW_SECONDS": value})
