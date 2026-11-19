# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime recreates at most once a window, on the window's start (OMN-25912).

Run with ``-k recreate_window``. On 2026-11-19 the dev runtime started at 04:33:39Z,
04:54:06Z, 05:12:28Z and 05:38:11Z, each start 7 to 14 minutes to ready; the regression
below replays those four recorded starts through the declared policy.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.events import Phase, PhaseStatus
from deploy_agent.job_state import JobStore

from omnibase_infra.enums.enum_dev_recreate_reason import EnumDevRecreateReason
from omnibase_infra.enums.enum_dev_recreate_verdict import EnumDevRecreateVerdict
from omnibase_infra.handlers.dev_recreate_window.handler_dev_recreate_window import (
    DevRecreateWindowGate,
    HandlerDevRecreateWindow,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_recreate_request import ModelDevRecreateRequest
from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    newest_dev_recreate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    Path(__file__).parent / "fixtures" / "recreate_window_four_starts_omn25912.json"
)
POLICY_ENV = REPO_ROOT / "docker" / "runtime-policy.env"
HANDLER_SOURCE = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "handlers"
    / "dev_recreate_window"
    / "handler_dev_recreate_window.py"
)
MERGE = "gha/omnibase_infra/pr-1"
IDLE = "agent/idle-converge"


def _declared_policy() -> ModelDevRecreatePolicy:
    env: dict[str, str] = {}
    for line in POLICY_ENV.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    return ModelDevRecreatePolicy.from_env(env)


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


class _Bus:
    """A bus that accepts every announcement and remembers it."""

    def __init__(self) -> None:
        self.sent: list[ModelDevRecreateAnnouncement] = []

    def __call__(self, announcement: ModelDevRecreateAnnouncement) -> bool:
        self.sent.append(announcement)
        return True


def test_recreate_window_two_hour_simulation_with_a_merge_every_four_minutes() -> None:
    """AC2: a merge every four minutes for two hours recreates at most once an hour, on the hour."""
    policy = _declared_policy()
    bus = _Bus()
    gate = DevRecreateWindowGate(policy, bus, clock=lambda: _at("2026-11-19T06:00:20Z"))
    first = datetime(2026, 11, 19, 4, 0, 1, tzinfo=UTC)
    pending = False
    recreates: list[datetime] = []
    for second in range(0, 2 * 3600 + 1, 15):
        now = first + timedelta(seconds=second)
        if second % (4 * 60) == 0:
            pending = True
        if pending and gate.permit(MERGE, now):
            recreates.append(now)
            pending = False
    per_hour: dict[int, int] = {}
    for moment in recreates:
        per_hour[moment.hour] = per_hour.get(moment.hour, 0) + 1
    assert recreates, (
        "a merge every four minutes must still recreate once the window opens"
    )
    assert max(per_hour.values()) == 1, per_hour
    for moment in recreates:
        start = moment.replace(minute=0, second=0, microsecond=0)
        assert moment - start <= timedelta(seconds=policy.start_grace_seconds), moment
    assert len(bus.sent) >= len(recreates)


def test_recreate_window_no_recreate_off_the_hour_without_a_pinned_fix() -> None:
    policy = _declared_policy()
    handler = HandlerDevRecreateWindow()
    for minute in (6, 12, 33, 59):
        decision = handler.handle(
            ModelDevRecreateRequest(
                policy=policy,
                now=datetime(2026, 11, 19, 5, minute, 0, tzinfo=UTC),
                requested_by=IDLE,
            )
        )
        assert decision.verdict is EnumDevRecreateVerdict.HOLD
        assert decision.reason is EnumDevRecreateReason.WINDOW_MISSED
        assert decision.next_recreate_at == datetime(2026, 11, 19, 6, 0, tzinfo=UTC)


def test_recreate_window_second_ask_in_a_spent_window_holds_to_the_next_start() -> None:
    policy = _declared_policy()
    decision = HandlerDevRecreateWindow().handle(
        ModelDevRecreateRequest(
            policy=policy,
            now=datetime(2026, 11, 19, 5, 0, 40, tzinfo=UTC),
            requested_by=IDLE,
            last_recreate_at=datetime(2026, 11, 19, 5, 0, 5, tzinfo=UTC),
        )
    )
    assert decision.verdict is EnumDevRecreateVerdict.HOLD
    assert decision.reason is EnumDevRecreateReason.WINDOW_SPENT


def test_recreate_window_a_pinned_fix_recreates_immediately_off_the_hour() -> None:
    policy = _declared_policy()
    assert policy.pinned_fix_requester_prefixes, "the pinned-fix requester is policy"
    requester = policy.pinned_fix_requester_prefixes[0] + "omn-1"
    bus = _Bus()
    gate = DevRecreateWindowGate(
        policy,
        bus,
        clock=lambda: datetime(2026, 11, 19, 5, 33, tzinfo=UTC),
        last_recreate_at=datetime(2026, 11, 19, 5, 0, 5, tzinfo=UTC),
    )
    assert gate.permit(requester, datetime(2026, 11, 19, 5, 33, tzinfo=UTC)) is True
    assert bus.sent[-1].immediate is True
    assert bus.sent[-1].reason is EnumDevRecreateReason.PINNED_FIX


def test_recreate_window_values_come_from_the_policy_not_the_handler() -> None:
    half_hour = ModelDevRecreatePolicy(
        window_seconds=1800, offset_seconds=0, start_grace_seconds=60
    )
    handler = HandlerDevRecreateWindow()
    decision = handler.handle(
        ModelDevRecreateRequest(
            policy=half_hour,
            now=datetime(2026, 11, 19, 5, 30, 10, tzinfo=UTC),
            requested_by=IDLE,
        )
    )
    assert decision.verdict is EnumDevRecreateVerdict.RECREATE
    offset = ModelDevRecreatePolicy(
        window_seconds=3600, offset_seconds=600, start_grace_seconds=60
    )
    held = handler.handle(
        ModelDevRecreateRequest(
            policy=offset,
            now=datetime(2026, 11, 19, 5, 0, 10, tzinfo=UTC),
            requested_by=IDLE,
        )
    )
    assert held.next_recreate_at == datetime(2026, 11, 19, 5, 10, tzinfo=UTC)


def test_recreate_window_handler_source_writes_no_window_value() -> None:
    source = HANDLER_SOURCE.read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", source, flags=re.S)
    code = re.sub(r"#.*", "", code)
    assert not re.search(r"\b(3600|1800|900|86400|300)\b", code)
    assert not re.search(r"agent/|operator/|gha/", code)


def test_recreate_window_an_undeclared_policy_refuses() -> None:
    with pytest.raises(ValueError, match="not declared"):
        ModelDevRecreatePolicy.from_env({})


def test_recreate_window_declared_in_the_rendered_policy_env() -> None:
    policy = _declared_policy()
    assert (policy.window_seconds, policy.offset_seconds) == (3600, 0)


def test_recreate_window_four_recorded_starts_are_held_to_the_window() -> None:
    """AC3: the four starts recorded on h201 would have been held, not run."""
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    starts = recorded["starts"]
    stamps = [_at(item["at"]) for item in starts]
    assert stamps[-1] - stamps[0] < timedelta(minutes=65), (
        "four starts inside 65 minutes"
    )
    dev_jobs = [
        job
        for job in recorded["jobs"]
        if job["runtime_lane"] == "dev"
        and job["runtime_phase"] in ("success", "failed")
    ]
    seeded = max(
        _at(job["accepted_at"])
        for job in dev_jobs
        if job["accepted_at"] < starts[0]["at"]
    )
    bus = _Bus()
    gate = DevRecreateWindowGate(
        _declared_policy(),
        bus,
        last_recreate_at=seeded,
        clock=lambda: _at("2026-11-19T06:00:20Z"),
    )
    held_until = []
    for item, stamp in zip(starts, stamps, strict=True):
        assert gate.permit(item["requested_by"], stamp) is False, item["at"]
        held_until.append(bus.sent[-1].next_recreate_at)
    assert held_until == [
        _at("2026-11-19T05:00:00Z"),
        _at("2026-11-19T05:00:00Z"),
        _at("2026-11-19T06:00:00Z"),
        _at("2026-11-19T06:00:00Z"),
    ]
    assert gate.permit(MERGE, _at("2026-11-19T06:00:30Z")) is True


def _agent_with(
    gate: DevRecreateWindowGate | None, jobs: Path | None = None
) -> agent_mod.DeployAgent:
    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    if jobs is not None:
        agent.job_store = JobStore(jobs)
    agent.__dict__.update(
        _load_gate=None, _load_gate_last_verdict=None, _recreate_gate=gate
    )
    return agent


def test_recreate_window_install_path_idle_converge_and_restore_wait(
    tmp_path: Path,
) -> None:
    clock = [_at("2026-11-19T05:12:28Z")]
    gate = DevRecreateWindowGate(_declared_policy(), _Bus(), clock=lambda: clock[0])
    agent = _agent_with(gate, tmp_path)
    assert agent._idle_converge_gate_open("agent/idle-converge") is False
    assert agent._idle_converge_gate_open("agent/idle-restore") is False
    clock[0] = _at("2026-11-19T06:00:20Z")
    assert agent._idle_converge_gate_open("agent/idle-converge") is True
    assert agent._idle_converge_gate_open("agent/idle-restore") is False, (
        "one per window"
    )


def test_recreate_window_agent_without_a_gate_is_unchanged() -> None:
    assert _agent_with(None)._idle_converge_gate_open("agent/idle-converge") is True


def test_recreate_window_job_records_show_the_newest_recreate(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    runtime_job = UUID("46498207-11b6-4e21-a509-a51652ec50ad")
    store.accept(
        runtime_job,
        {
            "correlation_id": str(runtime_job),
            "requested_by": MERGE,
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "services": [],
            "requested_at": "2026-11-19T04:43:47Z",
        },
    )
    assert newest_dev_recreate(tmp_path) is None, "no runtime phase has run"
    store.update_phase(runtime_job, Phase.RUNTIME, PhaseStatus.SUCCESS)
    assert newest_dev_recreate(tmp_path) is not None


def test_recreate_window_a_merge_recreate_in_the_job_records_spends_the_window() -> (
    None
):
    """A recreate the gate did not permit (a merge's job) still spends its window."""
    clock = [_at("2026-11-19T06:00:30Z")]
    gate = DevRecreateWindowGate(_declared_policy(), _Bus(), clock=lambda: clock[0])
    merge_job_accepted = _at("2026-11-19T06:00:05Z")
    assert gate.allow("agent/idle-converge", merge_job_accepted) is False
    clock[0] = _at("2026-11-19T07:00:30Z")
    assert gate.allow("agent/idle-converge", merge_job_accepted) is True


def test_recreate_window_install_path_branch_install_reapply_waits(
    tmp_path: Path,
) -> None:
    """Two of the four recorded starts were the reapply's docker restart."""
    from omnibase_infra.nodes.node_dev_roll_events_effect import (
        runtime_dev_branch_install_reapply as reapply,
    )

    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    kinds = {item["source_kind"] for item in recorded["starts"]}
    assert kinds == {"branch_install_reapply", "merge_recreate"}
    assert recorded["starts"][0]["requested_by"] == reapply.REAPPLY_REQUESTER
    clock = [_at("2026-11-19T05:12:28Z")]
    bus = _Bus()

    def build(
        root: Path, last: datetime | None, announced: datetime | None
    ) -> DevRecreateWindowGate:
        return DevRecreateWindowGate(
            _declared_policy(),
            bus,
            last_recreate_at=last,
            clock=lambda: clock[0],
            announced_for=announced,
        )

    for _ in range(3):
        assert reapply.window_holds_apply(tmp_path, repo_root=REPO_ROOT, build=build)
    assert len(bus.sent) == 1, "the held restart names the next recreate once"
    assert bus.sent[0].next_recreate_at == _at("2026-11-19T06:00:00Z")
    clock[0] = _at("2026-11-19T06:00:30Z")
    assert not reapply.window_holds_apply(tmp_path, repo_root=REPO_ROOT, build=build)
    assert bus.sent[-1].immediate is True, "announced before the restart runs"

    calls: list[str] = []
    argv = ["apply", "--state-dir", str(tmp_path)]
    held = reapply.main(
        argv,
        handler_main=lambda a: calls.append("ran") or 0,
        window=lambda *a, **k: True,
    )
    assert held == 0 and calls == [], "a held apply does not run the handler"
    reapply.main(
        argv,
        handler_main=lambda a: calls.append("ran") or 0,
        window=lambda *a, **k: False,
    )
    assert calls == ["ran"]
