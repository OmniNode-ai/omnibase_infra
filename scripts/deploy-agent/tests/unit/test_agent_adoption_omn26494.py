# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A deploy agent adopts a code ref only after a one-shot self-check of its idle ticks (OMN-26494).

omnibase_infra#6080 (OMN-25912) shipped an idle tick that raised a ``ValidationError``; the dev
agent restarted 111 times and took no job for about 4.7 hours. Four mechanisms stand between
that merge and the outage now, one test group each (AC1 to AC4):

* the self-check runs the idle restore, idle converge and stability follow ticks once each,
  against a fake clock and a dry-run bus, and exits non-zero when one raises;
* the agent adopts a ref only when the self-check of that ref exits zero;
* five restarts inside ten minutes after an adoption return the agent to the previous ref and
  append one alarm row naming both refs;
* the recorded journal of 2026-11-20 (``fixtures/dev_agent_requested_by_crash_omn26106.txt``)
  replays through the self-check: it fails on the broken calling convention and passes on the fix.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor, load_dev_boot_guard
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    STABILITY_FOLLOW_REQUESTER,
)

from .test_idle_converge_omn19509 import _agent
from .test_stability_follows_dev_tag_omn19998 import LAB_FENCE

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
JOURNAL = (
    Path(__file__).parent / "fixtures" / "dev_agent_requested_by_crash_omn26106.txt"
)
RESTORE_REQUESTER = "agent/idle-restore"
OLD = "a" * 40
NEW = "b" * 40
BOUNDARY = EnumSelfUpdateBoundary.IDLE_HEARTBEAT
AGENT_DIR = str(REPO_ROOT / "scripts" / "deploy-agent")


ADOPTION_MODULE = "nodes.node_dev_roll_events_effect.runtime_dev_agent_adoption"
SELF_CHECK_FILE = (
    REPO_ROOT
    / "src/omnibase_infra/nodes/node_dev_roll_events_effect"
    / "runtime_dev_agent_self_check.py"
)


def _adoption() -> ModuleType:
    return load_dev_boot_guard(ADOPTION_MODULE)


_REAL_CANDIDATE_SELF_CHECK = _adoption().candidate_self_check
_REAL_SETTINGS_AT = _adoption().settings_at


def _settings() -> object:
    return _REAL_SETTINGS_AT(REPO_ROOT)


def _self_check_module() -> ModuleType:
    """The self-check entry, loaded by path as the agent's host entries are."""
    name = (
        "omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_agent_self_check"
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SELF_CHECK_FILE)
        assert spec is not None and spec.loader is not None
        module = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(autouse=True)
def _real_candidate_self_check(
    monkeypatch: pytest.MonkeyPatch, _candidate_self_check_passes: None
) -> None:
    """These tests prove the self-check's own gate: put the real one back over the conftest stub."""
    adoption = _adoption()
    monkeypatch.setattr(adoption, "candidate_self_check", _REAL_CANDIDATE_SELF_CHECK)
    monkeypatch.setattr(adoption, "settings_at", _REAL_SETTINGS_AT)


def _agent_to_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeployAgent:
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", LAB_FENCE)
    return _agent(tmp_path, monkeypatch, omnimarket_to_202=True)


# --- AC1 ----------------------------------------------------------------------------------


def test_agent_self_check_runs_every_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _self_check_module()
    agent = _agent_to_check(tmp_path, monkeypatch)
    ran: list[str] = []

    def spying(real: Callable[..., object], tick: str) -> Callable[..., object]:
        def spy(*args: object) -> object:
            ran.append(tick)
            return real(*args)

        return spy

    for tick in (
        "_maybe_idle_restore",
        "_maybe_idle_converge",
        "_maybe_stability_follow",
    ):
        monkeypatch.setattr(agent, tick, spying(getattr(agent, tick), tick))

    report = check.run_self_check(agent, REPO_ROOT)

    assert report.failures == {}
    assert report.exit_code == 0
    assert ran.count("_maybe_idle_restore") >= 1
    assert ran.count("_maybe_idle_converge") == 1
    assert ran.count("_maybe_stability_follow") >= 1
    # The real window gate saw each tick's own requester, on the dry-run announcement bus.
    assert set(report.announced) == {
        RESTORE_REQUESTER,
        IDLE_CONVERGE_REQUESTER,
        STABILITY_FOLLOW_REQUESTER,
    }
    assert report.dispatched == 3
    assert agent.executor.calls == []  # nothing reached a real deploy


def test_agent_self_check_runs_every_tick_and_exits_nonzero_when_one_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _self_check_module()
    agent = _agent_to_check(tmp_path, monkeypatch)

    def boom(now: object) -> None:
        raise RuntimeError("the stability follow tick raised")

    monkeypatch.setattr(agent, "_maybe_stability_follow", boom)

    report = check.run_self_check(agent, REPO_ROOT)

    assert set(report.failures) == {"stability_follow"}
    assert "the stability follow tick raised" in report.failures["stability_follow"]
    assert report.exit_code == 1
    # The ticks before it still ran: one raise does not hide the others.
    assert RESTORE_REQUESTER in report.announced
    assert IDLE_CONVERGE_REQUESTER in report.announced


# --- AC2 ----------------------------------------------------------------------------------


class _Git:
    """A runner for the adoption guard: the self-check's exit code is the case under test."""

    def __init__(self, self_check_code: int) -> None:
        self.self_check_code = self_check_code
        self.argvs: list[list[str]] = []

    def __call__(
        self, argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.argvs.append(list(argv))
        if argv[0] != "git" and "runtime_dev_agent_self_check.py" in argv[1]:
            return subprocess.CompletedProcess(
                argv, self.self_check_code, "", "tick idle_converge raised"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_agent_keeps_previous_ref_on_failed_self_check(tmp_path: Path) -> None:
    adoption = _adoption()
    run = _Git(self_check_code=1)
    synced: list[str] = []

    adopted = adoption.adopts(
        AGENT_DIR, tmp_path, (OLD, NEW), run, lambda d: synced.append(d)
    )

    assert adopted is False
    # The clone went back to the ref the process loaded, and its deps with it.
    assert ["git", "-C", AGENT_DIR, "reset", "--hard", OLD] in run.argvs
    assert synced == [
        AGENT_DIR,
        AGENT_DIR,
    ]  # candidate deps for the check, then the old ones
    state = adoption.read_state(tmp_path, _settings())
    assert state.current_ref == OLD
    assert (NEW, "self-check-exit-1") in state.refused
    # The refusal holds: the same remote ref is not pulled or checked again.
    assert adoption.refused_reason(tmp_path, _settings(), NEW) is not None
    assert adoption.refused_reason(tmp_path, _settings(), "c" * 40) is None


def test_agent_keeps_previous_ref_on_failed_self_check_in_self_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declare_loaded_code_sha: Callable[[str], None],
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_DIR", AGENT_DIR)
    declare_loaded_code_sha(OLD)
    executor = DeployExecutor(migration_record_dir=tmp_path)
    run = _Git(self_check_code=1)

    def responder(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "status" in argv and "--porcelain" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "rev-parse" in argv:
            ref = NEW if any(part.startswith("origin/") for part in argv) else OLD
            return subprocess.CompletedProcess(argv, 0, ref, "")
        return run(argv, timeout, **kwargs)

    with (
        patch("deploy_agent.executor._run", side_effect=responder),
        patch("deploy_agent.executor._uv_sync_after_pull"),
        patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=None),
        patch("os.execv") as execv,
    ):
        executor.self_update(boundary=BOUNDARY)
        executor.self_update(boundary=BOUNDARY)

    execv.assert_not_called()
    resets = [a for a in run.argvs if a[:2] == ["git", "-C"] and "reset" in a]
    checks = [a for a in run.argvs if a[0] != "git"]
    assert len(resets) == 2  # the clone goes back to the loaded ref after each pull
    assert len(checks) == 1  # the refused ref is not self-checked a second time


def test_agent_adopts_a_ref_whose_self_check_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declare_loaded_code_sha: Callable[[str], None],
) -> None:
    """Positive control: the same path with a passing self-check re-execs and records the adoption."""
    monkeypatch.setenv("DEPLOY_AGENT_DIR", AGENT_DIR)
    declare_loaded_code_sha(OLD)
    executor = DeployExecutor(migration_record_dir=tmp_path)
    run = _Git(self_check_code=0)

    def responder(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "status" in argv and "--porcelain" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "rev-parse" in argv:
            ref = NEW if any(part.startswith("origin/") for part in argv) else OLD
            return subprocess.CompletedProcess(argv, 0, ref, "")
        return run(argv, timeout, **kwargs)

    with (
        patch("deploy_agent.executor._run", side_effect=responder),
        patch("deploy_agent.executor._uv_sync_after_pull"),
        patch("deploy_agent.executor._sync_own_unit_after_pull", return_value=None),
        patch("os.execv") as execv,
    ):
        executor.self_update(boundary=BOUNDARY)

    execv.assert_called_once()
    state = _adoption().read_state(tmp_path, _settings())
    assert (state.current_ref, state.previous_ref) == (NEW, OLD)


# --- AC3 ----------------------------------------------------------------------------------


def _storm(
    tmp_path: Path, offsets: list[float]
) -> tuple[list[bool], list[str], list[tuple[str, str]]]:
    """Adopt NEW over OLD, then start NEW at each offset (seconds); what each start did."""
    adoption = _adoption()
    adoption.adopts(AGENT_DIR, tmp_path, (OLD, NEW), _Git(0), lambda d: None)
    reverted: list[str] = []
    alarms: list[tuple[str, str]] = []

    def alarm(adopted: str, previous: str) -> str:
        alarms.append((adopted, previous))
        return ""

    results = [
        adoption.record_boot(
            tmp_path,
            _settings(),
            NEW,
            1000.0 + offset,
            revert=reverted.append,
            alarm=alarm,
        )
        for offset in offsets
    ]
    return results, reverted, alarms


def test_agent_reverts_after_restart_storm_bounds_are_five_in_ten_minutes() -> None:
    adoption = _adoption()
    settings = adoption.settings_at(REPO_ROOT)
    assert (settings.restart_limit, settings.storm_window_seconds) == (5, 600)


def test_agent_reverts_after_restart_storm(tmp_path: Path) -> None:
    # The adoption's own start, then five restarts inside ten minutes.
    results, reverted, alarms = _storm(tmp_path, [0, 5, 10, 15, 20, 25])

    assert results == [False, False, False, False, False, True]
    assert reverted == [OLD]
    assert alarms == [(NEW, OLD)]  # one alarm row, naming both refs
    state = _adoption().read_state(
        tmp_path, _settings()
    )  # the state a restart reads back
    assert (state.current_ref, state.previous_ref) == (OLD, "")
    assert (NEW, "restart-storm") in state.refused


def test_agent_reverts_after_restart_storm_appends_one_row_per_storm(
    tmp_path: Path,
) -> None:
    _, reverted, alarms = _storm(tmp_path, [0, 5, 10, 15, 20, 25])
    adoption = _adoption()
    # The agent restarts on the previous ref and keeps restarting for another reason.
    more = [
        adoption.record_boot(
            tmp_path,
            _settings(),
            OLD,
            1100.0 + n,
            revert=reverted.append,
            alarm=lambda a, p: "x",
        )
        for n in range(8)
    ]

    assert (
        more == [False] * 8
    )  # no previous ref is left to return to, and no second alarm
    assert alarms == [(NEW, OLD)]
    assert reverted == [OLD]


def test_agent_reverts_after_restart_storm_not_for_four_restarts_or_a_slow_loop(
    tmp_path: Path,
) -> None:
    four, reverted, alarms = _storm(tmp_path, [0, 5, 10, 15, 20])
    assert four == [False] * 5 and reverted == [] and alarms == []

    slow_dir = tmp_path / "slow"
    slow, reverted, alarms = _storm(slow_dir, [0, 200, 400, 600, 800, 1000])
    assert slow == [False] * 6 and reverted == [] and alarms == []


def test_agent_reverts_after_restart_storm_unreadable_state_fails_fast(
    tmp_path: Path,
) -> None:
    adoption = _adoption()
    adoption.state_path(tmp_path, _settings()).write_text("{not json", encoding="utf-8")

    with pytest.raises(adoption.AdoptionStateUnreadableError):
        adoption.record_boot(
            tmp_path,
            _settings(),
            NEW,
            1.0,
            revert=lambda ref: None,
            alarm=lambda a, p: "",
        )
    with pytest.raises(adoption.AdoptionStateUnreadableError):
        adoption.refused_reason(tmp_path, _settings(), NEW)
    assert adoption.state_path(tmp_path, _settings()).read_text() == "{not json"


def test_agent_reverts_after_restart_storm_alarm_row_names_both_refs() -> None:
    load_dev_boot_guard(
        "nodes.node_deploy_job_reading_effect.runtime_deploy_job_reading"
    )
    import sys

    import yaml

    handler_module = sys.modules["omnibase_infra.handlers.handler_deploy_job_reading"]
    contract = yaml.safe_load(
        (
            REPO_ROOT
            / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
        ).read_text(encoding="utf-8")
    )
    from datetime import UTC, datetime

    row = handler_module.adoption_revert_alarm_row(
        contract["settings"],
        adopted_ref=NEW,
        previous_ref=OLD,
        now=datetime(2026, 11, 23, 1, 0, tzinfo=UTC),
    )

    cells = [cell.strip() for cell in row.split(" | ")]
    assert cells[0].endswith("2026-11-23T01:00:00Z") or "STATUS" in cells[0]
    assert "src=alarm" in cells
    assert "alarm=agent-adoption-reverted" in cells
    assert f"adopted_ref={NEW}" in cells
    assert f"previous_ref={OLD}" in cells


# --- AC4 ----------------------------------------------------------------------------------


def _journal_tokens() -> tuple[str, ...]:
    text = JOURNAL.read_text(encoding="utf-8")
    for token in ("ModelDevRecreateRequest", "requested_by", "string_too_short"):
        assert token in text, f"the journal fixture lost {token}"
    return ("ModelDevRecreateRequest", "requested_by", "string_too_short")


def test_agent_idle_tick_incident_replay_fails_on_the_broken_calling_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stability follow asking the window with no requester, as omnibase_infra#6080 left it."""
    check = _self_check_module()
    agent = _agent_to_check(tmp_path, monkeypatch)
    real_gate = DeployAgent._idle_converge_gate_open

    def pre_fix_gate(self: DeployAgent, who: str) -> bool:
        # Before OMN-26106 the stability follow called the gate with its default ``who=""``.
        return real_gate(self, "" if who == STABILITY_FOLLOW_REQUESTER else who)

    monkeypatch.setattr(DeployAgent, "_idle_converge_gate_open", pre_fix_gate)

    report = check.run_self_check(agent, REPO_ROOT)

    assert report.exit_code == 1
    assert set(report.failures) == {"stability_follow"}
    trace = report.failures["stability_follow"]
    for token in _journal_tokens():
        assert token in trace  # the same refusal the dev host's journal recorded


def test_agent_idle_tick_incident_replay_passes_on_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _self_check_module()
    agent = _agent_to_check(tmp_path, monkeypatch)
    assert agent_mod.DeployAgent is DeployAgent

    report = check.run_self_check(agent, REPO_ROOT)

    assert report.failures == {}
    assert report.exit_code == 0
