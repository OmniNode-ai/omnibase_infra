# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's one-shot self-check of a candidate code ref (OMN-26494).

Run as ``python <this file>`` by ``runtime_dev_agent_adoption.adopts`` with the candidate clone
on ``PYTHONPATH``: it builds the candidate's ``DeployAgent`` and runs its three idle ticks once
each, the idle restore, the idle converge and the stability follow, against a fake clock and a
dry run: no Docker, no git push, no Kafka publish, no host command and no ledger append. The
recreate window gate in front of every tick is the real one from the candidate's policy with a
recorder in place of its announcement publisher, so a tick that asks it with a request the
window model refuses raises here as it raised on the dev host (omnibase_infra#6080).

Exit 0 when no tick raised; exit 1 and one line per raised tick on stderr otherwise. The
agent's own seams stand in for what a tick reads (the runtime's presence, the followed refs,
the dev tags); the converge tick's decision is fixed to CONVERGE, so the tick body that asks
the gate runs whatever the host's routing says. The verdict rules themselves are pure and have
their own tests; what this runs is the code that wires them to the window and to the job path.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from operator import methodcaller
from pathlib import Path

logger = logging.getLogger(__name__)

#: The fake clock: a window start, so the real policy's verdict is RECREATE and the tick goes on
#: to its job path. The date is fixed, never today's.
FAKE_NOW = datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC)
_OLD = "1" * 40
_NEW = "3" * 40
#: One entry per tick: its name and the callable that runs it once.
TICKS = ("idle_restore", "idle_converge", "stability_follow")


@dataclass(frozen=True)
class SelfCheckReport:
    """What one self-check saw."""

    #: Each tick that raised, with its traceback.
    failures: dict[str, str]
    #: The requester of every recreate the dry-run window announced, in order.
    announced: tuple[str, ...]
    #: How many jobs the ticks handed to the (dry) job path.
    dispatched: int

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0


@contextmanager
def _swapped(target: object, name: str, value: object) -> Iterator[None]:
    """``target.name`` is ``value`` inside the block, and what it was after it."""
    before = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, before)


def run_self_check(agent: object, repo_root: Path) -> SelfCheckReport:
    """Run each idle tick of ``agent`` once, dry; no failures is a pass.

    Everything the ticks could touch is replaced for the call and
    restored after it. ``repo_root`` is the candidate checkout, whose window policy the gate reads.
    """
    from deploy_agent import agent as agent_mod
    from deploy_agent.events import EnumRuntimeLane
    from deploy_agent.executor import load_dev_boot_guard
    from deploy_agent.idle_converge import (
        EnumIdleConvergeVerdict,
        EnumRuntimePresence,
        ModelDevTag,
        ModelIdleConvergeDecision,
    )
    from deploy_agent.lane_policy import parse_allowed_lanes
    from deploy_agent.routing import ModelLaneFlags

    roll = load_dev_boot_guard(agent_mod.DEV_ROLL_EVENTS)
    announced: list[str] = []

    def announce(event: object) -> bool:
        announced.append(str(getattr(event, "requested_by", "")))
        return True

    def fresh_gate() -> object:
        # One gate per tick: a tick that recreates spends the window, and the next tick's
        # ask would then be held without reaching the request the window model validates.
        return roll.recreate_gate_at(
            repo_root, None, announce=announce, clock=lambda: FAKE_NOW
        )

    commands: list[object] = []
    tag = ModelDevTag(name="dev-20260105000000-" + _NEW[:12], sha=_NEW, cut_at=FAKE_NOW)
    presence = {"value": (EnumRuntimePresence.DOWN, "self-check: runtime down")}
    failures: dict[str, str] = {}

    store = vars(agent)["job_store"]
    decision = ModelIdleConvergeDecision(
        verdict=EnumIdleConvergeVerdict.CONVERGE, head_ref=_NEW, detail="self-check"
    )
    overrides: list[tuple[object, str, object]] = [
        (agent_mod, "publish_result", lambda payload, cfg: False),
        (agent_mod, "publish_dev_tag_heartbeat", lambda payload, cfg: True),
        (agent, "_execute_command", commands.append),
        (agent, "_job_reading", lambda *args: None),
        (agent, "_idle_converge_now", lambda: FAKE_NOW),
        (agent, "_idle_converge_started_at", FAKE_NOW - timedelta(days=1)),
        (agent, "_lane_flags", ModelLaneFlags()),
        (
            agent,
            "_allowed_lanes",
            parse_allowed_lanes(
                f"{EnumRuntimeLane.DEV.value},{EnumRuntimeLane.STABILITY_TEST.value}"
            ),
        ),
        (agent, "_idle_read_runtime_presence", lambda: presence["value"]),
        (agent, "_idle_probe_blocker", lambda now: None),
        (agent, "_idle_read_running_ref", lambda repo: _NEW),
        (agent, "_idle_read_head_ref", lambda repo: _NEW),
        (agent, "_idle_read_dev_tags", lambda repo: [tag]),
        (agent, "_idle_running_contains", lambda *args: False),
        (agent, "_stability_read_running_ref", lambda repo: _OLD),
        (store, "latest_terminal_status", lambda lane: "success"),
        # A tick's accepted job stays accepted (nothing runs it), so none counts as in flight.
        (store, "has_active_job", lambda: False),
        (agent, "_decide_idle_converge", lambda repository, now: decision),
    ]

    def restore() -> None:
        presence["value"] = (EnumRuntimePresence.DOWN, "self-check: runtime down")
        methodcaller("_maybe_idle_restore", FAKE_NOW)(agent)

    def converge() -> None:
        presence["value"] = (EnumRuntimePresence.RUNNING, "self-check: running")
        methodcaller("_maybe_idle_converge")(agent)

    def follow() -> None:
        methodcaller("_maybe_stability_follow", FAKE_NOW)(agent)

    ticks: dict[str, Callable[[], None]] = {
        "idle_restore": restore,
        "idle_converge": converge,
        "stability_follow": follow,
    }
    with ExitStack() as stack:
        for target, name, value in overrides:
            stack.enter_context(_swapped(target, name, value))
        for name in TICKS:
            vars(agent)["_recreate_gate"] = fresh_gate()
            try:
                ticks[name]()
            except Exception:
                logger.exception("self-check: tick %s raised", name)
                failures[name] = traceback.format_exc()
        vars(agent).pop("_recreate_gate", None)
    return SelfCheckReport(failures, tuple(announced), len(commands))


def main() -> int:
    """Build the candidate's agent in a scratch state directory and run the check."""
    scratch = tempfile.mkdtemp(prefix="deploy-agent-self-check-")
    for name in ("ONEX_LEDGER_BUS_APPEND_COMMAND",):
        os.environ.pop(name, None)
    from deploy_agent import agent as agent_mod

    root = Path(__file__).resolve().parents[4]
    with _swapped(agent_mod, "STATE_DIR", Path(scratch) / "jobs"):
        agent = agent_mod.DeployAgent(skip_self_update=True)
    try:
        report = run_self_check(agent, root)
    finally:
        for pool in ("_job_pool", "_lag_pool", "_settle_pool"):
            getattr(agent, pool).shutdown(wait=False)
    failures = report.failures
    for name, trace in failures.items():
        sys.stderr.write(f"self-check: tick {name} raised\n{trace}\n")
    sys.stdout.write(
        f"self-check: {len(TICKS) - len(failures)} of {len(TICKS)} idle ticks ran clean\n"
    )
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
