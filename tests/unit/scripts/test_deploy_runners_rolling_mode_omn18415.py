# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the rolling (one-at-a-time, busy-checked) fleet deploy mode (OMN-18415).

Container env is frozen at creation, so an env change reaches a live runner only
through a RECREATE. The default deploy path recreates every service in one
compose call, which takes the whole fleet down at once; ``--soft`` never
recreates and so cannot carry an env change at all. Neither is usable for an env
roll, and the procedure that is -- one service at a time, skipping any runner
executing someone else's job -- was a hand-typed loop until this ticket.

The properties asserted here are the ones whose absence is invisible in a green
run and expensive in a live one:

- the roll names exactly ONE service per compose call, and never sweeps orphans
  (an orphan sweep mid-roll deletes containers the pass has not reached);
- the busy check consults two independent signals and fails CLOSED -- an API
  error, an unreadable process list, or a disagreement all skip the runner. A
  fail-open process check killed a live job once already;
- the roll re-checks busy immediately before stopping a container, because a job
  can start between selection and recreate;
- a runner that does not come back HALTS the roll instead of cascading;
- ``--soft`` and ``--rolling`` are refused together rather than silently letting
  the one that cannot carry an env change win.

These are source-level assertions against a bash script, which is a weaker
instrument than an execution test -- so each one pins a distinctive construct
and is paired with a control that fails if the construct is renamed away.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rolling_block(script_text: str) -> str:
    """The rolling mode's own functions, isolated from the batch deploy path."""
    start = script_text.index("fleet_services() {")
    end = script_text.index("main() {")
    block = script_text[start:end]
    assert len(block) > 500, "rolling_deploy() body is suspiciously small"
    return block


def test_the_script_parses(script_text: str) -> None:
    """Control for every source assertion below: a syntactically broken script
    would still satisfy string matching.
    """
    result = subprocess.run(
        ["bash", "-n", str(DEPLOY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_rolling_flag_is_parsed_and_documented(script_text: str) -> None:
    assert "--rolling)    ROLLING_DEPLOY=true ;;" in script_text
    assert "ROLLING_DEPLOY=false" in script_text
    assert "--rolling" in script_text[: script_text.index("set -euo pipefail")], (
        "the mode must appear in the script's own header usage block"
    )


def test_soft_and_rolling_are_mutually_exclusive(script_text: str) -> None:
    """--soft cannot carry a container-env change. Accepting both would run the
    one that cannot do the job and report success.
    """
    assert '"${SOFT_DEPLOY}" && "${ROLLING_DEPLOY}"' in script_text
    guard = script_text[script_text.index('"${SOFT_DEPLOY}" && "${ROLLING_DEPLOY}"') :][
        :400
    ]
    assert "err " in guard, "the combination must be refused, not warned about"


# --- blast radius: one service per call, no orphan sweep --------------------


def test_the_roll_recreates_exactly_one_named_service_per_compose_call(
    rolling_block: str,
) -> None:
    up_calls = re.findall(r"up -d[^\n\"]*", rolling_block)
    assert up_calls, "no compose `up` call found in the rolling path"
    for call in up_calls:
        assert "--no-deps" in call, f"missing --no-deps: {call!r}"
        assert "${name}" in call, (
            f"the roll must name a single service explicitly: {call!r}"
        )


def _executable_lines(block: str) -> str:
    """Drop comment lines: the rolling path explains in prose why it does not
    sweep orphans, and that explanation must not read as the flag itself.
    """
    return "\n".join(
        line for line in block.splitlines() if not line.lstrip().startswith("#")
    )


def test_the_roll_never_sweeps_orphans(rolling_block: str) -> None:
    assert "--remove-orphans" not in _executable_lines(rolling_block), (
        "an orphan sweep during a partial roll deletes containers the pass has "
        "not reached yet"
    )


def test_the_orphan_control_is_meaningful(script_text: str) -> None:
    """Positive control: the batch path DOES use --remove-orphans, so the
    absence above is a property of the rolling path, not of the whole file.
    """
    assert "--remove-orphans" in script_text


# --- the busy check fails closed --------------------------------------------


def test_the_busy_check_consults_two_independent_signals(script_text: str) -> None:
    start = script_text.index("runner_is_idle() {")
    end = script_text.index("wait_for_runner_online() {")
    body = script_text[start:end]
    assert "github_runner_state" in body, (
        "GitHub's authoritative busy flag is not consulted"
    )
    assert "docker top" in body, "the container process list is not consulted"
    assert "Runner.Worker" in body, "the process check does not look for a job worker"


def test_every_unknown_path_in_the_busy_check_returns_not_idle(
    script_text: str,
) -> None:
    """Fail-closed means UNKNOWN skips. Each guarded branch must return
    non-zero; a bare `return 0` reachable from a failed probe is the fail-open
    shape this pins against.
    """
    start = script_text.index("runner_is_idle() {")
    end = script_text.index("wait_for_runner_online() {")
    body = script_text[start:end]
    # Exactly one success return, and it is the last statement in the function.
    successes = re.findall(r"^\s*return 0\s*$", body, flags=re.MULTILINE)
    assert len(successes) == 1, (
        f"expected exactly one success return in runner_is_idle, found {len(successes)}"
    )
    assert body.rstrip().rstrip("}").rstrip().endswith("return 0"), (
        "the success return must be the last statement, after every guard"
    )
    # One skip return per failure branch: GitHub state not online/idle, the
    # process probe failing, an empty (unreadable) process answer, and a
    # non-zero worker count.
    assert body.count("return 1") >= 4, (
        f"expected a skip return on each failure branch, found {body.count('return 1')}"
    )
    assert "|| return 1" in body


def test_an_unresolvable_github_state_is_reported_as_unknown(
    script_text: str,
) -> None:
    start = script_text.index("github_runner_state() {")
    end = script_text.index("runner_is_idle() {")
    body = script_text[start:end]
    assert body.count("unknown unknown") >= 2, (
        "an API failure and a missing runner must both resolve to unknown"
    )


def test_busy_is_rechecked_immediately_before_stopping(rolling_block: str) -> None:
    """A job can start between the selection pass and the recreate."""
    start = rolling_block.index("roll_one_runner() {")
    recreate = rolling_block.index("up -d", start)
    prelude = rolling_block[start:recreate]
    assert "runner_is_idle" in prelude, (
        "roll_one_runner must re-check idleness before it touches the container"
    )
    assert "return 2" in prelude, "a busy runner must be skipped, not recreated"


# --- failure handling -------------------------------------------------------


def test_a_runner_that_does_not_return_halts_the_roll(rolling_block: str) -> None:
    assert "wait_for_runner_online" in rolling_block
    assert "HALTED at" in rolling_block
    halt = rolling_block[rolling_block.index("HALTED at") - 200 :][:600]
    assert "err " in halt, "the halt must exit non-zero, not warn and continue"
    assert "do NOT retry blindly" in halt, (
        "the halt message must carry the wedged-removal hazard, which is what "
        "makes a blind retry worse than stopping"
    )


def test_skipped_runners_are_retried_and_then_reported(rolling_block: str) -> None:
    assert "ROLL_SKIP_RETRY_PASSES" in rolling_block
    assert "Still busy after" in rolling_block
    assert "Rolled ${done_count}/${total} runners." in rolling_block


def test_the_fleet_list_comes_from_compose_and_is_cross_checked(
    script_text: str,
) -> None:
    """Deriving service names from a counter would invent names the compose
    file does not declare if the two ever disagreed.
    """
    start = script_text.index("fleet_services() {")
    end = script_text.index("github_runner_state() {")
    body = script_text[start:end]
    assert "${COMPOSE_FILE}" in body, "the list must come from the compose file"
    assert "RUNNER_COUNT" in body and "err " in body, (
        "a disagreement between compose and runner_fleet.yaml must fail closed"
    )
