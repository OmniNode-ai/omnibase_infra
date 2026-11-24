# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""When the deploy agent may adopt a code ref, and when it returns to the previous one (OMN-26494).

The deploy agent adopts new code by pulling its own clone and re-execing. omnibase_infra#6080
(OMN-25912) shipped an idle tick that raised, so deploy-agent-dev.service restarted 111 times and
accepted no job for about 4.7 hours; nothing ran the new code's idle ticks before the running
process loaded it, and nothing returned the agent to the code that worked.

Two rules live here, both loaded by file path through ``load_dev_boot_guard`` like the node's
other host entries (the agent cannot import the node as a package). The bounds are the node
contract's ``settings`` (``adoption_*``), read from the checkout's contract file.

* **Adopt only a self-checked ref.** ``adopts`` runs the candidate clone's one-shot self-check
  (``runtime_dev_agent_self_check``) in a subprocess. Non-zero exit: the clone is put back on the
  ref the process loaded, the refusal and its reason are kept, and the agent goes on running the
  previous ref. A ref refused earlier is put back without a second check until the remote moves.
* **Return after a restart storm.** ``guard_boot`` runs at every start. Five restarts of an
  adopted ref inside ten minutes put the clone back on the previous ref, append one alarm row
  naming both refs through the node's ledger route, and restart on the previous ref.

State (current ref, previous ref, boot times, refused refs) is one JSON record in the agent's
state directory, written by the agent process alone (atomic rename), so it survives a restart. A
record that exists and cannot be read raises ``AdoptionStateUnreadableError``: the agent never
falls back to the new ref on a guess.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

NODE_CONTRACT = Path(
    "src/omnibase_infra/nodes/node_dev_roll_events_effect/contract.yaml"
)
SELF_CHECK_ENTRY = Path(
    "src/omnibase_infra/nodes/node_dev_roll_events_effect/runtime_dev_agent_self_check.py"
)
#: The refusal reason recorded for a ref that restarted the agent in a storm.
STORM_REASON = "restart-storm"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
#: Restores the clone to a ref; raises when it cannot.
Reverter = Callable[[str], None]
#: Appends the alarm row for (adopted ref, previous ref); returns "" when taken, else why not.
Alarmer = Callable[[str, str], str]


class AdoptionStateUnreadableError(RuntimeError):
    """The adoption record exists and could not be read; nothing is guessed from it."""


@dataclass(frozen=True)
class AdoptionSettings:
    """The contract's ``adoption_*`` settings."""

    restart_limit: int
    storm_window_seconds: int
    state_file: str
    self_check_timeout_seconds: int
    git_timeout_seconds: int


@dataclass(frozen=True)
class AdoptionState:
    """What the agent knows about the refs it runs, kept across restarts."""

    current_ref: str
    previous_ref: str
    #: Start times (epoch seconds) of ``current_ref``; the first is the adoption's own start.
    boots: tuple[float, ...]
    alarmed: bool
    #: ``(ref, reason)`` for every ref the agent refused or left.
    refused: tuple[tuple[str, str], ...]


_EMPTY = AdoptionState("", "", (), False, ())


def settings_at(repo_root: Path) -> AdoptionSettings:
    """The adoption settings of the checkout at ``repo_root``; raises when one is missing."""
    declared = yaml.safe_load((repo_root / NODE_CONTRACT).read_text(encoding="utf-8"))[
        "settings"
    ]
    return AdoptionSettings(
        restart_limit=int(declared["adoption_restart_limit"]),
        storm_window_seconds=int(declared["adoption_storm_window_seconds"]),
        state_file=str(declared["adoption_state_file"]),
        self_check_timeout_seconds=int(declared["adoption_self_check_timeout_seconds"]),
        git_timeout_seconds=int(declared["adoption_git_timeout_seconds"]),
    )


def _root(agent_dir: str) -> Path:
    """The checkout root of the agent's directory (``<root>/scripts/deploy-agent``)."""
    return Path(agent_dir).resolve().parents[1]


def state_path(state_dir: Path, settings: AdoptionSettings) -> Path:
    return state_dir / settings.state_file


def read_state(state_dir: Path, settings: AdoptionSettings) -> AdoptionState | None:
    """The record, ``None`` when none was ever written; raises when it exists and is unreadable."""
    path = state_path(state_dir, settings)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return AdoptionState(
            current_ref=str(raw["current_ref"]),
            previous_ref=str(raw["previous_ref"]),
            boots=tuple(float(item) for item in raw["boots"]),
            alarmed=bool(raw["alarmed"]),
            refused=tuple((str(ref), str(why)) for ref, why in raw["refused"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AdoptionStateUnreadableError(
            f"ADOPTION_STATE_UNREADABLE: {path} exists and could not be read "
            f"({type(exc).__name__}: {exc}); refusing to guess which ref is safe. "
            "Remove the file once the running ref is known."
        ) from exc


def write_state(
    state_dir: Path, settings: AdoptionSettings, state: AdoptionState
) -> None:
    """Write the record atomically: a reader sees the old record or the new one."""
    payload = {
        "current_ref": state.current_ref,
        "previous_ref": state.previous_ref,
        "boots": list(state.boots),
        "alarmed": state.alarmed,
        "refused": [list(item) for item in state.refused],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_dir, prefix=f".{settings.state_file}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        Path(tmp).replace(state_path(state_dir, settings))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def refused_reason(
    state_dir: Path | None, settings: AdoptionSettings, ref: str
) -> str | None:
    """Why ``ref`` was refused earlier, ``None`` when it was not."""
    state = read_state(state_dir, settings) if state_dir is not None else None
    if state is None:
        return None
    return next((why for bad, why in state.refused if bad == ref), None)


def _refuse(
    state_dir: Path | None,
    settings: AdoptionSettings,
    ref: str,
    why: str,
    loaded_ref: str,
) -> None:
    if state_dir is None:
        return
    state = read_state(state_dir, settings) or replace(_EMPTY, current_ref=loaded_ref)
    if all(bad != ref for bad, _ in state.refused):
        state = replace(state, refused=(*state.refused, (ref, why)))
    write_state(state_dir, settings, state)


def candidate_self_check(
    agent_dir: str, run: Runner, settings: AdoptionSettings
) -> subprocess.CompletedProcess[str]:
    """Run the one-shot self-check of the clone at ``agent_dir`` in a subprocess.

    The subprocess gets its own scratch state directory (``DEPLOY_AGENT_STATE_DIR``), so no tick
    reads or writes the agent's job records.
    """
    with tempfile.TemporaryDirectory(prefix="deploy-agent-self-check-") as scratch:
        env = dict(os.environ)
        env["DEPLOY_AGENT_STATE_DIR"] = str(Path(scratch) / "jobs")
        env["PYTHONPATH"] = os.pathsep.join(
            [agent_dir, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
        return run(
            [sys.executable, str(_root(agent_dir) / SELF_CHECK_ENTRY)],
            timeout=settings.self_check_timeout_seconds,
            cwd=agent_dir,
            env=env,
        )


def adopts(
    agent_dir: str,
    state_dir: Path | None,
    refs: tuple[str, str],
    run: Runner,
    resync: Callable[[str], None],
) -> bool:
    """Whether the agent may re-exec onto ``candidate_ref``, the clone's HEAD after the pull.

    ``refs`` is ``(loaded_ref, candidate_ref)``: the ref the process loaded and the candidate.

    A ref refused earlier is put back without a second check. Otherwise ``resync(agent_dir)``
    installs the clone's dependencies so the candidate's imports resolve, and the
    candidate's self-check runs. A non-zero exit, a timeout or a self-check that could not start
    refuses the ref: the clone goes back to ``loaded_ref``, ``resync`` restores that ref's
    dependencies, the reason is kept and the agent stays where it is. On success the adoption is
    recorded (previous ref = ``loaded_ref``) before the caller re-execs.
    """
    loaded_ref, candidate_ref = refs
    settings = settings_at(_root(agent_dir))
    earlier = refused_reason(state_dir, settings, candidate_ref)
    if earlier is not None:
        logger.info(
            "self_update: holding %s, refused earlier (%s); staying on %s",
            candidate_ref[:12],
            earlier,
            loaded_ref[:12],
        )
        git_revert(agent_dir, loaded_ref, run, settings)
        return False
    resync(agent_dir)
    try:
        done = candidate_self_check(agent_dir, run, settings)
        code, tail = done.returncode, (done.stdout + done.stderr).strip()[-400:]
    except (OSError, subprocess.SubprocessError) as exc:
        code, tail = -1, f"{type(exc).__name__}: {exc}"
    if code == 0:
        logger.info(
            "self_update: self-check of %s exited 0, adopting it over %s",
            candidate_ref[:12],
            loaded_ref[:12],
        )
        if state_dir is not None:
            known = read_state(state_dir, settings) or _EMPTY
            write_state(
                state_dir,
                settings,
                AdoptionState(candidate_ref, loaded_ref, (), False, known.refused),
            )
        return True
    logger.error(
        "self_update: self-check of %s exited %s, staying on %s: %s "
        "friction_type=self_check_refused_adoption",
        candidate_ref[:12],
        code,
        loaded_ref[:12],
        tail.replace("\n", " | "),
    )
    _refuse(state_dir, settings, candidate_ref, f"self-check-exit-{code}", loaded_ref)
    git_revert(agent_dir, loaded_ref, run, settings)
    resync(agent_dir)
    return False


def git_revert(
    agent_dir: str, ref: str, run: Runner, settings: AdoptionSettings
) -> None:
    """Put the clone's checked-out branch back on ``ref``; raises when git refuses.

    ``self_update`` runs this only after its dirty check, so no tracked work is discarded.
    """
    done = run(
        ["git", "-C", agent_dir, "reset", "--hard", ref],
        timeout=settings.git_timeout_seconds,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"ADOPTION_REVERT_FAILED: git reset --hard {ref[:12]} exited "
            f"{done.returncode}: {done.stderr.strip()[:200]}"
        )


def record_boot(
    state_dir: Path,
    settings: AdoptionSettings,
    loaded_ref: str,
    now: float,
    *,
    revert: Reverter,
    alarm: Alarmer,
) -> bool:
    """Record this start; True when a storm returned the clone to the previous ref.

    A start of a ref the record does not hold as current is a new adoption (previous ref =
    the record's current ref). Restarts are the starts after the first of one ref; the fifth
    inside the storm window reverts to the previous ref, appends one alarm row naming both
    refs and keeps the storm's ref refused. The row's request id is fixed by the two refs, so
    a start that repeats the revert after a crash appends it once.
    """
    state = read_state(state_dir, settings)
    if state is None or state.current_ref != loaded_ref:
        write_state(
            state_dir,
            settings,
            AdoptionState(
                current_ref=loaded_ref,
                previous_ref=state.current_ref if state else "",
                boots=(now,),
                alarmed=False,
                refused=state.refused if state else (),
            ),
        )
        return False
    limit, window = settings.restart_limit, settings.storm_window_seconds
    boots = (*state.boots, now)[-(limit + 1) :]
    state = replace(state, boots=boots)
    restarts = boots[1:]
    storm = len(restarts) >= limit and restarts[-1] - restarts[-limit] <= window
    if not storm:
        write_state(state_dir, settings, state)
        return False
    if not state.previous_ref:
        write_state(state_dir, settings, state)
        logger.error(
            "adoption: %d restarts of %s inside %ds and no previous ref is known, "
            "so nothing to return to friction_type=restart_storm_no_previous_ref",
            limit,
            loaded_ref[:12],
            window,
        )
        return False
    logger.error(
        "adoption: %d restarts of %s inside %ds, returning to %s "
        "friction_type=restart_storm_revert",
        limit,
        loaded_ref[:12],
        window,
        state.previous_ref[:12],
    )
    revert(state.previous_ref)
    failure = alarm(loaded_ref, state.previous_ref)
    if failure:
        logger.error(
            "adoption: the alarm row for %s -> %s was not appended (%s) "
            "friction_type=restart_storm_alarm_not_appended",
            loaded_ref[:12],
            state.previous_ref[:12],
            failure,
        )
    write_state(
        state_dir,
        settings,
        AdoptionState(
            current_ref=state.previous_ref,
            previous_ref="",
            boots=(),
            alarmed=not failure,
            refused=(*state.refused, (loaded_ref, STORM_REASON)),
        ),
    )
    return True


def guard_boot(
    execution: object,
    loaded_ref: str,
    alarm: Alarmer,
    reexec: Callable[[str, list[str]], None],
) -> None:
    """The agent's startup step: journal the loaded ref, record this start, and on a storm
    return to the previous ref and re-exec onto it.

    ``execution`` is the agent module (``STATE_DIR``, ``DEPLOY_AGENT_DIR``), ``alarm`` the deploy
    job reading's ``record_adoption_revert`` and ``reexec`` ``os.execv``.
    """
    agent_dir = str(vars(execution)["DEPLOY_AGENT_DIR"])
    settings = settings_at(_root(agent_dir))
    logger.info("Deploy agent loaded code sha: %s", loaded_ref)

    def revert(ref: str) -> None:
        git_revert(agent_dir, ref, _run, settings)

    stormed = record_boot(
        Path(vars(execution)["STATE_DIR"]),
        settings,
        loaded_ref,
        time.time(),
        revert=revert,
        alarm=alarm,
    )
    if stormed:
        reexec(sys.executable, [sys.executable, *sys.argv])


def _run(
    cmd: list[str], timeout: int, **kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, timeout=timeout, capture_output=True, text=True, check=False
    )
