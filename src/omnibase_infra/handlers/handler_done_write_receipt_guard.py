# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Done-write receipt guard: the one check every Linear Done writer calls.

OMN-20368. Operator ruling (2026-10-02): a ticket reaches Done only on a PASS
dod_verify whose checks bind every acceptance criterion. The rule itself is
``omnibase_core.handlers.handler_done_write_receipt_gate`` (lifted out of the evidence-
autoclose closer so that omnimarket and this repo evaluate one implementation).
This module is the infra half: it runs the verifier the way the closer does,
reads the verdict off either declared receipt arm, and hands verdict plus the
ticket's current description to the core rule.

Fail-closed at every step. A verifier that cannot be launched, times out,
prints no JSON, declares no receipt arm, or reaches no verdict is a refusal:
"I could not check" never resolves to "so I will write".

Callers: the sync-revert watchdog re-flip, and the generic Linear adapters
(``AdapterTicketLinear``, ``AdapterLinearGraphQLProjectTracker``) for any write
whose target state is a Done state. Writes to any other state do not touch
this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from omnibase_core.handlers.handler_done_write_receipt_gate import (
    evaluate_done_write_receipt,
    extract_dod_verify_verdict,
)
from omnibase_core.models.ticket.model_done_write_decision import (
    ModelDoneWriteDecision,
)

logger = logging.getLogger(__name__)

#: Linear's workflow-state type for every Done-like state.
LINEAR_COMPLETED_STATE_TYPE = "completed"
#: The state NAME a Done write carries when the caller does not know the type.
LINEAR_DONE_STATE_NAME = "done"
#: A verifier run executes the contract's checks, so it is slow by design.
DEFAULT_DOD_VERIFY_TIMEOUT_SECONDS = 300.0

# Injectable runner signature (the real one execs the dispatch venv's `onex`;
# tests inject fakes with the same shape).
TypeRunDodVerifyCommand = Callable[
    [str, str, float], Awaitable[tuple[dict[str, object] | None, int, str]]
]


def dod_verify_argv(ticket_id: str) -> list[str]:
    """Argv that dispatches the verifier from THIS process's own environment.

    OMN-16846. This used to be ``["uv", "run", "onex", "skill", "dod_verify",
    ticket_id]``, which does not name an environment at all — ``uv run``
    re-resolves the project at the subprocess's cwd and uses that project's
    venv. The sweep's cwd is the omnibase_infra product clone, so the verifier
    landed in the PRODUCT venv, and the only way to make the verifier
    resolvable was to co-install its provider (omnimarket, which ships
    ``node_dod_verify``) into that same venv.

    That is the collision. dod_verify's behaviour checks are ``uv run pytest``
    with ``cwd: "${OMNI_HOME}/omnibase_infra"``, so they resolve the very same
    venv, where ``tests/conftest.py`` calls ``assert_venv_purity()``
    (OMN-15620) and correctly refuses an undeclared ``onex.nodes`` provider.
    Both halves are right; only their collapse onto one venv is wrong. Run
    33194402437 is the receipt: all three OMN-16759 behaviour checks recorded
    ``FAILED ... Canonical venv is IMPURE``, ``behavior_proving=0``, without
    a single test having executed.

    Dispatching through ``sys.executable``'s sibling ``onex`` makes the
    verifier's environment a property of how the SWEEP was composed rather
    than of where it happens to be standing. The dispatch venv can then carry
    the co-installed omnimarket while the product clone's venv stays pure and
    the behaviour checks actually run. It is also strictly more determinate
    for the existing local path, where both resolved to the same venv anyway.

    A missing sibling ``onex`` raises ``FileNotFoundError`` at exec time,
    which the caller already converts into a named per-ticket error; the
    sweep job additionally asserts the binary resolves before any ticket is
    scanned, so the loud failure comes first.
    """
    return [
        str(Path(sys.executable).parent / "onex"),
        "skill",
        "dod_verify",
        ticket_id,
        "--execution-audience",
        "hosted",
    ]


async def reap_timed_out_process(proc: asyncio.subprocess.Process) -> None:
    """Kill and reap a subprocess whose ``communicate()`` await timed out.

    ``asyncio.wait_for`` cancels the *await*, not the child process: without
    this, a `gh`/`onex` invocation that hangs past its timeout keeps running
    with its stdout/stderr pipes held open, leaking a process per timeout on
    every scheduled sweep tick.
    """
    if proc.returncode is not None:
        return
    proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        logger.warning(
            "Timed-out subprocess (pid=%s) did not exit after kill()", proc.pid
        )


async def run_dod_verify_command(
    ticket_id: str, cwd: str, timeout: float
) -> tuple[dict[str, object] | None, int, str]:
    # OMN-16846: resolved from THIS process's interpreter, not re-resolved
    # by `uv run` from the cwd. See `dod_verify_argv`.
    args = dod_verify_argv(ticket_id)
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        # Process creation itself can raise (missing executable, invalid
        # cwd) before there is any `proc` to reap — must be caught here,
        # not only around `communicate()` below.
        return None, -1, f"OS error launching dod_verify for {ticket_id}: {exc}"
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        await reap_timed_out_process(proc)
        return None, -1, f"Timeout running dod_verify for {ticket_id}"
    except OSError as exc:
        await reap_timed_out_process(proc)
        return None, -1, f"OS error running dod_verify for {ticket_id}: {exc}"
    exit_code = proc.returncode if proc.returncode is not None else -1
    stderr_text = stderr.decode(errors="replace").strip()
    # Parse stdout REGARDLESS of exit code (OMN-16736). `onex skill` exits
    # non-zero whenever the verdict is `failed` — i.e. on every genuine
    # evidence GAP — while still printing a complete, valid ModelSkillResult
    # on stdout. A prior revision discarded stdout the moment the exit code
    # was non-zero, so every real gap was misrecorded as
    # ERROR_VERIFY_NONZERO_EXIT ("the verifier crashed") instead of
    # GAP_POSTED ("the ticket is not proven"). Those are different facts and
    # only one of them is actionable by the ticket's owner.
    try:
        parsed = json.loads(stdout.decode(errors="replace"))
    except json.JSONDecodeError as exc:
        # No parseable verdict. Prefer stderr when the process also failed —
        # that is where a dispatch error (missing node, bad contract) lands.
        return (
            None,
            exit_code,
            stderr_text if exit_code != 0 else f"Invalid dod_verify JSON: {exc}",
        )
    if not isinstance(parsed, dict):
        return None, exit_code, "dod_verify output was not a JSON object"
    return parsed, exit_code, ""


def is_done_state(name: str | None = None, state_type: str | None = None) -> bool:
    """Whether a Linear workflow state is a Done state.

    ``state_type`` is Linear's own classification (``completed``); ``name`` is
    the human label. Either one says Done. Canceled and Duplicate are a
    different type and are not Done.
    """
    if state_type is not None and state_type.strip().lower() == (
        LINEAR_COMPLETED_STATE_TYPE
    ):
        return True
    return name is not None and name.strip().lower() == LINEAR_DONE_STATE_NAME


class DoneWriteRefusedError(Exception):
    """Raised when the Done-write receipt gate refuses a write.

    Attributes:
        ticket_id: The Linear identifier the Done write targeted.
        decision: The structured refusal verdict (``allowed=False`` + reason).
    """

    def __init__(self, ticket_id: str, decision: ModelDoneWriteDecision) -> None:
        self.ticket_id = ticket_id
        self.decision = decision
        super().__init__(f"Refusing Done write for {ticket_id}: {decision.reason}")


class DoneWriteReceiptGuard:
    """Run the verifier and apply the core Done-write rule to its verdict."""

    def __init__(
        self,
        *,
        run_dod_verify: TypeRunDodVerifyCommand | None = None,
        cwd: str = "",
        timeout_seconds: float = DEFAULT_DOD_VERIFY_TIMEOUT_SECONDS,
    ) -> None:
        self._run_dod_verify = run_dod_verify or run_dod_verify_command
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds

    async def decide(
        self, *, ticket_id: str, description: str
    ) -> ModelDoneWriteDecision:
        """Verdict for one Done write. Never raises on a verifier failure."""
        if not ticket_id.strip():
            return ModelDoneWriteDecision(
                allowed=False,
                reason=(
                    "no ticket id: a Done write needs the issue identifier so "
                    "its dod_verify receipt can be read (OMN-20368)."
                ),
            )
        result, exit_code, error = await self._run_dod_verify(
            ticket_id, self._cwd, self._timeout_seconds
        )
        if result is None:
            return ModelDoneWriteDecision(
                allowed=False,
                reason=(
                    f"dod_verify reached no output for {ticket_id} "
                    f"(exit_code={exit_code}): {error}"
                ),
            )
        verdict, why = extract_dod_verify_verdict(result)
        if verdict is None:
            return ModelDoneWriteDecision(
                allowed=False,
                reason=f"dod_verify exit_code={exit_code}: {why}",
            )
        return evaluate_done_write_receipt(
            ticket_id=ticket_id, description=description, verdict=verdict
        )

    async def enforce(
        self, *, ticket_id: str, description: str
    ) -> ModelDoneWriteDecision:
        """:meth:`decide`, raising :class:`DoneWriteRefusedError` on a refusal."""
        decision = await self.decide(ticket_id=ticket_id, description=description)
        if not decision.allowed:
            raise DoneWriteRefusedError(ticket_id, decision)
        return decision


__all__ = [
    "DEFAULT_DOD_VERIFY_TIMEOUT_SECONDS",
    "DoneWriteReceiptGuard",
    "DoneWriteRefusedError",
    "LINEAR_COMPLETED_STATE_TYPE",
    "LINEAR_DONE_STATE_NAME",
    "TypeRunDodVerifyCommand",
    "dod_verify_argv",
    "is_done_state",
    "reap_timed_out_process",
    "run_dod_verify_command",
]
