# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Self-healing hook for the omnimarket drift refusal (OMN-17190).

## Why this exists

Before this module, a drifted venv turned every ``onex skill`` / ``onex node``
/ ``onex delegate`` invocation into a refusal that told a human to run a repair
command by hand. The refusal was correct -- dispatching from a stale build
produces results that are not evidence -- but the *hand-run repair* was the
defect. Operator direction, 2026-08-30:

    "Why is anything hand built? ... We need a process that either
    (1) disconnects the local installation from the canonical clones, or
    (2) automatically pulls the clones whenever a PR is merged and refreshes
    the venv."

Option 2 was chosen: the local install tracks the canonical clones, and closing
the gap is automatic. So the guard now *repairs and continues* instead of
refusing, and refuses only when the repair itself cannot complete.

## What is deliberately NOT here

No bypass environment variable. The OMN-13930 override
(``ONEX_ALLOW_OMNIMARKET_DRIFT``) still exists on the guard, for an operator who
knowingly accepts results from an unverified build; it is unchanged and this
module never reads it. But a reconcile that *fails* is a different condition --
the venv is broken -- and adding a "proceed anyway" switch for that would only
move the breakage to the next dispatch, with a receipt that looks clean.

## Layering

This module is a thin, typed adapter over ``scripts/reconcile-workspace-venvs.sh``.
It holds no reconciliation policy of its own: which venvs exist, which layers
they have, and what order to repair them in are all the script's business (see
its header for the two-layer composition rule). Keeping the policy in one place
means the tick, ``pull-all.sh``, the SessionStart line, and this guard all heal
a venv the same way -- there is exactly one definition of "reconciled".

## Success is a readback, not an exit status (OMN-18663)

This module used to return ``ok=True`` whenever the reconciler subprocess exited
0. That is not the same claim, and on 2026-09-18 the difference showed twice on
the shared plugin CLI venv (omni_home/CLAUDE.md rule 11): the reconciler exited
0 and the guard, re-checking the very interpreter it was protecting, still found
drift and refused with "a reconcile ran, reported SUCCESS, and the venv is STILL
drifted".

The exit status could not have carried that claim. ``reconcile-workspace-venvs.sh``
says so in its own header (OMN-17307): it exits on its collaborators' statuses
and deliberately never re-reads the venv, because the readback belongs one layer
up in ``scripts/reconcile-host.sh``. This adapter calls the repair script
DIRECTLY, so on this path that layer simply did not exist. And the reconciler's
surface set is the dispatch venv, the gate venv and any hook venv carrying a
``uv.lock`` -- the interpreter running the CLI is not necessarily among them, so
a completely successful reconcile can leave the caller's own venv untouched.

So ``ok`` is now DERIVED from a readback of the interpreter the caller actually
runs on, using the same installed-commit-vs-canonical-clone comparison
:mod:`omnibase_infra.cli.omnimarket_drift_guard` makes. The two cannot disagree,
because there is one comparison. A reconcile whose result the caller's venv does
not carry reports ``ok=False`` naming both values, which is a true statement
about the venv rather than a true statement about a subprocess.

Every attempt carries a ``run_id``. A refusal that says "a reconcile ran" and
cannot say WHICH is not reproducible from the message alone; the id appears in
the log line, in the outcome, and in the guard's refusal text.

## Interim by design

The node-based successor named on OMN-17190 replaces the subprocess call below
with a dispatch to a NodeEffect reconcile publisher, and replaces
:class:`ModelReconcileOutcome` with the effect's typed result. The call
signature here is already shaped for that swap.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "READBACK_SCRIPT_RELATIVE_PATH",
    "RECONCILE_SCRIPT_RELATIVE_PATH",
    "ModelReconcileOutcome",
    "ReconcileFn",
    "make_workspace_reconciler",
    "new_reconcile_run_id",
    "reconcile_workspace_venvs",
]

logger = logging.getLogger(__name__)

RECONCILE_SCRIPT_RELATIVE_PATH = "omnibase_infra/scripts/reconcile-workspace-venvs.sh"

# The readback that decides whether an exit-0 reconcile actually landed on THIS
# interpreter (OMN-18663). Same comparison as the in-process guard: the venv's
# installed omnimarket VCS commit against the canonical clone's HEAD.
READBACK_SCRIPT_RELATIVE_PATH = "omnibase_infra/scripts/venv_readback.py"

# Bounded: purely local metadata and git reads, but a wedged interpreter must
# surface as a refusal rather than as a CLI that never returns.
_READBACK_TIMEOUT_SECONDS = 120

# The reconcile can reinstall a package set; on a cold venv that is a real
# install, not a metadata touch. Bounded so a stalled uv lock-wait (the
# OMN-15590 failure mode: uv takes an exclusive flock on <venv>/.lock, waits
# forever, and prints nothing) surfaces as a refusal naming the command rather
# than as a CLI that never returns.
_RECONCILE_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class ModelReconcileOutcome:
    """Result of one reconcile attempt.

    Attributes:
        ok: True only when the reconciler exited 0 AND the readback proves the
            calling interpreter now carries the canonical commit. An exit status
            alone is never enough (OMN-18663): the reconciler exits on its
            collaborators' statuses and does not necessarily own the venv this
            process is running in.
        command: The exact invocation, rendered for a human to re-run. Present
            on success and failure alike so a refusal can always name it.
        detail: One-line reason, empty on success.
        run_id: Identifies THIS attempt. Carried into the guard's refusal text
            so a message saying "a reconcile ran" names which one, and so the
            log line and the refusal can be tied together afterwards.
    """

    ok: bool
    command: str
    detail: str
    run_id: str


def new_reconcile_run_id() -> str:
    """Mint an identifier for one reconcile attempt: when, plus a unique tail."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"reconcile-{stamp}-{uuid.uuid4().hex[:8]}"


# The guard depends on this shape, not on this module's implementation, so the
# node-based successor can supply its own producer without touching the guard.
ReconcileFn = Callable[[], ModelReconcileOutcome]


def _readback(omni_home: str, target_python: str, run_id: str) -> tuple[bool, str]:
    """Ask whether ``target_python`` now carries the canonical clone's commit.

    Returns ``(proved, detail)``. ``proved`` is True only on an IN_SYNC
    readback; every other answer -- drifted, unprobeable, readback script
    missing -- is False, because none of them is evidence the repair landed.
    """
    readback = Path(omni_home) / READBACK_SCRIPT_RELATIVE_PATH
    if not readback.is_file():
        return False, (
            f"the reconcile exited 0 but its result could not be proven: no "
            f"readback at {readback} (OMN-18663). An exit status is not evidence "
            f"that this interpreter carries the canonical commit"
        )
    argv = [
        sys.executable,
        str(readback),
        "--python",
        target_python,
        "--clone",
        str(Path(omni_home) / "omnimarket"),
        "--label",
        f"post-reconcile readback ({run_id})",
    ]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_READBACK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"the post-reconcile readback could not run: {exc}"
    if result.returncode == 0:
        return True, ""

    # Carry the readback's own two values through rather than restating them:
    # the whole point of the refusal is that a reader can see what is installed
    # and what was expected without re-running anything.
    lines = [
        line.strip()
        for line in (result.stderr or result.stdout).splitlines()
        if line.strip()
    ]
    marker = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(("DRIFTED:", "INDETERMINATE:"))
        ),
        None,
    )
    reported = "; ".join(lines[marker : marker + 4]) if marker is not None else ""
    return False, (
        reported
        or f"the post-reconcile readback exited {result.returncode} with no readable report"
    )


def reconcile_workspace_venvs(
    omni_home: str, *, target_python: str | None = None
) -> ModelReconcileOutcome:
    """Run the workspace reconciler once against ``omni_home``, then PROVE it.

    ``target_python`` is the interpreter whose venv the caller actually needs
    repaired; it defaults to the running one, which is the interpreter the drift
    guard protects. It is an argument rather than an assumption so a caller
    reconciling on behalf of another venv can say so, and so a test can hand in
    a fixture interpreter.

    Never raises: every failure mode is reported as an outcome, because the
    caller (a guard on the CLI hot path) has to turn it into a refusal message
    rather than a traceback.
    """
    run_id = new_reconcile_run_id()
    python_bin = target_python or sys.executable
    script = Path(omni_home) / RECONCILE_SCRIPT_RELATIVE_PATH
    # The root travels as an explicit argument, not as an environment override.
    # That keeps this module free of any ``os.environ`` access (the
    # ``check-env-reads`` hook correctly forbids it under ``src/``) and lets the
    # subprocess inherit PATH untouched, so ``uv`` and ``git`` resolve exactly
    # as they do for the caller.
    argv = ["bash", str(script), "--omni-home", omni_home]
    command = " ".join(argv)

    if not script.is_file():
        return ModelReconcileOutcome(
            ok=False,
            command=command,
            detail=f"reconciler not found at {script}",
            run_id=run_id,
        )

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_RECONCILE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ModelReconcileOutcome(
            ok=False,
            command=command,
            detail=(
                f"reconcile exceeded {_RECONCILE_TIMEOUT_SECONDS}s and was killed "
                "(most likely another process holds the exclusive uv lock on the "
                "venv -- uv waits on it forever and prints nothing)"
            ),
            run_id=run_id,
        )
    except OSError as exc:
        return ModelReconcileOutcome(
            ok=False, command=command, detail=str(exc), run_id=run_id
        )

    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()
        return ModelReconcileOutcome(
            ok=False,
            command=command,
            detail=(tail[-1] if tail else f"reconciler exited {result.returncode}"),
            run_id=run_id,
        )

    # Exit 0 means the reconciler finished the surfaces IT owns. Whether THIS
    # interpreter is one of them is a different question, and it is the only one
    # the caller cares about (OMN-18663).
    proved, detail = _readback(omni_home, python_bin, run_id)
    if not proved:
        return ModelReconcileOutcome(
            ok=False,
            command=command,
            detail=(
                f"the reconciler exited 0 but {python_bin} does not carry the "
                f"canonical commit: {detail}. Either the reconciler does not own "
                f"this interpreter's venv (it reconciles the dispatch venv, the "
                f"gate venv, and hook venvs carrying a uv.lock) or its install "
                f"did not land. Repair this venv directly: "
                f"OMNI_HOME={omni_home} bash {Path(omni_home) / 'omnibase_infra/scripts/check-omnimarket-venv-drift.sh'} "
                f"--repair {python_bin}"
            ),
            run_id=run_id,
        )

    logger.info(
        "omnimarket drift self-healed via %s and proven by readback (run %s)",
        command,
        run_id,
    )
    return ModelReconcileOutcome(ok=True, command=command, detail="", run_id=run_id)


def make_workspace_reconciler(omni_home: str | None) -> ReconcileFn | None:
    """Bind a zero-argument reconciler for ``check_omnimarket_drift``.

    Returns ``None`` when ``omni_home`` is unset. The guard already fails open
    in that case (it cannot determine a canonical clone to compare against), so
    there is nothing to heal and nothing to bind.

    The return type is the ``ReconcileFn`` alias rather than a concrete class:
    the node-based successor swaps the body for a node dispatch without
    touching a single call site.
    """
    if not omni_home:
        return None

    def _reconcile() -> ModelReconcileOutcome:
        return reconcile_workspace_venvs(omni_home)

    return _reconcile
