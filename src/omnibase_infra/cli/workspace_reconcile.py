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

## Interim by design

The node-based successor named on OMN-17190 replaces the subprocess call below
with a dispatch to a NodeEffect reconcile publisher, and replaces
:class:`ModelReconcileOutcome` with the effect's typed result. The call
signature here is already shaped for that swap.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RECONCILE_SCRIPT_RELATIVE_PATH",
    "ModelReconcileOutcome",
    "ReconcileFn",
    "make_workspace_reconciler",
    "reconcile_workspace_venvs",
]

logger = logging.getLogger(__name__)

RECONCILE_SCRIPT_RELATIVE_PATH = "omnibase_infra/scripts/reconcile-workspace-venvs.sh"

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
        ok: True only when the reconciler exited 0. Any other exit -- including
            a timeout or a missing script -- is False.
        command: The exact invocation, rendered for a human to re-run. Present
            on success and failure alike so a refusal can always name it.
        detail: One-line reason, empty on success.
    """

    ok: bool
    command: str
    detail: str


# The guard depends on this shape, not on this module's implementation, so the
# node-based successor can supply its own producer without touching the guard.
ReconcileFn = Callable[[], ModelReconcileOutcome]


def reconcile_workspace_venvs(omni_home: str) -> ModelReconcileOutcome:
    """Run the workspace reconciler once against ``omni_home``.

    Never raises: every failure mode is reported as an outcome, because the
    caller (a guard on the CLI hot path) has to turn it into a refusal message
    rather than a traceback.
    """
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
        )
    except OSError as exc:
        return ModelReconcileOutcome(ok=False, command=command, detail=str(exc))

    if result.returncode == 0:
        logger.info("omnimarket drift self-healed via %s", command)
        return ModelReconcileOutcome(ok=True, command=command, detail="")

    tail = (result.stdout + result.stderr).strip().splitlines()
    return ModelReconcileOutcome(
        ok=False,
        command=command,
        detail=(tail[-1] if tail else f"reconciler exited {result.returncode}"),
    )


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
