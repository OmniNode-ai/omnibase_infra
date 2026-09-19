# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18815 follow-up: the auto-repair only runs when it CAN converge, and
otherwise names the surface that is actually blocking.

Found by the change's own first live run, one minute after it merged. The
receipts log recorded:

    [reconcile]   provider   : DRIFT -- does not carry omnimarket 2219992325c7
    [reconcile]   repair     : running .../repair-plugin-venv.sh (marker-writing path)
    [reconcile]   repair     : ran and exited 0, but the venv STILL does not carry

The readback did its job -- it refused to call that repaired -- but the repair
could never have succeeded. ``repair-plugin-venv.sh`` reaches
``ensure-plugin-venv.sh``, whose ``uv sync --frozen`` installs the rev the
omniclaude LOCK names. The drift verdict compares against the CLONE head. At
that moment the lock named ``133dd018`` and the clone head was ``2219992325c7``,
four commits apart, so no number of rebuilds could have closed it.

Left alone that is a full venv rebuild on every tick -- at most once per
throttle interval, forever, always failing, on a surface other lanes are
using. The rebuild is not free and the failure is not informative after the
first one.

So the precondition is now checked before the rebuild is spent: the repair
runs only when the lock carries the clone head, which is exactly when
rebuilding from the lock can reach it. When it does not, the blocker is the
LOCK, and the line says so and names the workflow that owns advancing it
rather than implying the venv is at fault.

This is not a weakening of the OMN-18815 behaviour. The repair still runs in
the case it was built for, and the drift is still reported in both cases --
what changes is that an impossible repair is not attempted and is not blamed
on the wrong surface.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_surface_census_omn18663 import (
    _CLONE_VERSION,
    _cli_venv,
    _copy_collaborators,
    _give_the_clone_a_pyproject,
    _teach_dispatch_python_to_run_programs,
)
from tests.scripts.test_reconcile_workspace_venvs import _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"


def _repair_recorder(ws: _Workspace) -> Path:
    log = ws.root / "repair-plugin-venv.calls"
    script = ws.root / "omniclaude" / "scripts" / "repair-plugin-venv.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f'#!/usr/bin/env bash\nprintf "ran\\n" >> "{log}"\nexit 0\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return log


def _omniclaude_lock(ws: _Workspace, rev: str) -> Path:
    """Write an omniclaude lock naming ``rev`` for omnimarket.

    The real file is a uv lock; only the rev fragment is read here, and it is
    read the same way the guard and the dispatch gate read it -- out of the
    git source URL -- so a fixture that matched a different spelling would
    prove nothing about the live file.
    """
    lock = ws.root / "omniclaude" / "uv.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        "[[package]]\n"
        'name = "omnimarket"\n'
        f'source = {{ git = "https://github.com/OmniNode-ai/omnimarket.git?rev={rev}#{rev}" }}\n',
        encoding="utf-8",
    )
    return lock


def _run(
    ws: _Workspace, *args: str, home: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{ws.bin_dir}:{env['PATH']}"
    env["OMNI_HOME"] = str(ws.root)
    env["UV_SHIM_LOG"] = str(ws.uv_log)
    env["ORDER_LOG"] = str(ws.order_log)
    env["ONEX_RECONCILE_INSTALL_SCRIPT"] = str(ws.install_script)
    env["ONEX_DISPATCH_BASE_PYTHON"] = str(ws.brew_python)
    env.pop("CLAUDE_PLUGIN_DATA", None)
    if home is not None:
        env["HOME"] = str(home)
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    workspace = _Workspace(tmp_path)
    _give_the_clone_a_pyproject(workspace)
    _copy_collaborators(workspace)
    _teach_dispatch_python_to_run_programs(workspace)
    return workspace


def test_no_rebuild_is_spent_when_the_lock_cannot_reach_the_clone_head(
    ws: _Workspace,
) -> None:
    """The live failure, as a test.

    Lock behind the clone head, venv behind both. Rebuilding from the lock
    lands on the lock's rev, which is not the clone head, so the repair cannot
    close the drift and must not be attempted.
    """
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    _omniclaude_lock(ws, "d" * 40)
    log = _repair_recorder(ws)

    result = _run(ws, home=ws.root)

    assert not log.exists(), (
        "a rebuild was spent that could not have converged: the lock names a "
        "rev other than the clone head, so uv sync --frozen lands somewhere "
        "that is still drift"
    )
    combined = result.stdout + result.stderr
    assert "provider   : DRIFT" in combined, "the drift must still be reported"


def test_the_blocking_surface_is_named_and_it_is_the_lock(ws: _Workspace) -> None:
    """Blaming the venv sends the next reader to rebuild it again."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    _omniclaude_lock(ws, "d" * 40)
    _repair_recorder(ws)

    result = _run(ws, home=ws.root)

    combined = result.stdout + result.stderr
    assert "sibling-lock-refresh" in combined, (
        "the line does not name the workflow that owns advancing the lock, so "
        "the reader is left with a drifted venv and no owner"
    )


def test_the_repair_still_runs_when_the_lock_carries_the_clone_head(
    ws: _Workspace,
) -> None:
    """The case OMN-18815 was built for is untouched.

    Without this the fix would read as 'stop repairing', which is the opposite
    of the change it is amending.
    """
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    _omniclaude_lock(ws, ws.market_head)
    log = _repair_recorder(ws)

    _run(ws, home=ws.root)

    assert log.exists(), (
        "the lock carries the clone head, so a rebuild from it reaches the "
        "clone head and must still be attempted"
    )


def test_an_unreadable_lock_does_not_silently_spend_a_rebuild(
    ws: _Workspace,
) -> None:
    """No omniclaude lock at all is the shape a fresh or partial checkout has.

    Fails toward NOT spending the rebuild: an unknown precondition is not a
    satisfied one, and the drift is still reported either way, so nothing is
    hidden by declining.
    """
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _repair_recorder(ws)

    result = _run(ws, home=ws.root)

    assert not log.exists(), (
        "a rebuild was spent with no lock to prove it could converge"
    )
    assert "provider   : DRIFT" in (result.stdout + result.stderr)


def test_the_script_still_parses() -> None:
    """Positive control: every assertion above reads output or side effects,
    and a script with a syntax error produces neither."""
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
