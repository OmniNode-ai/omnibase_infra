# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18815: a drifted plugin CLI venv is REPAIRED on a clone advance, not
merely reported, and the repair goes through the marker-writing path.

OMN-18663 made this venv visible: before it, the reconciler reported zero
failures while the CLI venv sat a bump behind and the in-process drift guard
refused every plugin-path delegation. Visible was the right first step and it
is not enough. Measured 2026-09-19, the venv spent about 545 of 740 elapsed
minutes drifted -- roughly 74% of the day -- across windows of 8h31m and 34m,
and BOTH were closed by a lane typing a repair command it read off this
script's own output. A remedy nobody runs is a remedy in name.

WHICH REPAIR, AND WHY IT IS NOT A DETAIL. Two scripts will make the installed
commit equal the clone head and they are not interchangeable:

* ``repair-plugin-venv.sh`` delegates to ``ensure-plugin-venv.sh``, whose
  ``uv sync --frozen`` rebuilds the venv from the lock AND writes the
  ``.built-from`` marker. It is the only writer of that marker.
* ``check-omnimarket-venv-drift.sh --repair`` installs the clone head through
  a targeted ``--no-deps`` install and touches the marker nowhere.

Taking the second converges the drift guard and leaves the skew gate red on a
stale marker -- the exact split this lane hit on 2026-09-19 and had to correct
by hand. It has also downgraded siblings before. So the path is pinned here,
positively and negatively, rather than left to whoever edits the script next.

THE OWNERSHIP BOUNDARY IS KEPT, NOT MOVED. This script still does not write
that venv; rule 11's table says ``repair-plugin-venv.sh`` owns it. What changes
is that the reconciler CALLS the owner instead of printing its name.
"""

from __future__ import annotations

import os
import subprocess
import sys
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
    """Stand in for ``repair-plugin-venv.sh`` and record every invocation.

    A recorder rather than the real script: the real one builds a venv from a
    lock over the network, which is neither available nor the thing under
    test. What IS under test is whether the reconciler reaches for the owner
    at all, and with what.
    """
    log = ws.root / "repair-plugin-venv.calls"
    script = ws.root / "omniclaude" / "scripts" / "repair-plugin-venv.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "argv=%s CLAUDE_PLUGIN_DATA=%s PATH=%s\\n" "$*" "${{CLAUDE_PLUGIN_DATA:-}}" "${{PATH}}" >> "{log}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return log


def _failing_repair(ws: _Workspace) -> Path:
    log = ws.root / "repair-plugin-venv.calls"
    script = ws.root / "omniclaude" / "scripts" / "repair-plugin-venv.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "argv=%s CLAUDE_PLUGIN_DATA=%s\\n" "$*" "${{CLAUDE_PLUGIN_DATA:-}}" >> "{log}"\n'
        'echo "could not reach the index" >&2\n'
        "exit 7\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return log


def _run(
    ws: _Workspace, *args: str, home: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the reconciler. No verb is REPAIR mode -- the script's default --
    and ``--check`` is the read-only one."""
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


# --------------------------------------------------------------------------- #
# AC1 — a drifted venv is repaired, not narrated
# --------------------------------------------------------------------------- #


def test_a_drifted_cli_venv_is_repaired_rather_than_only_reported(
    ws: _Workspace,
) -> None:
    """AC1. The defect is that the remedy was a sentence."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _repair_recorder(ws)

    _run(ws, home=ws.root)

    assert log.exists(), (
        "the reconciler found the venv drifted and never called the owner's "
        "repair -- which is the 2026-09-19 failure verbatim: a remedy printed "
        "and left for a human to notice"
    )


def test_the_repair_targets_the_drifted_interpreter(ws: _Workspace) -> None:
    """AC1. A repair that does not name the venv it found is a repair of
    whatever happened to be default, which on this host is a different venv."""
    venv = _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _repair_recorder(ws)

    _run(ws, home=ws.root)

    called = log.read_text(encoding="utf-8")
    # The owner script resolves its target from CLAUDE_PLUGIN_DATA, not from
    # argv, so that is where the targeting has to be asserted. A repair that
    # left it unset would rebuild whatever the default resolves to, which on a
    # real host is a DIFFERENT venv from the one just found drifted.
    assert f"CLAUDE_PLUGIN_DATA={venv.parent}" in called, (
        f"the repair call {called!r} does not target the drifted venv at {venv}"
    )
    assert str(venv / "bin") in called, (
        "the repair was not given the venv's bin on PATH; repair-plugin-venv.sh "
        "refuses to run without it, and a detached tick has no interactive PATH"
    )


# --------------------------------------------------------------------------- #
# AC2 — the MARKER-WRITING path, and not the other one
# --------------------------------------------------------------------------- #


def test_the_repair_goes_through_the_marker_writing_path(ws: _Workspace) -> None:
    """AC2, positively. Only ``ensure-plugin-venv.sh``, reached through
    ``repair-plugin-venv.sh``, writes ``.built-from``."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _repair_recorder(ws)

    result = _run(ws, home=ws.root)

    # INVOKED, not merely mentioned. The script already prints this name in
    # its remedy text, so an output-only assertion would pass today and would
    # keep passing if the call were later removed.
    assert log.exists(), "repair-plugin-venv.sh was named but never run"
    combined = result.stdout + result.stderr
    assert "repair-plugin-venv.sh" in combined


def test_the_targeted_repair_is_never_invoked_for_this_venv(
    ws: _Workspace,
) -> None:
    """AC2, negatively, and this is the half that matters.

    ``check-omnimarket-venv-drift.sh --repair`` converges the commit and
    leaves the marker stale, so the skew gate stays red while the drift guard
    goes green -- two gates disagreeing about one venv, which is the condition
    OMN-18753 exists to end rather than to reproduce somewhere else.
    """
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    _repair_recorder(ws)
    drift_log = ws.root / "check-omnimarket-venv-drift.calls"
    stub = ws.infra / "scripts" / "check-omnimarket-venv-drift.sh"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{drift_log}"\nexit 0\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    _run(ws, home=ws.root)

    assert not drift_log.exists(), (
        "the reconciler invoked check-omnimarket-venv-drift.sh, which installs "
        "the clone head WITHOUT writing the .built-from marker; that converges "
        "one gate and leaves the skew gate red on a stale marker"
    )


# --------------------------------------------------------------------------- #
# AC3 / AC5 — do nothing when there is nothing to do, and survive a failure
# --------------------------------------------------------------------------- #


def test_an_in_sync_cli_venv_is_not_rebuilt(ws: _Workspace) -> None:
    """AC3. The tick runs on every tool call behind a throttle; rebuilding a
    converged venv each time would spend minutes to change nothing."""
    _cli_venv(ws.root, commit=ws.market_head, version=_CLONE_VERSION)
    log = _repair_recorder(ws)

    _run(ws, home=ws.root)

    assert not log.exists(), (
        "the reconciler rebuilt a venv that already carried the clone head"
    )


def test_check_mode_reports_and_never_repairs(ws: _Workspace) -> None:
    """AC3. ``--check`` is the read-only verb; a check that mutates is not one."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _repair_recorder(ws)

    result = _run(ws, "--check", home=ws.root)

    assert not log.exists(), "--check repaired the venv; it must only report"
    combined = result.stdout + result.stderr
    assert "provider   : DRIFT" in combined


def test_a_failed_repair_is_reported_and_does_not_abort_the_run(
    ws: _Workspace,
) -> None:
    """AC5. The tick runs detached and exits 0 on every path. A repair that
    cannot reach the index must leave a line naming the venv, not take the
    whole reconcile down with it."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _failing_repair(ws)

    result = _run(ws, home=ws.root)

    assert log.exists(), "the failing repair was never reached"
    combined = result.stdout + result.stderr
    assert "repair-plugin-venv.sh" in combined
    assert "could not reach the index" in combined, (
        "the repair's own failure detail was swallowed; a line saying only "
        "that a repair failed sends the next reader back to reproduce it"
    )
    assert "FAILED" in combined.upper(), (
        "a failed repair left no failure line; a silent failure reads exactly "
        "like a repair that was never needed"
    )


# --------------------------------------------------------------------------- #
# AC6 — the hook interpreter is not in scope
# --------------------------------------------------------------------------- #


def test_the_hook_interpreter_venv_is_not_rebuilt_by_this_path(
    ws: _Workspace,
) -> None:
    """AC6. That venv is ``uv sync`` in its own clone. Two owners for one venv
    is the defect OMN-18752 spent a day unpicking."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")
    log = _repair_recorder(ws)

    _run(ws, home=ws.root)

    if log.exists():
        called = log.read_text(encoding="utf-8")
        assert "omniclaude/.venv" not in called, (
            "the CLI venv repair was pointed at the hook interpreter venv"
        )


def test_the_script_still_parses(ws: _Workspace) -> None:
    """A positive control for every assertion above.

    Each one reads the script's output or its side effects, and a script with
    a syntax error produces neither -- so an empty log would read as 'did not
    repair' rather than 'did not run'.
    """
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_python_used_by_this_module_is_the_test_interpreter() -> None:
    """Guards the fixture, not the subject: the census shim execs
    ``sys.executable`` and a mismatch there fails these tests for a reason
    that has nothing to do with the reconciler."""
    assert Path(sys.executable).exists()
