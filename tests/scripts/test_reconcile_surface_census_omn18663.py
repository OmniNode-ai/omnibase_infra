# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconciler's verdict covers every venv that serves `onex` (OMN-18663).

Two surfaces were invisible to that verdict, in opposite ways. Both were
measured live on 2026-09-18 at ~15:45Z, while a repair tick was reporting
success.

**The dispatch venv's lock leg asked a question the repair never answers.** The
repair syncs with ``--python "$base_python"`` -- the brew interpreter CLAUDE.md
rule 11 requires for the macOS Local Network grant. The check omitted it, so uv
resolved the interpreter its own way, from ``.python-version`` (3.12 in this
project) while the venv is correctly on 3.13, and answered "would replace this
environment". Single variable, everything else identical, on the real venv::

    uv sync --frozen --check --inexact                      -> exit 1
    uv sync --frozen --check --inexact --python <brew 3.13>  -> exit 0

A permanent DRIFT verdict no repair could clear.

**The plugin CLI venv was omitted entirely, and silently.** ``hook_venv_projects``
requires a ``uv.lock``; that venv is built from a requirements.txt by
``repair-plugin-venv.sh`` and has none, so the loop skipped it without a word.
It is the venv rule 11's table calls the CLI venv, and it serves ``onex`` for
every lane reaching the CLI through the plugin. It sat one omnimarket bump
behind the clone while the reconciler reported zero failures.

It is still not owned here -- one directory, one owner -- so what these tests
pin is that it is NAMED, that its lock layer is reported unowned, and that its
provider layer is read back and is verdict-bearing in ``--check``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import (
    _SCRIPT,
    _git,
    _Workspace,
)

#: The version the fixture clone declares, and therefore the version the
#: readback expects a venv installed from its HEAD to carry.
_CLONE_VERSION = "0.4.121"

pytestmark = pytest.mark.unit


def _cli_venv(root: Path, *, commit: str | None, version: str | None) -> Path:
    """A plugin-CLI-venv-shaped directory: a venv, deliberately no uv.lock.

    Its python answers the readback probe (``-c``) with canned facts and execs
    the real interpreter for anything else, because the readback is a real
    program that has to really run.
    """
    venv = root / ".claude" / "plugins" / "data" / "onex-omninode-tools" / ".venv"
    (venv / "bin").mkdir(parents=True)
    facts = json.dumps({"omnimarket": {"version": version, "commit": commit}})
    python = venv / "bin" / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-c" ]]; then\n'
        f"  printf '%s' {json.dumps(facts)}\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == "-" ]]; then\n'
        "  cat >/dev/null 2>&1 || true\n"
        f"  printf '%s\\n' {json.dumps(commit or '')}\n"
        "  exit 0\n"
        "fi\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    return venv


def _teach_dispatch_python_to_run_programs(ws: _Workspace) -> None:
    """Let the harness's dispatch interpreter execute a real script.

    The census runs ``venv_readback.py`` on the DISPATCH interpreter rather than
    on the venv it is inspecting, so in this fixture that shim has to stop being
    a pure echo. It keeps answering the here-doc commit probe exactly as before.
    """
    python = ws.dispatch_venv / "bin" / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-" ]]; then\n'
        "  cat >/dev/null 2>&1 || true\n"
        f"  printf '%s\\n' '{ws.market_head}'\n"
        "  exit 0\n"
        "fi\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)


def _give_the_clone_a_pyproject(ws: _Workspace) -> None:
    """The readback reads [project].version out of the ref being checked.

    The shared harness's fake omnimarket clone is a bare text repo, so without
    this the readback cannot resolve an expected version and fails closed --
    correctly, but for a fixture reason rather than the one under test.
    """
    (ws.omnimarket / "pyproject.toml").write_text(
        f'[project]\nname = "omnimarket"\nversion = "{_CLONE_VERSION}"\n',
        encoding="utf-8",
    )
    _git("add", "pyproject.toml", cwd=ws.omnimarket)
    _git("commit", "--quiet", "-m", "pyproject", cwd=ws.omnimarket)
    ws.market_head = _git("rev-parse", "HEAD", cwd=ws.omnimarket)


def _copy_collaborators(ws: _Workspace) -> None:
    """Put the readback beside the script copy the harness runs."""
    repo_root = Path(__file__).resolve().parents[2]
    target = ws.infra / "scripts" / "venv_readback.py"
    target.write_text(
        (repo_root / "scripts" / "venv_readback.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


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
    # The census must not depend on this being exported: a cron or SessionStart
    # tick does not carry it, which is exactly how the surface stayed invisible.
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
    # After the clone moved: this rewrites the dispatch shim to echo the new
    # HEAD, so the dispatch venv is not itself drift in every test below.
    _teach_dispatch_python_to_run_programs(workspace)
    return workspace


# --------------------------------------------------------------------------- #
# The check must ask the question the repair answers
# --------------------------------------------------------------------------- #
def _dispatch_lines(ws: _Workspace) -> list[str]:
    """Every logged uv invocation that targeted the dispatch venv."""
    return [
        line
        for line in ws.uv_log.read_text(encoding="utf-8").splitlines()
        if f"env={ws.dispatch_venv}" in line
    ]


def test_the_dispatch_lock_check_names_the_interpreter_the_repair_pins(
    ws: _Workspace,
) -> None:
    """THE DEFECT. Without --python, uv answers about a different environment."""
    _run(ws, "--check")

    checks = [line for line in _dispatch_lines(ws) if "--check" in line]
    assert checks, "the dispatch venv's lock was never checked at all"
    for line in checks:
        assert "--python" in line, (
            "the check omits the interpreter the repair pins, so uv resolves it "
            "from .python-version and reports a replace that is not real"
        )
        assert str(ws.brew_python) in line


def test_the_check_and_the_repair_ask_uv_the_same_question(ws: _Workspace) -> None:
    """Flag for flag on the dispatch venv, so the two cannot disagree."""
    _run(ws, "--check")
    check_flags = {
        flag
        for line in _dispatch_lines(ws)
        if "--check" in line
        for flag in line.split()
        if flag.startswith("--")
    } - {"--check"}

    ws.uv_log.write_text("", encoding="utf-8")
    ws.set_installed_commit(None)
    _run(ws)
    repair_flags = {
        flag
        for line in _dispatch_lines(ws)
        if "--check" not in line and "sync" in line
        for flag in line.split()
        if flag.startswith("--")
    }

    assert check_flags <= repair_flags, (
        f"the check asks uv something the repair never answers: "
        f"{sorted(check_flags - repair_flags)}"
    )
    for shared in ("--inexact", "--python"):
        assert shared in check_flags and shared in repair_flags


# --------------------------------------------------------------------------- #
# The CLI venv is named, never silently omitted
# --------------------------------------------------------------------------- #
def test_check_names_the_unowned_cli_venv_and_its_owner(ws: _Workspace) -> None:
    _cli_venv(ws.root, commit=ws.market_head, version=_CLONE_VERSION)

    result = _run(ws, "--check", home=ws.root)

    combined = result.stdout + result.stderr
    assert "CLI venv (not owned here)" in combined
    assert "repair-plugin-venv.sh" in combined


def test_a_drifted_cli_venv_is_drift_not_in_sync(ws: _Workspace) -> None:
    """THE DEFECT: it sat a bump behind while the verdict reported no failures."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")

    result = _run(ws, "--check", home=ws.root)

    combined = result.stdout + result.stderr
    assert "provider   : DRIFT" in combined
    assert "verdict: DRIFT" in combined
    # The remedy is named, and it is the sanctioned one rather than this script.
    assert "check-omnimarket-venv-drift.sh" in combined


def test_an_in_sync_cli_venv_does_not_by_itself_cause_drift(ws: _Workspace) -> None:
    _cli_venv(ws.root, commit=ws.market_head, version=_CLONE_VERSION)

    result = _run(ws, "--check", home=ws.root)

    combined = result.stdout + result.stderr
    assert "provider   : in sync" in combined
    assert "verdict: IN_SYNC" in combined


def test_an_unreadable_cli_venv_fails_closed(ws: _Workspace) -> None:
    """A venv that cannot answer has proven nothing; it is not in sync."""
    venv = _cli_venv(ws.root, commit=ws.market_head, version=_CLONE_VERSION)
    broken = venv / "bin" / "python"
    broken.write_text("#!/usr/bin/env bash\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)

    result = _run(ws, "--check", home=ws.root)

    assert "verdict: DRIFT" in (result.stdout + result.stderr)


def test_the_census_does_not_need_claude_plugin_data_exported(
    ws: _Workspace,
) -> None:
    """A tick does not carry it, which is how the surface stayed invisible."""
    _cli_venv(ws.root, commit=ws.market_head, version=_CLONE_VERSION)

    result = _run(ws, "--check", home=ws.root)

    assert "onex-omninode-tools" in (result.stdout + result.stderr)


def test_a_candidate_carrying_a_lock_is_left_to_the_hook_venv_pass(
    ws: _Workspace,
) -> None:
    """One surface, one owner: a locked candidate is reconciled, not censused."""
    venv = _cli_venv(ws.root, commit=ws.market_head, version=_CLONE_VERSION)
    (venv.parent / "uv.lock").write_text("plugin-lock\n", encoding="utf-8")

    result = _run(ws, "--check", home=ws.root)

    assert "CLI venv (not owned here)" not in (result.stdout + result.stderr)


def test_the_repair_path_reports_the_cli_venv_too(ws: _Workspace) -> None:
    """Report-only there, on the same terms the clone leg is report-only."""
    _cli_venv(ws.root, commit="c" * 40, version="0.4.120")

    result = _run(ws, home=ws.root)

    combined = result.stdout + result.stderr
    assert "CLI venv (not owned here)" in combined
    assert "provider   : DRIFT" in combined
    # The exit code still answers for the surfaces this script WRITES.
    assert result.returncode == 0, combined


def test_no_cli_venv_present_is_silent_not_a_failure(ws: _Workspace) -> None:
    result = _run(ws, "--check", home=ws.root)

    combined = result.stdout + result.stderr
    assert "CLI venv (not owned here)" not in combined
    assert "verdict: IN_SYNC" in combined
