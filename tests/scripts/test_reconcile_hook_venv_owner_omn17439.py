# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The hook-venv ownership guard must actually guard (OMN-17439).

WHAT WENT WRONG

OMN-17366 moved the ownership mechanics into ``scripts/reconcile_privilege_lib.sh``,
where they are exported with an ``rp_`` prefix. One call site was not updated::

    # scripts/reconcile-workspace-venvs.sh:615
    hook_owner="$(surface_owner "$project/.venv" || true)"

Observed live on `.201` running the reconcile exactly as the `:19` cron does::

    [reconcile] cli venv: reconciled
    .../reconcile-workspace-venvs.sh: line 615: surface_owner: command not found
    [reconcile] hook venv: already in sync (/data/omninode/omniclaude/.venv)
    [reconcile-host] VERDICT: IN_SYNC — every surface proven at target.

The run exits 0 and reports IN_SYNC.

WHY IT IS WORSE THAN A STRAY ERROR LINE

The call is wrapped in ``|| true``, so the failure is swallowed and
``hook_owner`` becomes the empty string. The guard immediately below then skips
itself on its own emptiness check::

    if [[ -n "$hook_owner" && "$hook_owner" != "$SURFACE_OWNER" ]]; then

So a hook venv owned by a different user stops being refused and is written with
the ``RUN_AS`` prefix planned from the CLI venv's owner -- exactly the hazard
that block's own comment says it exists to prevent. A guard that reports success
while guarding nothing is the whole subject of OMN-17305, reproduced inside its
own fix.

WHY NOTHING CAUGHT IT

Bash resolves function names at call time. ``bash -n`` parses this file happily,
shellcheck has no undefined-function check, and no existing test exercises a
hook venv with a foreign owner. The only place it surfaces is a live host, on
stderr, in the middle of a run that reports success -- which is why the fix
below ships with a gate rather than only a corrected spelling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import _make_uv_shim, _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"
_LIB = _REPO_ROOT / "scripts" / "reconcile_privilege_lib.sh"
_GATE = _REPO_ROOT / "scripts" / "check_reconciler_privilege.py"

_EXIT_INDETERMINATE = 3
_FOREIGN = "someone-else"


@pytest.fixture
def workspace(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "omni_home")


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "fakehome"
    home.mkdir()
    return home


def _stat_shim_foreign_for(shims: Path, foreign_path: Path) -> None:
    """A ``stat`` that reports a foreign owner for ONE path only.

    Selective on purpose. A shim that lied about every path would make the CLI
    venv foreign too, and the run would refuse in ``plan_privileges`` long
    before it reached the hook-venv branch under test -- passing for a reason
    that has nothing to do with this defect.
    """
    real = shutil.which("stat") or "/usr/bin/stat"
    shim = shims / "stat"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do\n'
        f'  case "$a" in {foreign_path}*) echo {_FOREIGN}; exit 0 ;; esac\n'
        "done\n"
        f'exec {real} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)


def _run(
    workspace: _Workspace, home: Path, shims: Path
) -> subprocess.CompletedProcess[str]:
    env = workspace.env()
    env["HOME"] = str(home)
    env["PATH"] = f"{shims}:/usr/bin:/bin"
    env["ONEX_RECONCILE_UV_BIN"] = str(home / ".local" / "bin" / "uv")
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


# --------------------------------------------------------------------------- #
# AC1 -- the branch that was silently dead
# --------------------------------------------------------------------------- #
def test_a_hook_venv_owned_by_someone_else_is_refused(
    workspace: _Workspace, fake_home: Path, tmp_path: Path
) -> None:
    """RED against the ``surface_owner`` spelling.

    Before the fix the lookup failed, ``|| true`` swallowed it, ``hook_owner``
    was empty, and the guard's own ``-n`` test skipped the refusal -- so this
    reconciled a venv belonging to another user and reported success.
    """
    _make_uv_shim(fake_home / ".local" / "bin")
    shims = tmp_path / "shims"
    shims.mkdir()
    _stat_shim_foreign_for(shims, workspace.omniclaude / ".venv")

    result = _run(workspace, fake_home, shims)

    assert result.returncode == _EXIT_INDETERMINATE, result.stdout + result.stderr
    assert _FOREIGN in result.stdout, (
        "the hook venv's foreign owner was never resolved, so the guard below it "
        "skipped itself -- OMN-17439"
    )
    assert "is owned by" in result.stdout


def test_no_command_not_found_escapes_a_run(
    workspace: _Workspace, fake_home: Path, tmp_path: Path
) -> None:
    """AC3, asserted mechanically rather than by reading a cron log.

    ``command not found`` on stderr inside a run that exits 0 is the exact
    signature this ticket is about: bash resolves names at call time, so a
    renamed helper stays invisible until the branch executes on a host.
    """
    _make_uv_shim(fake_home / ".local" / "bin")
    shims = tmp_path / "shims"
    shims.mkdir()
    _stat_shim_foreign_for(shims, workspace.omniclaude / ".venv")

    result = _run(workspace, fake_home, shims)

    combined = result.stdout + result.stderr
    assert "command not found" not in combined, combined


# --------------------------------------------------------------------------- #
# AC2 -- the gate, stated as the general rule
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(_GATE), "--repo-root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def _fixture_repo(tmp_path: Path, venv_body: str) -> Path:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(_LIB, scripts / _LIB.name)
    (scripts / "reconcile-workspace-venvs.sh").write_text(venv_body, encoding="utf-8")
    return root


_PREAMBLE = (
    "#!/usr/bin/env bash\n"
    'ONEX_RECONCILE_UV_BIN="${ONEX_RECONCILE_UV_BIN:-}"\n'
    "resolve_uv() { :; }\n"
    'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
    "as_owner true\n"
)


def test_the_gate_rejects_an_unprefixed_library_call(tmp_path: Path) -> None:
    """The exact line that shipped, rejected.

    Stated as the general rule -- call the library's functions by their exported
    names -- rather than as a special case for ``surface_owner``, because the
    next rename will otherwise reproduce this with a different name.
    """
    root = _fixture_repo(
        tmp_path, _PREAMBLE + 'hook_owner="$(surface_owner "$p/.venv" || true)"\n'
    )

    result = _gate(root)

    assert result.returncode == 1
    assert "surface_owner" in result.stderr
    assert "rp_surface_owner" in result.stderr


def test_the_gate_accepts_the_prefixed_call(tmp_path: Path) -> None:
    """The positive control: the corrected spelling passes."""
    root = _fixture_repo(
        tmp_path, _PREAMBLE + 'hook_owner="$(rp_surface_owner "$p/.venv" || true)"\n'
    )

    result = _gate(root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_allows_a_locally_defined_wrapper_of_the_same_name(
    tmp_path: Path,
) -> None:
    """A name the script defines itself is its own, not a leftover.

    ``reconcile-workspace-venvs.sh`` legitimately defines ``plan_privileges`` as
    a thin policy wrapper around the library's ``rp_plan_privileges``. Flagging
    that would force the wrapper to be renamed for the gate's benefit, which is
    how a gate starts costing more than it catches.
    """
    root = _fixture_repo(
        tmp_path,
        _PREAMBLE
        + 'plan_privileges() { rp_plan_privileges "$1"; }\nplan_privileges /x\n',
    )

    result = _gate(root)

    assert result.returncode == 0, result.stdout + result.stderr
