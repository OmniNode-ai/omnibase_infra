# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How ``reconcile-workspace-venvs.sh`` finds ``uv``, and who it writes as (OMN-17335).

THE INCIDENT THESE PIN DOWN

The first live ``.201`` run of the host reconciler refused with::

    [reconcile] INDETERMINATE: `uv` is not on PATH; every sync below is a uv operation.

and the ticket filed against it concluded that ``uv`` was not installed on that
machine at all, citing::

    $ command -v uv                      # as the operator
    (nothing)
    $ sudo bash -lc 'command -v uv'      # as root
    (nothing)

Both probes were invalid. ``uv 0.11.5`` was installed the whole time at
``~/.local/bin/uv``; a **non-interactive** shell never sources the profile that
puts that directory on PATH, and the cron.d unit pins an explicit minimal PATH
that cannot reach a user-local install by construction. The venv's own
``pyvenv.cfg`` even recorded ``uv = 0.11.5`` -- it could not have been built by a
tool that was never installed.

So an empty result was read as evidence of absence, inside the one family of
scripts written to stop precisely that. These tests hold the fix: resolution is
an ordered list of candidates, and a refusal names every candidate it tried.

The second half is the defect that only appeared while fixing the first. The
cron unit runs as **root**; the venv and every file in it are owned by the
operator. Putting ``uv`` on root's PATH would have "fixed" the refusal by having
root write root-owned files into a user-owned venv, after which the owner's own
reconcile fails on permissions -- a loud, correct failure traded for a quiet,
latent one. So package operations run as the surface owner or do not run.

WHY THESE ARE RED AGAINST THE PRE-FIX SCRIPT

Every test below fails on the shipped-before version, which resolved uv as
``UV_BIN="$(command -v uv)"`` and exited INDETERMINATE on an empty result:

* the override and owner-home cases got exit 3 instead of reconciling;
* the refusal case got a one-line message naming only PATH, so the assertion
  that every candidate is named finds nothing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import _make_uv_shim, _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"

_EXIT_OK = 0
_EXIT_INDETERMINATE = 3


def _uv_free_env(workspace: _Workspace, home: Path) -> dict[str, str]:
    """The workspace environment with every route to a real ``uv`` removed.

    PATH is rebuilt from scratch rather than filtered: this developer machine
    has ``uv`` at ``~/.local/bin/uv`` -- the same shape as the ``.201`` host --
    so inheriting the ambient PATH would let the test find a real uv and pass
    for the wrong reason. ``/usr/bin`` and ``/bin`` are kept because the script
    genuinely needs ``git``, ``stat``, ``id`` and ``dirname``.

    HOME is redirected too, since the owner-home candidate resolves through it
    for the current user.
    """
    env = workspace.env()
    env["PATH"] = "/usr/bin:/bin"
    env["HOME"] = str(home)
    env.pop("ONEX_RECONCILE_UV_BIN", None)
    return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "omni_home")


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "fakehome"
    home.mkdir()
    return home


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #
def test_uv_absent_from_path_is_resolved_from_the_explicit_override(
    workspace: _Workspace, fake_home: Path, tmp_path: Path
) -> None:
    """PATH holds no uv, ONEX_RECONCILE_UV_BIN does -- and the reconcile runs.

    This is the operator's escape hatch on a host whose PATH cannot be changed.
    It moves WHICH uv runs; it never moves whether the sync has to succeed.
    """
    override_bin = tmp_path / "override"
    _make_uv_shim(override_bin)

    env = _uv_free_env(workspace, fake_home)
    env["ONEX_RECONCILE_UV_BIN"] = str(override_bin / "uv")

    result = _run(env, "--verbose")

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert "is not on PATH" not in result.stdout
    # Proven by the shim's own log, not by the exit code: the override was the
    # binary that actually ran.
    assert workspace.uv_calls(), "the override uv was never invoked"


def test_uv_under_the_surface_owners_home_is_found_when_path_cannot_see_it(
    workspace: _Workspace, fake_home: Path
) -> None:
    """The exact `.201` shape: uv installed at ~/.local/bin, invisible to cron PATH.

    Nothing is on PATH and no override is set. The only uv on the machine lives
    under the home directory of the user that owns the venv -- which is where
    the standalone uv installer puts it, and which is what the original probe
    failed to look at before declaring the tool missing.
    """
    local_bin = fake_home / ".local" / "bin"
    _make_uv_shim(local_bin)

    env = _uv_free_env(workspace, fake_home)

    result = _run(env, "--verbose")

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert workspace.uv_calls(), "the owner-home uv was never invoked"
    assert str(local_bin / "uv") in result.stdout


def test_no_uv_anywhere_still_refuses_and_names_every_candidate_it_tried(
    workspace: _Workspace, fake_home: Path
) -> None:
    """Fail-closed is preserved -- and the refusal is now actionable.

    The contract that a reconcile which cannot run must NOT report success is
    unchanged. What changes is that the message can no longer be misread as
    "uv is not installed": it names each place that was searched, so the next
    reader can see that PATH was only one of them.
    """
    env = _uv_free_env(workspace, fake_home)

    result = _run(env)

    assert result.returncode == _EXIT_INDETERMINATE
    assert "no usable `uv`" in result.stdout
    assert "Searched, in order:" in result.stdout
    # Each candidate source appears by name.
    assert "ONEX_RECONCILE_UV_BIN" in result.stdout
    assert "PATH=" in result.stdout
    assert str(fake_home / ".local" / "bin" / "uv") in result.stdout
    # And it says the thing the original ticket got wrong.
    assert "'not on PATH' is NOT 'not installed'" in result.stdout.replace('"', "'")


def test_check_mode_is_fail_closed_on_a_missing_uv_too(
    workspace: _Workspace, fake_home: Path
) -> None:
    """``--check`` must not report IN_SYNC when it could not look.

    ``--check`` mutates nothing, so it is tempting to let it degrade to a
    verdict. It must not: an unanswerable question is INDETERMINATE, never a
    pass. That is the OMN-13418 health-gate rule applied to host state, and it
    is what keeps the floor marker from being stamped on an unproven surface.
    """
    env = _uv_free_env(workspace, fake_home)

    result = _run(env, "--check")

    assert result.returncode == _EXIT_INDETERMINATE
    assert "IN_SYNC" not in result.stdout


def test_the_override_wins_over_a_uv_that_is_on_path(
    workspace: _Workspace, fake_home: Path, tmp_path: Path
) -> None:
    """Order is a contract, not an accident.

    The override is first so an operator can redirect a host whose PATH holds a
    uv they do not want used -- a stale one, or one from another project's venv.
    Asserted by giving the two shims different log files and reading back which
    one recorded the call.
    """
    path_bin = tmp_path / "onpath"
    _make_uv_shim(path_bin)
    override_bin = tmp_path / "override"
    _make_uv_shim(override_bin)

    env = _uv_free_env(workspace, fake_home)
    env["PATH"] = f"{path_bin}:/usr/bin:/bin"
    env["ONEX_RECONCILE_UV_BIN"] = str(override_bin / "uv")

    result = _run(env, "--verbose")

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert f"uv resolved to {override_bin / 'uv'}" in result.stdout


# --------------------------------------------------------------------------- #
# Writing as the surface owner
# --------------------------------------------------------------------------- #
def test_package_operations_run_directly_when_this_process_owns_the_surface(
    workspace: _Workspace, fake_home: Path
) -> None:
    """The developer-machine case: owner == current user, so no privilege drop.

    The guard must be invisible when it has nothing to do. A reconciler that
    started shelling through ``runuser`` on every laptop would be a regression,
    and the empty-array expansion it relies on is exactly the kind of bash
    detail that breaks under ``set -u`` without a test noticing.
    """
    local_bin = fake_home / ".local" / "bin"
    _make_uv_shim(local_bin)
    env = _uv_free_env(workspace, fake_home)

    result = _run(env, "--verbose")

    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert "writing as" not in result.stdout, (
        "no privilege drop should be announced when the process already owns "
        "the surface"
    )
    assert workspace.uv_calls()


def test_an_unwritable_surface_owner_is_refused_rather_than_written_as_the_wrong_user(
    workspace: _Workspace, fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner differs, and this process cannot become them -> INDETERMINATE.

    Simulated by putting a ``stat`` shim ahead of the real one that reports a
    different owner, which is the only way to model the root-cron-vs-operator
    split without actually being root. The assertion is that the script refuses
    and says whose surface it is -- never that it writes anyway.
    """
    local_bin = fake_home / ".local" / "bin"
    _make_uv_shim(local_bin)

    shim_bin = fake_home / "statshim"
    shim_bin.mkdir(parents=True)
    stat_shim = shim_bin / "stat"
    stat_shim.write_text(
        "#!/usr/bin/env bash\n"
        "# Report a foreign owner for any query, real behaviour otherwise.\n"
        'if [[ "$1" == "-c" && "$2" == "%U" ]]; then echo someone-else; exit 0; fi\n'
        'if [[ "$1" == "-f" && "$2" == "%Su" ]]; then echo someone-else; exit 0; fi\n'
        'exec /usr/bin/stat "$@"\n',
        encoding="utf-8",
    )
    stat_shim.chmod(0o755)

    env = _uv_free_env(workspace, fake_home)
    env["PATH"] = f"{shim_bin}:/usr/bin:/bin"

    result = _run(env)

    assert result.returncode == _EXIT_INDETERMINATE
    assert "someone-else" in result.stdout
    assert "cannot become that user" in result.stdout
    # The whole point: it did not proceed to write.
    assert not workspace.uv_calls(), "the reconciler wrote to a surface it does not own"


def test_check_mode_still_works_on_a_surface_this_process_does_not_own(
    workspace: _Workspace, fake_home: Path
) -> None:
    """A read-only probe must not be blocked by an ownership mismatch.

    ``--check`` writes nothing, so refusing it would only teach people to stop
    running the read-only probe -- and the SessionStart status line depends on
    it. The ownership rule guards writes, not reads.

    ``uv`` is supplied through the override rather than through the owner's home
    on purpose: with a foreign owner the home-directory candidate is
    unresolvable by construction, so leaving uv to be discovered there would
    make this test fail on uv resolution and never reach the ownership branch it
    exists to cover.
    """
    local_bin = fake_home / ".local" / "bin"
    _make_uv_shim(local_bin)

    shim_bin = fake_home / "statshim"
    shim_bin.mkdir(parents=True)
    stat_shim = shim_bin / "stat"
    stat_shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" && "$2" == "%U" ]]; then echo someone-else; exit 0; fi\n'
        'if [[ "$1" == "-f" && "$2" == "%Su" ]]; then echo someone-else; exit 0; fi\n'
        'exec /usr/bin/stat "$@"\n',
        encoding="utf-8",
    )
    stat_shim.chmod(0o755)

    env = _uv_free_env(workspace, fake_home)
    env["PATH"] = f"{shim_bin}:/usr/bin:/bin"
    env["ONEX_RECONCILE_UV_BIN"] = str(local_bin / "uv")

    result = _run(env, "--check")

    assert result.returncode != _EXIT_INDETERMINATE, result.stdout + result.stderr
    assert "cannot become that user" not in result.stdout


# --------------------------------------------------------------------------- #
# The gate itself (CLAUDE.md rule 5: it ships wired, and it can fail)
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(_REPO_ROOT / "scripts" / "check_reconciler_privilege.py"),
            "--repo-root",
            str(repo_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_shipped_reconciler_that_resolved_uv_from_path_only_is_rejected(
    tmp_path: Path,
) -> None:
    """Incident replay (OMN-15547 registry case ``omn17335-uv-resolved-from-path-only``).

    The fixture is not written for this test -- it is the verbatim reconciler as
    it shipped on dev at ``1b66d58f``, the bytes the ``.201`` maintenance cron
    actually executed every hour. Line 210 is the entire defect::

        UV_BIN="$(command -v uv 2>/dev/null || true)"

    with an INDETERMINATE exit on an empty result. The cron unit pins a minimal
    PATH that cannot reach a user-local install, so this refused hourly on a host
    that had ``uv 0.11.5`` at ``~/.local/bin/uv`` the entire time.

    A synthetic three-line reproduction would prove the regex works. Driving the
    real gate over the real artifact proves it would have caught the thing that
    shipped -- which is the only claim worth making, since this file was
    reviewed, merged and scheduled with nothing objecting.
    """
    captured = (
        _REPO_ROOT
        / "tests"
        / "fixtures"
        / "omn17335"
        / "reconcile-workspace-venvs.path-only-uv.sh.captured"
    )
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_bytes(
        captured.read_bytes()
    )

    result = _gate(scratch)

    assert result.returncode == 1, result.stdout + result.stderr
    # Both halves of the incident are named, not just one.
    assert "no resolve_uv() function" in result.stderr
    assert "ONEX_RECONCILE_UV_BIN" in result.stderr
    assert "without as_owner" in result.stderr


def test_the_gate_rejects_a_uv_sync_that_skips_the_owner_helper(tmp_path: Path) -> None:
    """Mutation proof: delete one ``as_owner`` and the build must go red.

    A gate that cannot be made to fail is decoration. This mutates the real
    script rather than a toy fixture, so the assertion tracks the file that
    actually ships.
    """
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    source = (_REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh").read_text(
        encoding="utf-8"
    )
    mutated = source.replace(
        'as_owner env -u PYTHONPATH "$UV_BIN" sync --frozen --inexact',
        'env -u PYTHONPATH "$UV_BIN" sync --frozen --inexact',
    )
    assert mutated != source, "the mutation target moved; update this test"
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_text(
        mutated, encoding="utf-8"
    )

    result = _gate(scratch)

    assert result.returncode == 1
    assert "without as_owner" in result.stderr


def test_the_gate_rejects_path_only_uv_resolution(tmp_path: Path) -> None:
    """The pre-fix resolution must not be re-introducible.

    Reproduces the shipped-before code exactly -- ``command -v uv`` and nothing
    else -- and asserts the gate names both missing properties.
    """
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_text(
        "#!/usr/bin/env bash\n"
        'UV_BIN="$(command -v uv 2>/dev/null || true)"\n'
        'if [[ -z "$UV_BIN" ]]; then exit 3; fi\n',
        encoding="utf-8",
    )

    result = _gate(scratch)

    assert result.returncode == 1
    assert "no resolve_uv() function" in result.stderr
    assert "ONEX_RECONCILE_UV_BIN" in result.stderr


def test_the_gate_rejects_uv_called_by_name(tmp_path: Path) -> None:
    """Reaching ``uv`` off PATH by name defeats the resolution it just did."""
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_text(
        "#!/usr/bin/env bash\n"
        'ONEX_RECONCILE_UV_BIN="${ONEX_RECONCILE_UV_BIN:-}"\n'
        "resolve_uv() { :; }\n"
        'as_owner() { "$@"; }\n'
        "as_owner uv sync --frozen\n",
        encoding="utf-8",
    )

    result = _gate(scratch)

    assert result.returncode == 1
    assert "by name" in result.stderr


def test_the_gate_does_not_flag_a_command_quoted_inside_an_error_message(
    tmp_path: Path,
) -> None:
    """Refusals print the command to run by hand; that is documentation.

    Without this the gate would punish the very messages that make a refusal
    actionable, and the cheapest way to make it quiet would be to delete them.
    """
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_text(
        "#!/usr/bin/env bash\n"
        'ONEX_RECONCILE_UV_BIN="${ONEX_RECONCILE_UV_BIN:-}"\n'
        "resolve_uv() { :; }\n"
        'as_owner() { "$@"; }\n'
        'fail "run by hand:" \\\n'
        '  "  cd $DIR && env -u PYTHONPATH uv sync --frozen --inexact"\n',
        encoding="utf-8",
    )

    result = _gate(scratch)

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_is_wired_into_precommit_and_ci() -> None:
    """Rule 5: detection that is not wired is advisory and gets ignored.

    Asserted against the config files themselves, so deleting the hook or the CI
    step fails a test rather than silently disarming the check.
    """
    precommit = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check-reconciler-privilege" in precommit
    assert "scripts/check_reconciler_privilege.py" in precommit

    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/check_reconciler_privilege.py" in ci
