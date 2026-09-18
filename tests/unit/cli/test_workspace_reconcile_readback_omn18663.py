# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconcile outcome is a readback, not an exit status (OMN-18663).

THE DEFECT, in the guard's own words: "A reconcile ran, reported SUCCESS, and
the venv is STILL drifted." Seen twice on 2026-09-18 against the shared plugin
CLI venv (omni_home/CLAUDE.md rule 11).

Why it happened: :func:`reconcile_workspace_venvs` returned ``ok=True`` whenever
``scripts/reconcile-workspace-venvs.sh`` exited 0. That script says in its own
header (OMN-17307) that it does NOT prove its own work -- it exits on its
collaborators' statuses and the readback belongs one layer up, in
``scripts/reconcile-host.sh``. This adapter calls the repair script directly, so
on this path that layer did not exist. The reconciler also owns a FIXED set of
venvs (the dispatch venv, the gate venv, hook venvs carrying a ``uv.lock``), and
the interpreter running the CLI need not be one of them -- so a completely
successful reconcile can leave the caller's own venv untouched and still exit 0.

These tests pin the fix: ``ok`` is derived from a readback of the interpreter
the caller runs on, using the same comparison the guard makes, and every attempt
carries a run id that the guard's refusals name.

Fully hermetic: the reconciler is a stub script, the canonical clone is a real
local git repo, and the "venv" is a shim that answers the metadata probe.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.cli.omnimarket_drift_guard import (
    OmnimarketDriftError,
    check_omnimarket_drift,
)
from omnibase_infra.cli.workspace_reconcile import (
    READBACK_SCRIPT_RELATIVE_PATH,
    RECONCILE_SCRIPT_RELATIVE_PATH,
    ModelReconcileOutcome,
    new_reconcile_run_id,
    reconcile_workspace_venvs,
)

pytestmark = pytest.mark.unit


def _scrubbed_git_env() -> dict[str, str]:
    """A git environment that cannot reach out of ``tmp_path`` (OMN-14891).

    git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into every hook
    environment, and those OVERRIDE both ``cwd=`` and ``git -C``: a fixture that
    shells out to git while running under a pre-commit or pre-push hook would
    mutate the REAL invoking worktree. ``scrub_git_location_env`` removes them.

    It also removes every ``GIT_CONFIG*`` key, including the conftest fixture's
    protective ones (OMN-16584), so the neutral overrides are put back after the
    scrub -- otherwise a developer's ``[tag] gpgsign = true`` (or any global
    hook config) leaks into these fixtures.
    """
    env = scrub_git_location_env(os.environ)
    # Named literally as well as scrubbed: the OMN-14891 guard verifies a
    # module-local scrubber by reading the keys it drops, and a delegated call
    # is invisible to that check.
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EDITOR"] = "true"
    return env


_REPO_ROOT = Path(__file__).resolve().parents[3]
_STALE_SHA = "c" * 40


def _build_workspace(
    root: Path, *, reconciler_exit: int, version: str = "0.4.119"
) -> tuple[Path, str]:
    """An $OMNI_HOME with a real omnimarket clone and a stub reconciler."""
    omni_home = root / "omni_home"
    clone = omni_home / "omnimarket"
    clone.mkdir(parents=True)

    def run(*args: str) -> None:
        subprocess.run(
            list(args),
            cwd=clone,
            check=True,
            capture_output=True,
            env=_scrubbed_git_env(),
        )

    run("git", "init", "--quiet", "-b", "dev")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (clone / "pyproject.toml").write_text(
        f'[project]\nname = "omnimarket"\nversion = "{version}"\n', encoding="utf-8"
    )
    run("git", "add", "pyproject.toml")
    run("git", "commit", "--quiet", "-m", "init")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        env=_scrubbed_git_env(),
    ).stdout.strip()

    script = omni_home / RECONCILE_SCRIPT_RELATIVE_PATH
    script.parent.mkdir(parents=True, exist_ok=True)
    # The whole point: it reports a result without changing the caller's venv.
    script.write_text(
        "#!/usr/bin/env bash\n"
        "echo '[reconcile] dispatch venv: reconciled'\n"
        f"exit {reconciler_exit}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    shutil.copy2(
        _REPO_ROOT / "scripts" / "venv_readback.py",
        omni_home / READBACK_SCRIPT_RELATIVE_PATH,
    )
    return omni_home, head


def _fake_python(root: Path, commit: str | None, version: str | None) -> Path:
    shim = root / "fake-python"
    facts = {"omnimarket": {"version": version, "commit": commit}}
    shim.write_text(
        "#!/usr/bin/env bash\nprintf '%s' " + repr(json.dumps(facts)) + "\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


# --------------------------------------------------------------------------- #
# ok is derived from the readback
# --------------------------------------------------------------------------- #
def test_exit_zero_reconcile_that_leaves_this_venv_drifted_is_not_ok(
    tmp_path: Path,
) -> None:
    """THE REPRO. Before OMN-18663 this returned ok=True."""
    omni_home, head = _build_workspace(tmp_path, reconciler_exit=0)
    python_bin = _fake_python(tmp_path, _STALE_SHA, "0.4.118")

    outcome = reconcile_workspace_venvs(str(omni_home), target_python=str(python_bin))

    assert outcome.ok is False
    assert _STALE_SHA in outcome.detail
    assert head in outcome.detail
    # Actionable: the repair for THIS venv, not a generic pointer.
    assert "check-omnimarket-venv-drift.sh" in outcome.detail


def test_exit_zero_reconcile_whose_result_this_venv_carries_is_ok(
    tmp_path: Path,
) -> None:
    omni_home, head = _build_workspace(tmp_path, reconciler_exit=0)
    python_bin = _fake_python(tmp_path, head, "0.4.119")

    outcome = reconcile_workspace_venvs(str(omni_home), target_python=str(python_bin))

    assert outcome.ok is True, outcome.detail
    assert outcome.detail == ""


def test_absent_omnimarket_after_an_exit_zero_reconcile_is_not_ok(
    tmp_path: Path,
) -> None:
    omni_home, _ = _build_workspace(tmp_path, reconciler_exit=0)
    python_bin = _fake_python(tmp_path, None, None)

    assert (
        reconcile_workspace_venvs(str(omni_home), target_python=str(python_bin)).ok
        is False
    )


def test_a_failing_reconciler_is_still_not_ok(tmp_path: Path) -> None:
    omni_home, head = _build_workspace(tmp_path, reconciler_exit=2)
    python_bin = _fake_python(tmp_path, head, "0.4.119")

    outcome = reconcile_workspace_venvs(str(omni_home), target_python=str(python_bin))

    assert outcome.ok is False
    assert outcome.run_id


def test_a_missing_readback_fails_closed(tmp_path: Path) -> None:
    """No readback means no proof, and no proof is not a pass."""
    omni_home, head = _build_workspace(tmp_path, reconciler_exit=0)
    (omni_home / READBACK_SCRIPT_RELATIVE_PATH).unlink()
    python_bin = _fake_python(tmp_path, head, "0.4.119")

    outcome = reconcile_workspace_venvs(str(omni_home), target_python=str(python_bin))

    assert outcome.ok is False
    assert "readback" in outcome.detail


def test_a_missing_reconciler_is_not_ok(tmp_path: Path) -> None:
    omni_home, _ = _build_workspace(tmp_path, reconciler_exit=0)
    (omni_home / RECONCILE_SCRIPT_RELATIVE_PATH).unlink()

    outcome = reconcile_workspace_venvs(str(omni_home))

    assert outcome.ok is False
    assert outcome.run_id


def test_the_readback_target_defaults_to_the_running_interpreter(
    tmp_path: Path,
) -> None:
    """The guard protects THIS interpreter, so that is what gets proven."""
    omni_home, _ = _build_workspace(tmp_path, reconciler_exit=0)

    outcome = reconcile_workspace_venvs(str(omni_home))

    # sys.executable is not installed from this throwaway clone, so the honest
    # answer is "not proven" -- never a pass by default.
    assert outcome.ok is False
    assert sys.executable in outcome.detail


# --------------------------------------------------------------------------- #
# Every attempt is identifiable (AC3)
# --------------------------------------------------------------------------- #
def test_run_ids_are_unique_and_timestamped() -> None:
    first, second = new_reconcile_run_id(), new_reconcile_run_id()
    assert first != second
    assert first.startswith("reconcile-")


def test_outcomes_carry_a_run_id(tmp_path: Path) -> None:
    omni_home, head = _build_workspace(tmp_path, reconciler_exit=0)
    python_bin = _fake_python(tmp_path, head, "0.4.119")

    assert reconcile_workspace_venvs(
        str(omni_home), target_python=str(python_bin)
    ).run_id.startswith("reconcile-")


# --------------------------------------------------------------------------- #
# The guard names the run it disagrees with
# --------------------------------------------------------------------------- #
def _guard_workspace(root: Path) -> tuple[str, str]:
    clone = root / "omnimarket"
    clone.mkdir(parents=True)

    def run(*args: str) -> None:
        subprocess.run(
            list(args),
            cwd=clone,
            check=True,
            capture_output=True,
            env=_scrubbed_git_env(),
        )

    run("git", "init", "--quiet", "-b", "dev")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (clone / "f.txt").write_text("x", encoding="utf-8")
    run("git", "add", "f.txt")
    run("git", "commit", "--quiet", "-m", "init")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        env=_scrubbed_git_env(),
    ).stdout.strip()
    return str(root), head


def test_a_failed_reconcile_refusal_names_the_run(tmp_path: Path) -> None:
    omni_home, _ = _guard_workspace(tmp_path)
    outcome = ModelReconcileOutcome(
        ok=False,
        command="bash reconcile-workspace-venvs.sh",
        detail="uv sync failed",
        run_id="reconcile-20260918T153012Z-deadbeef",
    )

    with pytest.raises(OmnimarketDriftError) as excinfo:
        check_omnimarket_drift(omni_home, reconcile=lambda: outcome)

    assert outcome.run_id in str(excinfo.value)


def test_a_success_the_guard_disagrees_with_names_the_run(tmp_path: Path) -> None:
    """A refusal that says 'a reconcile ran' must say WHICH one."""
    omni_home, _ = _guard_workspace(tmp_path)
    outcome = ModelReconcileOutcome(
        ok=True,
        command="bash reconcile-workspace-venvs.sh",
        detail="",
        run_id="reconcile-20260918T153012Z-cafef00d",
    )

    with pytest.raises(OmnimarketDriftError) as excinfo:
        check_omnimarket_drift(omni_home, reconcile=lambda: outcome)

    message = str(excinfo.value)
    assert outcome.run_id in message
    assert "STILL drifted" in message
