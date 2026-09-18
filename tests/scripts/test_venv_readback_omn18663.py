# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/venv_readback.py (OMN-18663).

The defect these cover: every surface in the omnimarket repair path decided
success from an exit status. ``uv pip install`` exits 0 for "already satisfied"
exactly as readily as for a real install, so a repair that changed nothing
reported SUCCESS and the drift guard went on refusing the same interpreter --
"a reconcile ran, reported SUCCESS, and the venv is STILL drifted", observed
twice on 2026-09-18 against the shared plugin CLI venv.

Fully hermetic: the "canonical clone" is a real local git repo with a real
pyproject.toml, and the "venv" is a fake python shim that prints canned
importlib.metadata facts. No network, no uv, no real install.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
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


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "venv_readback.py"

sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from venv_readback import (
    EXIT_DRIFTED,
    EXIT_IN_SYNC,
    EXIT_INDETERMINATE,
    ReadbackOutcome,
    ReadbackRow,
    ReadbackVerdict,
    classify_commit,
    classify_sibling,
    classify_version,
    outcome_for,
)

_SHA_A = "a" * 40
_SHA_B = "b" * 40


# --------------------------------------------------------------------------- #
# Pure classification
# --------------------------------------------------------------------------- #
def test_matching_commit_is_a_match() -> None:
    row = classify_commit(_SHA_A, _SHA_A)
    assert row.verdict is ReadbackVerdict.MATCH


def test_different_commit_is_a_mismatch_carrying_both_values() -> None:
    row = classify_commit(_SHA_A, _SHA_B)
    assert row.verdict is ReadbackVerdict.MISMATCH
    assert row.installed == _SHA_A
    assert row.expected == _SHA_B


def test_absent_commit_is_absent_not_a_match() -> None:
    """A PyPI wheel carries no vcs_info; that is drift, never a pass."""
    assert classify_commit(None, _SHA_A).verdict is ReadbackVerdict.ABSENT


def test_version_mismatch_is_drift_even_when_the_commit_matches() -> None:
    assert classify_version("0.4.118", "0.4.119").verdict is ReadbackVerdict.MISMATCH


def test_sibling_inside_its_declared_range_matches() -> None:
    row = classify_sibling("omnibase-compat>=0.5.7,<0.6.0", "0.5.7")
    assert row.verdict is ReadbackVerdict.MATCH


def test_sibling_below_its_declared_range_is_a_mismatch() -> None:
    """The 2026-09-18 downgrade shape: 0.5.5 installed under a >=0.5.7 pin."""
    row = classify_sibling("omnibase-compat>=0.5.7,<0.6.0", "0.5.5")
    assert row.verdict is ReadbackVerdict.MISMATCH


def test_absent_sibling_is_absent() -> None:
    row = classify_sibling("omninode-memory>=0.18.0", None)
    assert row.verdict is ReadbackVerdict.ABSENT


def test_unparseable_requirement_is_unreadable_not_a_pass() -> None:
    row = classify_sibling("!!! not a requirement", "1.0.0")
    assert row.verdict is ReadbackVerdict.UNREADABLE


# --------------------------------------------------------------------------- #
# Reduction — fail closed
# --------------------------------------------------------------------------- #
def test_all_matching_rows_are_in_sync() -> None:
    rows = [classify_commit(_SHA_A, _SHA_A), classify_version("1.0.0", "1.0.0")]
    assert outcome_for(rows) is ReadbackOutcome.IN_SYNC


def test_one_mismatch_drifts_the_whole_readback() -> None:
    rows = [classify_commit(_SHA_A, _SHA_B), classify_version("1.0.0", "1.0.0")]
    assert outcome_for(rows) is ReadbackOutcome.DRIFTED


def test_an_unreadable_row_is_indeterminate_not_drifted_and_not_in_sync() -> None:
    rows = [
        classify_commit(_SHA_A, _SHA_A),
        ReadbackRow("x", "y", "z", ReadbackVerdict.UNREADABLE),
    ]
    assert outcome_for(rows) is ReadbackOutcome.INDETERMINATE


def test_an_empty_row_set_is_indeterminate() -> None:
    """A readback that checked nothing has proven nothing."""
    assert outcome_for([]) is ReadbackOutcome.INDETERMINATE


# --------------------------------------------------------------------------- #
# End to end, against a fake clone and a fake interpreter
# --------------------------------------------------------------------------- #
def _make_clone(root: Path, *, version: str) -> tuple[Path, str]:
    """A real git clone whose pyproject.toml declares ``version``."""
    clone = root / "omnimarket"
    clone.mkdir()

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
        f'[project]\nname = "omnimarket"\nversion = "{version}"\n'
        'dependencies = ["omnibase-compat>=0.5.7,<0.6.0"]\n',
        encoding="utf-8",
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
    return clone, head


def _make_fake_python(root: Path, facts: dict[str, dict[str, str | None]]) -> Path:
    """A fake interpreter that answers the readback probe with canned facts."""
    shim = root / "fake-python"
    shim.write_text(
        "#!/usr/bin/env bash\nprintf '%s' " + repr(json.dumps(facts)) + "\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _run(
    clone: Path, python_bin: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--python",
            str(python_bin),
            "--clone",
            str(clone),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_venv_carrying_the_ref_reads_back_in_sync(tmp_path: Path) -> None:
    clone, head = _make_clone(tmp_path, version="0.4.119")
    python_bin = _make_fake_python(
        tmp_path,
        {
            "omnimarket": {"version": "0.4.119", "commit": head},
            "omnibase-compat": {"version": "0.5.7", "commit": None},
        },
    )
    result = _run(clone, python_bin, "--sibling", "omnibase-compat>=0.5.7,<0.6.0")
    assert result.returncode == EXIT_IN_SYNC, result.stdout + result.stderr
    assert "IN_SYNC" in result.stdout


def test_venv_left_at_the_old_commit_reads_back_drifted(tmp_path: Path) -> None:
    """THE DEFECT: the install reported success and the venv did not move."""
    clone, head = _make_clone(tmp_path, version="0.4.119")
    python_bin = _make_fake_python(
        tmp_path,
        {"omnimarket": {"version": "0.4.118", "commit": _SHA_A}},
    )
    result = _run(clone, python_bin)
    assert result.returncode == EXIT_DRIFTED
    # Both values, so the reader needs nothing else to act.
    assert _SHA_A in result.stderr
    assert head in result.stderr


def test_absent_omnimarket_reads_back_drifted(tmp_path: Path) -> None:
    clone, _ = _make_clone(tmp_path, version="0.4.119")
    python_bin = _make_fake_python(
        tmp_path, {"omnimarket": {"version": None, "commit": None}}
    )
    assert _run(clone, python_bin).returncode == EXIT_DRIFTED


def test_downgraded_sibling_reads_back_drifted(tmp_path: Path) -> None:
    """The OMN-18675 shape: omnimarket right, a sibling moved backwards."""
    clone, head = _make_clone(tmp_path, version="0.4.119")
    python_bin = _make_fake_python(
        tmp_path,
        {
            "omnimarket": {"version": "0.4.119", "commit": head},
            "omnibase-compat": {"version": "0.5.5", "commit": None},
        },
    )
    result = _run(clone, python_bin, "--sibling", "omnibase-compat>=0.5.7,<0.6.0")
    assert result.returncode == EXIT_DRIFTED
    assert "omnibase-compat" in result.stderr


def test_unprobeable_interpreter_is_indeterminate_not_in_sync(tmp_path: Path) -> None:
    """Fail closed: an interpreter that cannot answer has proven nothing."""
    clone, _ = _make_clone(tmp_path, version="0.4.119")
    broken = tmp_path / "broken-python"
    broken.write_text("#!/usr/bin/env bash\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)
    result = _run(clone, broken)
    assert result.returncode == EXIT_INDETERMINATE
    assert "INDETERMINATE" in result.stderr


def test_unknown_ref_is_indeterminate(tmp_path: Path) -> None:
    clone, _ = _make_clone(tmp_path, version="0.4.119")
    python_bin = _make_fake_python(
        tmp_path, {"omnimarket": {"version": "0.4.119", "commit": _SHA_A}}
    )
    result = _run(clone, python_bin, "--ref", _SHA_B)
    assert result.returncode == EXIT_INDETERMINATE


# --------------------------------------------------------------------------- #
# No bypass (omni_home/CLAUDE.md rule 10)
# --------------------------------------------------------------------------- #
_BYPASS_SHAPES = ("--force", "--allow", "--skip", "--ignore", "--no-readback", "--yes")


def test_help_advertises_no_bypass_flag() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for shape in _BYPASS_SHAPES:
        assert shape not in result.stdout, (
            f"{shape} would let a readback be told to pass"
        )


def test_source_declares_no_bypass_option() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    for shape in _BYPASS_SHAPES:
        assert f'"{shape}' not in source
