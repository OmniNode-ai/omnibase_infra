# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/venv_install_plan.py (OMN-18675).

The defect these cover: on 2026-09-18 a venv-drift repair ran
``uv pip install --no-deps omnibase-compat==0.5.5`` against a venv that already
carried 0.5.7, uv did exactly as instructed, and the resulting downgrade
removed a module the installed ``omnibase_infra`` imports from an ``onex.cli``
entry point — taking the whole ``onex`` CLI down for every lane on the host.

Fully hermetic: ``uv`` is a fake shell script on PATH that records its argv and
prints a canned change plan, so the refusal is driven end-to-end without
mutating any real environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "venv_install_plan.py"

sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from venv_install_plan import (
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_UV_ERROR,
    PlanVerdict,
    classify,
    normalize_name,
    parse_plan,
)

# A real uv dry-run plan, as emitted for the 2026-09-18 repair.
_DOWNGRADE_PLAN = """\
Resolved 3 packages in 812ms
Would download 2 packages
Would install 3 packages
 - omnibase-compat==0.5.7
 + omnibase-compat==0.5.5
 - omninode-memory==0.18.0
 + omninode-memory==0.15.0
 - omnimarket==0.4.118
 + omnimarket==0.4.118 (from git+https://github.com/OmniNode-ai/omnimarket.git@abc123)
"""

_SAFE_PLAN = """\
Resolved 1 package in 120ms
Would install 1 package
 - omnimarket==0.4.117
 + omnimarket==0.4.118
"""

_NOOP_PLAN = "Resolved 3 packages in 90ms\nAudited 3 packages in 1ms\n"


def _write_fake_uv(tmp_path: Path, plan: str) -> Path:
    """Put a fake `uv` on PATH that records argv and prints ``plan``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    argv_log = tmp_path / "uv-argv.log"
    fake = bin_dir / "uv"
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text(plan, encoding="utf-8")
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{argv_log}"\n'
        f'cat "{plan_file}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return bin_dir


def _run(bin_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


# --------------------------------------------------------------------------
# parsing / classification
# --------------------------------------------------------------------------


def test_normalize_name_follows_pep503() -> None:
    assert normalize_name("Omnibase_Compat") == "omnibase-compat"
    assert normalize_name("omninode.memory") == "omninode-memory"


def test_parse_plan_pairs_removed_and_added_versions() -> None:
    changes = {c.name: c for c in parse_plan(_DOWNGRADE_PLAN)}
    assert changes["omnibase-compat"].before == "0.5.7"
    assert changes["omnibase-compat"].after == "0.5.5"
    assert changes["omnibase-compat"].verdict is PlanVerdict.DOWNGRADE
    assert changes["omninode-memory"].verdict is PlanVerdict.DOWNGRADE


def test_a_vcs_reinstall_at_the_same_version_is_not_a_downgrade() -> None:
    """The git-sourced omnimarket line is the whole point of the install."""
    changes = {c.name: c for c in parse_plan(_DOWNGRADE_PLAN)}
    assert changes["omnimarket"].verdict is PlanVerdict.REINSTALL


def test_classify_covers_every_verdict() -> None:
    assert classify("p", None, "1.0").verdict is PlanVerdict.INSTALL
    assert classify("p", "1.0", None).verdict is PlanVerdict.REMOVE
    assert classify("p", "1.0", "1.0").verdict is PlanVerdict.REINSTALL
    assert classify("p", "1.0", "2.0").verdict is PlanVerdict.UPGRADE
    assert classify("p", "2.0", "1.0").verdict is PlanVerdict.DOWNGRADE


def test_an_unorderable_version_pair_fails_closed() -> None:
    """An ordering we cannot prove is not evidence the move is safe."""
    assert classify("p", "not-a-version", "0.1").verdict is PlanVerdict.UNKNOWN


def test_version_ordering_is_not_lexicographic() -> None:
    """0.5.10 > 0.5.7 numerically but sorts lower as a string."""
    assert classify("p", "0.5.7", "0.5.10").verdict is PlanVerdict.UPGRADE
    assert classify("p", "0.5.10", "0.5.7").verdict is PlanVerdict.DOWNGRADE


# --------------------------------------------------------------------------
# end-to-end refusal — the OMN-18675 falsifier
# --------------------------------------------------------------------------


def test_refuses_a_plan_that_downgrades_an_installed_package(tmp_path: Path) -> None:
    """AC1/AC3: a repair that would move a sibling backwards is refused."""
    bin_dir = _write_fake_uv(tmp_path, _DOWNGRADE_PLAN)
    result = _run(
        bin_dir,
        "--python",
        sys.executable,
        "--no-deps",
        "--apply",
        "--label",
        "step 1",
        "--",
        "omnibase-compat==0.5.5",
    )
    assert result.returncode == EXIT_REFUSED, result.stdout + result.stderr
    assert "REFUSED" in result.stderr
    # Names the packages, so the operator can act without re-deriving them.
    assert "omnibase-compat" in result.stderr
    assert "omninode-memory" in result.stderr
    assert "0.5.7 -> 0.5.5" in result.stderr


def test_refusal_does_not_apply_anything(tmp_path: Path) -> None:
    """A refusal must leave the venv untouched: uv is never run without --dry-run."""
    bin_dir = _write_fake_uv(tmp_path, _DOWNGRADE_PLAN)
    result = _run(
        bin_dir,
        "--python",
        sys.executable,
        "--apply",
        "--",
        "omnibase-compat==0.5.5",
    )
    assert result.returncode == EXIT_REFUSED
    invocations = (tmp_path / "uv-argv.log").read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1, invocations
    assert "--dry-run" in invocations[0]


def test_there_is_no_flag_that_permits_a_downgrade(tmp_path: Path) -> None:
    """Rule 10: a bypass flag would be the first thing a blocked lane reached for."""
    help_text = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for forbidden in ("--force", "--allow-downgrade", "--skip", "--no-verify"):
        assert forbidden not in help_text


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_safe_plan_applies_and_leaves_unrelated_pins_untouched(tmp_path: Path) -> None:
    """AC1 happy path: only the intended package appears in the change table."""
    bin_dir = _write_fake_uv(tmp_path, _SAFE_PLAN)
    result = _run(
        bin_dir,
        "--python",
        sys.executable,
        "--no-deps",
        "--apply",
        "--",
        "omnimarket==0.4.118",
    )
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert "UPGRADE" in result.stdout
    assert "omnimarket  0.4.117 -> 0.4.118" in result.stdout
    assert "omnibase-compat" not in result.stdout
    assert "omninode-memory" not in result.stdout
    invocations = (tmp_path / "uv-argv.log").read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2, invocations
    assert "--dry-run" in invocations[0]
    assert "--dry-run" not in invocations[1]


def test_without_apply_nothing_is_installed(tmp_path: Path) -> None:
    bin_dir = _write_fake_uv(tmp_path, _SAFE_PLAN)
    result = _run(bin_dir, "--python", sys.executable, "--", "omnimarket==0.4.118")
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert "PLAN ONLY" in result.stdout
    invocations = (tmp_path / "uv-argv.log").read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "--dry-run" in invocations[0]


def test_a_no_op_plan_is_reported_as_no_change(tmp_path: Path) -> None:
    bin_dir = _write_fake_uv(tmp_path, _NOOP_PLAN)
    result = _run(
        bin_dir, "--python", sys.executable, "--apply", "--", "omnimarket==0.4.118"
    )
    assert result.returncode == EXIT_OK, result.stdout + result.stderr
    assert "no change" in result.stdout


def test_a_failing_uv_is_an_error_not_an_empty_plan(tmp_path: Path) -> None:
    """An empty result is not evidence of absence (omni_home/CLAUDE.md rule 16)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text(
        '#!/usr/bin/env bash\necho "error: no solution found" >&2\nexit 1\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    result = _run(
        bin_dir, "--python", sys.executable, "--apply", "--", "omnimarket==0.4.118"
    )
    assert result.returncode == EXIT_UV_ERROR
    assert "could not resolve" in result.stderr
