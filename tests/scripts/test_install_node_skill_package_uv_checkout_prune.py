# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pruning the uv git checkouts a verified install leaves behind (OMN-24308).

The reconcile's additive pass installs ``omnimarket @ git+https://...@<sha>``
about 45 times a day. uv keeps one full git checkout per revision it resolves
under ``<uv cache dir>/git-v0/checkouts/<repo hash>/<revision folder>`` (the
newer uv uses ``git-v1`` and writes a marker file ``<revision folder>.ok``
beside the folder) and never removes it. The checkout is created even by the
dry-run plan inside ``stage``, and the installed package is a copy, so after a
verified install the checkout is dead weight. These tests exercise the recording
and the prune without a network or a real venv.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = (
    _REPO_ROOT
    / "src/omnibase_infra/handlers/venv_atomic_install/atomic_venv_install.py"
)

_SPEC = importlib.util.spec_from_file_location("atomic_venv_install", _LIB)
assert _SPEC is not None and _SPEC.loader is not None
plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plan)

_SHA = "0123456789abcdef0123456789abcdef01234567"
_OTHER = "fedcba9876543210fedcba9876543210fedcba9876"


def _seed(cache: Path, layout: str, repo: str, sha: str, marker: bool) -> Path:
    folder = cache / layout / "checkouts" / repo / sha[:7]
    folder.mkdir(parents=True)
    (folder / "file").write_text("x", encoding="utf-8")
    if marker:
        (folder.parent / (sha[:7] + ".ok")).write_text("ok", encoding="utf-8")
    return folder


def _recorded(stage: Path) -> None:
    plan.record_git_revisions(
        stage,
        [
            f"omnimarket @ git+https://github.com/OmniNode-ai/omnimarket.git@{_SHA}",
            "anthropic",
        ],
    )


# git-v0 is the older uv layout (no marker file); git-v1 is the one measured with uv 0.13.6 on 2026-11-13, which writes
# `<revision>.ok` beside each checkout folder.
@pytest.mark.parametrize(("layout", "marker"), [("git-v0", False), ("git-v1", True)])
def test_finalize_removes_installed_ref_checkout_of_a_recorded_git_revision_and_only_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, marker: bool
) -> None:
    """Only the recorded revision's checkout (and its marker) is removed."""
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    stage.mkdir()
    recorded = _seed(cache, layout, "repoA", _SHA, marker)
    other_a = _seed(cache, layout, "repoA", _OTHER, marker)
    other_b = _seed(cache, layout, "repoB", _OTHER, marker)
    (cache / "archive-v0" / "x").mkdir(parents=True)
    (cache / "archive-v0" / "x" / "file").write_text("x", encoding="utf-8")
    _recorded(stage)
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))

    rc = plan.run_command("finalize", stage_dir=stage)

    assert rc == plan.EXIT_OK
    assert not recorded.exists(), "the recorded checkout was not pruned"
    assert not recorded.with_name(recorded.name + ".ok").exists()
    if marker:
        assert other_a.with_name(other_a.name + ".ok").exists()
    assert other_a.exists(), "an unrecorded checkout in the same repo was pruned"
    assert other_b.exists(), "an unrecorded checkout in another repo was pruned"
    assert (cache / "archive-v0" / "x" / "file").exists(), "archive-v0 was touched"
    assert not stage.exists(), "the stage dir was not removed"


def test_record_git_revisions_reads_only_git_requirements(tmp_path: Path) -> None:
    """Only git requirements are recorded; duplicates collapse; shas lowercase."""
    stage = tmp_path / "stage"
    shas = plan.record_git_revisions(
        stage,
        [
            f"omnimarket @ git+https://github.com/OmniNode-ai/omnimarket.git@{_SHA}",
            f"omnimarket @ git+https://github.com/OmniNode-ai/omnimarket.git@{_SHA}",
            "anthropic",
            "radon",
        ],
    )
    assert shas == [_SHA]
    assert (stage / "git-revisions.txt").read_text(encoding="utf-8").splitlines() == [
        _SHA
    ]

    empty = tmp_path / "empty"
    assert plan.record_git_revisions(empty, ["anthropic", "radon"]) == []
    assert not (empty / "git-revisions.txt").exists()

    upper = tmp_path / "upper"
    got = plan.record_git_revisions(
        upper, [f"pkg @ git+https://example.com/x.git@{_SHA.upper()}"]
    )
    assert got == [_SHA]

    short = tmp_path / "short"
    assert (
        plan.record_git_revisions(short, ["pkg @ git+https://example.com/x.git@abc123"])
        == []
    )


def test_dry_run_or_failed_install_rollback_leaves_the_checkout_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rolled back install removes nothing from the uv cache."""
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    stage.mkdir()
    recorded = _seed(cache, "git-v0", "repoA", _SHA, False)
    _recorded(stage)
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))

    rc = plan.run_command("rollback", stage_dir=stage)

    assert rc == plan.EXIT_OK
    assert recorded.exists(), "rollback pruned a checkout it must not touch"


def test_finalize_never_fails_when_uv_is_missing_or_the_cache_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing uv or an absent cache dir still finalizes and removes the stage."""
    stage = tmp_path / "stage"
    stage.mkdir()
    _recorded(stage)

    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    assert plan.run_command("finalize", stage_dir=stage) == plan.EXIT_OK
    assert not stage.exists()

    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    _recorded(stage2)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "absent"))
    assert plan.run_command("finalize", stage_dir=stage2) == plan.EXIT_OK
    assert not stage2.exists()


def test_short_or_symlinked_revision_folders_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short folder name and a symlink are never pruned."""
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    stage.mkdir()
    short = cache / "git-v0" / "checkouts" / "repoA" / "abc"
    short.mkdir(parents=True)
    (short / "file").write_text("x", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    (target / "file").write_text("x", encoding="utf-8")
    link = cache / "git-v0" / "checkouts" / "repoA" / _SHA[:7]
    link.symlink_to(target)
    _recorded(stage)
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))

    rc = plan.run_command("finalize", stage_dir=stage)

    assert rc == plan.EXIT_OK
    assert short.exists(), "a short revision folder was pruned"
    assert link.is_symlink(), "a symlinked revision folder was pruned"
    assert (target / "file").exists(), "the symlink target was disturbed"


def test_dry_run_or_failed_stage_records_the_revision_and_removes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The revision is recorded before the dry-run plan creates the checkout."""
    cache = tmp_path / "cache"
    stage = tmp_path / "stage"
    stage.mkdir()
    recorded = _seed(cache, "git-v0", "repoA", _SHA, False)
    monkeypatch.setenv("UV_CACHE_DIR", str(cache))

    def fake_write_wrapper(python_bin: str, stage_dir: Path) -> Path:
        return stage_dir / "python"

    class _FakePlan:
        def _run_uv(
            self, *args: Any, **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="", stderr=""
            )

    monkeypatch.setattr(plan, "write_wrapper", fake_write_wrapper)
    monkeypatch.setattr(plan, "_load_plan_tool", lambda: _FakePlan())

    plan.stage(
        "/nonexistent/python",
        stage,
        [f"omnimarket @ git+https://github.com/OmniNode-ai/omnimarket.git@{_SHA}"],
        no_deps=True,
        label="t",
    )

    assert (stage / "git-revisions.txt").read_text(encoding="utf-8").splitlines() == [
        _SHA
    ]
    # The stage failed at its plan: nothing prunes until finalize, which the script
    # never reaches on a failed install or a dry run.
    assert recorded.exists(), "a failed stage pruned a checkout"
