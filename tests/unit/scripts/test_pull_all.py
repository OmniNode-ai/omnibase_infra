# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for pull-all.sh and bare-clone-sync infrastructure."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
PULL_ALL = SCRIPTS_DIR / "pull-all.sh"
PLIST = SCRIPTS_DIR / "ai.omninode.bare-clone-sync.plist"
INSTALL_SCRIPT = SCRIPTS_DIR / "install-bare-clone-sync.sh"

# OMN-15590: the caller-checkable terminal completion signal. A caller must be
# able to distinguish {clean, drift-timeout, drift-fail} runs from ONE
# machine-parseable line, without reading prose banners.
RESULT_LINE_PREFIX = "PULL-ALL-RESULT: "

# Harness-side backstop for the bounding tests. NOT the mechanism under test:
# the assertions require pull-all.sh to return on its own well inside this.
# It exists so an UNBOUNDED script fails the suite instead of wedging it.
HARNESS_BACKSTOP_SECONDS = 90.0


def _hermetic_git_env() -> dict[str, str]:
    """Environment for child ``git`` calls, hermetic against a pre-push hook.

    A ``git push`` pre-push hook (the OMN-13973 governed full-suite escalation
    that runs this suite when ``src/omnibase_infra/topics/`` changes) exports
    ``GIT_DIR`` / ``GIT_INDEX_FILE`` / ``GIT_WORK_TREE`` into its environment.
    Those variables take precedence over BOTH the ``cwd=`` argument and
    ``git -C``, so an inherited ``GIT_DIR`` would silently redirect every
    ``git init``/``add``/``commit`` in these fixtures onto the REAL surrounding
    worktree -- rewriting its HEAD to a ``t@t`` "init" commit and dropping its
    tracked files. Stripping every ``GIT_*`` var makes each child git resolve its
    repository purely from its target directory. The tests/unit autouse fixture
    strips these globally too; this keeps the fixture self-contained. (OMN-14744)
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _assert_under_tmp(path: Path) -> Path:
    """Guard: refuse any git op whose target is not under the pytest tmp sandbox.

    Belt-and-suspenders companion to :func:`_hermetic_git_env`. Runs BEFORE the
    destructive ``add``/``commit`` (``git init`` is first and creates ``.git`` in
    the target), so if the env strip ever regresses this fails loud and CLOSED
    instead of corrupting the surrounding worktree. (OMN-14744)
    """
    resolved = path.resolve()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    assert resolved == tmp_root or tmp_root in resolved.parents, (
        f"OMN-14744 guard: refusing git op on {resolved} outside tmp sandbox "
        f"{tmp_root} -- a leaked GIT_DIR/GIT_WORK_TREE would corrupt the real repo"
    )
    return resolved


def _git(args: list[str], *, cwd: Path) -> None:
    """Run ``git`` hermetically inside ``cwd`` (asserted under the tmp sandbox)."""
    target = _assert_under_tmp(cwd)
    subprocess.run(
        ["git", "-C", str(target), *args],
        check=True,
        env=_hermetic_git_env(),
    )


def _make_omniclaude_source(
    root: Path, file_contents: dict[str, str] | None = None
) -> Path:
    """Create a fake omniclaude repo with a minimal plugins/onex/ tree.

    Returns the path to the repo root. Commits an initial state so the
    refresh logic in pull-all.sh can `git archive HEAD`.
    """
    omniclaude = root / "omniclaude"
    onex = omniclaude / "plugins" / "onex"
    (onex / "skills").mkdir(parents=True)
    (onex / "hooks").mkdir(parents=True)
    (onex / "lib").mkdir(parents=True)
    (onex / "agents").mkdir(parents=True)

    defaults = {
        "plugins/onex/skills/example.md": "# example skill\n",
        "plugins/onex/hooks/example.sh": "#!/bin/sh\necho hooked\n",
        "plugins/onex/lib/example.py": "VERSION = 'v1'\n",
        "plugins/onex/agents/example.yaml": "name: example\n",
    }
    for rel, contents in (file_contents or defaults).items():
        p = omniclaude / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)

    # Initialize as a git repo so `git archive HEAD` works. Also set up a
    # bare upstream so `git pull --ff-only` succeeds (otherwise pull-all.sh
    # reports the repo as FAILED and exits 1, which is unrelated to the
    # cache-refresh logic we are testing). All git ops run hermetically inside
    # tmp (see _git / _hermetic_git_env) so a pre-push-hook GIT_DIR leak can
    # never redirect them onto the surrounding worktree. (OMN-14744)
    _git(["init", "-q", "--initial-branch=main"], cwd=omniclaude)
    # Fail CLOSED if the repo did not actually land in tmp: `git init` above must
    # have created omniclaude/.git. If a leaked GIT_DIR ever redirected init, no
    # .git exists here and we abort BEFORE the destructive add/commit.
    assert (omniclaude / ".git").is_dir(), (
        f"OMN-14744 guard: `git init` did not create {omniclaude / '.git'} -- a "
        "leaked GIT_DIR redirected it onto another repository"
    )
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "add", "."], cwd=omniclaude)
    _git(
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=omniclaude,
    )

    upstream = root / "omniclaude.git"
    _git(
        ["init", "-q", "--bare", "--initial-branch=main", str(upstream)],
        cwd=root,
    )
    _git(["remote", "add", "origin", str(upstream)], cwd=omniclaude)
    _git(["push", "-q", "-u", "origin", "main"], cwd=omniclaude)
    _git(["switch", "-q", "-c", "dev"], cwd=omniclaude)
    _git(["push", "-q", "-u", "origin", "dev"], cwd=omniclaude)
    _git(["switch", "-q", "main"], cwd=omniclaude)
    return omniclaude


def _commit_file(repo: Path, rel_path: str, contents: str, message: str) -> None:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    _git(["add", rel_path], cwd=repo)
    _git(
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", message],
        cwd=repo,
    )


def _make_versioned_cache(
    home: Path, initial_files: dict[str, str] | None = None
) -> Path:
    """Create a versioned plugin cache mirroring the real layout.

    Layout: <home>/.claude/plugins/cache/omninode-tools/onex/<version>/
    """
    cache = home / ".claude" / "plugins" / "cache" / "omninode-tools" / "onex" / "2.2.5"
    (cache / "skills").mkdir(parents=True)
    if initial_files:
        for rel, contents in initial_files.items():
            p = cache / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(contents)
    # Seed the .deployed-commit marker so detection prefers this path
    # over any fallback directory search.
    (cache / ".deployed-commit").write_text("0" * 40)
    return cache


def _make_simple_repo_source(root: Path, name: str) -> Path:
    """Create a minimal `<root>/<name>` repo with main+dev branches and a bare
    `<root>/<name>.git` upstream, so pull-all.sh can fetch+fast-forward it and
    report "OK". No plugins/onex scaffolding -- used for repos other than
    omniclaude where the plugin-cache-refresh content is irrelevant.
    """
    repo = root / name
    repo.mkdir(parents=True)
    (repo / "README.md").write_text(f"# {name}\n")

    _git(["init", "-q", "--initial-branch=main"], cwd=repo)
    assert (repo / ".git").is_dir(), (
        f"OMN-14744 guard: `git init` did not create {repo / '.git'}"
    )
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "add", "."], cwd=repo)
    _git(
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo,
    )

    upstream = root / f"{name}.git"
    _git(["init", "-q", "--bare", "--initial-branch=main", str(upstream)], cwd=root)
    _git(["remote", "add", "origin", str(upstream)], cwd=repo)
    _git(["push", "-q", "-u", "origin", "main"], cwd=repo)
    _git(["switch", "-q", "-c", "dev"], cwd=repo)
    _git(["push", "-q", "-u", "origin", "dev"], cwd=repo)
    _git(["switch", "-q", "main"], cwd=repo)
    return repo


def _make_main_only_repo_source(root: Path, name: str) -> Path:
    """Create a minimal `<root>/<name>` repo whose bare `<root>/<name>.git`
    upstream carries ONLY `main` -- no `dev` ref exists locally or on origin
    (OMN-16502). Models `omnigemini`, the one registry repo that never had a
    `dev` branch cut, left checked out on `main`, clean tree.
    """
    repo = root / name
    repo.mkdir(parents=True)
    (repo / "README.md").write_text(f"# {name}\n")

    _git(["init", "-q", "--initial-branch=main"], cwd=repo)
    assert (repo / ".git").is_dir(), (
        f"OMN-14744 guard: `git init` did not create {repo / '.git'}"
    )
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "add", "."], cwd=repo)
    _git(
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo,
    )

    upstream = root / f"{name}.git"
    _git(["init", "-q", "--bare", "--initial-branch=main", str(upstream)], cwd=root)
    _git(["remote", "add", "origin", str(upstream)], cwd=repo)
    _git(["push", "-q", "-u", "origin", "main"], cwd=repo)
    return repo


def _make_fake_infra_with_drift_stub(
    omni_home: Path,
    *,
    behavior: str = "ok",
    with_venv: bool = True,
    hang_seconds: int = 300,
) -> tuple[Path, Path]:
    """Create `$OMNI_HOME/omnibase_infra/scripts/check-omnimarket-venv-drift.sh`
    as a recording stub (never the real script -- the real script's own
    detect/repair logic is covered by tests/scripts/test_check_omnimarket_venv_drift.py;
    this only proves pull-all.sh's WIRING: does it invoke the drift script at
    the right time, with the right args, and handle failure without aborting).

    `behavior`:
      * ``"ok"``    -> stub exits 0.
      * ``"fail"``  -> stub exits 1.
      * ``"hang"``  -> stub sleeps ``hang_seconds`` (far past any test bound)
        and never returns on its own. Models the OMN-15590 field failure.
      * ``"hang_with_grandchild"`` -> stub spawns a DETACHED grandchild that
        also sleeps ``hang_seconds``, records that grandchild's pid to
        ``<infra_dir>/.drift-grandchild.pid``, then sleeps itself. Models the
        real process shape (``pull-all.sh`` -> drift script -> ``uv``/``git``)
        so orphan cleanup is proven against a grandchild, not just the direct
        child -- killing only the direct child is exactly the OMN-15590
        orphaning defect (PIDs 61908/62077/62078 in the field report).

    `with_venv`: when False, no `.venv/bin/python` is created (skip-guard case).
    `hang_seconds`: sleep length for the hanging behaviors.

    Returns (infra_dir, calls_log) -- calls_log records one line per
    invocation ("<args> | OMNI_HOME=<value>") so tests can assert whether/how
    the stub fired without depending on real venv or network state.
    """
    infra_dir = omni_home / "omnibase_infra"
    scripts_dir = infra_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    calls_log = infra_dir / ".drift-repair-calls.log"
    grandchild_pid_file = infra_dir / ".drift-grandchild.pid"

    record = f'echo "$@ | OMNI_HOME=$OMNI_HOME" >> "{calls_log}"\n'
    if behavior == "hang":
        body = f"sleep {hang_seconds}\n"
    elif behavior == "hang_with_grandchild":
        body = (
            f"( sleep {hang_seconds} ) &\n"
            f'echo "$!" > "{grandchild_pid_file}"\n'
            f"sleep {hang_seconds}\n"
        )
    elif behavior == "fail":
        body = "exit 1\n"
    else:
        body = "exit 0\n"

    stub = scripts_dir / "check-omnimarket-venv-drift.sh"
    stub.write_text("#!/usr/bin/env bash\n" + record + body)
    stub.chmod(0o755)

    if with_venv:
        venv_python = infra_dir / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/usr/bin/env bash\nexit 0\n")
        venv_python.chmod(0o755)

    return infra_dir, calls_log


def _run_pull_all(
    omni_home: Path,
    fake_home: Path,
    repos: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke pull-all.sh with controlled OMNI_HOME and HOME.

    ``timeout`` is a TEST-HARNESS backstop only: if pull-all.sh itself is
    unbounded, the test must fail rather than wedge the suite forever. It is
    deliberately NOT the mechanism under test -- the assertions check that the
    script returns on its own, well inside this backstop.
    """
    # Build from the hermetic (GIT_*-stripped) env: pull-all.sh shells out to git
    # on repos under OMNI_HOME, so an inherited GIT_DIR from a pre-push hook would
    # likewise redirect its git ops onto the real worktree. (OMN-14744)
    env = {
        **_hermetic_git_env(),
        "OMNI_HOME": str(omni_home),
        "HOME": str(fake_home),
        "LANG": "C",
        "LC_ALL": "C",
        "LC_CTYPE": "C",
    }
    # Drop CLAUDE_PLUGIN_ROOT so the auto-detection path is exercised.
    env.pop("CLAUDE_PLUGIN_ROOT", None)
    if extra_env:
        env.update(extra_env)
    args = ["bash", str(PULL_ALL), *(repos or ["omniclaude"])]
    return subprocess.run(
        args, capture_output=True, text=True, env=env, check=False, timeout=timeout
    )


def _parse_result_line(stdout: str) -> dict[str, str]:
    """Parse pull-all.sh's machine-checkable terminal result line (OMN-15590 AC4).

    Shape: ``PULL-ALL-RESULT: overall=OK repos_ok=1 ... drift_repair=OK ...``
    Returns the key=value pairs as a dict; empty dict when the line is absent
    (which is itself the AC4 failure mode -- a run whose completion a caller
    cannot determine without parsing prose).
    """
    for line in stdout.splitlines():
        if line.startswith(RESULT_LINE_PREFIX):
            fields = line[len(RESULT_LINE_PREFIX) :].split()
            return dict(f.split("=", 1) for f in fields if "=" in f)
    return {}


def _pid_alive(pid: int) -> bool:
    """True when ``pid`` still exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


@pytest.mark.unit
class TestPullAllScript:
    """Tests for pull-all.sh."""

    def test_script_exists_and_executable(self) -> None:
        assert PULL_ALL.exists(), f"pull-all.sh not found at {PULL_ALL}"
        assert os.access(PULL_ALL, os.X_OK), "pull-all.sh is not executable"

    def test_script_has_spdx_header(self) -> None:
        content = PULL_ALL.read_text()
        assert "SPDX-License-Identifier" in content

    def test_script_uses_parallel_execution(self) -> None:
        """Verify pull-all.sh uses parallel fetching (OMN-6869)."""
        content = PULL_ALL.read_text()
        # Should background repo fetches with &
        assert "_pull_one" in content, (
            "Expected _pull_one helper for parallel execution"
        )
        assert "wait" in content, "Expected 'wait' for parallel job synchronization"

    def test_script_uses_temp_dir_for_results(self) -> None:
        """Verify parallel results are aggregated via temp files."""
        content = PULL_ALL.read_text()
        assert "mktemp -d" in content, "Expected mktemp -d for result aggregation"
        assert "trap" in content, "Expected trap for temp dir cleanup"

    def test_script_pulls_main_and_dev_and_leaves_repo_on_dev(
        self, tmp_path: Path
    ) -> None:
        """pull-all.sh fast-forwards both long-lived branches and ends on dev."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omniclaude = _make_omniclaude_source(omni_home)
        upstream = omni_home / "omniclaude.git"
        writer = tmp_path / "writer"
        _git(["clone", "-q", str(upstream), str(writer)], cwd=tmp_path)

        _commit_file(writer, "main-only.txt", "main\n", "main update")
        _git(["push", "-q", "origin", "main"], cwd=writer)
        _git(["switch", "-q", "dev"], cwd=writer)
        _commit_file(writer, "dev-only.txt", "dev\n", "dev update")
        _git(["push", "-q", "origin", "dev"], cwd=writer)

        result = _run_pull_all(omni_home, fake_home)

        assert result.returncode == 0, result.stderr
        assert "left on dev" in result.stdout
        current_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=omniclaude, text=True
        ).strip()
        main_sha = subprocess.check_output(
            ["git", "rev-parse", "main"], cwd=omniclaude, text=True
        ).strip()
        origin_main_sha = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=omniclaude, text=True
        ).strip()
        dev_sha = subprocess.check_output(
            ["git", "rev-parse", "dev"], cwd=omniclaude, text=True
        ).strip()
        origin_dev_sha = subprocess.check_output(
            ["git", "rev-parse", "origin/dev"], cwd=omniclaude, text=True
        ).strip()

        assert current_branch == "dev"
        assert main_sha == origin_main_sha
        assert dev_sha == origin_dev_sha

    def test_script_refuses_dirty_repo_before_branch_switch(
        self, tmp_path: Path
    ) -> None:
        """Local uncommitted work blocks branch switching and remains in place."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omniclaude = _make_omniclaude_source(omni_home)
        dirty_file = omniclaude / "local-notes.txt"
        dirty_file.write_text("do not lose this\n")

        result = _run_pull_all(omni_home, fake_home)

        assert result.returncode != 0
        assert "dirty worktree" in result.stdout
        assert dirty_file.read_text() == "do not lose this\n"
        current_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=omniclaude, text=True
        ).strip()
        assert current_branch == "main"

    def test_missing_repo_warns_and_exits_zero(self) -> None:
        """Absent repo emits a WARN line and exits 0 (OMN-13055)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["bash", str(PULL_ALL), "nonexistent_repo_xyz"],
                capture_output=True,
                text=True,
                env={**_hermetic_git_env(), "OMNI_HOME": tmpdir},
                check=False,
            )
            assert result.returncode == 0, (
                f"Expected exit 0 for absent repo; got {result.returncode}. "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )
            assert "MISSING" in result.stdout or "not cloned" in result.stdout, (
                f"Expected warning about absent repo; stdout={result.stdout!r}"
            )

    def test_missing_repo_mixed_with_present_exits_zero(self, tmp_path: Path) -> None:
        """One absent + one OK repo: exit 0, absent repo warned (OMN-13055)."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(omni_home)

        result = _run_pull_all(
            omni_home, fake_home, repos=["omniclaude", "nonexistent_xyz"]
        )
        assert result.returncode == 0, (
            f"Expected exit 0 when present repos all OK; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "MISSING" in result.stdout or "not cloned" in result.stdout, (
            f"Expected warning for absent repo; stdout={result.stdout!r}"
        )
        # The present repo must have been processed successfully.
        assert "OK" in result.stdout, (
            f"Expected OK for omniclaude; stdout={result.stdout!r}"
        )

    def test_failed_present_repo_still_exits_nonzero(self, tmp_path: Path) -> None:
        """When a present repo fails (dirty), exit 1 even if other repos are absent."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omniclaude = _make_omniclaude_source(omni_home)
        # Make omniclaude dirty so it fails.
        (omniclaude / "dirty.txt").write_text("uncommitted\n")

        result = _run_pull_all(omni_home, fake_home, repos=["omniclaude", "absent_xyz"])
        assert result.returncode != 0, (
            f"Expected non-zero exit when a present repo fails; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "dirty worktree" in result.stdout

    def test_script_handles_bare_repo_no_crash(self) -> None:
        """Create a bare git repo and verify pull-all.sh does not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_name = "test_repo"
            repo_path = Path(tmpdir) / repo_name
            subprocess.run(
                ["git", "init", "--bare", str(repo_path)],
                capture_output=True,
                check=True,
                env=_hermetic_git_env(),
            )

            result = subprocess.run(
                ["bash", str(PULL_ALL), repo_name],
                capture_output=True,
                text=True,
                env={**_hermetic_git_env(), "OMNI_HOME": tmpdir},
                check=False,
            )
            # Script should complete (exit 0 or 1) without crashing.
            # The bare repo has no main ref and no remote, so it will
            # report FAILED — that's expected. The key is no crash.
            assert result.returncode in (0, 1)


@pytest.mark.unit
class TestMainOnlyRepo:
    """A canonical repo whose origin carries only `main` -- no `dev` branch
    exists anywhere (OMN-16502; `omnigemini` is the live instance).

    Field failure this closes: `_pull_one` ran `git fetch --prune origin main
    dev` unconditionally, which fails WHOLESALE ("fatal: couldn't find remote
    ref dev") when either ref is absent -- so a main-only repo was marked
    FAILED on every single pull-all.sh run (the sole failure in the OMN-16500
    proof run, 2026-08-24T23:37Z), dragging down `repos_failed` and the
    process exit code even though nothing was actually wrong with the repo.
    """

    def test_main_only_repo_reports_ok_and_stays_on_main(self, tmp_path: Path) -> None:
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        repo = _make_main_only_repo_source(omni_home, "omnigemini")

        # Advance origin/main so the fetch+fast-forward has something real to
        # prove, not just a no-op.
        writer = tmp_path / "writer"
        _git(
            ["clone", "-q", str(omni_home / "omnigemini.git"), str(writer)],
            cwd=tmp_path,
        )
        _commit_file(writer, "main-advance.txt", "main\n", "main advance")
        _git(["push", "-q", "origin", "main"], cwd=writer)

        result = _run_pull_all(omni_home, fake_home, repos=["omnigemini"])

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "OK       omnigemini" in result.stdout
        assert "main-only repo" in result.stdout
        assert "FAILED" not in result.stdout

        current_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip()
        assert current_branch == "main"

        main_sha = subprocess.check_output(
            ["git", "rev-parse", "main"], cwd=repo, text=True
        ).strip()
        origin_main_sha = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=repo, text=True
        ).strip()
        assert main_sha == origin_main_sha

        # No dev branch is fabricated locally for a main-only repo.
        dev_ref = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", "refs/heads/dev"],
            cwd=repo,
            check=False,
        )
        assert dev_ref.returncode != 0, (
            "no local dev branch should exist for a main-only repo"
        )

        fields = _parse_result_line(result.stdout)
        assert fields.get("overall") == "OK", f"fields={fields!r}"
        assert fields.get("repos_ok") == "1", f"fields={fields!r}"
        assert fields.get("repos_failed") == "0", f"fields={fields!r}"

    def test_main_only_repo_mixed_with_normal_repo_exits_zero(
        self, tmp_path: Path
    ) -> None:
        """AC: a full pull-all.sh run with a main-only repo present exits 0,
        alongside a normal main+dev repo (OMN-16502)."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_main_only_repo_source(omni_home, "omnigemini")
        _make_simple_repo_source(omni_home, "omnibase_core")

        result = _run_pull_all(
            omni_home, fake_home, repos=["omnigemini", "omnibase_core"]
        )

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "OK       omnigemini" in result.stdout
        assert "OK       omnibase_core" in result.stdout
        fields = _parse_result_line(result.stdout)
        assert fields.get("overall") == "OK", f"fields={fields!r}"
        assert fields.get("repos_ok") == "2", f"fields={fields!r}"
        assert fields.get("repos_failed") == "0", f"fields={fields!r}"


@pytest.mark.unit
class TestPluginCacheRefresh:
    """Tests for the plugin cache refresh logic in pull-all.sh (OMN-7369)."""

    def test_detects_versioned_cache_path(self, tmp_path: Path) -> None:
        """Refresh triggers when cache lives under cache/omninode-tools/onex/<ver>/."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(omni_home)
        cache = _make_versioned_cache(fake_home)

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0, (
            f"pull-all.sh failed: stdout={result.stdout} stderr={result.stderr}"
        )
        assert "Plugin cache refreshed" in result.stdout, (
            f"Cache refresh did not trigger; output: {result.stdout}"
        )
        # The real commit should replace the 0...0 placeholder.
        deployed = (cache / ".deployed-commit").read_text().strip()
        assert deployed != "0" * 40
        assert len(deployed) == 40

    def test_refresh_copies_all_plugin_subdirs(self, tmp_path: Path) -> None:
        """Refresh copies hooks, lib, agents — not just skills/."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(omni_home)
        cache = _make_versioned_cache(fake_home)

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0, result.stderr

        # Every subdir from the source must appear in the refreshed cache.
        assert (cache / "skills" / "example.md").exists()
        assert (cache / "hooks" / "example.sh").exists()
        assert (cache / "lib" / "example.py").exists()
        assert (cache / "agents" / "example.yaml").exists()

    def test_content_hash_written_after_refresh(self, tmp_path: Path) -> None:
        """.content-hash is computed and stored after a refresh."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(omni_home)
        cache = _make_versioned_cache(fake_home)

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0, result.stderr

        hash_file = cache / ".content-hash"
        assert hash_file.exists(), ".content-hash was not created"
        content = hash_file.read_text().strip()
        # shasum produces a 40-char hex digest.
        assert len(content) == 40
        assert all(c in "0123456789abcdef" for c in content)

    def test_deployed_commit_preserved_alongside_hash(self, tmp_path: Path) -> None:
        """Existing .deployed-commit behavior is preserved."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omniclaude = _make_omniclaude_source(omni_home)
        cache = _make_versioned_cache(fake_home)

        expected_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=omniclaude, text=True
        ).strip()

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0, result.stderr

        assert (cache / ".deployed-commit").read_text().strip() == expected_commit
        assert (cache / ".content-hash").exists()

    def test_no_cache_skips_cleanly(self, tmp_path: Path) -> None:
        """Missing plugin cache is a no-op, not an error."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        # No cache created.

        _make_omniclaude_source(omni_home)

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0, (
            f"pull-all.sh should succeed with no cache; "
            f"stdout={result.stdout} stderr={result.stderr}"
        )
        assert "Plugin cache refreshed" not in result.stdout
        assert "WARN: Plugin cache refresh failed" not in result.stdout

    def test_refresh_is_idempotent(self, tmp_path: Path) -> None:
        """Second run with no changes does not re-trigger a refresh."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(omni_home)
        _make_versioned_cache(fake_home)

        first = _run_pull_all(omni_home, fake_home)
        assert first.returncode == 0
        assert "Plugin cache refreshed" in first.stdout

        second = _run_pull_all(omni_home, fake_home)
        assert second.returncode == 0
        # On the second run, commit + content hash both match — no refresh.
        assert "Plugin cache refreshed" not in second.stdout


@pytest.mark.unit
class TestLaunchdPlist:
    """Tests for the bare-clone-sync launchd plist."""

    def test_plist_exists(self) -> None:
        assert PLIST.exists(), f"plist not found at {PLIST}"

    def test_plist_is_well_formed(self) -> None:
        content = PLIST.read_text()
        assert content.startswith("<?xml version=")
        assert '<plist version="1.0">' in content
        assert "</plist>" in content

    def test_plist_has_correct_label(self) -> None:
        content = PLIST.read_text()
        assert "ai.omninode.bare-clone-sync" in content

    def test_plist_interval_is_1800(self) -> None:
        content = PLIST.read_text()
        assert "<integer>1800</integer>" in content

    def test_plist_references_pull_all(self) -> None:
        content = PLIST.read_text()
        assert "pull-all.sh" in content


@pytest.mark.unit
class TestInstallScript:
    """Tests for install-bare-clone-sync.sh."""

    def test_install_script_exists_and_executable(self) -> None:
        assert INSTALL_SCRIPT.exists()
        assert os.access(INSTALL_SCRIPT, os.X_OK)

    def test_install_script_has_spdx_header(self) -> None:
        content = INSTALL_SCRIPT.read_text()
        assert "SPDX-License-Identifier" in content

    def test_install_script_supports_uninstall(self) -> None:
        content = INSTALL_SCRIPT.read_text()
        assert "uninstall" in content

    def test_install_script_uses_launchctl(self) -> None:
        content = INSTALL_SCRIPT.read_text()
        assert "launchctl" in content


# Minimal, hermetic pre-commit config: a single `repo: local` hook with
# `language: system` needs no network and no environment build, so
# `pre-commit install-hooks` is effectively a no-op in tests.
_PRECOMMIT_CONFIG = """repos:
  - repo: local
    hooks:
      - id: noop
        name: noop
        entry: "true"
        language: system
        pass_filenames: false
"""


@pytest.mark.unit
class TestPreCommitHookInstall:
    """pull-all.sh installs the pre-commit git hook in repos that ship a config.

    Regression guard for the OMN-14099 bypass leak: the hook was never installed
    in canonical clones, so every .pre-commit-config.yaml was decoration and CI
    became the first (most expensive) catch layer.
    """

    @pytest.fixture(autouse=True)
    def _require_precommit(self) -> None:
        if shutil.which("pre-commit") is None:
            pytest.skip("pre-commit binary not on PATH")

    @staticmethod
    def _hook_path(repo: Path) -> Path:
        return repo / ".git" / "hooks" / "pre-commit"

    def test_installs_hook_when_config_present(self, tmp_path: Path) -> None:
        """A repo with a .pre-commit-config.yaml gets a pre-commit-managed hook."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omniclaude = _make_omniclaude_source(
            omni_home,
            file_contents={".pre-commit-config.yaml": _PRECOMMIT_CONFIG},
        )
        # No hook before the run.
        assert not self._hook_path(omniclaude).exists()

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "HOOK     omniclaude" in result.stdout, result.stdout

        hook = self._hook_path(omniclaude)
        assert hook.exists(), "pre-commit hook was not written"
        assert "File generated by pre-commit" in hook.read_text(), (
            "hook file is not pre-commit-managed"
        )

    def test_second_run_is_idempotent_noop(self, tmp_path: Path) -> None:
        """Once installed, a subsequent pull-all does not reinstall the hook."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(
            omni_home,
            file_contents={".pre-commit-config.yaml": _PRECOMMIT_CONFIG},
        )

        first = _run_pull_all(omni_home, fake_home)
        assert first.returncode == 0
        assert "HOOK     omniclaude" in first.stdout

        second = _run_pull_all(omni_home, fake_home)
        assert second.returncode == 0
        # Already pre-commit-managed -> the install branch is skipped entirely.
        assert "HOOK     omniclaude" not in second.stdout, second.stdout

    def test_no_config_means_no_hook(self, tmp_path: Path) -> None:
        """A repo without a .pre-commit-config.yaml is left untouched."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omniclaude = _make_omniclaude_source(omni_home)  # defaults: no config

        result = _run_pull_all(omni_home, fake_home)
        assert result.returncode == 0
        assert "HOOK     omniclaude" not in result.stdout
        assert not self._hook_path(omniclaude).exists()


@pytest.mark.unit
class TestOmnimarketDriftRepair:
    """pull-all.sh wires check-omnimarket-venv-drift.sh --repair after an
    omnimarket pull (OMN-15242).

    Root cause this closes: the OMN-14060 pre-flight guard detects venv drift
    against the canonical omnibase_infra venv but never repairs it, and the
    canonical `git pull` on omnimarket IS the event that creates the drift.
    Two same-day 2026-07-27 incidents bricked every onex CLI consumer on this
    Mac until a human ran the repair by hand. Scope: interactive/session path
    only -- preregistered battery runs use the frozen-environment mechanism
    (OMN-15265) and must never be auto-repaired mid-run; this hook lives only
    in pull-all.sh, never in a battery driver.

    Every fixture here stubs check-omnimarket-venv-drift.sh (see
    `_make_fake_infra_with_drift_stub`) rather than using the real script --
    the real script's detect/repair correctness is covered by
    tests/scripts/test_check_omnimarket_venv_drift.py. These tests only prove
    pull-all.sh's wiring: invoked at the right time, with the right args, and
    fail-loud-but-not-fatal on repair failure.
    """

    def test_repairs_omnimarket_venv_drift_after_successful_pull(
        self, tmp_path: Path
    ) -> None:
        """A successful omnimarket pull triggers --repair against the
        canonical omnibase_infra venv, and the invocation is logged/attributable.
        """
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_simple_repo_source(omni_home, "omnimarket")
        infra_dir, calls_log = _make_fake_infra_with_drift_stub(omni_home)

        result = _run_pull_all(omni_home, fake_home, repos=["omnimarket"])

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert calls_log.exists(), (
            "check-omnimarket-venv-drift.sh was never invoked after the "
            f"omnimarket pull; stdout={result.stdout!r}"
        )
        call = calls_log.read_text()
        assert "--repair" in call
        assert str(infra_dir / ".venv" / "bin" / "python") in call
        assert f"OMNI_HOME={omni_home}" in call
        # Attributable: pull-all.sh's own stdout names the action.
        assert "omnimarket venv drift" in result.stdout.lower()

    def test_repair_failure_prints_loud_banner_and_is_fatal(
        self, tmp_path: Path
    ) -> None:
        """A repair failure prints an unmissable banner naming the manual
        command and the ticket, AND propagates a non-zero exit (OMN-15590 AC2b).

        CONTRACT CHANGE (OMN-15590): this test previously asserted
        ``returncode == 0`` -- "fail-loud-but-not-fatal". That was the defect:
        a drift-repair failure printed a banner, fell through without touching
        ``FAILED[]``, and produced an exit-0, green-looking run indistinguishable
        from a complete sync. A caller running the documented "sync first" step
        had no way to tell. The repo pull itself still succeeded, so the repo
        line stays ``OK`` -- what changed is that the STAGE failure is now
        carried into the terminal summary and the exit code.
        """
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_simple_repo_source(omni_home, "omnimarket")
        _make_fake_infra_with_drift_stub(omni_home, behavior="fail")

        result = _run_pull_all(
            omni_home, fake_home, repos=["omnimarket"], timeout=HARNESS_BACKSTOP_SECONDS
        )

        assert result.returncode != 0, (
            "a drift-repair failure must surface as a non-zero exit (AC2b); "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "OMN-15242" in result.stdout
        assert "check-omnimarket-venv-drift.sh --repair" in result.stdout
        # The repo pull itself succeeded -- only the drift STAGE failed.
        assert "OK       omnimarket" in result.stdout

    def test_skip_guard_when_omnibase_infra_absent(self, tmp_path: Path) -> None:
        """No local omnibase_infra clone at all -- clean no-op, no crash."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_simple_repo_source(omni_home, "omnimarket")
        # No omnibase_infra directory created at all.

        result = _run_pull_all(omni_home, fake_home, repos=["omnimarket"])

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert not (omni_home / "omnibase_infra").exists()

    def test_skip_guard_when_infra_venv_absent(self, tmp_path: Path) -> None:
        """omnibase_infra clone present but no canonical venv -- skip, no crash."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_simple_repo_source(omni_home, "omnimarket")
        _infra_dir, calls_log = _make_fake_infra_with_drift_stub(
            omni_home, with_venv=False
        )

        result = _run_pull_all(omni_home, fake_home, repos=["omnimarket"])

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert not calls_log.exists(), (
            "drift script must not be invoked when the canonical venv is absent"
        )

    def test_no_repair_when_omnimarket_not_in_this_run(self, tmp_path: Path) -> None:
        """omnimarket absent from the requested repo list -- never triggers,
        even though a fully-wired omnibase_infra + venv is present."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_omniclaude_source(omni_home)
        _infra_dir, calls_log = _make_fake_infra_with_drift_stub(omni_home)

        result = _run_pull_all(omni_home, fake_home, repos=["omniclaude"])

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert not calls_log.exists()

    def test_no_repair_when_omnimarket_pull_fails(self, tmp_path: Path) -> None:
        """A dirty (FAILED) omnimarket pull must not trigger a repair -- there
        is no fresh canonical SHA to repair against."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        omnimarket = _make_simple_repo_source(omni_home, "omnimarket")
        (omnimarket / "dirty.txt").write_text("uncommitted\n")
        _infra_dir, calls_log = _make_fake_infra_with_drift_stub(omni_home)

        result = _run_pull_all(omni_home, fake_home, repos=["omnimarket"])

        assert result.returncode != 0
        assert "dirty worktree" in result.stdout
        assert not calls_log.exists()


@pytest.mark.unit
class TestDriftRepairBounding:
    """The drift-repair stage is bounded, fatal on failure, orphan-free, and
    every run ends with a caller-checkable completion signal (OMN-15590).

    Field failure this closes (remote-gate-readiness run ``wf_c69db51c-74d``,
    2026-07-31, host ``stickybeatz-studio``): the stage at pull-all.sh:246 was
    invoked with no bound of any kind. It overran the caller's 3-minute
    timeout, left three orphaned bash processes behind, and never reached the
    plugin-cache / pre-commit / summary stages -- so a caller running the
    documented "sync first" step saw a partially-executed sync with NO failure
    surfaced. Its failure path was independently non-fatal: a non-zero drift
    exit printed a banner and fell through without touching ``FAILED[]``.

    ROOT CAUSE (AC6), established by controlled experiment on the gate host
    2026-08-02 -- not inferred: the repair chain terminates in
    ``uv pip install --python <canonical infra venv>``, and uv takes an
    EXCLUSIVE flock on ``<venv>/.lock`` for the whole install. uv has no
    lock-acquisition timeout and prints nothing at default verbosity while it
    waits. Holding that lock from a second process made an otherwise-2-second
    install sit silent and running past 30s, then complete in milliseconds the
    instant the lock was released. ``.200`` runs many sessions against that one
    shared canonical venv, so any concurrent uv operation (including a peer's
    own pull-all.sh, which the readiness probe itself runs) blocks this stage
    for the peer's entire duration, unbounded. Resolve/network was excluded by
    measurement on the same host: step-1 git+HTTPS resolve 1.93s, step-2 leaf
    resolve 43ms, ``git ls-remote`` 0.19s.

    Mechanism note: macOS ships neither coreutils ``timeout``/``gtimeout`` nor
    ``setsid`` (verified on the gate host), and GNU ``timeout`` would in any
    case only signal the direct child -- leaving the uv/git grandchildren
    orphaned, which is the exact field symptom. pull-all.sh therefore runs the
    stage under bash job control so it owns its own process group, and kills
    the GROUP on timeout.
    """

    # Short bound so the suite stays fast; the value is the script's declared
    # env override, which is itself part of the contract (AC1: "an explicit
    # timeout with a declared value").
    BOUND_SECONDS = 5

    def _fixture(self, tmp_path: Path) -> tuple[Path, Path]:
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        _make_simple_repo_source(omni_home, "omnimarket")
        return omni_home, fake_home

    def _env(self) -> dict[str, str]:
        return {
            "PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS": str(self.BOUND_SECONDS),
        }

    def test_ac1_hanging_drift_repair_returns_within_bound(
        self, tmp_path: Path
    ) -> None:
        """AC1 -- with the drift script stubbed to sleep far past the bound,
        pull-all.sh returns within ``bound + delta``, not indefinitely."""
        omni_home, fake_home = self._fixture(tmp_path)
        _make_fake_infra_with_drift_stub(omni_home, behavior="hang")

        started = time.monotonic()
        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env=self._env(),
            # Harness backstop only. Generous vs. the bound so a SLOW return
            # still fails on the elapsed assertion below (a real signal) rather
            # than on a TimeoutExpired (which would also be a failure, just a
            # noisier one).
            timeout=HARNESS_BACKSTOP_SECONDS,
        )
        elapsed = time.monotonic() - started

        # delta budget: the watchdog's 1s poll granularity + its TERM->KILL
        # escalation grace + the remaining stages of the script.
        assert elapsed < self.BOUND_SECONDS + 60, (
            f"drift-repair stage was not bounded: pull-all.sh took {elapsed:.1f}s "
            f"against a declared {self.BOUND_SECONDS}s bound; "
            f"stdout={result.stdout!r}"
        )

    def test_ac2a_timeout_named_in_summary_and_exits_nonzero(
        self, tmp_path: Path
    ) -> None:
        """AC2a -- a stub that sleeps past the bound: the terminal summary
        names the drift stage as failed AND the exit code is non-zero."""
        omni_home, fake_home = self._fixture(tmp_path)
        _make_fake_infra_with_drift_stub(omni_home, behavior="hang")

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env=self._env(),
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert result.returncode != 0, (
            f"a timed-out drift stage must exit non-zero; stdout={result.stdout!r}"
        )
        fields = _parse_result_line(result.stdout)
        assert fields.get("drift_repair") == "TIMEOUT", (
            f"summary must name the drift stage as timed out; fields={fields!r} "
            f"stdout={result.stdout!r}"
        )
        assert fields.get("overall") == "FAILED", f"fields={fields!r}"
        assert str(self.BOUND_SECONDS) in result.stdout, (
            "the bound's value must be stated in the output (AC1: declared value)"
        )

    def test_ac2b_nonzero_drift_exit_named_in_summary_and_exits_nonzero(
        self, tmp_path: Path
    ) -> None:
        """AC2b -- a stub that exits 1 immediately: same outcome as AC2a.

        RED against pre-fix behavior: today the banner prints and pull-all.sh
        exits 0 with the stage absent from every aggregate.
        """
        omni_home, fake_home = self._fixture(tmp_path)
        _make_fake_infra_with_drift_stub(omni_home, behavior="fail")

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env=self._env(),
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert result.returncode != 0, (
            f"a failed drift stage must exit non-zero; stdout={result.stdout!r}"
        )
        fields = _parse_result_line(result.stdout)
        assert fields.get("drift_repair") == "FAILED", (
            f"summary must name the drift stage as failed; fields={fields!r} "
            f"stdout={result.stdout!r}"
        )
        assert fields.get("overall") == "FAILED", f"fields={fields!r}"
        # The repo pull itself succeeded; the aggregate must not misreport it.
        assert fields.get("repos_failed") == "0", f"fields={fields!r}"

    def test_ac3_no_orphaned_descendants_after_timeout(self, tmp_path: Path) -> None:
        """AC3 -- after the timeout case, no descendant of the timed-out stage
        survives. Proven against a GRANDCHILD, because killing only the direct
        child is precisely the field defect (3 orphaned bash PIDs)."""
        omni_home, fake_home = self._fixture(tmp_path)
        infra_dir, _calls_log = _make_fake_infra_with_drift_stub(
            omni_home, behavior="hang_with_grandchild"
        )
        pid_file = infra_dir / ".drift-grandchild.pid"

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env=self._env(),
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert pid_file.exists(), (
            "stub never recorded its grandchild pid -- the fixture did not "
            f"exercise the orphan path; stdout={result.stdout!r}"
        )
        grandchild_pid = int(pid_file.read_text().strip())

        # Give the KILL escalation a moment to be reaped, then assert absence.
        deadline = time.monotonic() + 15
        while _pid_alive(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.5)

        alive = _pid_alive(grandchild_pid)
        if alive:  # pragma: no cover - only on regression
            # Best-effort cleanup so a failing assertion does not leak the
            # very orphan it is complaining about into the CI runner.
            with contextlib.suppress(OSError):
                os.kill(grandchild_pid, signal.SIGKILL)
        assert not alive, (
            f"grandchild pid {grandchild_pid} survived the bounded stage -- the "
            "timeout killed only the direct child, leaving an orphan (AC3)"
        )

    def test_ac4_result_line_distinguishes_clean_timeout_and_fail(
        self, tmp_path: Path
    ) -> None:
        """AC4 -- one machine-parseable completion line is emitted on EVERY
        exit path and correctly distinguishes the three run classes."""
        seen: dict[str, dict[str, str]] = {}
        for case, behavior in (
            ("clean", "ok"),
            ("timeout", "hang"),
            ("fail", "fail"),
        ):
            case_root = tmp_path / case
            case_root.mkdir()
            omni_home, fake_home = self._fixture(case_root)
            _make_fake_infra_with_drift_stub(omni_home, behavior=behavior)
            result = _run_pull_all(
                omni_home,
                fake_home,
                repos=["omnimarket"],
                extra_env=self._env(),
                timeout=HARNESS_BACKSTOP_SECONDS,
            )
            fields = _parse_result_line(result.stdout)
            assert fields, (
                f"[{case}] no {RESULT_LINE_PREFIX!r} line -- a caller cannot "
                f"determine completion without parsing prose; "
                f"stdout={result.stdout!r}"
            )
            seen[case] = fields

        assert seen["clean"]["overall"] == "OK", seen
        assert seen["clean"]["drift_repair"] == "OK", seen
        assert seen["timeout"]["drift_repair"] == "TIMEOUT", seen
        assert seen["fail"]["drift_repair"] == "FAILED", seen
        # The three are mutually distinguishable, not merely present.
        assert len({tuple(sorted(f.items())) for f in seen.values()}) == 3, seen

    def test_ac5_later_stages_reported_when_drift_stage_fails(
        self, tmp_path: Path
    ) -> None:
        """AC5 -- plugin-cache refresh and pre-commit install are not silently
        skipped when the drift stage fails: their disposition is stated."""
        for behavior in ("hang", "fail"):
            case_root = tmp_path / behavior
            case_root.mkdir()
            omni_home, fake_home = self._fixture(case_root)
            _make_fake_infra_with_drift_stub(omni_home, behavior=behavior)

            result = _run_pull_all(
                omni_home,
                fake_home,
                repos=["omnimarket"],
                extra_env=self._env(),
                timeout=HARNESS_BACKSTOP_SECONDS,
            )
            fields = _parse_result_line(result.stdout)
            for stage in ("plugin_cache", "precommit_hooks"):
                assert stage in fields, (
                    f"[{behavior}] stage {stage!r} absent from the completion "
                    f"line -- silently skipped; fields={fields!r}"
                )
                assert fields[stage] != "PENDING", (
                    f"[{behavior}] stage {stage!r} never ran after the drift "
                    f"stage failed (silent skip); fields={fields!r}"
                )

    def test_clean_run_still_exits_zero_with_ok_result_line(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: the new fatality must not make healthy runs red."""
        omni_home, fake_home = self._fixture(tmp_path)
        _make_fake_infra_with_drift_stub(omni_home, behavior="ok")

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env=self._env(),
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        fields = _parse_result_line(result.stdout)
        assert fields.get("overall") == "OK", f"fields={fields!r}"
        assert fields.get("drift_repair") == "OK", f"fields={fields!r}"

    def test_bound_default_is_declared_in_the_script(self) -> None:
        """AC1 -- the bound is an explicit, declared value in the script, not
        an implicit or caller-supplied one."""
        source = PULL_ALL.read_text()
        assert "PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS" in source, (
            "the bound must be a named, overridable, declared value"
        )
        match = re.search(
            r"PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS:-(\d+)\}",
            source,
        )
        assert match, "no declared default for the drift-repair bound"
        assert int(match.group(1)) > 0

    def test_plugin_content_hash_uses_batched_exec(self) -> None:
        r"""Regression guard for the SECOND unbounded stall found while proving
        this ticket (OMN-15590).

        ``_plugin_content_hash`` used ``find ... -exec shasum {} \;`` -- one
        ``shasum`` (perl) process per file. The live plugin cache on the gate
        host holds 53,057 files, so that form forked ~53k interpreters and the
        stage ran past 10 minutes without finishing, wedging pull-all.sh after
        the drift stage. The batched ``+`` form hashes the same tree in 7.9s
        and produces identical output. Reintroducing ``\;`` silently restores
        a multi-minute-to-unbounded stage, so it is asserted mechanically.
        """
        source = PULL_ALL.read_text()
        assert '-exec "$hasher" {} +' in source, (
            "plugin content hash must batch (`-exec ... {} +`)"
        )
        assert r'-exec "$hasher" {} \;' not in source, (
            r"the per-file `-exec ... {} \;` form spawns one process per "
            "file (53k on the gate host) and stalls the plugin-cache stage"
        )

    def test_every_unbounded_call_site_has_a_declared_bound(self) -> None:
        """All three long-running / network-bound stages carry an explicit,
        named, overridable bound -- not just the one the ticket names."""
        source = PULL_ALL.read_text()
        for var in (
            "PULL_ALL_DRIFT_REPAIR_TIMEOUT_SECONDS",
            "PULL_ALL_HOOK_ENV_TIMEOUT_SECONDS",
            "PULL_ALL_PLUGIN_HASH_TIMEOUT_SECONDS",
        ):
            assert re.search(rf"{var}:-\d+\}}", source), (
                f"{var} has no declared default bound"
            )

    def test_precommit_stage_reports_warn_when_install_fails(
        self, tmp_path: Path
    ) -> None:
        """The completion line must not report ``precommit_hooks=OK`` on a run
        whose hook installs failed (OMN-15590).

        Found on the gate host: a real run printed seven
        ``WARN <repo> (pre-commit install failed ...)`` lines and still
        summarised the stage as OK, because the install runs in a subshell that
        cannot assign to the parent's stage variable. That is the same
        "green-looking incomplete run" defect this ticket closes, one stage
        over -- so it is asserted, not assumed.
        """
        omni_home, fake_home = self._fixture(tmp_path)
        _make_fake_infra_with_drift_stub(omni_home, behavior="ok")
        # omnimarket ships no .pre-commit-config.yaml in the fixture; add one so
        # the stage has something to act on. It must be COMMITTED (an
        # uncommitted file trips the dirty-worktree refusal) and it must be on
        # DEV (pull-all.sh leaves every repo on dev, and the hook loop runs
        # after that switch -- a config committed only on main is invisible).
        repo = omni_home / "omnimarket"
        _git(["switch", "-q", "dev"], cwd=repo)
        _commit_file(repo, ".pre-commit-config.yaml", "repos: []\n", "pre-commit cfg")
        _git(["push", "-q", "origin", "dev"], cwd=repo)
        _git(["switch", "-q", "main"], cwd=repo)

        # PATH shim: a `pre-commit` that exists (so the stage is not skipped as
        # UNAVAILABLE) but fails on `install`.
        shim_dir = tmp_path / "shim"
        shim_dir.mkdir()
        shim = shim_dir / "pre-commit"
        shim.write_text("#!/usr/bin/env bash\nexit 1\n")
        shim.chmod(0o755)

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env={
                **self._env(),
                "PATH": f"{shim_dir}:{os.environ.get('PATH', '')}",
            },
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert "pre-commit install failed" in result.stdout, (
            f"fixture did not exercise the failure path; stdout={result.stdout!r}"
        )
        fields = _parse_result_line(result.stdout)
        assert fields.get("precommit_hooks") == "WARN", (
            "a run with failed hook installs must not summarise the stage as "
            f"OK; fields={fields!r}"
        )
        # Still non-fatal: hook install is a convenience layer, not the sync's
        # core job. Visibility changed, fatality did not.
        assert result.returncode == 0, f"stdout={result.stdout!r}"


def _make_diverged_main_repo(omni_home: Path, name: str) -> Path:
    """A repo whose local main can never fast-forward again (OMN-16500).

    Models the release-synced-main shape: local main holds an unpushed
    "promotion" commit while origin/main was rewritten (force-pushed) to a
    different history. Left checked out on dev, clean tree.
    """
    repo = _make_simple_repo_source(omni_home, name)
    # local promotion commit on main -- never pushed, orphaned by the rewrite
    _commit_file(repo, "promotion.txt", "promotion\n", "promotion commit")
    # rewrite remote main to a history that does not contain that commit
    _git(["switch", "-q", "dev"], cwd=repo)
    _commit_file(repo, "dev-advance.txt", "dev\n", "dev advance")
    _git(["push", "-q", "origin", "dev"], cwd=repo)
    _git(["push", "-q", "-f", "origin", "dev:main"], cwd=repo)
    return repo


def _make_converge_stub(omni_home: Path, *, behavior: str = "ok") -> tuple[Path, Path]:
    """Stub `$OMNI_HOME/omniclaude/scripts/converge-canonical-clone.sh`.

    Never the real script -- its own behavior is covered by omniclaude's
    tests/scripts/test_converge_canonical_clone.py; these tests prove only
    pull-all.sh's WIRING (invoked at the right time, right args, failure
    handled). The "ok" stub performs the one ref move the real script would.
    Returns (stub_path, calls_log).
    """
    scripts_dir = omni_home / "omniclaude" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    calls_log = omni_home / "omniclaude" / ".converge-calls.log"
    stub = scripts_dir / "converge-canonical-clone.sh"
    if behavior == "ok":
        body = 'git -C "$OMNI_HOME/$1" branch -f main origin/main\n'
    else:
        body = "exit 1\n"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "invoked_as=$0 | $@ | OMNI_HOME=$OMNI_HOME" >> "{calls_log}"\n' + body
    )
    stub.chmod(0o755)
    return stub, calls_log


@pytest.mark.unit
class TestMainConvergeWiring:
    """pull-all.sh routes a non-fast-forwardable main through the sanctioned
    convergence script and continues to the dev pull (OMN-16500).

    Field failure this closes: under the release-synced-main policy,
    origin/main is rewritten to the release tag on every release, so the local
    main of a canonical clone -- still carrying the pre-rewrite promotion
    commits -- can NEVER fast-forward again. pull-all.sh failed at
    "fast-forward main" in 9 of 12 registry clones (verified 2026-08-24) and
    returned before pulling dev, breaking the documented "sync first" step.
    main is a release-pointer branch never worked on locally, so converging is
    always correct -- but the ref move must go through
    converge-canonical-clone.sh --branch (canonical-clone guard, OMN-16496),
    which preserves the orphaned commits first. dev stays strictly ff-only.
    """

    def test_nonff_main_is_converged_and_run_continues_to_dev(
        self, tmp_path: Path
    ) -> None:
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        repo = _make_diverged_main_repo(omni_home, "omnimarket")
        _stub, calls_log = _make_converge_stub(omni_home)
        _make_fake_infra_with_drift_stub(omni_home)

        result = _run_pull_all(
            omni_home, fake_home, repos=["omnimarket"], timeout=HARNESS_BACKSTOP_SECONDS
        )

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert calls_log.exists(), (
            f"converge script never invoked; stdout={result.stdout!r}"
        )
        call = calls_log.read_text()
        assert "omnimarket --branch main --execute" in call
        assert f"OMNI_HOME={omni_home}" in call
        # Snapshot invocation (OMN-16500 race fix): pull-all itself switches
        # the canonical omniclaude clone between main (the release pointer,
        # which can predate or lack the script) and dev while the parallel
        # pulls run, so the script must be invoked from a run-start snapshot,
        # never through the omniclaude working tree mid-run. The 2026-08-24
        # proof run lost omnimemory and omnibase_spi to exactly this race.
        invoked_as = call.split("invoked_as=", 1)[1].split(" | ", 1)[0]
        assert not invoked_as.startswith(str(omni_home / "omniclaude")), (
            f"converge script was invoked through the omniclaude working tree "
            f"({invoked_as}) instead of a run-start snapshot -- racy against "
            f"pull-all's own omniclaude branch switching"
        )

        # main converged, dev pulled, repo left on dev, OK line says what happened
        main_sha = subprocess.check_output(
            ["git", "rev-parse", "main"], cwd=repo, text=True
        ).strip()
        origin_main_sha = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=repo, text=True
        ).strip()
        assert main_sha == origin_main_sha
        assert (
            subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=repo, text=True
            ).strip()
            == "dev"
        )
        assert "OK       omnimarket" in result.stdout
        assert "main converged" in result.stdout
        fields = _parse_result_line(result.stdout)
        assert fields.get("overall") == "OK", f"fields={fields!r}"
        assert fields.get("repos_failed") == "0", f"fields={fields!r}"

    def test_converge_failure_fails_the_repo(self, tmp_path: Path) -> None:
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_diverged_main_repo(omni_home, "omnimarket")
        _make_converge_stub(omni_home, behavior="fail")

        result = _run_pull_all(
            omni_home, fake_home, repos=["omnimarket"], timeout=HARNESS_BACKSTOP_SECONDS
        )

        assert result.returncode != 0, (
            f"a failed convergence must fail the run; stdout={result.stdout!r}"
        )
        assert "FAILED   omnimarket" in result.stdout
        assert "converge-canonical-clone.sh" in result.stdout

    def test_missing_converge_script_fails_with_actionable_path(
        self, tmp_path: Path
    ) -> None:
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _make_diverged_main_repo(omni_home, "omnimarket")
        # no omniclaude clone at all -> no sanctioned script to route through

        result = _run_pull_all(
            omni_home, fake_home, repos=["omnimarket"], timeout=HARNESS_BACKSTOP_SECONDS
        )

        assert result.returncode != 0
        assert "FAILED   omnimarket" in result.stdout
        assert "converge-canonical-clone.sh" in result.stdout, (
            f"failure must name the missing sanctioned script; stdout={result.stdout!r}"
        )

    def test_nonff_dev_stays_strict_and_is_never_converged(
        self, tmp_path: Path
    ) -> None:
        """A dev that cannot fast-forward is a real problem -- pull-all must
        FAIL it, not converge it. Only main is a release pointer."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        repo = _make_simple_repo_source(omni_home, "omnimarket")
        upstream = omni_home / "omnimarket.git"
        _stub, calls_log = _make_converge_stub(omni_home)

        # local dev gains an unpushed commit; origin/dev advances independently
        _git(["switch", "-q", "dev"], cwd=repo)
        _commit_file(repo, "local-dev.txt", "local\n", "local dev commit")
        writer = tmp_path / "writer"
        _git(["clone", "-q", str(upstream), str(writer)], cwd=tmp_path)
        _git(["switch", "-q", "--track", "-c", "dev", "origin/dev"], cwd=writer)
        _commit_file(writer, "remote-dev.txt", "remote\n", "remote dev commit")
        _git(["push", "-q", "origin", "dev"], cwd=writer)

        local_dev_before = subprocess.check_output(
            ["git", "rev-parse", "dev"], cwd=repo, text=True
        ).strip()

        result = _run_pull_all(
            omni_home, fake_home, repos=["omnimarket"], timeout=HARNESS_BACKSTOP_SECONDS
        )

        assert result.returncode != 0, (
            f"a diverged dev must fail the run; stdout={result.stdout!r}"
        )
        assert "FAILED   omnimarket" in result.stdout
        assert "fast-forward dev" in result.stdout
        # the converge script must never have been aimed at dev
        if calls_log.exists():
            assert "--branch dev" not in calls_log.read_text()
        # and local dev must be exactly where the user left it
        local_dev_after = subprocess.check_output(
            ["git", "rev-parse", "dev"], cwd=repo, text=True
        ).strip()
        assert local_dev_after == local_dev_before


@pytest.mark.unit
class TestGuardManagedHooksPath:
    """The pre-commit stage must not report guard-managed clones as broken
    (OMN-16500).

    On this fleet, canonical clones set ``core.hooksPath`` to the
    canonical-clone guard directory; the guard CHAINS to the installed hook or
    invokes ``pre-commit hook-impl`` itself (OMN-15071), so commit-time
    enforcement is ACTIVE. ``pre-commit install`` refuses to write hook files
    while ``core.hooksPath`` is set, so attempting it produced a false
    ``WARN <repo> (pre-commit install failed -- commit-time enforcement
    inactive)`` and downgraded the stage.
    """

    GUARD_MARKER = "Canonical-clone worktree-discipline guard"

    def _repo_with_config_on_dev(self, omni_home: Path, name: str) -> Path:
        repo = _make_simple_repo_source(omni_home, name)
        _git(["switch", "-q", "dev"], cwd=repo)
        _commit_file(repo, ".pre-commit-config.yaml", "repos: []\n", "cfg")
        _git(["push", "-q", "origin", "dev"], cwd=repo)
        _git(["switch", "-q", "main"], cwd=repo)
        return repo

    def _install_guard_hooks_path(self, omni_home: Path, repo: Path) -> None:
        guard_dir = omni_home / "scripts" / "git-hooks" / "canonical-clone"
        guard_dir.mkdir(parents=True, exist_ok=True)
        hook = guard_dir / "pre-commit"
        hook.write_text(
            "#!/usr/bin/env bash\n"
            f"# {self.GUARD_MARKER}, chained into the real hook chain\n"
            "exit 0\n"
        )
        hook.chmod(0o755)
        _git(["config", "core.hooksPath", str(guard_dir)], cwd=repo)

    def test_guard_managed_repo_is_not_warned_and_stage_stays_ok(
        self, tmp_path: Path
    ) -> None:
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        repo = self._repo_with_config_on_dev(omni_home, "omnimarket")
        self._install_guard_hooks_path(omni_home, repo)
        _make_fake_infra_with_drift_stub(omni_home)

        # a pre-commit on PATH that FAILS on install proves the stage never
        # attempts an install against a guard-managed clone
        shim_dir = tmp_path / "shim"
        shim_dir.mkdir()
        shim = shim_dir / "pre-commit"
        shim.write_text("#!/usr/bin/env bash\nexit 1\n")
        shim.chmod(0o755)

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env={"PATH": f"{shim_dir}:{os.environ.get('PATH', '')}"},
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert result.returncode == 0, (
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "pre-commit install failed" not in result.stdout, (
            f"guard-managed clone falsely warned; stdout={result.stdout!r}"
        )
        fields = _parse_result_line(result.stdout)
        assert fields.get("precommit_hooks") == "OK", f"fields={fields!r}"

    def test_non_guard_hookspath_still_warns(self, tmp_path: Path) -> None:
        """An arbitrary core.hooksPath (no guard, no chain) keeps the WARN --
        there enforcement genuinely is inactive."""
        omni_home = tmp_path / "omni_home"
        omni_home.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        repo = self._repo_with_config_on_dev(omni_home, "omnimarket")
        other_dir = tmp_path / "other-hooks"
        other_dir.mkdir()
        (other_dir / "pre-commit").write_text("#!/usr/bin/env bash\nexit 0\n")
        (other_dir / "pre-commit").chmod(0o755)
        _git(["config", "core.hooksPath", str(other_dir)], cwd=repo)
        _make_fake_infra_with_drift_stub(omni_home)

        shim_dir = tmp_path / "shim"
        shim_dir.mkdir()
        shim = shim_dir / "pre-commit"
        shim.write_text("#!/usr/bin/env bash\nexit 1\n")
        shim.chmod(0o755)

        result = _run_pull_all(
            omni_home,
            fake_home,
            repos=["omnimarket"],
            extra_env={"PATH": f"{shim_dir}:{os.environ.get('PATH', '')}"},
            timeout=HARNESS_BACKSTOP_SECONDS,
        )

        assert "pre-commit install failed" in result.stdout, (
            f"non-guard hooksPath must keep warning; stdout={result.stdout!r}"
        )
        fields = _parse_result_line(result.stdout)
        assert fields.get("precommit_hooks") == "WARN", f"fields={fields!r}"
