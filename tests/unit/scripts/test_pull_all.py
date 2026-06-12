# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for pull-all.sh and bare-clone-sync infrastructure."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
PULL_ALL = SCRIPTS_DIR / "pull-all.sh"
PLIST = SCRIPTS_DIR / "ai.omninode.bare-clone-sync.plist"
INSTALL_SCRIPT = SCRIPTS_DIR / "install-bare-clone-sync.sh"


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
    # cache-refresh logic we are testing).
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"], cwd=omniclaude, check=True
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=omniclaude,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        cwd=omniclaude,
        check=True,
    )

    upstream = root / "omniclaude.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", "--initial-branch=main", str(upstream)],
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(upstream)], cwd=omniclaude, check=True
    )
    subprocess.run(
        ["git", "push", "-q", "-u", "origin", "main"], cwd=omniclaude, check=True
    )
    subprocess.run(["git", "switch", "-q", "-c", "dev"], cwd=omniclaude, check=True)
    subprocess.run(
        ["git", "push", "-q", "-u", "origin", "dev"], cwd=omniclaude, check=True
    )
    subprocess.run(["git", "switch", "-q", "main"], cwd=omniclaude, check=True)
    return omniclaude


def _commit_file(repo: Path, rel_path: str, contents: str, message: str) -> None:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    subprocess.run(["git", "add", rel_path], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
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


def _run_pull_all(
    omni_home: Path, fake_home: Path, repos: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke pull-all.sh with controlled OMNI_HOME and HOME."""
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "HOME": str(fake_home),
        "LANG": "C",
        "LC_ALL": "C",
        "LC_CTYPE": "C",
    }
    # Drop CLAUDE_PLUGIN_ROOT so the auto-detection path is exercised.
    env.pop("CLAUDE_PLUGIN_ROOT", None)
    args = ["bash", str(PULL_ALL), *(repos or ["omniclaude"])]
    return subprocess.run(args, capture_output=True, text=True, env=env, check=False)


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
        subprocess.run(["git", "clone", "-q", str(upstream), str(writer)], check=True)

        _commit_file(writer, "main-only.txt", "main\n", "main update")
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=writer, check=True)
        subprocess.run(["git", "switch", "-q", "dev"], cwd=writer, check=True)
        _commit_file(writer, "dev-only.txt", "dev\n", "dev update")
        subprocess.run(["git", "push", "-q", "origin", "dev"], cwd=writer, check=True)

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
                env={**os.environ, "OMNI_HOME": tmpdir},
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
            )

            result = subprocess.run(
                ["bash", str(PULL_ALL), repo_name],
                capture_output=True,
                text=True,
                env={**os.environ, "OMNI_HOME": tmpdir},
                check=False,
            )
            # Script should complete (exit 0 or 1) without crashing.
            # The bare repo has no main ref and no remote, so it will
            # report FAILED — that's expected. The key is no crash.
            assert result.returncode in (0, 1)


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
