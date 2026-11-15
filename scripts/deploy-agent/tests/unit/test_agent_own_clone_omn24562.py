# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent resets only its OWN deploy-source clone (OMN-24562).

On h201 the dev agent's ``git reset --hard <pin>`` ran against
the host's canonical clone, which then sat
32 commits ahead of dev and was refused by the clone-sync unit and the
remote-lane runner. Every test builds real git repositories; the agent code runs
in a child process configured the way a unit configures it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod

AGENT_DIR = Path(__file__).resolve().parents[2]


class _HandlerModule:
    """The handler module, loaded on first use: a checkout without it fails per test."""

    def __getattr__(self, name: str) -> Any:
        return getattr(
            executor_mod.load_dev_boot_guard("handler_deploy_source_clone"), name
        )


dsc = _HandlerModule()


def _ensure(settings: object) -> Any:
    """Create when absent. ``apply_request`` is not the attribute conftest stubs."""
    return dsc.apply_request(settings)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(
        repo,
        "-c", "user.email=t@example.invalid", "-c", "user.name=t",
        "commit", "-q", "-m", name,
    )  # fmt: skip
    return _git(repo, "rev-parse", "HEAD")


class _World:
    """A bare origin, and a canonical clone of it on dev."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.origin = root / "origin.git"
        self.canonical = root / "registry" / "omnibase_infra"
        self.agent_clone = root / "state" / "deploy-source" / "omnibase_infra"
        seed = root / "seed"
        subprocess.run(["git", "init", "-q", "-b", "dev", str(seed)], check=True)
        self.dev_sha = _commit(seed, "one")
        pinned_branch = "pin-work"
        _git(seed, "checkout", "-q", "-b", pinned_branch)
        self.pinned_sha = _commit(seed, "pinned")
        _git(seed, "checkout", "-q", "dev")
        subprocess.run(
            ["git", "clone", "-q", "--bare", str(seed), str(self.origin)], check=True
        )
        self.canonical.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "clone", "-q", "-b", "dev", str(self.origin), str(self.canonical)],
            check=True,
        )

    def settings(self, mirror_url: str = "", served_sha: str = "") -> object:
        return dsc.ModelDeploySourceClone(
            operation="ensure",
            clone_dir=str(self.agent_clone),
            canonical_clone=str(self.canonical),
            mirror_url=mirror_url,
            tracking_ref="dev",
            served_sha=served_sha,
            lab_overlay_source=str(self.root / "overlay-source"),
        )

    def canonical_snapshot(self) -> tuple[str, str, str, str]:
        return (
            _git(self.canonical, "rev-parse", "HEAD"),
            _git(self.canonical, "symbolic-ref", "--short", "HEAD"),
            _git(self.canonical, "status", "--porcelain=v1", "--ignored"),
            (self.canonical / "one").read_text(encoding="utf-8"),
        )


@pytest.fixture
def world(tmp_path: Path) -> _World:
    return _World(tmp_path)


def _pinned_job(world: _World, repo_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a pinned job's git phase in a fresh agent process, as a unit starts it."""
    code = textwrap.dedent(
        f"""
        from deploy_agent import executor
        executor.load_dev_boot_guard("handler_deploy_source_clone").ensure_clone(
            executor.deploy_source(), served_sha={world.dev_sha!r})  # start-up creates the agent's own clone
        from deploy_agent.events import EnumRuntimeLane
        from deploy_agent.executor import DeployExecutor
        sha = DeployExecutor().git_pull(
            {world.pinned_sha!r}, lane=EnumRuntimeLane.DEV,
            on_phase_update=lambda phase, status: None)
        print("PULLED", sha)
        """
    )
    env = {
        "PATH": __import__("os").environ["PATH"],
        "PYTHONPATH": str(AGENT_DIR),
        "DEPLOY_AGENT_REPO_DIR": str(repo_dir),
        "DEPLOY_AGENT_CANONICAL_CLONE": str(world.canonical),
        "DEPLOY_AGENT_LAB_OVERLAY_SOURCE": str(world.root / "overlay-source"),
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        "ONEX_LANE_LOCK_DIR": str(world.root / "locks"),
        "HOME": str(world.root),
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(world.root),
        capture_output=True,
        text=True,
        check=False,
    )


class TestPinnedJobLeavesCanonicalCloneAlone:
    def test_canonical_clone_is_byte_identical_and_agent_clone_sits_on_the_pin(
        self, world: _World
    ) -> None:
        before = world.canonical_snapshot()
        result = _pinned_job(world, world.agent_clone)
        assert result.returncode == 0, result.stderr
        assert f"PULLED {world.pinned_sha}" in result.stdout
        assert _git(world.agent_clone, "rev-parse", "HEAD") == world.pinned_sha
        assert world.canonical_snapshot() == before
        assert before[0] == world.dev_sha

    def test_a_deploy_source_that_is_the_canonical_clone_runs_nothing(
        self, world: _World
    ) -> None:
        """The pre-change configuration (the clone named as the deploy source)."""
        before = world.canonical_snapshot()
        result = _pinned_job(world, world.canonical)
        assert result.returncode != 0
        assert "DeploySourceIsCanonicalCloneError" in result.stderr
        assert world.canonical_snapshot() == before


class TestMigrationOnStart:
    def test_first_start_creates_the_clone_at_the_served_commit(
        self, world: _World
    ) -> None:
        assert _ensure(world.settings(served_sha=world.dev_sha)).created is True
        assert _git(world.agent_clone, "rev-parse", "HEAD") == world.dev_sha
        assert _git(world.agent_clone, "symbolic-ref", "--short", "HEAD") == "dev"

    def test_second_start_leaves_the_clone_and_its_pin_alone(
        self, world: _World
    ) -> None:
        assert _pinned_job(world, world.agent_clone).returncode == 0
        assert _ensure(world.settings()).created is False
        assert _git(world.agent_clone, "rev-parse", "HEAD") == world.pinned_sha


class TestMirror:
    def test_unreachable_mirror_falls_back_to_origin_and_counts(
        self, world: _World, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = world.settings(mirror_url=str(world.root / "no-such-mirror.git"))
        with caplog.at_level("WARNING"):
            assert _ensure(settings).created is True
        assert _git(world.agent_clone, "remote", "get-url", "origin") == str(
            world.origin
        )
        assert dsc.read_mirror_fallback_count(world.agent_clone) == 1
        count_file = world.agent_clone.parent / dsc.MIRROR_FALLBACK_COUNT_FILE
        assert count_file.read_text(encoding="utf-8").strip() == "1"
        assert "fallback to origin" in caplog.text

    def test_reachable_mirror_is_the_fetch_url_and_github_stays_the_push_url(
        self, world: _World
    ) -> None:
        mirror = world.root / "mirror.git"
        subprocess.run(
            ["git", "clone", "-q", "--bare", str(world.origin), str(mirror)], check=True
        )
        assert _ensure(world.settings(str(mirror))).created is True
        assert _git(world.agent_clone, "remote", "get-url", "origin") == str(mirror)
        assert _git(world.agent_clone, "remote", "get-url", "--push", "origin") == str(
            world.origin
        )
        assert dsc.read_mirror_fallback_count(world.agent_clone) == 0


def _resolve(env: dict[str, str]) -> object:
    return dsc.apply_request(dsc.settings(env))


def _env(world: _World, repo_dir: Path, **extra: str) -> dict[str, str]:
    return {
        "DEPLOY_AGENT_REPO_DIR": str(repo_dir),
        "DEPLOY_AGENT_CANONICAL_CLONE": str(world.canonical),
        "DEPLOY_AGENT_LAB_OVERLAY_SOURCE": str(world.root / "overlay-source"),
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        **extra,
    }


class TestGuard:
    def test_a_path_equal_to_the_canonical_clone_is_refused(
        self, world: _World
    ) -> None:
        with pytest.raises(dsc.DeploySourceIsCanonicalCloneError):
            _resolve(_env(world, world.canonical))

    def test_a_symlink_to_the_canonical_clone_is_refused(
        self, world: _World, tmp_path: Path
    ) -> None:
        link = tmp_path / "link"
        link.symlink_to(world.canonical)
        with pytest.raises(dsc.DeploySourceIsCanonicalCloneError):
            _resolve(_env(world, link))

    def test_the_registry_clone_is_canonical_too(self, world: _World) -> None:
        other = world.root / "elsewhere" / "omnibase_infra"
        env = _env(world, world.canonical, OMNI_HOME=str(world.canonical.parent))
        env["DEPLOY_AGENT_CANONICAL_CLONE"] = str(other)
        with pytest.raises(dsc.DeploySourceIsCanonicalCloneError):
            _resolve(env)

    def test_unset_settings_have_no_default(self) -> None:
        with pytest.raises(dsc.DeploySourceCloneError):
            dsc.settings({})

    def test_an_own_clone_path_is_accepted(self, world: _World) -> None:
        settings = _resolve(_env(world, world.agent_clone))
        assert settings.clone_dir == str(world.agent_clone)
