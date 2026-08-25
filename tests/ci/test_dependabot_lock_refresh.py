# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the Dependabot runner-image lock auto-regen (OMN-16553).

Dependabot bumps pyproject.toml/uv.lock but can never run repo scripts, so it
can never regenerate docker/runners/runner-image.lock.json itself -- this
module is the bot remediation step. Two things matter most:

* the same-repo/dependabot-branch guard never lets a push target anything
  else (the whole point of holding a repo-write App token), and
* the regen logic actually detects and fixes drift against a real git
  history, using ONLY the two PR-authored manifest files from the PR ref and
  everything else from the trusted checkout -- proven against a real local
  git repository fixture, not mocked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
MODULE_PATH = SCRIPTS_CI / "dependabot_lock_refresh.py"
INCIDENT_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn16553"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "dependabot_lock_refresh", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


# --------------------------------------------------------------------------- #
# is_same_repo_dependabot_branch: the guard that gates every push             #
# --------------------------------------------------------------------------- #


def test_same_repo_dependabot_branch_accepted() -> None:
    module = _load_module()
    pr = {
        "headRefName": "dependabot/pip/kafka-python-gte-2.3.2-and-lt-4.0.0",
        "headRepository": {"name": "omnibase_infra"},
        "headRepositoryOwner": {"login": "OmniNode-ai"},
    }
    assert module.is_same_repo_dependabot_branch(pr, "OmniNode-ai/omnibase_infra")


def test_fork_pr_rejected_even_if_branch_name_matches() -> None:
    module = _load_module()
    pr = {
        "headRefName": "dependabot/pip/kafka-python-gte-2.3.2-and-lt-4.0.0",
        "headRepository": {"name": "omnibase_infra"},
        "headRepositoryOwner": {"login": "some-fork-owner"},
    }
    assert not module.is_same_repo_dependabot_branch(pr, "OmniNode-ai/omnibase_infra")


def test_non_dependabot_branch_rejected_even_if_same_repo() -> None:
    module = _load_module()
    pr = {
        "headRefName": "jonah/omn-99999-unrelated-feature",
        "headRepository": {"name": "omnibase_infra"},
        "headRepositoryOwner": {"login": "OmniNode-ai"},
    }
    assert not module.is_same_repo_dependabot_branch(pr, "OmniNode-ai/omnibase_infra")


def test_missing_head_repo_fields_rejected() -> None:
    module = _load_module()
    pr = {
        "headRefName": "dependabot/pip/foo",
        "headRepository": {},
        "headRepositoryOwner": {},
    }
    assert not module.is_same_repo_dependabot_branch(pr, "OmniNode-ai/omnibase_infra")


# --------------------------------------------------------------------------- #
# regenerate_for_ref: proven against a real local git repo fixture            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def git_fixture(tmp_path: Path) -> Path:
    """A minimal local git repo shaped like the real trust boundary.

    ``base`` carries the trusted shared-env inputs + a starting lock file;
    ``dependabot/pip/example`` (the "PR" branch) carries only a
    pyproject.toml byte change -- exactly what a real Dependabot bump does.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "base", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)

    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'x'\nversion = '1'\n", encoding="utf-8"
    )
    (repo / "uv.lock").write_text("# lock v1\n", encoding="utf-8")
    (repo / ".github" / "actions" / "setup-python-uv").mkdir(parents=True)
    (repo / ".github" / "actions" / "setup-python-uv" / "action.yml").write_text(
        "name: setup\n", encoding="utf-8"
    )
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "scripts" / "ci" / "ci_env_digest.py").write_text(
        "# digest script\n", encoding="utf-8"
    )
    (repo / "scripts" / "ci" / "ensure_ci_env.sh").write_text(
        "#!/bin/sh\n", encoding="utf-8"
    )

    module = _load_module()
    lock_dir = repo / "docker" / "runners"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "runner-image.lock.json"
    lock_path.write_text(
        '{"base_image_digest": "sha256:' + "0" * 64 + '", "gh_version": "1.0", '
        '"image_version": 1, "kubectl_version": "1.0", "python_version": "3.12", '
        '"runner_version": "1.0", "shared_env_install_args": "--frozen", '
        '"uv_version": "0.6.14", "shared_env_digest": "", "identity_digest": ""}\n',
        encoding="utf-8",
    )
    # Bring the fixture lock into a genuinely self-consistent starting state,
    # exercising the real generate_lock path rather than hand-computing digests.
    module.runner_image_identity.generate_lock(repo, lock_path)

    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    _git("checkout", "-q", "-b", "dependabot/pip/example", cwd=repo)
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'x'\nversion = '2'\n", encoding="utf-8"
    )
    _git("commit", "-q", "-am", "bump", cwd=repo)
    _git("checkout", "-q", "base", cwd=repo)

    # regenerate_for_ref reads the PR branch via `origin/<ref>` -- give the
    # repo a same-directory "origin" remote pointing at itself so `git fetch
    # origin ...` and `git show origin/<ref>:...` resolve exactly as they do
    # against the real GitHub remote in CI.
    _git("remote", "add", "origin", str(repo), cwd=repo)
    return repo


def test_regenerate_for_ref_detects_manifest_drift(git_fixture: Path) -> None:
    module = _load_module()
    new_lock = module.regenerate_for_ref(git_fixture, "dependabot/pip/example")
    assert new_lock is not None
    parsed = json.loads(new_lock)
    old = json.loads(
        (git_fixture / "docker" / "runners" / "runner-image.lock.json").read_text()
    )
    assert parsed["shared_env_digest"] != old["shared_env_digest"]
    assert parsed["identity_digest"] != old["identity_digest"]
    # Non-manifest fields are preserved verbatim from the PR branch's lock file.
    assert parsed["image_version"] == old["image_version"]
    assert parsed["python_version"] == old["python_version"]


def test_regenerate_for_ref_is_none_when_manifest_unchanged(git_fixture: Path) -> None:
    module = _load_module()
    _git("branch", "dependabot/pip/no-op", "base", cwd=git_fixture)
    assert module.regenerate_for_ref(git_fixture, "dependabot/pip/no-op") is None


def test_regenerate_for_ref_never_mutates_the_calling_checkout(
    git_fixture: Path,
) -> None:
    """regenerate_for_ref must never move the caller's own HEAD/working tree."""
    module = _load_module()
    before_head = _git("rev-parse", "HEAD", cwd=git_fixture).stdout.strip()
    before_status = _git("status", "--porcelain", cwd=git_fixture).stdout
    module.regenerate_for_ref(git_fixture, "dependabot/pip/example")
    after_head = _git("rev-parse", "HEAD", cwd=git_fixture).stdout.strip()
    after_status = _git("status", "--porcelain", cwd=git_fixture).stdout
    assert before_head == after_head
    assert before_status == after_status


# --------------------------------------------------------------------------- #
# Incident replay (OMN-16553, tests/incident_replays/registry.yaml R1-R5):    #
# the real bytes that broke Dependabot PR #2883, live 2026-08-25.             #
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_omn16553_replay_detects_real_pr2883_manifest_drift(tmp_path: Path) -> None:
    """The exact bytes that broke PR #2883's runner-image-build-smoke run.

    ``pr2883-{pyproject.toml,uv.lock,runner-image.lock.json}.captured`` are
    the verbatim ``git show`` output of those three files at PR #2883's real
    head commit (``edc32c1fa41e25cf0ef9aef07273e1133beb3cad``,
    ``dependabot/pip/kafka-python-gte-2.3.2-and-lt-4.0.0``), which failed
    live with "runner image shared_env_digest is stale
    (recorded='8c8cf47e7d5c90a88a258f46',
    recomputed='d4474468b118409953224dca')" (job 97701790777, run
    32815140802). Staged as a real ``dependabot/*`` branch, ``regenerate_for_ref``
    must detect the drift (R5, false_green direction: a guard that wrongly
    reported this real state as "already fresh" would have left the PR red
    forever with no automated fix).
    """
    pyproject = INCIDENT_FIXTURES / "pr2883-pyproject.toml.captured"
    uv_lock = INCIDENT_FIXTURES / "pr2883-uv.lock.captured"
    lock_json = INCIDENT_FIXTURES / "pr2883-runner-image.lock.json.captured"

    # R1: bytes are unmodified since capture -- pin all three, even though the
    # registry entry's own artifact.sha256 only covers pyproject.toml.
    assert (
        _sha256(pyproject)
        == "0fb3d144655f766e612e457a0fa7c5b7a7bf4e6e0a9d12df000991b6078ac0f3"
    )
    assert (
        _sha256(uv_lock)
        == "95788bf849afb479f747ba45bb46d91ec5536ce8d7a2bd4e71b310b29726281f"
    )
    assert (
        _sha256(lock_json)
        == "e17074f0b2ddfe3af1eeb79835464782ad6457539f6b4f26763894c000ae3616"
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "base", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)

    # Trusted shared-env inputs: this incident never touched them, so their
    # content is irrelevant -- regenerate_for_ref only requires them present.
    (repo / ".github" / "actions" / "setup-python-uv").mkdir(parents=True)
    (repo / ".github" / "actions" / "setup-python-uv" / "action.yml").write_text(
        "name: setup\n", encoding="utf-8"
    )
    (repo / "scripts" / "ci").mkdir(parents=True)
    (repo / "scripts" / "ci" / "ci_env_digest.py").write_text(
        "# digest script\n", encoding="utf-8"
    )
    (repo / "scripts" / "ci" / "ensure_ci_env.sh").write_text(
        "#!/bin/sh\n", encoding="utf-8"
    )
    # base branch's own manifest/lock content is never read by
    # regenerate_for_ref (only origin/<head_ref>'s copies are) -- placeholders.
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (repo / "uv.lock").write_text("# lock\n", encoding="utf-8")
    (repo / "docker" / "runners").mkdir(parents=True)
    (repo / "docker" / "runners" / "runner-image.lock.json").write_text(
        "{}", encoding="utf-8"
    )
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)

    head_ref = "dependabot/pip/kafka-python-gte-2.3.2-and-lt-4.0.0"
    _git("checkout", "-q", "-b", head_ref, cwd=repo)
    shutil.copyfile(pyproject, repo / "pyproject.toml")
    shutil.copyfile(uv_lock, repo / "uv.lock")
    shutil.copyfile(lock_json, repo / "docker" / "runners" / "runner-image.lock.json")
    _git("commit", "-q", "-am", "real PR #2883 bump (captured)", cwd=repo)
    _git("checkout", "-q", "base", cwd=repo)
    _git("remote", "add", "origin", str(repo), cwd=repo)

    module = _load_module()
    result = module.regenerate_for_ref(repo, head_ref)

    assert result is not None, (
        "regenerate_for_ref reported the real PR #2883 manifest state as "
        "already fresh -- the exact false-green miss this replay exists to catch"
    )
    new_lock = json.loads(result)
    old_lock = json.loads(lock_json.read_text(encoding="utf-8"))
    assert new_lock["shared_env_digest"] != old_lock["shared_env_digest"]
    assert new_lock["identity_digest"] != old_lock["identity_digest"]
    assert new_lock["shared_env_digest"] != "8c8cf47e7d5c90a88a258f46", (
        "must not reproduce the stale recorded digest from the real failure"
    )
