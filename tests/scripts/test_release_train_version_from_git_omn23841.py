# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The core release cut takes its version from the newest tag and writes no version line.

RED STATE THIS REPRODUCES (OMN-23841)
-------------------------------------
omnibase_core takes its version from git, so its ``pyproject.toml`` carries
``dynamic = ["version"]`` and no static ``version`` line. The train read
``[project].version`` for the current number and refused such a repo every night
with ``version_unreadable``; when it did cut a repo it also rewrote that line
whenever the manifest was not ahead of the newest tag.

EVERY GREEN ASSERTION HERE HAS A FLIPPED SIBLING
------------------------------------------------
"a tree with no static version cuts the next patch of the newest tag" is paired
with "a tree with a static version still compares it to the tag and bumps it",
and "no tag to derive from refuses" is paired with "a manifest with neither a
static version nor a dynamic declaration still refuses".
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_POLICY_PATH = _REPO_ROOT / "config" / "release_train_policy.yaml"


def _load(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load(_MODULE, "release_train_under_test_omn23841")

_DYNAMIC_MANIFEST = """\
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "omnibase-core"
dynamic = ["version"]
requires-python = ">=3.12"
"""

_STATIC_MANIFEST = """\
[project]
name = "omnibase-core"
version = "0.47.60"
requires-python = ">=3.12"
"""

_NO_VERSION_MANIFEST = """\
[project]
name = "omnibase-core"
requires-python = ">=3.12"
"""


def _git(clone: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(clone), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    ).stdout


def _tree(tmp_path: Path, manifest: str, *, tag: str | None) -> Path:
    """A clone whose ``origin/dev`` is a commit past ``tag`` that touches ``src``."""
    clone = tmp_path / "omnibase_core"
    clone.mkdir()
    _git(clone, "init", "-q", "-b", "dev")
    _git(clone, "config", "user.email", "test@example.invalid")
    _git(clone, "config", "user.name", "test")
    (clone / "pyproject.toml").write_text(manifest, encoding="utf-8")
    (clone / "src").mkdir()
    (clone / "src" / "a.py").write_text("A = 1\n", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-q", "-m", "chore: first")
    if tag is not None:
        _git(clone, "tag", tag)
    (clone / "src" / "a.py").write_text("A = 2\n", encoding="utf-8")
    _git(clone, "commit", "-q", "-am", "feat: change core (#1)")
    _git(clone, "update-ref", "refs/remotes/origin/dev", "HEAD")
    return clone


def _core_policy() -> Any:
    return rt.load_policy(_POLICY_PATH)["omnibase_core"]


@pytest.fixture(autouse=True)
def _green_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rt, "default_required_contexts", lambda repo, branch: ["CI Summary"]
    )
    monkeypatch.setattr(rt, "default_gating_sha", lambda repo, sha: "f" * 40)
    monkeypatch.setattr(
        rt,
        "default_check_runs",
        lambda repo, sha: [
            {"name": "CI Summary", "status": "completed", "conclusion": "success"}
        ],
    )


@pytest.mark.unit
def test_dynamic_version_tree_cuts_next_patch_of_newest_tag(tmp_path: Path) -> None:
    clone = _tree(tmp_path, _DYNAMIC_MANIFEST, tag="v0.47.61")
    policy = _core_policy()

    facts = rt.collect_repo_facts(policy, clone)
    decision = rt.decide(policy=policy, facts=facts)

    assert facts.version_from_git is True
    assert facts.dev_version == "v0.47.61"
    assert decision.verdict is rt.EnumTrainVerdict.CUT
    assert decision.candidate_version == "0.47.62"
    assert decision.needs_bump is False


@pytest.mark.unit
def test_dynamic_version_candidate_follows_the_newest_tag_not_any_file(
    tmp_path: Path,
) -> None:
    clone = _tree(tmp_path, _DYNAMIC_MANIFEST, tag="v0.47.61")
    _git(clone, "tag", "v0.48.3", "refs/remotes/origin/dev")
    policy = _core_policy()

    decision = rt.decide(policy=policy, facts=rt.collect_repo_facts(policy, clone))

    assert decision.candidate_version == "0.48.4"


@pytest.mark.unit
def test_dynamic_version_tree_with_no_tag_refuses(tmp_path: Path) -> None:
    clone = _tree(tmp_path, _DYNAMIC_MANIFEST, tag=None)
    policy = _core_policy()

    decision = rt.decide(policy=policy, facts=rt.collect_repo_facts(policy, clone))

    assert decision.verdict is rt.EnumTrainVerdict.REFUSE
    assert decision.reason is rt.EnumTrainReason.VERSION_UNREADABLE
    assert "got ''" in decision.detail


@pytest.mark.unit
def test_bump_refuses_and_writes_nothing_on_a_dynamic_version_tree(
    tmp_path: Path,
) -> None:
    clone = _tree(tmp_path, _DYNAMIC_MANIFEST, tag="v0.47.61")

    with pytest.raises(rt.ReleaseTrainConfigError):
        rt.bump_declared_version(clone, "0.47.62")

    assert (clone / "pyproject.toml").read_text(encoding="utf-8") == _DYNAMIC_MANIFEST
    assert _git(clone, "status", "--porcelain") == ""


@pytest.mark.unit
def test_static_version_behind_the_tag_still_bumps(tmp_path: Path) -> None:
    clone = _tree(tmp_path, _STATIC_MANIFEST, tag="v0.47.61")
    policy = _core_policy()

    facts = rt.collect_repo_facts(policy, clone)
    decision = rt.decide(policy=policy, facts=facts)

    assert facts.version_from_git is False
    assert facts.dev_version == "0.47.60"
    assert decision.verdict is rt.EnumTrainVerdict.CUT
    assert decision.candidate_version == "0.47.62"
    assert decision.needs_bump is True


@pytest.mark.unit
def test_manifest_with_no_version_and_no_dynamic_declaration_still_refuses(
    tmp_path: Path,
) -> None:
    clone = _tree(tmp_path, _NO_VERSION_MANIFEST, tag="v0.47.61")
    policy = _core_policy()

    with pytest.raises(rt.ReleaseTrainConfigError, match=r"declares no \[project\]"):
        rt.collect_repo_facts(policy, clone)


@pytest.mark.unit
def test_cut_job_bumps_the_manifest_only_when_the_decision_asks() -> None:
    workflow = (
        _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"
    ).read_text(encoding="utf-8")

    guard = 'if [ "${NEEDS_BUMP}" = "true" ]; then'
    assert workflow.count(guard) == 1
    bump_call = workflow.index('release_train.py" bump-version')
    assert workflow.index(guard) < bump_call
