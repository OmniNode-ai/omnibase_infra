# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19376: the self-update dirty gate must ignore TRACKED regenerable byproducts.

OMN-16442 taught the gate to ignore UNTRACKED byproducts
(``workspace/deploy-source-refs.json``) via ``--untracked-files=no``. That
exemption cannot reach ``workspace/sibling-pin-comparison.json`` and
``workspace/sibling-vcs-provenance.json``: both are committed PLACEHOLDER
files kept TRACKED on purpose (a Dockerfile COPY needs them to resolve on a
fresh checkout), and both get rewritten as a byproduct of every deploy job by
``compute_workspace_provenance.py`` / the ``stage_workspace_*`` family. Being
tracked, ``--untracked-files=no`` does not exclude them, so the last job's
rewrite always read as a genuine tracked modification and every self-update
boundary since 05:09Z on 2026-09-24 silently skipped.

Same test shape as ``test_executor_self_update_untracked_omn16442.py``: real
git repositories, only ``uv sync`` and ``os.execv`` stubbed on the pull path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import SELF_UPDATE_KNOWN_BYPRODUCT_PATHS, DeployExecutor
from deploy_agent.executor import _run as real_run
from deploy_agent.loaded_code import record_loaded_code_sha

TRACKING_BRANCH = "dev"

# Real placeholder content, unmodified from the committed files, so the
# fixture is not a fictional shape.
_PLACEHOLDER_CONTENTS = {
    "workspace/sibling-pin-comparison.json": (
        '{\n  "lock_source": "placeholder",\n  "allow_drift": false,\n'
        '  "drift_count": 0,\n  "comparisons": [],\n'
        '  "_note": "placeholder"\n}\n'
    ),
    "workspace/sibling-vcs-provenance.json": (
        '{\n  "siblings": {},\n  "_note": "placeholder"\n}\n'
    ),
}


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_clone(tmp_path: Path, *, extra_commit: bool) -> Path:
    """Build an origin repo on ``dev``, with both byproduct placeholders
    committed at the repo root, plus a clone that tracks it.

    ``extra_commit=True`` advances origin one commit past the clone, so the
    clone is genuinely BEHIND ``origin/dev``.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--initial-branch", TRACKING_BRANCH)
    _git(origin, "config", "user.email", "agent@example.invalid")
    _git(origin, "config", "user.name", "deploy agent test")
    (origin / "agent.py").write_text("print('v1')\n", encoding="utf-8")
    workspace = origin / "workspace"
    workspace.mkdir()
    for rel_path, content in _PLACEHOLDER_CONTENTS.items():
        (origin / rel_path).write_text(content, encoding="utf-8")
    _git(origin, "add", "agent.py", *_PLACEHOLDER_CONTENTS.keys())
    _git(origin, "commit", "-m", "v1")

    # ``self_update`` runs from a subdirectory of the clone -- exactly like
    # the real agent, whose DEPLOY_AGENT_DIR is scripts/deploy-agent inside
    # the omnibase_infra clone. This is what makes a bare relative pathspec
    # (resolved against the ``-C`` dir) fail to match, and the ":/<path>"
    # magic pathspec (resolved against the repo root) succeed.
    (origin / "scripts").mkdir()
    (origin / "scripts" / "deploy-agent").mkdir()
    (origin / "scripts" / "deploy-agent" / ".keep").write_text("", encoding="utf-8")
    _git(origin, "add", "scripts/deploy-agent/.keep")
    _git(origin, "commit", "-m", "add agent subdir")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    _git(clone, "config", "user.email", "agent@example.invalid")
    _git(clone, "config", "user.name", "deploy agent test")

    if extra_commit:
        (origin / "agent.py").write_text("print('v2')\n", encoding="utf-8")
        _git(origin, "add", "agent.py")
        _git(origin, "commit", "-m", "v2")

    return clone


def _rewrite_byproducts_like_a_job(clone: Path) -> None:
    """Recreate what a deploy job's build/provenance steps do to these files."""
    for rel_path in SELF_UPDATE_KNOWN_BYPRODUCT_PATHS:
        real_path = clone / rel_path
        real_path.write_text(
            '{\n  "lock_source": "real",\n  "comparisons": [{"drift": true}]\n}\n',
            encoding="utf-8",
        )
    tracked_diff = _git(clone, "status", "--porcelain", "--untracked-files=no")
    assert "sibling-pin-comparison.json" in tracked_diff
    assert "sibling-vcs-provenance.json" in tracked_diff


@pytest.mark.unit
def test_declared_byproduct_paths_match_the_committed_placeholders() -> None:
    """The fixture and the production constant must agree on what a job writes."""
    assert set(SELF_UPDATE_KNOWN_BYPRODUCT_PATHS) == set(_PLACEHOLDER_CONTENTS)


@pytest.mark.unit
def test_job_rewritten_byproducts_do_not_block_and_agent_reports_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A clone at origin/dev, dirtied only by a job's byproduct rewrite, proceeds."""
    clone = _make_clone(tmp_path, extra_commit=False)
    agent_dir = clone / "scripts" / "deploy-agent"
    _rewrite_byproducts_like_a_job(clone)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    record_loaded_code_sha(str(clone))

    executor = DeployExecutor()
    with caplog.at_level("INFO"), patch("os.execv") as mock_execv:
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert f"already at origin/{TRACKING_BRANCH}" in caplog.text
    assert "tracked modifications" not in caplog.text
    assert "discarded a job's rewrite of known byproduct" in caplog.text
    mock_execv.assert_not_called()
    # Discarded back to the committed placeholder, not left dirty.
    assert _git(clone, "status", "--porcelain", "--untracked-files=no") == ""
    for rel_path, content in _PLACEHOLDER_CONTENTS.items():
        assert (clone / rel_path).read_text(encoding="utf-8") == content


@pytest.mark.unit
def test_job_rewritten_byproducts_do_not_block_the_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Behind + byproduct-dirty-only: the agent really pulls and re-execs."""
    clone = _make_clone(tmp_path, extra_commit=True)
    agent_dir = clone / "scripts" / "deploy-agent"
    _rewrite_byproducts_like_a_job(clone)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    before = _git(clone, "rev-parse", "HEAD")
    record_loaded_code_sha(str(clone))

    def _run_git_for_real(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        if cmd[0] == "uv":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )
        return real_run(cmd, timeout, **kwargs)

    executor = DeployExecutor()
    with (
        caplog.at_level("INFO"),
        patch("deploy_agent.executor._run", side_effect=_run_git_for_real),
        patch("os.execv") as mock_execv,
    ):
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert f"behind origin/{TRACKING_BRANCH}" in caplog.text
    after = _git(clone, "rev-parse", "HEAD")
    assert after != before
    assert after == _git(clone, "rev-parse", f"origin/{TRACKING_BRANCH}")
    mock_execv.assert_called_once()


@pytest.mark.unit
def test_real_tracked_source_edit_still_blocks_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AC3: a genuine tracked change elsewhere must still skip self-update.

    The exemption is scoped to exactly the two declared paths; it must not
    widen into "any file under workspace/" or "any tracked change at all".
    """
    clone = _make_clone(tmp_path, extra_commit=True)
    agent_dir = clone / "scripts" / "deploy-agent"
    (clone / "agent.py").write_text("print('local edit')\n", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    before = _git(clone, "rev-parse", "HEAD")
    record_loaded_code_sha(str(clone))

    executor = DeployExecutor()
    with caplog.at_level("INFO"), patch("os.execv") as mock_execv:
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert "tracked modifications" in caplog.text
    assert "agent.py" in caplog.text
    assert _git(clone, "rev-parse", "HEAD") == before
    mock_execv.assert_not_called()


@pytest.mark.unit
def test_byproduct_rewrite_plus_a_real_tracked_edit_still_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A job byproduct AND a genuine tracked edit together: still a skip.

    The byproduct is discarded, but the genuine edit remains and must still
    be reported and still block -- the exemption must not paper over an
    unrelated real change that happens to land in the same dirty tree.
    """
    clone = _make_clone(tmp_path, extra_commit=True)
    agent_dir = clone / "scripts" / "deploy-agent"
    _rewrite_byproducts_like_a_job(clone)
    (clone / "agent.py").write_text("print('local edit')\n", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    before = _git(clone, "rev-parse", "HEAD")
    record_loaded_code_sha(str(clone))

    executor = DeployExecutor()
    with caplog.at_level("INFO"), patch("os.execv") as mock_execv:
        executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

    assert "tracked modifications" in caplog.text
    assert "agent.py" in caplog.text
    assert _git(clone, "rev-parse", "HEAD") == before
    mock_execv.assert_not_called()
    # The byproducts were still discarded even though the overall update skipped.
    for rel_path, content in _PLACEHOLDER_CONTENTS.items():
        assert (clone / rel_path).read_text(encoding="utf-8") == content
