# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep candidate sibling and build-context provenance on dev-merge tags.

OMN-19998: unpinned siblings need remote-tracking refs to engage OMN-17427's
tag following. The build checkout must also fetch its HEAD tag after the
concurrent dev-merge-tag workflow has published it.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = (
    _REPO_ROOT / ".github" / "workflows" / "build-workspace-candidate-runtime.yml"
)
_FETCH_STEP = "Fetch the build context's dev-merge tag"
_STAGING_STEP = "Stage workspace sibling trees into the build context"
_CHECKOUT_STEP = "Checkout omnibase_infra (the consuming repo / build context)"


@pytest.fixture(scope="module")
def candidate_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return cast(
        "list[dict[str, Any]]",
        workflow["jobs"]["build-workspace-candidate"]["steps"],
    )


def _step_index(steps: list[dict[str, Any]], name: str) -> int:
    indices = [index for index, step in enumerate(steps) if step.get("name") == name]
    assert len(indices) == 1, f"expected exactly one step named {name!r}: {indices}"
    return indices[0]


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Candidate regression test",
            "-c",
            "user.email=candidate@example.com",
            *args,
        ],
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_checkout(tmp_path: Path, *, with_tag: bool) -> tuple[Path, str]:
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    checkout = tmp_path / "checkout"
    _git(tmp_path, "-c", "init.defaultBranch=dev", "init", "--bare", str(origin))
    _git(tmp_path, "clone", str(origin), str(work))
    (work / "source.txt").write_text("candidate source\n", encoding="utf-8")
    _git(work, "add", "source.txt")
    _git(work, "commit", "-m", "Candidate source")
    _git(work, "push", "origin", "dev")
    sha12 = _git(work, "rev-parse", "HEAD")[:12]
    tag = f"dev-20261101000000-{sha12}"
    if with_tag:
        unrelated = "dev-20261101000000-000000000000"
        _git(work, "tag", tag)
        _git(work, "tag", unrelated)
        _git(work, "push", "origin", f"refs/tags/{tag}", f"refs/tags/{unrelated}")
    _git(tmp_path, "clone", "--no-tags", str(origin), str(checkout))
    assert _git(checkout, "tag", "--points-at", "HEAD") == ""
    return checkout, tag


def _run_tag_fetch(
    checkout: Path, steps: list[dict[str, Any]], *, attempts: int = 1
) -> subprocess.CompletedProcess[str]:
    script = steps[_step_index(steps, _FETCH_STEP)]["run"]
    assert isinstance(script, str)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=checkout,
        env=scrub_git_location_env(os.environ)
        | {"DEV_TAG_WAIT_ATTEMPTS": str(attempts), "DEV_TAG_WAIT_SECONDS": "0"},
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
def test_unpinned_siblings_are_armed_with_the_remote_tracking_ref(
    candidate_steps: list[dict[str, Any]],
) -> None:
    staging = candidate_steps[_step_index(candidate_steps, _STAGING_STEP)]
    deploy_ref = staging["env"]["DEPLOY_REF"]
    assert "inputs.pinned_revision" in deploy_ref
    assert "format('origin/{0}', inputs.sibling_ref)" in deploy_ref
    assert deploy_ref != "${{ inputs.pinned_revision || inputs.sibling_ref }}"


@pytest.mark.unit
def test_follow_dev_tag_engages_only_on_a_remote_tracking_ref() -> None:
    source = (_REPO_ROOT / "scripts/runtime_build/deploy_source_ref.py").read_text(
        encoding="utf-8"
    )
    assert 'follow_dev_tag and effective_ref.startswith("origin/")' in source


@pytest.mark.unit
def test_the_tag_fetch_runs_before_staging(
    candidate_steps: list[dict[str, Any]],
) -> None:
    fetch_index = _step_index(candidate_steps, _FETCH_STEP)
    assert fetch_index + 1 == _step_index(candidate_steps, _STAGING_STEP)
    assert _step_index(candidate_steps, _CHECKOUT_STEP) < fetch_index
    fetch = candidate_steps[fetch_index]
    assert "if" not in fetch
    assert fetch["env"] == {
        "DEV_TAG_WAIT_ATTEMPTS": "20",
        "DEV_TAG_WAIT_SECONDS": "15",
    }


@pytest.mark.unit
def test_the_tag_fetch_brings_the_head_tag_into_the_checkout(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
) -> None:
    checkout, tag = _make_checkout(tmp_path, with_tag=True)
    result = _run_tag_fetch(checkout, candidate_steps)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(checkout, "tag", "--points-at", "HEAD", "--list", "dev-*") == tag


@pytest.mark.unit
def test_the_tag_fetch_warns_and_continues_when_no_tag_exists(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
) -> None:
    checkout, _ = _make_checkout(tmp_path, with_tag=False)
    result = _run_tag_fetch(checkout, candidate_steps, attempts=2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::" in result.stdout
    assert _git(checkout, "tag", "--points-at", "HEAD") == ""


@pytest.mark.unit
def test_the_checker_reads_the_fetched_tag_as_a_dev_merge_tag(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, tag = _make_checkout(tmp_path, with_tag=True)
    result = _run_tag_fetch(checkout, candidate_steps)
    assert result.returncode == 0, result.stdout + result.stderr

    source = _REPO_ROOT / "src/omnibase_infra/validation/checker_first_party_sources.py"
    spec = importlib.util.spec_from_file_location(
        "candidate_dev_merge_tag_checker", source
    )
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, checker)
    spec.loader.exec_module(checker)
    record = checker._build_context_record(checkout, checker.load_contract())
    assert record["dev_merge_tag"] == tag
