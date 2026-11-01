# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep candidate sibling and build-context provenance on dev-merge tags.

OMN-19998: unpinned siblings need remote-tracking refs to engage OMN-17427's
tag following. The build checkout must also fetch its HEAD tag after the
concurrent dev-merge-tag workflow has published it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
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
_CHECKOUT_STEP = "actions/checkout@"


@pytest.fixture(scope="module")
def candidate_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return cast(
        "list[dict[str, Any]]",
        workflow["jobs"]["build-workspace-candidate"]["steps"],
    )


def _step_index(steps: list[dict[str, Any]], marker: str) -> int:
    indices = [
        index
        for index, step in enumerate(steps)
        if step.get("name") == marker or step.get("uses", "").startswith(marker)
    ]
    assert len(indices) == 1, f"expected exactly one step {marker!r}: {indices}"
    return indices[0]


def _evaluate_deploy_ref(expression: str, inputs: dict[str, str]) -> str:
    """Evaluate the workflow's `a || b` expression with GitHub's empty-string falsiness."""
    body = expression.strip().removeprefix("${{").removesuffix("}}").strip()
    result = ""
    for operand in body.split("||"):
        operand = operand.strip()
        if operand.startswith("format("):
            match = re.fullmatch(r"format\('([^']*)', inputs\.(\w+)\)", operand)
            assert match is not None, operand
            result = match.group(1).replace("{0}", inputs[match.group(2)])
        else:
            assert operand.startswith("inputs."), operand
            result = inputs[operand.removeprefix("inputs.")]
        if result:
            return result
    return result


def _load_script(path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


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


def _make_checkout(
    tmp_path: Path, *, with_tag: bool, extra_tags: tuple[str, ...] = ()
) -> tuple[Path, str]:
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
    for name in extra_tags:
        name = name.format(sha12=sha12)
        _git(work, "tag", name)
        _git(work, "push", "origin", f"refs/tags/{name}")
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
    unpinned = {"pinned_revision": "", "sibling_ref": "dev"}
    assert _evaluate_deploy_ref(deploy_ref, unpinned) == "origin/dev"
    pinned = {"pinned_revision": "a" * 40, "sibling_ref": "dev"}
    assert _evaluate_deploy_ref(deploy_ref, pinned) == "a" * 40


@pytest.mark.unit
@pytest.mark.parametrize(
    "ref_kind", ["remote", "branch", "sha", "fallback", "same_fallback"]
)
def test_follow_dev_tag_engages_only_on_a_remote_tracking_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref_kind: str
) -> None:
    checkout, tag = _make_checkout(tmp_path, with_tag=True)
    tagged = _git(checkout, "rev-parse", "HEAD")
    work = tmp_path / "work"
    (work / "source.txt").write_text("untagged branch advance\n", encoding="utf-8")
    _git(work, "add", "source.txt")
    _git(work, "commit", "-m", "Advance beyond the dev tag")
    _git(work, "push", "origin", "dev")
    branch_head = _git(work, "rev-parse", "HEAD")
    _git(checkout, "fetch", "origin", "dev")
    _git(checkout, "checkout", "-B", "dev", "origin/dev")

    primary = {
        "remote": "origin/dev",
        "branch": "dev",
        "sha": branch_head,
        "fallback": "f" * 40,
        "same_fallback": "origin/dev",
    }[ref_kind]
    fallback = "origin/dev" if ref_kind in {"fallback", "same_fallback"} else None
    resolver = _load_script(
        _REPO_ROOT / "scripts/runtime_build/deploy_source_ref.py", monkeypatch
    )
    result = resolver.clean_checkout(
        checkout,
        primary,
        fallback_ref=fallback,
        follow_dev_tag=True,
        worktree_root=tmp_path / "trees",
    )
    follows = ref_kind in {"remote", "fallback", "same_fallback"}
    expected = tagged if follows else branch_head
    assert tagged != branch_head
    assert result.head_sha == result.expected_sha == expected
    assert _git(Path(result.path), "rev-parse", "HEAD") == expected
    assert result.ref == (tag if follows else primary)
    assert result.dev_tag_from == ("origin/dev" if follows else "")
    assert result.fallback_from == (primary if ref_kind == "fallback" else "")
    assert (Path(result.path) / "source.txt").read_text() == (
        "candidate source\n" if follows else "untagged branch advance\n"
    )


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
def test_the_tag_fetch_fails_fast_when_no_tag_exists(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
) -> None:
    checkout, _ = _make_checkout(tmp_path, with_tag=False)
    result = _run_tag_fetch(checkout, candidate_steps, attempts=2)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::" in result.stdout
    assert _git(checkout, "tag", "--points-at", "HEAD") == ""


@pytest.mark.unit
@pytest.mark.parametrize("points_at_head", [True, False])
def test_the_checker_reads_the_fetched_tag_as_a_dev_merge_tag(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    points_at_head: bool,
) -> None:
    checkout, tag = _make_checkout(tmp_path, with_tag=points_at_head)
    if not points_at_head:
        # Even a name with HEAD's suffix must not stamp a different commit.
        work = tmp_path / "work"
        (work / "source.txt").write_text("different commit\n", encoding="utf-8")
        _git(work, "add", "source.txt")
        _git(work, "commit", "-m", "Different tag target")
        _git(work, "tag", tag)
        _git(work, "push", "origin", f"refs/tags/{tag}")
    result = _run_tag_fetch(checkout, candidate_steps)
    assert (result.returncode == 0) == points_at_head, result.stdout + result.stderr
    assert ("::error::" not in result.stdout) == points_at_head
    if points_at_head:
        assert _git(checkout, "rev-parse", tag) == _git(checkout, "rev-parse", "HEAD")
    else:
        # A same-suffix tag on another commit is never fetched into the checkout.
        assert _git(checkout, "tag", "--list", "dev-*") == ""

    source = _REPO_ROOT / "src/omnibase_infra/validation/checker_first_party_sources.py"
    checker = _load_script(source, monkeypatch)
    staging = tmp_path / "staging.json"
    expected = tmp_path / "expected.json"
    staging.write_text('{"siblings": {}}\n', encoding="utf-8")
    expected.write_text('{"repos": {}}\n', encoding="utf-8")
    checker.stamp(staging, expected, checkout)
    record = json.loads(staging.read_text())["build_context"]["omnibase_infra"]
    assert record["dev_merge_tag"] == (tag if points_at_head else "")
    assert record["vcs_ref"] == _git(checkout, "rev-parse", "HEAD")
    assert record["vcs_dirty"] is False


@pytest.mark.unit
def test_the_tag_fetch_picks_the_newest_timestamped_tag_and_ignores_malformed_names(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
) -> None:
    checkout, tag = _make_checkout(
        tmp_path,
        with_tag=True,
        extra_tags=("dev-hotfix-{sha12}", "dev-20250101000000-{sha12}"),
    )
    newer = tag.replace("20261101000000", "20261102000000")
    _git(tmp_path / "work", "tag", newer)
    _git(tmp_path / "work", "push", "origin", f"refs/tags/{newer}")
    result = _run_tag_fetch(checkout, candidate_steps)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(checkout, "tag", "--points-at", "HEAD", "--list", "dev-*") == newer


@pytest.mark.unit
def test_the_tag_fetch_retries_until_a_late_tag_appears(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
) -> None:
    checkout, tag = _make_checkout(tmp_path, with_tag=False)
    work = tmp_path / "work"
    _git(work, "tag", tag)
    script = candidate_steps[_step_index(candidate_steps, _FETCH_STEP)]["run"]
    # Publish the tag from the step's own sleep, i.e. after the first poll.
    shim = tmp_path / "bin"
    shim.mkdir()
    sleeper = shim / "sleep"
    sleeper.write_text(
        f'#!/bin/sh\ngit -C "{work}" push origin "refs/tags/{tag}" >/dev/null 2>&1\n',
        encoding="utf-8",
    )
    sleeper.chmod(0o755)
    env = scrub_git_location_env(os.environ) | {
        "DEV_TAG_WAIT_ATTEMPTS": "3",
        "DEV_TAG_WAIT_SECONDS": "1",
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::error::" not in result.stdout
    assert _git(checkout, "tag", "--points-at", "HEAD", "--list", "dev-*") == tag


@pytest.mark.unit
def test_the_tag_fetch_fails_when_origin_stays_unreachable(
    tmp_path: Path,
    candidate_steps: list[dict[str, Any]],
) -> None:
    checkout, _ = _make_checkout(tmp_path, with_tag=True)
    _git(checkout, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    result = _run_tag_fetch(checkout, candidate_steps, attempts=2)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["ls-remote", "fetch"])
def test_the_tag_fetch_recovers_after_a_transient_git_error(
    tmp_path: Path, candidate_steps: list[dict[str, Any]], operation: str
) -> None:
    checkout, tag = _make_checkout(tmp_path, with_tag=True)
    real_git = shutil.which("git")
    assert real_git is not None
    shim = tmp_path / "bin"
    shim.mkdir()
    wrapper = shim / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "$FAIL_OPERATION" ] && [ ! -f "$FAIL_STATE" ]; then\n'
        '  touch "$FAIL_STATE"\n'
        "  exit 128\n"
        "fi\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    script = candidate_steps[_step_index(candidate_steps, _FETCH_STEP)]["run"]
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=checkout,
        env=scrub_git_location_env(os.environ)
        | {
            "DEV_TAG_WAIT_ATTEMPTS": "2",
            "DEV_TAG_WAIT_SECONDS": "0",
            "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
            "REAL_GIT": real_git,
            "FAIL_OPERATION": operation,
            "FAIL_STATE": str(tmp_path / "failed-once"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (tmp_path / "failed-once").is_file()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::error::" not in result.stdout
    assert _git(checkout, "tag", "--points-at", "HEAD", "--list", "dev-*") == tag
