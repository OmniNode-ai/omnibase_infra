# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wiring proof for the post-release dev version bump job (OMN-13912).

Operating Rule #5: detection that is not wired as a gate is advisory and gets
ignored. The same is true of a *disarm* step — a correct
``post_release_dev_bump.py`` that no workflow calls fixes nothing. The unit
suite proves the decision; this file proves the release train actually performs
it, and performs it under the conditions that matter:

* the job exists and depends on ``release``,
* it is gated on ``needs.release.outputs.version`` — NOT on
  ``needs.release.result == 'success'``. That distinction is the whole reason
  v0.38.10 and v0.38.11 stayed armed: both PUBLISHED fine and then the release
  job went red on the unrelated "Sync main to release tag" GH006 (OMN-16343),
  which would have skipped a success-gated disarm exactly when it was needed,
* it never fires for an rc tag,
* it lands the bump through a PR against ``dev`` (never a direct push to a
  protected branch), and
* the PR it opens carries an ``OMN-`` reference, which the ``pr-title`` CI job
  requires of every PR in this repo.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
BUMP_SCRIPT = REPO_ROOT / "scripts" / "ci" / "post_release_dev_bump.py"

JOB_ID = "post-release-dev-bump"
LOCK_FILE = "docker/runners/runner-image.lock.json"


def _load_release_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _job() -> dict[str, Any]:
    jobs = _load_release_workflow()["jobs"]
    assert JOB_ID in jobs, (
        f"release.yml has no `{JOB_ID}` job -- publishing from dev HEAD arms the "
        "OMN-13412 release-identity gate against dev and nothing disarms it"
    )
    job = jobs[JOB_ID]
    assert isinstance(job, dict)
    return job


def _job_text() -> str:
    """The job's own YAML slice, for step-level string assertions."""
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    start = text.index(f"  {JOB_ID}:")
    remainder = text[start + 1 :]
    # Next top-level job header (two-space indent, non-comment).
    match = re.search(r"\n  [a-z0-9][a-z0-9-]*:\n", remainder)
    end = start + 1 + (match.start() if match else len(remainder))
    return text[start:end]


def test_the_bump_script_exists() -> None:
    assert BUMP_SCRIPT.is_file()


def test_job_runs_after_the_release_job() -> None:
    assert _job()["needs"] == "release"


def test_main_sync_app_token_can_write_tagged_workflows() -> None:
    """The release tag may include .github/workflows changes (OMN-17272)."""
    release_steps = _load_release_workflow()["jobs"]["release"]["steps"]
    mint_step = next(
        step
        for step in release_steps
        if step.get("name")
        == "Mint onexbot-occ-writer app token (release main-sync push identity)"
    )
    assert mint_step["with"]["permission-workflows"] == "write"


def test_job_is_gated_on_publish_output_not_on_overall_job_success() -> None:
    condition = " ".join(_job()["if"].split())
    # The published-version output is the publish proof.
    assert "needs.release.outputs.version != ''" in condition
    # A main-sync GH006 (OMN-16343) must NOT skip the disarm.
    assert "needs.release.result == 'success'" not in condition
    assert "always()" in condition


def test_job_never_fires_for_a_release_candidate() -> None:
    condition = " ".join(_job()["if"].split())
    assert "!contains(needs.release.outputs.version, 'rc')" in condition


def test_job_invokes_the_bump_script_in_both_decide_and_apply_modes() -> None:
    body = _job_text()
    assert "scripts/ci/post_release_dev_bump.py --released" in body
    assert "--apply" in body


def test_job_relocks_uv_so_the_lockfile_version_moves_with_pyproject() -> None:
    # uv.lock carries this package's own version; bumping pyproject alone
    # produces a lockfile-drift failure on the bump PR itself.
    assert "uv lock" in _job_text()


def test_job_regenerates_the_runner_image_lock_after_relocking() -> None:
    # uv.lock bytes feed the runner image identity (OMN-19998): a re-lock with
    # no regenerate leaves docker/runners/runner-image.lock.json stale, and the
    # bump PR goes red on tests/ci/test_runner_image_identity.py and on
    # runner-image-build-smoke. 0.38.72, 0.38.73, 0.38.74 and 0.38.75 each
    # needed a hand-pushed regenerate.
    body = _job_text()
    relock = body.index("uv lock")
    regenerate = body.index("scripts/ci/runner_image_identity.py --mode generate")
    assert regenerate > relock


def test_bump_commit_stages_the_regenerated_runner_image_lock() -> None:
    body = _job_text()
    add_lines = [line for line in body.splitlines() if "git add" in line]
    assert add_lines, "the bump job must stage its changes"
    assert any("docker/runners/runner-image.lock.json" in line for line in add_lines)


@pytest.mark.live_contact("tests/ci/fixtures/omn19998_bump_pr_stale_runner_lock.json")
def test_recorded_bump_pr_red_is_cleared_by_the_job_regenerate(
    recorded_response: dict[str, Any],
) -> None:
    # The 0.38.75 bump PR committed only pyproject.toml and uv.lock, and its
    # runner-image-build-smoke named the remediation command; the job must
    # now run that command and stage the lock it rewrites.
    assert LOCK_FILE not in recorded_response["bump_commit_files"]
    error = recorded_response["runner_image_build_smoke_error"]
    assert "shared_env_digest is stale" in error
    remediation = error.rsplit("run ", 1)[1]
    body = _job_text()
    assert body.index(remediation) > body.index("uv lock")
    add_line = next(line for line in body.splitlines() if "git add" in line)
    for path in (*recorded_response["bump_commit_files"], LOCK_FILE):
        assert path in add_line.split()
    assert all(
        failure.startswith("tests.ci.test_runner_image_identity::")
        for failure in recorded_response["ratchet_new_failures"]
    )


def test_bump_lands_through_a_pr_against_dev_never_a_direct_push() -> None:
    body = _job_text()
    assert "gh pr create" in body
    assert "--base dev" in body
    # A direct push to dev would bypass every required check on the branch.
    assert "git push origin dev" not in body
    assert "refs/heads/dev" not in body


def test_bump_pr_title_carries_a_ticket_reference_for_the_pr_title_gate() -> None:
    body = _job_text()
    title_lines = [line for line in body.splitlines() if "--title" in line]
    assert title_lines, "the bump PR must set an explicit title"
    assert all(re.search(r"OMN-\d+", line) for line in title_lines)


def test_bump_pr_open_is_idempotent_for_a_redispatched_tag() -> None:
    body = _job_text()
    assert "gh pr list" in body
    assert "--head" in body


def test_job_fails_loudly_when_the_app_credentials_are_absent() -> None:
    # A silently-skipped disarm is the failure mode this ticket exists for, so
    # a missing credential must be an error, not a warning-and-continue.
    body = _job_text()
    assert "::error::" in body
    assert "::warning::ONEXBOT_OCC_APP_ID" not in body
