# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26369: the merge read runs against a real git history and real job records.

No double: a clone with commits made by ``git``, job records written to disk, and the
production readers that decide whether an idle restore asks the window again.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    newest_dev_job_accepted,
    read_runtime_merge,
)

pytestmark = pytest.mark.integration

JOB_START = datetime(2026, 11, 21, 13, 0, 38, tzinfo=UTC)
RUNTIME_MERGE = datetime(2026, 11, 21, 14, 6, 14, tzinfo=UTC)


def _git(repo: Path, *args: str, **extra_env: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={**scrub_git_location_env(os.environ), **extra_env},
    )


def _commit(repo: Path, path: str, when: datetime) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(when.isoformat())
    stamp = when.strftime("%Y-%m-%dT%H:%M:%S+0000")
    _git(repo, "add", path)
    _git(
        repo,
        "commit",
        "-m",
        path,
        GIT_AUTHOR_DATE=stamp,
        GIT_COMMITTER_DATE=stamp,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )


def test_a_runtime_merge_after_the_newest_dev_job_is_read_from_disk(
    tmp_path: Path,
) -> None:
    state = tmp_path / "jobs-dev"
    state.mkdir()
    (state / "job.json").write_text(
        json.dumps(
            {
                "command": {"runtime_lane": "dev"},
                "accepted_at": JOB_START.isoformat(),
            }
        )
    )
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-b", "dev")
    _commit(repo, "README.md", JOB_START - timedelta(hours=1))
    _commit(repo, "docs/x.md", RUNTIME_MERGE + timedelta(minutes=30))
    _commit(repo, "src/omnibase_infra/b.py", RUNTIME_MERGE)

    built = newest_dev_job_accepted(state)

    assert built == JOB_START
    assert read_runtime_merge(repo, "HEAD", built) == RUNTIME_MERGE
    assert read_runtime_merge(repo, "HEAD", RUNTIME_MERGE) is None
