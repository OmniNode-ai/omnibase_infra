# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A merge that touches the runtime image restarts a down dev lane within minutes (OMN-26369).

Recorded on the lab dev lane, 2026-11-21 (``fixtures/deploy_agent_merge_trigger_omn26369.json``):
omnibase_infra#6131 merged at 13:45:48Z and #6133 at 14:06:14Z, and the deploy agent started no
job for either until its idle restore ``c8fd5fda`` at 15:00:27Z. No merge command reached the
agent (the chain that publishes it runs inside the dev runtime, which was down), the lane was
busy with idle restore ``831062d1`` until 14:15:56Z, and every later restore tick was held by
the dev recreate window (one recreate per hour, within 300 s of the hour), whose next opening
was 15:00Z.

The agent's own answer to a merge it never hears about: the window asks again, as
``agent/merge-restore``, for an idle restore it refused when the dev head carries a runtime-image
commit newer than the lane's newest dev job, and the window policy admits that requester off the
hour. The restore's own bounds, the health gate and the boot guard are not touched.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.enums.enum_dev_recreate_reason import EnumDevRecreateReason
from omnibase_infra.enums.enum_dev_recreate_verdict import EnumDevRecreateVerdict
from omnibase_infra.handlers.dev_recreate_window.handler_dev_recreate_window import (
    DevRecreateWindowGate,
    HandlerDevRecreateWindow,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_recreate_request import ModelDevRecreateRequest
from omnibase_infra.nodes.node_dev_roll_events_effect import runtime_dev_roll_events
from omnibase_infra.nodes.node_dev_roll_events_effect.runtime_dev_roll_events import (
    IDLE_RESTORE_REQUESTER,
    MERGE_RESTORE_REQUESTER,
    UPSTREAM_REF,
    is_runtime_image_path,
    newest_dev_job_accepted,
    read_runtime_merge,
    window_admits,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "deploy_agent_merge_trigger_omn26369.json"
    ).read_text()
)


def _t(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


MERGE_6131 = _t(FIXTURE["merges"][0]["merged_at"])
MERGE_6133 = _t(FIXTURE["merges"][1]["merged_at"])
JOB_831062D1 = FIXTURE["jobs"][0]
JOB_START = _t(JOB_831062D1["accepted_at"])
JOB_END = _t(JOB_831062D1["completed_at"])
LANE_DOWN_READ = _t(FIXTURE["lane_down_read_at"])
WINDOW_15 = _t("2026-11-21T15:00:00Z")


def _policy() -> ModelDevRecreatePolicy:
    env: dict[str, str] = {}
    for line in (REPO / "docker" / "runtime-policy.env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip()
    return ModelDevRecreatePolicy.from_env(env)


# --------------------------------------------------------------------------- #
# The recorded cause: the window held every restore until 15:00Z               #
# --------------------------------------------------------------------------- #
def test_the_recorded_outage_idle_restore_was_held_to_the_next_window() -> None:
    decision = HandlerDevRecreateWindow().handle(
        ModelDevRecreateRequest(
            policy=_policy(),
            now=LANE_DOWN_READ,
            requested_by=IDLE_RESTORE_REQUESTER,
            last_recreate_at=JOB_START,
        )
    )

    assert decision.verdict is EnumDevRecreateVerdict.HOLD
    assert decision.reason is EnumDevRecreateReason.WINDOW_MISSED
    assert decision.next_recreate_at == WINDOW_15


def test_the_window_admits_the_merge_restore_requester_off_the_hour() -> None:
    decision = HandlerDevRecreateWindow().handle(
        ModelDevRecreateRequest(
            policy=_policy(),
            now=LANE_DOWN_READ,
            requested_by=MERGE_RESTORE_REQUESTER,
            last_recreate_at=JOB_START,
        )
    )

    assert decision.verdict is EnumDevRecreateVerdict.RECREATE
    assert decision.reason is EnumDevRecreateReason.PINNED_FIX


# --------------------------------------------------------------------------- #
# The reads                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/omnibase_infra/handlers/handler_dev_boot_guard.py", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        ("contracts/services/runtime_policy.contract.yaml", True),
        ("docker/Dockerfile.runtime", True),
        ("docker/runtime-policy.env", True),
        ("docker/docker-compose.infra.yml", True),
        ("docker/migrations/0001_x.sql", True),
        ("scripts/deploy-agent/deploy_agent/agent.py", False),
        ("tests/unit/scripts/test_x.py", False),
        ("docs/patterns/service_catalog.md", False),
        ("CLAUDE.md", False),
    ],
)
def test_runtime_image_paths(path: str, expected: bool) -> None:
    assert is_runtime_image_path(path) is expected


def test_every_infra_source_the_runtime_image_copies_is_a_runtime_image_path() -> None:
    """The list cannot fall behind the Dockerfile: each infra-owned COPY source matches."""
    copied: set[str] = set()
    for line in (REPO / "docker" / "Dockerfile.runtime").read_text().splitlines():
        words = line.split()
        if (
            not words
            or words[0] != "COPY"
            or any(w.startswith("--from") for w in words)
        ):
            continue
        copied.update(w for w in words[1:-1] if not w.startswith("--"))
    owned = {
        w
        for w in copied
        if not w.startswith(("workspace/", "$", "/")) and w != "LICENSE"
    }
    assert {"pyproject.toml", "uv.lock", "src/", "contracts/"} <= owned
    for source in sorted(owned):
        probe = source if not source.endswith("/") else source + "x"
        assert is_runtime_image_path(probe), source


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


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-b", "dev")
    _commit(repo, "README.md", JOB_START - timedelta(hours=1))
    _commit(repo, "src/omnibase_infra/a.py", MERGE_6131)
    _commit(repo, "docs/x.md", MERGE_6133 + timedelta(minutes=30))
    _commit(repo, "src/omnibase_infra/b.py", MERGE_6133)
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    _git(repo, "config", "remote.origin.url", "https://example.invalid/x.git")
    _git(
        repo,
        "config",
        "remote.origin.fetch",
        "+refs/heads/*:refs/remotes/origin/*",
    )
    _git(repo, "config", "branch.dev.remote", "origin")
    _git(repo, "config", "branch.dev.merge", "refs/heads/dev")
    return repo


def test_read_runtime_merge_names_the_newest_runtime_commit_after_the_job(
    clone: Path,
) -> None:
    assert read_runtime_merge(clone, "origin/dev", JOB_START) == MERGE_6133


def test_read_runtime_merge_ignores_runtime_commits_the_job_already_checked_out(
    clone: Path,
) -> None:
    assert read_runtime_merge(clone, "origin/dev", MERGE_6133) is None


def test_read_runtime_merge_ignores_non_runtime_commits(clone: Path) -> None:
    after = MERGE_6133 + timedelta(minutes=1)
    assert read_runtime_merge(clone, "origin/dev", after) is None


def test_read_runtime_merge_follows_the_clones_own_upstream(clone: Path) -> None:
    assert read_runtime_merge(clone, UPSTREAM_REF, JOB_START) == MERGE_6133


def test_a_clone_with_no_upstream_reads_no_merge(clone: Path) -> None:
    _git(clone, "config", "--unset", "branch.dev.merge")

    assert read_runtime_merge(clone, UPSTREAM_REF, JOB_START) is None


def test_read_runtime_merge_unreadable_is_no_merge(tmp_path: Path) -> None:
    assert read_runtime_merge(tmp_path / "absent", "origin/dev", JOB_START) is None


def _write_jobs(state: Path) -> None:
    for job in FIXTURE["jobs"][:1]:
        (state / f"{job['id']}.json").write_text(
            json.dumps(
                {
                    "command": {
                        "runtime_lane": "dev",
                        "requested_by": job["requested_by"],
                    },
                    "accepted_at": job["accepted_at"],
                    "status": job["status"],
                    "phase_results": {"runtime": "failed"},
                }
            )
        )


def test_newest_dev_job_accepted_reads_the_job_records(tmp_path: Path) -> None:
    _write_jobs(tmp_path)
    (tmp_path / "another-lane.json").write_text(
        json.dumps(
            {
                "command": {"runtime_lane": "other-lane"},
                "accepted_at": "2026-11-21T17:00:00Z",
            }
        )
    )
    (tmp_path / "accept_backlog.json").write_text(json.dumps({"status": "healthy"}))

    assert newest_dev_job_accepted(tmp_path) == JOB_START
    assert newest_dev_job_accepted(tmp_path / "absent") is None


# --------------------------------------------------------------------------- #
# The window: the replay of the recorded outage                                #
# --------------------------------------------------------------------------- #
class _Bus:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    def __call__(self, announcement: Any) -> bool:
        self.sent.append(announcement)
        return True


def _agent(
    tmp_path: Path, at: datetime, bus: _Bus, *, jobs: bool = True
) -> SimpleNamespace:
    """An agent at a settable clock holding the window gate the checkout's policy builds."""
    if jobs:
        _write_jobs(tmp_path)
    gate = DevRecreateWindowGate(_policy(), bus, clock=lambda: at)
    return SimpleNamespace(
        _recreate_gate=gate, job_store=SimpleNamespace(state_dir=tmp_path)
    )


def _merge_read(
    monkeypatch: pytest.MonkeyPatch, merged: datetime | None
) -> list[datetime]:
    reads: list[datetime] = []

    def read(_clone: Path, ref: str, since: datetime, **_kw: Any) -> datetime | None:
        assert ref == UPSTREAM_REF, "the clone's own upstream, never a literal branch"
        reads.append(since)
        return merged if merged is not None and merged > since else None

    monkeypatch.setattr(runtime_dev_roll_events, "read_runtime_merge", read)
    return reads


def test_the_idle_restore_is_admitted_off_the_window_for_the_recorded_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay: 6133 merged 14:06:14Z, 831062d1 (13:00:38Z) built before it, window missed."""
    bus = _Bus()
    reads = _merge_read(monkeypatch, MERGE_6133)
    agent = _agent(tmp_path, LANE_DOWN_READ, bus)

    assert window_admits(agent, IDLE_RESTORE_REQUESTER) is True

    assert reads == [JOB_START], "the merge is read after the lane's newest dev job"
    assert bus.sent[-1].requested_by == MERGE_RESTORE_REQUESTER
    assert bus.sent[-1].immediate is True


def test_the_idle_restore_is_held_to_the_next_window_without_a_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus = _Bus()
    _merge_read(monkeypatch, None)
    agent = _agent(tmp_path, LANE_DOWN_READ, bus)

    assert window_admits(agent, IDLE_RESTORE_REQUESTER) is False
    assert bus.sent[-1].next_recreate_at == WINDOW_15
    assert bus.sent[-1].immediate is False


def test_a_merge_the_newest_job_already_built_does_not_open_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 15:00:27Z restore c8fd5fda staged both merges: nothing is waiting after it."""
    bus = _Bus()
    _merge_read(monkeypatch, MERGE_6133)
    _write_jobs(tmp_path)
    (tmp_path / "c8fd5fda.json").write_text(
        json.dumps(
            {
                "command": {"runtime_lane": "dev"},
                "accepted_at": FIXTURE["jobs"][1]["accepted_at"],
            }
        )
    )
    agent = _agent(tmp_path, _t("2026-11-21T15:40:00Z"), bus, jobs=False)

    assert window_admits(agent, IDLE_RESTORE_REQUESTER) is False


@pytest.mark.parametrize(
    "requester", ["agent/idle-converge", "agent/stability-follow-tag", "gha/x/pr-1"]
)
def test_only_the_idle_restore_asks_again_for_a_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requester: str
) -> None:
    reads = _merge_read(monkeypatch, MERGE_6133)
    agent = _agent(tmp_path, LANE_DOWN_READ, _Bus())

    assert window_admits(agent, requester) is False
    assert reads == []


def test_an_unannounced_merge_restore_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Announce first: a bus that refuses the announcement holds the restore."""
    _merge_read(monkeypatch, MERGE_6133)
    agent = _agent(tmp_path, LANE_DOWN_READ, _Bus())
    agent._recreate_gate = DevRecreateWindowGate(
        _policy(), lambda _a: False, clock=lambda: LANE_DOWN_READ
    )

    assert window_admits(agent, IDLE_RESTORE_REQUESTER) is False


def test_the_window_still_admits_the_idle_restore_at_the_hour_without_reading_a_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _merge_read(monkeypatch, MERGE_6133)
    agent = _agent(tmp_path, WINDOW_15 + timedelta(seconds=27), _Bus())

    assert window_admits(agent, IDLE_RESTORE_REQUESTER) is True
    assert reads == []
