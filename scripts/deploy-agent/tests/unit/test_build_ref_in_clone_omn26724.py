# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A queued build whose ref the deploy clone lacks is refused and the next one runs (OMN-26724).

THE RECORD
----------
2026-11-26 on h201: overdue dev builds published at 18:16Z by the old agent process carried a
sibling repository's sha as ``git_ref``. The merged-ref fence answers NOT_APPLICABLE for a signed
rollback, nothing checked the clone held the commit, and each window ended in "Could not parse
object", spending one stale record per window while 19 commands waited. node_build_repository_ref_compute
now decides, from what the clone holds, whether the build may start; the agent refuses the rest with a
typed reason and goes on to the next command.
"""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRejectionReason, EnumRuntimeLane
from deploy_agent.lineage_fence import GitRefResolver

pytestmark = pytest.mark.unit

ENTRY = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/handlers/build_repository_ref/agent_entry.py"
)
TOPIC = "onex.cmd.deploy.rebuild-requested.v1"
BAD_SHA = "a" * 40


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def clone(tmp_path: Path) -> tuple[Path, str]:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "dev")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    (origin / "f").write_text("1")
    _git(origin, "add", "f")
    _git(origin, "commit", "-q", "-m", "one")
    work = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True
    )
    return work, _git(work, "rev-parse", "HEAD")


def _check(repo: Path) -> Any:
    entry = runpy.run_path(str(ENTRY))

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )

    return entry["BuildRepositoryRefCheck"](str(repo), run)


def test_a_ref_the_clone_lacks_is_refused_with_a_typed_reason(
    clone: tuple[Path, str],
) -> None:
    repo, _ = clone
    decision = _check(repo)(BAD_SHA)
    assert decision.refuses is True
    assert decision.verdict.value == "missing"
    assert BAD_SHA in decision.reason


def test_a_real_ref_builds(clone: tuple[Path, str]) -> None:
    repo, sha = clone
    decision = _check(repo)(sha)
    assert decision.refuses is False
    assert decision.verdict.value == "held"


def test_a_miss_fetches_once_and_a_ref_that_arrives_builds(
    clone: tuple[Path, str],
) -> None:
    repo, _ = clone
    origin = repo.parent / "origin"
    (origin / "f").write_text("2")
    _git(origin, "commit", "-q", "-am", "two")
    new_sha = _git(origin, "rev-parse", "HEAD")
    decision = _check(repo)(new_sha)
    assert decision.refuses is False
    assert decision.verdict.value == "held_after_fetch"


def test_a_failed_fetch_leaves_the_ref_unverifiable_not_refused(
    clone: tuple[Path, str],
) -> None:
    repo, _ = clone
    _git(repo, "remote", "set-url", "origin", str(repo.parent / "nowhere"))
    decision = _check(repo)(BAD_SHA)
    assert decision.refuses is False
    assert decision.verdict.value == "unverifiable"


def _command(correlation: str, sha: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "correlation_id": correlation,
        "git_ref": sha,
        "requested_by": "agent/overdue-dev-merge-build",
        "scope": "runtime",
        "runtime_lane": "dev",
        "services": [],
        "rollback": {"actor": "agent/overdue-dev-merge-build", "reason": "overdue"},
        "_signature": "a" * 64,
    }
    payload.update(extra)
    return payload


def _consumer(repo: Path) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.ref_resolver = GitRefResolver(str(repo))
    consumer.ancestry_resolver = None
    consumer.on_rejected = Mock()
    return consumer


def _message(payload: dict[str, Any], offset: int) -> SimpleNamespace:
    return SimpleNamespace(value=payload, topic=TOPIC, partition=0, offset=offset)


def test_a_missing_ref_is_refused_and_the_next_command_runs_in_the_same_window(
    clone: tuple[Path, str],
) -> None:
    repo, good = clone
    consumer = _consumer(repo)
    stale = _command("aaaaaaaa-0000-0000-0000-000000000001", BAD_SHA)
    real = _command("aaaaaaaa-0000-0000-0000-000000000002", good)
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        first, reason = consumer._process_message(_message(stale, 1))
        second, second_reason = consumer._process_message(_message(real, 2))
    assert first is None
    assert reason == EnumRejectionReason.NOT_A_MERGED_REF.value
    assert second is not None and second_reason is None
    assert str(second.correlation_id).endswith("0002")
    consumer.job_store.accept.assert_called_once()


def test_a_pinned_image_promotion_is_not_checked_against_the_clone(
    clone: tuple[Path, str],
) -> None:
    repo, _ = clone
    consumer = _consumer(repo)
    promo = _command(
        "aaaaaaaa-0000-0000-0000-000000000003",
        BAD_SHA,
        image_ref="registry.invalid/omninode-runtime:1",
        rollback=None,
    )
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        _, reason = consumer._process_message(_message(promo, 3))
    assert reason != EnumRejectionReason.NOT_A_MERGED_REF.value
