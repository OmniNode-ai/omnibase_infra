# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The branch-install reapply entry and its seams (OMN-24013).

The deploy agent runs this module in place of the branch-install handler; when the
handler restarted containers it must publish the dev runtime start event for the
reapply. A failed publication never fails or delays the deploy.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_dev_roll_events_effect import (
    runtime_dev_branch_install_reapply as reapply,
)
from omnibase_infra.nodes.node_dev_roll_events_effect import runtime_dev_roll_events
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    EnumDevStartCause,
)

pytestmark = pytest.mark.unit

TOPIC_DEV_RUNTIME_STARTED = "onex.evt.omnimarket.dev-runtime-started.v1"


class _RecordingPublisher:
    """A publisher that records its started calls."""

    def __init__(self, *, raises: bool = False) -> None:
        self.started_calls: list[tuple[str, datetime, str]] = []
        self._raises = raises

    def started(self, job_id: str, started_at: datetime, cause: str) -> None:
        if self._raises:
            raise OSError("broker down")
        self.started_calls.append((job_id, started_at, cause))


def _write_job(state_dir: Path, correlation_id: str, status: str) -> None:
    (state_dir / f"{correlation_id}.json").write_text(
        json.dumps({"correlation_id": correlation_id, "status": status}),
        encoding="utf-8",
    )


def _handler_main(stdout: str, exit_code: int) -> Callable[[Sequence[str] | None], int]:
    def handler_main(argv: Sequence[str] | None) -> int:
        print(stdout, end="")
        return exit_code

    return handler_main


def _run(
    tmp_path: Path,
    stdout: str,
    *,
    exit_code: int = 0,
    publisher: _RecordingPublisher | None = None,
    build: Callable[[Path], _RecordingPublisher] | None = None,
    state_dir: Path | None = None,
) -> tuple[int, str]:
    state_dir = state_dir if state_dir is not None else tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    publisher = publisher if publisher is not None else _RecordingPublisher()
    build = build if build is not None else (lambda repo_root: publisher)
    captured: list[str] = []

    def announce(state_dir: Path, stdout: str, *, repo_root: Path) -> bool:
        captured.append(stdout)
        return reapply.announce_reapply_start(
            state_dir,
            stdout,
            repo_root=repo_root,
            build=build,
        )

    exit_code = reapply.main(
        ["apply", "--state-dir", str(state_dir)],
        handler_main=_handler_main(stdout, exit_code),
        announce=announce,
    )
    return exit_code, captured[0] if captured else ""


def test_a_restart_with_an_active_job_publishes_one_start_event(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_job(state_dir, "job-1", "in_progress")
    publisher = _RecordingPublisher()
    stdout = 'RECEIPT {"restarted": ["c1"]}\n'

    exit_code, captured = _run(tmp_path, stdout, exit_code=0, publisher=publisher)

    assert exit_code == 0
    assert captured == stdout
    assert len(publisher.started_calls) == 1
    job_id, _started_at, cause = publisher.started_calls[0]
    assert job_id == "job-1"
    assert cause == EnumDevStartCause.BRANCH_INSTALL_REAPPLY


def test_no_restart_publishes_nothing(tmp_path: Path) -> None:
    publisher = _RecordingPublisher()

    _run(tmp_path, "no receipt here\n", publisher=publisher)

    assert publisher.started_calls == []


def test_two_restarts_still_publish_one_start_event(tmp_path: Path) -> None:
    publisher = _RecordingPublisher()
    stdout = 'RECEIPT {"restarted": ["c1", "c2"]}\n'

    _run(tmp_path, stdout, publisher=publisher)

    assert len(publisher.started_calls) == 1


def test_no_active_job_uses_a_reapply_job_id(tmp_path: Path) -> None:
    publisher = _RecordingPublisher()
    stdout = 'RECEIPT {"restarted": ["c1"]}\n'

    _run(tmp_path, stdout, publisher=publisher)

    ((job_id, _started_at, _cause),) = publisher.started_calls
    assert job_id.startswith("branch-install-reapply/")


def test_a_success_job_is_not_active(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_job(state_dir, "done-1", "success")
    publisher = _RecordingPublisher()
    stdout = 'RECEIPT {"restarted": ["c1"]}\n'

    _run(tmp_path, stdout, publisher=publisher, state_dir=state_dir)

    ((job_id, _started_at, _cause),) = publisher.started_calls
    assert job_id.startswith("branch-install-reapply/")


def test_a_build_failure_never_fails_the_deploy(tmp_path: Path) -> None:
    def build(_repo_root: Path) -> _RecordingPublisher:
        raise OSError("no contract")

    publisher = _RecordingPublisher()
    stdout = 'RECEIPT {"restarted": ["c1"]}\n'

    exit_code, _captured = _run(
        tmp_path, stdout, exit_code=3, publisher=publisher, build=build
    )

    assert exit_code == 3
    assert publisher.started_calls == []


def test_a_publish_failure_never_fails_the_deploy(tmp_path: Path) -> None:
    publisher = _RecordingPublisher(raises=True)
    stdout = 'RECEIPT {"restarted": ["c1"]}\n'

    exit_code, _captured = _run(tmp_path, stdout, exit_code=3, publisher=publisher)

    assert exit_code == 3
    assert publisher.started_calls == []


def test_an_unparsable_receipt_counts_zero(tmp_path: Path) -> None:
    assert reapply.restart_count('RECEIPT {"restarted": ["c1"]}\n') == 1
    assert reapply.restart_count("RECEIPT not json\n") == 0
    assert reapply.restart_count('RECEIPT {"restarted": "not-a-list"}\n') == 0


def test_the_contract_declares_the_start_topic() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    declared = runtime_dev_roll_events.declaration_at(repo_root)

    topics = declared["topics"]
    assert isinstance(topics, dict)
    assert topics["dev_runtime_started"] == TOPIC_DEV_RUNTIME_STARTED
