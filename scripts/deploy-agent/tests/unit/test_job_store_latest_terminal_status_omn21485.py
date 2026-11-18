# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21485: the dev lane's deploy receipt, read out of the job store.

``decide_stability_follow`` takes the stability lane onto the omnimarket dev
tag only when the dev lane's own newest terminal deploy record is a success --
the receipt the post-merge lab guard reads. That receipt is
``JobStore.latest_terminal_status("dev")``, so these tests pin the store's own
rule: the newest terminal record OF THE LANE only. A superseded record never
ran, a job still accepted or in_progress has not finished, another lane's
record is not this lane's, an unparseable file is skipped -- and no terminal
record at all reads as ``None``, so a caller cannot tell a pass from a
not-yet-run and refuses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

DEV = EnumRuntimeLane.DEV.value
STABILITY = EnumRuntimeLane.STABILITY_TEST.value

#: Terminal records carry the wall clock ``complete`` stamped them with, so two
#: writes closer than this could share a microsecond and leave "newest"
#: undefined. A millisecond keeps the order the test asserts strict.
WRITE_GAP_SECONDS = 0.002


def _command(lane: str) -> dict[str, object]:
    return {"scope": "runtime", "runtime_lane": lane}


def _terminal_job(
    store: JobStore, lane: str, status: Literal["success", "failed"]
) -> None:
    correlation_id = uuid4()
    store.accept(correlation_id, _command(lane))
    store.complete(correlation_id, status=status)


class TestLatestTerminalStatus:
    def test_a_lane_with_no_records_has_no_receipt(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs")
        assert store.latest_terminal_status(DEV) is None

    def test_the_newest_terminal_record_of_the_lane_is_its_receipt(
        self, tmp_path: Path
    ) -> None:
        store = JobStore(tmp_path / "jobs")
        _terminal_job(store, DEV, "success")
        time.sleep(WRITE_GAP_SECONDS)
        _terminal_job(store, DEV, "failed")
        assert store.latest_terminal_status(DEV) == "failed"
        time.sleep(WRITE_GAP_SECONDS)
        _terminal_job(store, DEV, "success")
        assert store.latest_terminal_status(DEV) == "success"

    def test_a_newer_record_of_another_lane_is_not_this_lanes_receipt(
        self, tmp_path: Path
    ) -> None:
        store = JobStore(tmp_path / "jobs")
        _terminal_job(store, DEV, "failed")
        time.sleep(WRITE_GAP_SECONDS)
        _terminal_job(store, STABILITY, "success")
        assert store.latest_terminal_status(DEV) == "failed"

    def test_superseded_and_in_flight_records_never_count(self, tmp_path: Path) -> None:
        """A record that never ran, or has not finished, says nothing."""
        store = JobStore(tmp_path / "jobs")
        _terminal_job(store, DEV, "success")
        time.sleep(WRITE_GAP_SECONDS)
        # A superseded record is newer AND terminal -- and still not a receipt.
        superseded = uuid4()
        store.record_superseded(
            superseded,
            _command(DEV),
            superseded_by_sha="a" * 40,
            superseded_by_correlation_id=uuid4(),
        )
        active = uuid4()
        store.accept(active, _command(DEV))
        store.update_phase(active, Phase.RUNTIME, PhaseStatus.IN_PROGRESS)
        assert store.latest_terminal_status(DEV) == "success"

    def test_an_unparseable_record_is_skipped(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "jobs")
        _terminal_job(store, DEV, "success")
        (store.state_dir / "not-a-job.json").write_text("{", encoding="utf-8")
        assert store.latest_terminal_status(DEV) == "success"
        assert store.latest_terminal_status(STABILITY) is None
