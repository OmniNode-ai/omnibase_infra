# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23075 — the redeployed gateway runtime stops replaying failed reads.

omnibase_infra#5795 (merge commit f84141dbd) classifies a failed GitHub gateway
read as non-retryable. These tests hold the two readings taken on the lab dev
lane after the deploy agent redeployed the runtime from it, recorded in
``tests/fixtures/omn23075/gateway_redeploy_readings.json``:

* AC1: the deploy-agent job for the redeploy and the revision the gateway
  containers report, with the answer to whether the merge commit is an ancestor
  of that revision (read with ``git merge-base --is-ancestor`` on the lab host).
* AC2: every record of the gateway terminal topic over one hour after the
  redeploy, counted per command id.

The count helper is exercised on the pre-fix burst the fix was built from
(the same command answered ten times with a retryable job-log 404), which is the
named injected break: the reading must go red on it, so a green reading means
the replays stopped and not that the reader cannot see them.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "omn23075"
    / "gateway_redeploy_readings.json"
)
_MERGE_COMMIT = "f84141dbdfa06c818621bb7ae9d29ad985b6861f"
_JOB_LOG_404 = "HTTP 404"


class ModelTerminalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str
    envelope_timestamp: str
    status: str
    failure_class: str | None
    retryable: bool | None
    failure_reason: str | None


class ModelDeployReading(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    status: str
    requested_git_ref: str
    completed_at: str
    merge_commit: str
    merge_commit_is_ancestor_of_requested_git_ref: bool
    container_revisions: dict[str, str]


class ModelRedeployReadings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket: str
    terminal_topic: str
    window_start: str
    window_end: str
    unparsed_records: int
    deploy: ModelDeployReading
    records: list[ModelTerminalRecord]


def _load() -> ModelRedeployReadings:
    return ModelRedeployReadings.model_validate_json(_FIXTURE.read_text())


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)


def replayed_retryable_job_log_404s(
    records: list[ModelTerminalRecord],
) -> dict[str, int]:
    """Command ids answered more than once with a retryable job-log 404 failure."""
    per_id: Counter[str] = Counter(
        record.correlation_id
        for record in records
        if record.status == "failed"
        and record.retryable is True
        and _JOB_LOG_404 in (record.failure_reason or "")
    )
    return {cid: count for cid, count in per_id.items() if count > 1}


def _pre_fix_burst() -> list[ModelTerminalRecord]:
    """The shape the gateway topic had before the fix: one command, ten replays."""
    return [
        ModelTerminalRecord(
            correlation_id="cb9580b5-b45d-45bd-bcad-99bb5ded8b00",
            envelope_timestamp=f"2026-11-08T03:33:{second:02d}Z",
            status="failed",
            failure_class="HandlerExecutionError",
            retryable=True,
            failure_reason="job log not cached: HTTP 404, read_ceiling_hit=False",
        )
        for second in range(10)
    ]


@pytest.mark.unit
class TestAc1DeployAgentRun:
    def test_deploy_agent_job_reached_success(self) -> None:
        deploy = _load().deploy
        assert deploy.status == "success"

    def test_gateway_containers_report_the_deployed_revision(self) -> None:
        deploy = _load().deploy
        assert deploy.container_revisions
        assert set(deploy.container_revisions.values()) == {deploy.requested_git_ref}

    def test_deployed_revision_contains_the_merge_commit(self) -> None:
        deploy = _load().deploy
        assert deploy.merge_commit == _MERGE_COMMIT
        assert deploy.merge_commit_is_ancestor_of_requested_git_ref is True


@pytest.mark.unit
class TestAc2GatewayTerminals:
    def test_reading_covers_one_full_hour_after_the_redeploy(self) -> None:
        readings = _load()
        window = _at(readings.window_end) - _at(readings.window_start)
        assert window >= timedelta(hours=1)
        assert _at(readings.window_start) >= _at(readings.deploy.completed_at)

    def test_every_record_is_read_and_parsed(self) -> None:
        readings = _load()
        assert readings.unparsed_records == 0
        assert readings.records
        assert all(
            _at(readings.window_start)
            <= _at(record.envelope_timestamp)
            <= _at(readings.window_end)
            for record in readings.records
        )

    def test_no_job_log_404_is_replayed_as_retryable(self) -> None:
        assert replayed_retryable_job_log_404s(_load().records) == {}

    def test_reading_goes_red_on_the_pre_fix_burst(self) -> None:
        """Injected break: the count helper must see ten replays of one command."""
        replayed = replayed_retryable_job_log_404s(_pre_fix_burst())
        assert replayed == {"cb9580b5-b45d-45bd-bcad-99bb5ded8b00": 10}

    def test_reading_goes_red_when_the_recorded_window_gains_a_replay(self) -> None:
        records = _load().records
        broken = [
            *records,
            *(
                ModelTerminalRecord(
                    correlation_id="11111111-1111-4111-8111-111111111111",
                    envelope_timestamp=records[0].envelope_timestamp,
                    status="failed",
                    failure_class="HandlerExecutionError",
                    retryable=True,
                    failure_reason="job log not cached: HTTP 404, read_ceiling_hit=False",
                )
                for _ in range(2)
            ),
        ]
        assert replayed_retryable_job_log_404s(broken) == {
            "11111111-1111-4111-8111-111111111111": 2
        }
