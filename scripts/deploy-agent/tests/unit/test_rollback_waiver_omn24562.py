# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A stability deploy runs without a rollback point only over a lane read down (OMN-24562)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.job_state import JobState

from omnibase_infra.handlers.handler_stability_rollback_waiver import (
    HandlerStabilityRollbackWaiver,
    admit_rollback_point,
    classify_readings,
    take_readings,
)
from omnibase_infra.models.model_stability_rollback_waiver_request import (
    ModelStabilityRollbackWaiverRequest,
)
from tests.unit.test_auto_rollback import _agent, _check

pytestmark = pytest.mark.unit

_LANE = EnumRuntimeLane.STABILITY_TEST.value
_TIMEOUT = subprocess.TimeoutExpired(["curl"], 10)
_TARGETS = executor_mod.lane_config_for(
    EnumRuntimeLane.STABILITY_TEST
).runtime_health_targets
_PASS = (_check("t"), None)
_FAIL = (_check("t", healthy=False), None)


def read_lane_health(
    targets: Sequence[tuple[str, int]], probe: Mock
) -> tuple[str, str]:
    return classify_readings(take_readings(targets, probe))


def _probe(*answers: object) -> Mock:
    return Mock(side_effect=list(answers))


def test_reading_all_pass_is_healthy() -> None:
    assert read_lane_health(_TARGETS, _probe(_PASS, _PASS))[0] == "healthy"


def test_reading_definitive_failure_is_unhealthy() -> None:
    reading, detail = read_lane_health(_TARGETS, _probe(_FAIL, _PASS))
    assert reading == "unhealthy"
    assert "degraded" in detail


def test_reading_any_timeout_is_unknown_even_beside_a_failure() -> None:
    answers = _probe(_FAIL, (_check("t"), _TIMEOUT))
    assert read_lane_health(_TARGETS, answers)[0] == "unknown"


def test_reading_probe_error_is_unknown() -> None:
    probe = Mock(side_effect=OSError("docker gone"))
    reading, detail = read_lane_health(_TARGETS, probe)
    assert reading == "unknown"
    assert "docker gone" in detail


def test_reading_no_targets_is_unknown() -> None:
    assert read_lane_health((), _probe())[0] == "unknown"


@pytest.mark.parametrize("lane", ["prod", "dev"])
def test_prod_and_dev_are_never_probed_nor_captured(lane: str, tmp_path: Path) -> None:
    capture, probe = Mock(), Mock()
    point = admit_rollback_point(
        lane=lane,
        scope="runtime",
        correlation_id=uuid4(),
        record_dir=tmp_path,
        targets=_TARGETS,
        probe=probe,
        capture=capture,
    )
    assert point is None
    capture.assert_not_called()
    probe.assert_not_called()


def test_core_scope_is_never_probed_nor_captured(tmp_path: Path) -> None:
    capture, probe = Mock(), Mock()
    assert (
        admit_rollback_point(
            lane=_LANE,
            scope="core",
            correlation_id=uuid4(),
            record_dir=tmp_path,
            targets=_TARGETS,
            probe=probe,
            capture=capture,
        )
        is None
    )
    capture.assert_not_called()


def test_a_captured_point_is_returned_and_nothing_is_waived(tmp_path: Path) -> None:
    probe = Mock()
    point = admit_rollback_point(
        lane=_LANE,
        scope="runtime",
        correlation_id=uuid4(),
        record_dir=tmp_path,
        targets=_TARGETS,
        probe=probe,
        capture=lambda: "point",
    )
    assert point == "point"
    probe.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def _deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probes: list[object]
) -> tuple[JobState | None, Mock, Path, UUID]:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.STABILITY_TEST,
    )
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    # No point could be captured: the guard's own probe then reads the lane.
    monkeypatch.setattr(agent_mod, "capture_rollback_point", Mock(return_value=None))
    monkeypatch.setattr(agent_mod, "restore_rollback_point", Mock())
    executor._probe_runtime_health.side_effect = probes
    agent._run_deploy(cmd)
    record_dir = agent.job_store.state_dir / "rollback-waivers"
    return (
        agent.job_store.load(cmd.correlation_id),
        executor,
        record_dir,
        cmd.correlation_id,
    )


def test_unhealthy_lane_without_point_deploys_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, executor, record_dir, cid = _deploy(
        tmp_path, monkeypatch, [_FAIL] * len(_TARGETS)
    )
    assert job is not None and job.status == "success"
    executor.rebuild_scope.assert_called_once()
    record = json.loads((record_dir / f"{cid}.json").read_text(encoding="utf-8"))
    assert record["reading"] == "unhealthy"
    assert record["lane"] == _LANE


def test_healthy_lane_without_point_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, executor, record_dir, _ = _deploy(
        tmp_path, monkeypatch, [_PASS] * len(_TARGETS)
    )
    assert job is not None and job.status == "failed"
    assert "no healthy, retained rollback point" in job.errors[0]
    assert "reads healthy" in job.errors[0]
    assert not record_dir.exists()
    executor.rebuild_scope.assert_not_called()


def test_unknown_lane_without_point_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, executor, record_dir, _ = _deploy(
        tmp_path, monkeypatch, [(_check("t", healthy=False), _TIMEOUT)] * len(_TARGETS)
    )
    assert job is not None and job.status == "failed"
    assert "reads unknown" in job.errors[0]
    assert not record_dir.exists()
    executor.rebuild_scope.assert_not_called()


@pytest.mark.parametrize(
    ("readings", "admitted", "waived", "reading"),
    [
        ((("fail", "main down"), ("pass", "")), True, True, "unhealthy"),
        ((("pass", ""), ("pass", "")), False, False, "healthy"),
        ((("fail", "main down"), ("timeout", "slow")), False, False, "unknown"),
        ((), False, False, "unknown"),
    ],
)
def test_handle_decides_from_readings(
    readings: tuple[tuple[str, str], ...], admitted: bool, waived: bool, reading: str
) -> None:
    result = HandlerStabilityRollbackWaiver().handle(
        ModelStabilityRollbackWaiverRequest(
            lane=_LANE,
            scope="runtime",
            point_retained=False,
            readings=readings,
        )
    )
    assert (result.admitted, result.waived, result.reading) == (
        admitted,
        waived,
        reading,
    )
    assert bool(result.refusal) is (not admitted)


def test_handle_never_waives_prod() -> None:
    result = HandlerStabilityRollbackWaiver().handle(
        ModelStabilityRollbackWaiverRequest(
            lane="prod",
            scope="runtime",
            point_retained=False,
            readings=(("fail", "down"),),
        )
    )
    assert result.admitted and not result.waived
