# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rollback waiver writes its record to a real directory (OMN-24562).

Runs ``admit_rollback_point`` over a real filesystem record directory with a
stub health probe and Docker reader, so the decision, the refusal and the
record file are read back from disk rather than from mocks of the handler.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from omnibase_infra.handlers.handler_stability_rollback_waiver import (
    admit_rollback_point,
)

pytestmark = pytest.mark.integration

LANE = "stability"
_TARGETS = (("main", 8085),)


def _probe(*, service: str, port: int) -> tuple[SimpleNamespace, None]:
    return SimpleNamespace(status="fail", detail=f"{service}:{port} down"), None


def test_a_down_lane_is_waived_and_the_record_is_written(tmp_path: Path) -> None:
    correlation_id = uuid4()
    point = admit_rollback_point(
        lane=LANE,
        scope="runtime",
        correlation_id=correlation_id,
        record_dir=tmp_path / "rollback-waivers",
        targets=_TARGETS,
        probe=_probe,
        state=lambda _container: "unhealthy",
        capture=lambda: None,
    )
    assert point is None
    record = json.loads(
        (tmp_path / "rollback-waivers" / f"{correlation_id}.json").read_text()
    )
    assert record["correlation_id"] == str(correlation_id)
    assert record["lane"] == LANE
    assert record["reading"] == "unhealthy"


def test_a_serving_docker_state_refuses_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        admit_rollback_point(
            lane=LANE,
            scope="runtime",
            correlation_id=uuid4(),
            record_dir=tmp_path / "rollback-waivers",
            targets=_TARGETS,
            probe=_probe,
            state=lambda _container: "healthy",
            capture=lambda: None,
        )
    assert not (tmp_path / "rollback-waivers").exists()
