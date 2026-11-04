# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay excerpted census findings; start instants are synthesized, not recorded."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.handlers.handler_census_oneshot import (
    grade_oneshot,
    load_policy,
)
from omnibase_infra.models.model_oneshot_observation import ModelOneshotObservation


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/census_oneshot_fresh_omn19998.json"
)
def test_oneshot_recorded_statuses_are_pending_then_alarm(
    recorded_response: dict[str, Any],
) -> None:
    policy = load_policy(
        Path(__file__).resolve().parents[2]
        / "src/omnibase_infra/nodes/node_census_oneshot_effect/contract.yaml"
    )
    bound = policy.phase_timeouts["runtime"]
    response = recorded_response["response"]
    observed = datetime.fromisoformat(response["observed_at"])
    for row, age in zip(response["rows"], (0.25, 15), strict=True):
        request = ModelOneshotObservation(
            completion_timeout_seconds=bound,
            lane="dev",
            container=row["Names"],
            running=row["State"] == "running",
            docker_status=row["Status"],
            started_at=(observed - timedelta(seconds=age)).isoformat(),
            observed_at=observed,
        )
        fresh = grade_oneshot(request)
        assert not fresh.kind
        assert fresh.pending is not None
        assert fresh.pending.remaining_seconds == bound - age
        expired = grade_oneshot(
            request.model_copy(
                update={"observed_at": observed + timedelta(seconds=bound)}
            )
        )
        assert expired.kind == "oneshot_stuck"
        assert expired.pending is None
