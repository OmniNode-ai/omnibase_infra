# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Root integration coverage for non-dispatch savings projection fields."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.services.observability.savings_estimation.consumer import (
    ServiceSavingsEstimator,
)


def _service() -> ServiceSavingsEstimator:
    return ServiceSavingsEstimator(
        ConfigSavingsEstimation(
            kafka_bootstrap_servers="localhost:19092",
            grace_window_seconds=1.0,
            session_timeout_seconds=3600.0,
            max_sessions=100,
            finalized_session_cache_size=1000,
            schema_version="1.0",
        )
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_dispatch_session_populates_projection_costs() -> None:
    service = _service()

    with patch("time.monotonic", return_value=1000.0):
        service.ingest_event(
            "onex.evt.omniintelligence.llm-call-completed.v1",
            {
                "session_id": "session-non-dispatch-root",
                "model_id": "claude-sonnet-4",
                "prompt_tokens": 5000,
                "completion_tokens": 1000,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.hook-context-injected.v1",
            {
                "session_id": "session-non-dispatch-root",
                "tokens_injected": 300,
                "patterns_count": 3,
            },
        )
        service.ingest_event(
            "onex.evt.omniclaude.session-outcome.v1",
            {"session_id": "session-non-dispatch-root"},
        )

    with patch("time.monotonic", return_value=1002.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 1
    row = results[0]
    assert row["model_local"] == "claude-sonnet-4"
    assert row["model_cloud_baseline"] == "claude-opus-4-6"
    assert row["local_cost_usd"] == row["actual_cost_usd"]
    assert row["cloud_cost_usd"] == pytest.approx(
        row["actual_cost_usd"] + row["estimated_total_savings_usd"]
    )
    assert row["savings_usd"] == row["estimated_total_savings_usd"]