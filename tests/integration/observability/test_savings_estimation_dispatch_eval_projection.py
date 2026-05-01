# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dispatch-eval projection coverage for savings estimation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.services.observability.savings_estimation.consumer import (
    ServiceSavingsEstimator,
)

TOPIC_DISPATCH_OUTCOME = "onex.evt.omniintelligence.dispatch-outcome-evaluated.v1"


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
async def test_dispatch_eval_pass_projects_single_savings_estimate() -> None:
    service = _service()

    with patch("time.monotonic", return_value=1000.0):
        service.ingest_event(
            TOPIC_DISPATCH_OUTCOME,
            {
                "task_id": "t1",
                "dispatch_id": "d1",
                "ticket_id": "OMN-10388",
                "verdict": "pass",
                "quality_score": 0.85,
                "token_cost": 12_500,
                "dollars_cost": 0.031,
                "usage_source": "measured",
                "estimation_method": None,
                "source_payload_hash": "a" * 64,
                "model_local": "claude-sonnet-4",
                "model_cloud_baseline": "claude-opus-4-6",
                "model_calls": [
                    {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4",
                        "input_tokens": 10_000,
                        "output_tokens": 2_500,
                        "latency_ms": 750,
                        "cost_dollars": 0.031,
                    }
                ],
            },
        )

    with patch("time.monotonic", return_value=1002.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 1
    row = results[0]
    assert row["session_id"] == "t1"
    assert row["task_id"] == "t1"
    assert row["dispatch_id"] == "d1"
    assert row["ticket_id"] == "OMN-10388"
    assert row["model_local"] == "claude-sonnet-4"
    assert row["model_cloud_baseline"] == "claude-opus-4-6"
    assert row["usage_source"] == "MEASURED"
    assert row["source_payload_hash"] == "a" * 64
    assert row["savings_usd"] > 0
    assert row["estimated_total_savings_usd"] == row["savings_usd"]
    assert row["local_cost_usd"] == 0.031
    assert row["cloud_cost_usd"] == pytest.approx(0.031 + row["savings_usd"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_eval_replay_finalizes_once_for_dedupe_key() -> None:
    service = _service()
    payload = {
        "task_id": "t1",
        "dispatch_id": "d1",
        "verdict": "PASS",
        "quality_score": 0.85,
        "token_cost": 12_500,
        "dollars_cost": 0.031,
        "usage_source": "MEASURED",
        "source_payload_hash": "a" * 64,
    }

    with patch("time.monotonic", return_value=1000.0):
        service.ingest_event(TOPIC_DISPATCH_OUTCOME, payload)
        service.ingest_event(TOPIC_DISPATCH_OUTCOME, payload)

    with patch("time.monotonic", return_value=1002.0):
        results = await service.finalize_ready_sessions()

    assert len(results) == 1
    assert results[0]["session_id"] == "t1"
    assert results[0]["direct_tokens_saved"] == 12_500
