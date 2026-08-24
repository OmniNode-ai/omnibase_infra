# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts.edge_delegation_worker.worker_cycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from scripts.edge_delegation_worker.models import ModelDelegationEnvelope
from scripts.edge_delegation_worker.topic_constants import (
    DELEGATION_COMPLETED_TOPIC,
    DELEGATION_FAILED_TOPIC,
)
from scripts.edge_delegation_worker.worker_cycle import (
    SessionTerminatedError,
    UnsupportedRenewalModeError,
    run_single_cycle,
    run_worker_loop,
)
from tests.unit.scripts.edge_delegation_worker.fakes import FakeDelegationChannel

pytestmark = pytest.mark.unit


def _write_bearer_credential(
    path: Path, bearer_value: str = "pre-issued-token"
) -> Path:
    path.write_text(bearer_value, encoding="utf-8")
    path.chmod(0o600)
    return path


def _envelope(
    payload: dict[str, object],
    *,
    topic: str = "onex.cmd.omnibase-infra.delegation-request.v1",
) -> ModelDelegationEnvelope:
    return ModelDelegationEnvelope(
        correlation_id=uuid4(),
        source_topic=topic,
        event_type="omnibase-infra.delegation-request",
        payload=payload,
    )


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


@pytest.mark.asyncio
async def test_run_single_cycle_returns_false_on_empty_queue(
    http_client: httpx.AsyncClient,
) -> None:
    channel = FakeDelegationChannel()
    processed = await run_single_cycle(
        channel=channel,
        http_client=http_client,
        model_base="http://127.0.0.1:1",
        model_name="qwen-local",
    )
    assert processed is False


@pytest.mark.asyncio
async def test_run_single_cycle_success_publishes_completed_and_acks(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_json(
        {
            "choices": [
                {"message": {"content": "the answer"}, "finish_reason": "stop"}
            ],
        }
    )
    envelope = _envelope({"prompt": "what is the answer"})
    channel = FakeDelegationChannel([envelope])

    processed = await run_single_cycle(
        channel=channel,
        http_client=http_client,
        model_base=httpserver.url_for(""),  # type: ignore[attr-defined]
        model_name="qwen-local",
    )

    assert processed is True
    assert channel.acked == [envelope.correlation_id]
    assert not channel.nacked
    assert len(channel.published) == 1
    published = channel.published[0]
    assert published.topic == DELEGATION_COMPLETED_TOPIC
    assert published.correlation_id == envelope.correlation_id
    assert published.payload["content"] == "the answer"


@pytest.mark.asyncio
async def test_run_single_cycle_unusable_payload_publishes_failed_and_nacks(
    http_client: httpx.AsyncClient,
) -> None:
    envelope = _envelope({"unrelated_field": "no prompt or messages here"})
    channel = FakeDelegationChannel([envelope])

    processed = await run_single_cycle(
        channel=channel,
        http_client=http_client,
        model_base="http://127.0.0.1:1",
        model_name="qwen-local",
    )

    assert processed is True
    assert not channel.acked
    assert channel.nacked and channel.nacked[0][0] == envelope.correlation_id
    assert channel.published[0].topic == DELEGATION_FAILED_TOPIC


@pytest.mark.asyncio
async def test_run_single_cycle_model_failure_publishes_failed_and_nacks(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_data("bad request", status=400)
    envelope = _envelope({"prompt": "hello"})
    channel = FakeDelegationChannel([envelope])

    processed = await run_single_cycle(
        channel=channel,
        http_client=http_client,
        model_base=httpserver.url_for(""),  # type: ignore[attr-defined]
        model_name="qwen-local",
    )

    assert processed is True
    assert not channel.acked
    assert channel.nacked and channel.nacked[0][0] == envelope.correlation_id
    assert channel.published[0].topic == DELEGATION_FAILED_TOPIC


@pytest.mark.asyncio
async def test_run_worker_loop_end_to_end_bounded(
    httpserver: object, http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    """One full attach -> claim -> infer -> publish -> ack -> detach cycle."""
    session_id = str(uuid4())
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/attach", method="POST"
    ).respond_with_json(
        {
            "session": {
                "session_id": session_id,
                "expires_at": "2099-01-01T00:00:00+00:00",
                "heartbeat_interval_seconds": 3600,
            }
        }
    )
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/detach", method="POST"
    ).respond_with_json({"session_id": session_id, "detached": True})
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_json(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    )

    envelope = _envelope({"prompt": "hello"})
    channel = FakeDelegationChannel([envelope])
    credential_path = _write_bearer_credential(tmp_path / "cred")

    await run_worker_loop(
        api_base=httpserver.url_for(""),  # type: ignore[attr-defined]
        model_base=httpserver.url_for(""),  # type: ignore[attr-defined]
        model_name="qwen-local",
        credential_path=credential_path,
        edge_instance_id="edge-1",
        channel=channel,
        http_client=http_client,
        poll_interval_seconds=0.01,
        max_cycles=1,
    )

    assert channel.acked == [envelope.correlation_id]
    assert channel.published[0].topic == DELEGATION_COMPLETED_TOPIC


@pytest.mark.asyncio
async def test_run_worker_loop_rejects_unsupported_renewal_mode(
    httpserver: object, http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    session_id = str(uuid4())
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/attach", method="POST"
    ).respond_with_json(
        {
            "session": {
                "session_id": session_id,
                "expires_at": "2099-01-01T00:00:00+00:00",
                "heartbeat_interval_seconds": 3600,
            },
            "renewal": {
                "mode": "IN_PLACE_EXTEND",
                "renew_not_before": "2099-01-01T00:00:00+00:00",
                "renew_at": "2099-01-01T00:00:00+00:00",
                "session_expires_at": "2099-01-01T00:00:00+00:00",
            },
        }
    )
    channel = FakeDelegationChannel()
    credential_path = _write_bearer_credential(tmp_path / "cred")

    with pytest.raises(UnsupportedRenewalModeError):
        await run_worker_loop(
            api_base=httpserver.url_for(""),  # type: ignore[attr-defined]
            model_base=httpserver.url_for(""),  # type: ignore[attr-defined]
            model_name="qwen-local",
            credential_path=credential_path,
            edge_instance_id="edge-1",
            channel=channel,
            http_client=http_client,
            poll_interval_seconds=0.01,
            max_cycles=1,
        )


@pytest.mark.asyncio
async def test_run_worker_loop_stops_on_terminated_heartbeat(
    httpserver: object, http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    session_id = str(uuid4())
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/attach", method="POST"
    ).respond_with_json(
        {
            "session": {
                "session_id": session_id,
                "expires_at": "2099-01-01T00:00:00+00:00",
                "heartbeat_interval_seconds": 1,
            }
        }
    )
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/heartbeat", method="POST"
    ).respond_with_json({"session_id": session_id, "termination_reason": "REVOKED"})
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/detach", method="POST"
    ).respond_with_json({"session_id": session_id, "detached": True})

    channel = FakeDelegationChannel()
    credential_path = _write_bearer_credential(tmp_path / "cred")

    # Fake clock: attach's last_heartbeat is stamped at call #1; call #2 (the
    # loop's first "now") is already past heartbeat_interval_seconds=1, so
    # the very first iteration triggers the heartbeat deterministically --
    # no real sleeping needed.
    clock_calls = [datetime(2026, 1, 1, tzinfo=UTC)]
    _later = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)

    def fake_now() -> datetime:
        return clock_calls.pop(0) if clock_calls else _later

    with pytest.raises(SessionTerminatedError):
        await run_worker_loop(
            api_base=httpserver.url_for(""),  # type: ignore[attr-defined]
            model_base=httpserver.url_for(""),  # type: ignore[attr-defined]
            model_name="qwen-local",
            credential_path=credential_path,
            edge_instance_id="edge-1",
            channel=channel,
            http_client=http_client,
            poll_interval_seconds=0.01,
            max_cycles=5,
            now_fn=fake_now,
        )
