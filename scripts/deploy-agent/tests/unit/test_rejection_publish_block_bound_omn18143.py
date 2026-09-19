# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A rejection publish may not hold the job thread for a minute (OMN-18143).

WHAT WAS MEASURED
-----------------
On the .201 dev lane on 2026-09-19, of 1714 topics on the broker,
``onex.cmd.deploy.rebuild-requested.v1`` and
``onex.evt.deploy.rebuild-completed.v1`` are present and
``onex.evt.deploy.rebuild-rejected.v1`` is ABSENT. The two present ones are the
positive control for that probe, which ran read-only against the agent's own
Kafka configuration.

``KafkaProducer.send`` blocks up to ``max_block_ms`` resolving a topic's
metadata, and kafka-python's default is 60_000. So every rejection this agent
publishes waits a full minute and then raises. Two attempts for correlation
``63858212`` took exactly that, at 11:50:49Z and 11:51:49Z, while a completion
publish to the PRESENT topic succeeded at 11:49:34Z on the same config.

WHY IT MATTERS MORE SINCE COALESCING
-------------------------------------
Every blocking call in this process runs on ONE job thread
(``JOB_POOL_MAX_WORKERS = 1``), and coalescing made rejections common: before
it, the journal showed zero ``Rejecting command`` lines in fourteen days, so
the path effectively never fired. A superseded record carries a publish debt
that the retry loop replays every 30 s until the circuit breaker trips at ten
consecutive failures, so at the default block that is ten minutes of job-thread
time per superseded record, spent on a send that cannot land.

WHAT THIS DOES NOT CLAIM TO FIX
--------------------------------
The topic's absence, and the fact that nothing in any repository consumes it.
Both are recorded on OMN-18143. Bounding the cost of a failure is not the same
as making the publish succeed, and this test asserts only the first.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.events import EnumRejectionReason, ModelRebuildRejected, Scope

pytestmark = pytest.mark.unit


class _RecordingProducer:
    """Captures the kwargs the agent constructs its producer with."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def send(self, *args: Any, **kwargs: Any) -> None:
        return None

    def flush(self, timeout: float | None = None) -> None:
        return None

    def close(self) -> None:
        return None


def _agent() -> Any:
    instance = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    instance._kafka_config = _StubConfig()
    return instance


class _StubConfig:
    def producer_kwargs(self) -> dict[str, Any]:
        return {"bootstrap_servers": "broker:9092", "enable_idempotence": True}


def _event() -> ModelRebuildRejected:
    return ModelRebuildRejected(
        correlation_id=uuid4(),
        reason=EnumRejectionReason.BUSY,
        scope=Scope.FULL,
    )


def test_the_rejection_publisher_bounds_its_metadata_wait() -> None:
    """Without this, a missing topic costs 60 s of the single job thread."""
    with patch.object(agent_mod, "KafkaProducer", _RecordingProducer, create=True):
        with patch.dict(
            "sys.modules",
            {"kafka": type("m", (), {"KafkaProducer": _RecordingProducer})},
        ):
            published = _agent()._publish_rejection_event(_event())

    assert published is True
    assert (
        _RecordingProducer.last_kwargs.get("max_block_ms")
        == agent_mod.REJECTION_PUBLISH_MAX_BLOCK_MS
    ), (
        "the producer must carry an explicit max_block_ms; kafka-python's "
        "default of 60 s is how one unpublishable rejection occupies the job "
        "thread for ten minutes across the retry loop's attempts"
    )


def test_the_bound_is_well_under_the_default_and_above_a_healthy_publish() -> None:
    """A number with no relation to either end would be a guess, not a bound.

    Below kafka-python's 60 s default, or the change buys nothing. Far above
    the sub-second publish this lane achieves when the topic exists, or a
    healthy publish starts failing on timing.
    """
    assert agent_mod.REJECTION_PUBLISH_MAX_BLOCK_MS < 60_000
    assert agent_mod.REJECTION_PUBLISH_MAX_BLOCK_MS >= 5_000


def test_the_transport_config_is_still_taken_from_the_kafka_config() -> None:
    """The bound is an addition, never a replacement for the declared transport."""
    with patch.dict(
        "sys.modules",
        {"kafka": type("m", (), {"KafkaProducer": _RecordingProducer})},
    ):
        _agent()._publish_rejection_event(_event())

    assert _RecordingProducer.last_kwargs["bootstrap_servers"] == "broker:9092"
    assert _RecordingProducer.last_kwargs["enable_idempotence"] is True
