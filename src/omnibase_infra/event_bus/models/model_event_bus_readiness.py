# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event bus readiness status model.

Provides structured readiness information for Kafka event bus consumers,
separating readiness (can process events) from liveness (process alive).

This model is used by:
    - ``EventBusKafka.get_readiness_status()`` to report consumer readiness
    - ``RuntimeHostProcess.readiness_check()`` to aggregate readiness
    - ``ServiceHealth._handle_readiness()`` to serve ``/ready`` endpoint
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEventBusReadiness(BaseModel):
    """Structured readiness status for an event bus instance.

    Readiness is defined as: safe to receive traffic that depends on
    Kafka-driven orchestration. A bus is ready when all topics marked
    ``required_for_readiness=True`` at subscribe time have active
    consumers with partition assignments.

    Readiness is continuously evaluated, not a one-time gate. Loss of
    required consumer assignments flips readiness to False.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    is_ready: bool = Field(
        ...,
        description="Overall readiness: True only when all required topics are ready.",
    )
    consumers_started: bool = Field(
        ...,
        description="Whether the event bus has been started and consumers are running.",
    )
    assignments: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Current partition assignments per topic (topic -> partition list).",
    )
    consume_tasks_alive: dict[str, bool] = Field(
        default_factory=dict,
        description="Whether the consume loop task is alive per topic.",
    )
    required_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics marked as required_for_readiness at subscribe time.",
    )
    required_topics_ready: bool = Field(
        ...,
        description="Whether all required topics have active partition assignments.",
    )
    last_error: str = Field(
        default="",
        description="Last error encountered during readiness evaluation, empty if none.",
    )
