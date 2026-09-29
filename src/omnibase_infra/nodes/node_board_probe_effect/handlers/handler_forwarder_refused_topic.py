# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``forwarder_refused_topic`` board check (plan task S5).

Reads the lab lane's gateway forwarder and grades it:

* ``PASS``: the process is up, has run for at least one inbound-topic retry
  interval, its cloud leg is the broker the lane declares, and no inbound topic
  is refused;
* ``FAIL``: the process is down, or a topic is refused by the declared broker;
* ``INDETERMINATE``: the state could not be read, the lane declares no cloud
  broker, the forwarder's cloud leg is another broker, the log window is
  shorter than one retry interval, or the process has not yet run one retry
  interval with nothing refused. The lab proof grades INDETERMINATE as a
  failure (``ModelBoardProbeResult.as_lab_proof_check_result``).

A refused topic is attributed to a broker only when the forwarder's cloud leg
is the declared one, so a refusal on some other broker is INDETERMINATE, not a
FAIL of the lane under proof.

Ticket: OMN-19930
"""

from __future__ import annotations

from datetime import timedelta

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_forwarder_state_reader import (
    HandlerDockerForwarderStateReader,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    EnumBoardCheckId,
    EnumBoardCheckSurfaceClass,
    EnumBoardProbeOutcome,
    ModelBoardProbeResult,
    ModelForwarderRefusedTopicRequest,
    ModelForwarderStateObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.protocols import (
    ProtocolForwarderStateReader,
)

CHECK_ID = EnumBoardCheckId.FORWARDER_REFUSED_TOPIC
SURFACE_CLASS = EnumBoardCheckSurfaceClass.LAB_HARDWARE


def _brokers(bootstrap: str) -> frozenset[str]:
    """A bootstrap string as a set of ``host:port`` entries, order-free."""
    return frozenset(part.strip() for part in bootstrap.split(",") if part.strip())


def grade_forwarder_refused_topic(
    request: ModelForwarderRefusedTopicRequest,
    observation: ModelForwarderStateObservation,
) -> ModelBoardProbeResult:
    """Grade one forwarder observation against what its lane declares."""

    def result(
        outcome: EnumBoardProbeOutcome,
        *reasons: str,
        evidence: tuple[str, ...] = (),
    ) -> ModelBoardProbeResult:
        return ModelBoardProbeResult(
            check_id=CHECK_ID,
            surface_class=SURFACE_CLASS,
            subject=request.subject_lane,
            outcome=outcome,
            reasons=reasons,
            evidence_items=evidence,
            observed_at=observation.observed_at,
        )

    indeterminate = EnumBoardProbeOutcome.INDETERMINATE
    container = request.forwarder_container
    if not observation.read_ok:
        return result(
            indeterminate, f"{container}: state unreadable: {observation.read_error}"
        )
    if not observation.running:
        return result(EnumBoardProbeOutcome.FAIL, f"{container}: process is down")

    declared = _brokers(request.declared_cloud_broker)
    observed = _brokers(observation.observed_cloud_broker)
    if not declared:
        return result(
            indeterminate,
            f"lane {request.subject_lane} declares no cloud broker, so a refusal "
            "cannot be attributed to its cloud leg",
        )
    if observed != declared:
        return result(
            indeterminate,
            f"{container}: cloud leg {observation.observed_cloud_broker or '<unresolved>'} "
            f"is not the declared broker {request.declared_cloud_broker}",
        )
    if observation.refused_topics:
        return result(
            EnumBoardProbeOutcome.FAIL,
            *(
                f"{container}: the declared cloud broker refuses inbound topic {topic}"
                for topic in observation.refused_topics
            ),
            evidence=observation.refused_topics,
        )
    if observation.window_seconds < request.retry_interval_seconds:
        return result(
            indeterminate,
            f"log window {observation.window_seconds}s is shorter than one retry "
            f"interval ({request.retry_interval_seconds}s), so a refusal could be missed",
        )
    started_at = observation.started_at
    interval = timedelta(seconds=request.retry_interval_seconds)
    if started_at is None or observation.observed_at - started_at < interval:
        return result(
            indeterminate,
            f"{container}: has not run one retry interval "
            f"({request.retry_interval_seconds}s) since start {started_at}",
        )
    return result(
        EnumBoardProbeOutcome.PASS,
        f"{container}: up since {started_at}, cloud leg is the declared broker, "
        f"no inbound topic refused in the last {observation.window_seconds}s",
    )


class HandlerForwarderRefusedTopic:
    """Observe the forwarder through the injected reader, then grade it."""

    def __init__(self, reader: ProtocolForwarderStateReader | None = None) -> None:
        """Take the reader as a seam; the default reads docker on this host."""
        self._reader: ProtocolForwarderStateReader = (
            reader if reader is not None else HandlerDockerForwarderStateReader()
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler (host I/O)."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: effect, reads a container on the host."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelForwarderRefusedTopicRequest
    ) -> ModelBoardProbeResult:
        """Observe, then grade."""
        observation = await self._reader.observe(request)
        return grade_forwarder_refused_topic(request, observation)


__all__ = [
    "CHECK_ID",
    "SURFACE_CLASS",
    "HandlerForwarderRefusedTopic",
    "grade_forwarder_refused_topic",
]
