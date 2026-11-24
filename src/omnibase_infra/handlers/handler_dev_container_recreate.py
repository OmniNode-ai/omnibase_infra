# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Names each start of the dev runtime container and the cause it can prove (OMN-25912).

A start the restart policy makes, or a compose recreate no job asked for, leaves nothing on
the bus: the job records and ``dev_runtime_started`` events cover only what the deploy agent
did. This handler compares the container's reading now with the reading at the previous tick
and returns one ``dev_container_recreated`` event per new start:

* ``deploy_agent_job``: a ``dev_runtime_started`` event's start time is within the contract's
  tolerance of the container's StartedAt; the event's job id (and the manifest's issuer)
  ride in the event.
* ``restart_policy``: the same container started again with a higher RestartCount and no
  job event.
* ``unknown``: anything else. No cause is guessed.

The handler is pure: no clock, no I/O. The first reading has no previous one and names no
start.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_dev_recreate_cause import EnumDevRecreateCause
from omnibase_infra.models.model_dev_container_recreate_request import (
    ModelDevContainerRecreateRequest,
)
from omnibase_infra.models.model_dev_container_recreate_result import (
    ModelDevContainerRecreateResult,
)
from omnibase_infra.models.model_dev_container_recreated import (
    ModelDevContainerRecreated,
)

_KEY_NAMESPACE = "onex:dev-container-recreated"


class HandlerDevContainerRecreate:
    """Compare two container readings and name the start between them (pure)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelDevContainerRecreateRequest
    ) -> ModelDevContainerRecreateResult:
        reading, previous = request.reading, request.previous
        if previous is None or (
            previous.container_digest == reading.container_digest
            and previous.started_at == reading.started_at
        ):
            return ModelDevContainerRecreateResult()
        tolerance = timedelta(seconds=request.start_match_tolerance_seconds)
        matched = sorted(
            (
                start
                for start in request.starts
                if abs(start.started_at - reading.started_at) <= tolerance
            ),
            key=lambda start: abs(start.started_at - reading.started_at),
        )
        job_id = requested_by = ""
        if matched:
            cause = EnumDevRecreateCause.DEPLOY_AGENT_JOB
            job_id = matched[0].job_id
            requested_by = next(
                (m.issuer for m in request.manifests if m.job_id == job_id), ""
            )
        elif (
            previous.container_digest == reading.container_digest
            and reading.restart_count > previous.restart_count
        ):
            cause = EnumDevRecreateCause.RESTART_POLICY
        else:
            cause = EnumDevRecreateCause.UNKNOWN
        return ModelDevContainerRecreateResult(
            event=ModelDevContainerRecreated(
                event_id=uuid5(
                    NAMESPACE_URL,
                    f"{_KEY_NAMESPACE}:{reading.container_digest}:{reading.started_at.isoformat()}",
                ),
                runtime_lane="dev",
                container=reading.container,
                container_digest=reading.container_digest,
                started_at=reading.started_at,
                restart_count=reading.restart_count,
                exit_code=reading.exit_code,
                image_digest=reading.image_digest,
                image_ref=reading.image_ref,
                previous_image_digest=previous.image_digest,
                same_build=bool(reading.image_digest)
                and reading.image_digest == previous.image_digest,
                cause=cause,
                job_id=job_id,
                requested_by=requested_by,
            )
        )
