# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publish the overdue dev merge event and request its build, once, on the dev lane only (OMN-26316).

``HandlerDevMergeBuildOverdue`` judges; this handler acts on a ``REQUEST`` it was handed. It
reads no host. Its sender is the one the caller injects: in the deploy agent's idle tick the
sender publishes from the agent's side, outside the runtime the build replaces, and sends the
command through the signed build request path (``node_branch_deploy_command_effect``), never a
path of its own.

* A lane other than ``dev`` is refused with ``ProtocolConfigurationError`` before anything is sent.
* A decision that is not ``REQUEST`` (a job in flight, the failed head, inside the window, built)
  sends nothing.
* A sha already requested sends nothing; the command's correlation id is the decision's request
  id, one value per repository and sha, so a request a second tick repeats is a duplicate.
* A refused or failed send is a ``failed`` result, which the caller does not record as requested.
"""

from __future__ import annotations

from typing import Final

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.model_dev_merge_build_command import (
    ModelDevMergeBuildCommand,
)
from omnibase_infra.models.model_dev_merge_build_overdue_event import (
    ModelDevMergeBuildOverdueEvent,
)
from omnibase_infra.models.model_dev_merge_build_publish_request import (
    ModelDevMergeBuildPublishRequest,
)
from omnibase_infra.models.model_dev_merge_build_publish_result import (
    ModelDevMergeBuildPublishResult,
)
from omnibase_infra.protocols.protocol_dev_merge_build_sender import (
    ProtocolDevMergeBuildSender,
)

_DEV_LANE: Final = "dev"
_REQUESTER: Final = "agent/overdue-dev-merge-build"


class HandlerDevMergeBuildPublish:
    """Send the overdue event and the one signed dev build request of a merge."""

    def __init__(self, container: ModelONEXContainer) -> None:
        # The registry uses the interface as a key, never as a constructor.
        sender_key: type[object] = ProtocolDevMergeBuildSender
        sender = container.get_service(sender_key)
        if not isinstance(sender, ProtocolDevMergeBuildSender):
            raise TypeError("the dev merge build sender must implement its protocol")
        self._sender: ProtocolDevMergeBuildSender = sender

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDevMergeBuildPublishRequest
    ) -> ModelDevMergeBuildPublishResult:
        if request.runtime_lane != _DEV_LANE:
            raise ProtocolConfigurationError(
                f"an overdue dev merge build is requested on the dev lane only, not {request.runtime_lane!r}",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="request_overdue_dev_merge_build",
                ),
            )
        decision = request.decision
        request_id = decision.request_id
        if decision.outcome != "REQUEST" or request_id is None:
            return ModelDevMergeBuildPublishResult(
                status="skipped", reason="not_overdue"
            )
        if request.already_requested:
            return ModelDevMergeBuildPublishResult(
                status="skipped", reason="already_requested", request_id=request_id
            )
        if request.pr_number is None:
            return ModelDevMergeBuildPublishResult(
                status="skipped",
                reason="no_pull_request_number",
                request_id=request_id,
            )
        self._sender.publish_event(
            ModelDevMergeBuildOverdueEvent(
                repository=request.repository,
                merge_sha=decision.merge_sha,
                pr_number=request.pr_number,
                due_at=decision.due_at,
                request_id=request_id,
                runtime_lane=_DEV_LANE,
            )
        )
        status = self._sender.send_command(
            ModelDevMergeBuildCommand(
                correlation_id=request_id,
                repo=request.repository,
                pr_number=request.pr_number,
                git_ref=request.head_sha,
                rollback_actor=_REQUESTER,
                rollback_reason=f"merge {decision.merge_sha} had no dev build by {decision.due_at.isoformat()}",
                requested_by=_REQUESTER,
            )
        )
        if status == "published":
            return ModelDevMergeBuildPublishResult(
                status="requested", reason="published", request_id=request_id
            )
        return ModelDevMergeBuildPublishResult(
            status="failed",
            reason="send_refused" if status == "refused" else "send_failed",
            request_id=request_id,
        )
