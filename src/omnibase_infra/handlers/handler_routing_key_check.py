# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy lane routing key check and the router's refusal (OMN-26485).

THE FAILURE
-----------
``config/deploy_lane_routing.yaml`` routed omnimarket dev builds to its second instance
and marked it available (OMN-19510, OMN-21402, OMN-24213). Its agent answered a key check
of 9b0c12edb01c7b61 where the dev runtime and the default instance's agent answered
e30a85fc883bc113, so it dropped each command it was sent: its job record held no job that answered the five
held commands. None of the three changes that built the route and its availability read
an instance's key, so nothing refused the route.

THE CHECK
---------
For every instance the table marks available, compare the key check its agent's health
answer carries (``hmac_key_check``, OMN-24397) with the lane runtime's. An instance whose
value differs, or that has no answer, is a typed refusal naming both values. The check
reads no host: the caller supplies the answers.

THE ROUTE
---------
The command's route is the one both resolvers already apply (``deploy_agent.routing`` and
``scripts/ci/deploy_lane_verify_route.py``): a ``(runtime_lane, requester_repository)`` row,
else the default instance, and a row naming an unavailable instance takes the default
(OMN-21402). On top of it, the routed instance is tried first and the default instance
second; each is refused if it fails the check. The first that passes takes the command.
When the default instance itself fails, nothing is sent and every refusal is typed. The
availability flag and the OMN-21402 rule keep their behaviour: this only removes an
instance from the path, never adds one.
"""

from __future__ import annotations

import re
from typing import Final

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_lane_routing_table import (
    ModelDeployLaneRoutingTable,
)
from omnibase_infra.models.model_routing_key_check_request import (
    ModelRoutingKeyCheckRequest,
)
from omnibase_infra.models.model_routing_key_check_result import (
    ModelRoutingKeyCheckResult,
)
from omnibase_infra.models.model_routing_key_refusal import ModelRoutingKeyRefusal

#: ``requested_by`` of a CI-published command: ``gha/<repo>/...``.
_CI_REQUESTER_RE: Final = re.compile(r"^gha/([a-z][a-z0-9_]*)/")


def _requester_repository(requested_by: str) -> str | None:
    match = _CI_REQUESTER_RE.match(requested_by)
    return match.group(1) if match else None


def _routed_instance(
    table: ModelDeployLaneRoutingTable, runtime_lane: str, requested_by: str
) -> str:
    """The instance the table alone routes a command to (OMN-21402 included)."""
    repository = _requester_repository(requested_by)
    for route in table.routes:
        if (
            route.runtime_lane == runtime_lane
            and route.requester_repository == repository
        ):
            if table.instances[route.instance].available is False:
                return table.default_instance
            return route.instance
    return table.default_instance


class HandlerRoutingKeyCheck:
    """Judge a routing table against the key checks and route a command around a mismatch."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    @staticmethod
    def _refusal(
        request: ModelRoutingKeyCheckRequest, instance: str
    ) -> ModelRoutingKeyRefusal | None:
        seen = request.instance_key_checks.get(instance, "").strip()
        if seen == request.runtime_key_check:
            return None
        return ModelRoutingKeyRefusal(
            instance=instance,
            instance_key_check=seen,
            runtime_key_check=request.runtime_key_check,
            reason="key-check-differs" if seen else "key-check-unread",
        )

    def handle(
        self, request: ModelRoutingKeyCheckRequest
    ) -> ModelRoutingKeyCheckResult:
        table = request.table
        mismatches = tuple(
            refusal
            for name, spec in table.instances.items()
            if spec.available and (refusal := self._refusal(request, name)) is not None
        )
        outcome = "mismatch" if mismatches else "ok"
        if not request.runtime_lane:
            return ModelRoutingKeyCheckResult(outcome=outcome, mismatches=mismatches)

        routed = _routed_instance(table, request.runtime_lane, request.requested_by)
        path = (
            [routed]
            if routed == table.default_instance
            else [routed, table.default_instance]
        )
        refusals: list[ModelRoutingKeyRefusal] = []
        for instance in path:
            refusal = self._refusal(request, instance)
            if refusal is None:
                return ModelRoutingKeyCheckResult(
                    outcome=outcome,
                    mismatches=mismatches,
                    route_outcome="fell_back" if refusals else "routed",
                    routed_instance=instance,
                    refusals=tuple(refusals),
                )
            refusals.append(refusal)
        return ModelRoutingKeyCheckResult(
            outcome=outcome,
            mismatches=mismatches,
            route_outcome="refused",
            refusals=tuple(refusals),
        )
