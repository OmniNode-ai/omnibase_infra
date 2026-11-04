# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollback-safe deploy-agent image maintenance and unit installation (OMN-19998).

The host timer enters through the node's local-runtime command topic. Docker's
default dangling-only prune rechecks references during deletion; by-id removal
could instead untag an image retained for rollback after the inventory read.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_agent_image_prune_policy import (
    ModelDeployAgentImagePrunePolicy,
)
from omnibase_infra.models.model_deploy_agent_image_prune_request import (
    ModelDeployAgentImagePruneRequest,
)
from omnibase_infra.models.model_deploy_agent_image_prune_result import (
    ModelDeployAgentImagePruneResult,
)


def build_plan(
    images: list[dict[str, object]],
    inuse: set[str],
    *,
    policy: ModelDeployAgentImagePrunePolicy,
    now: datetime,
) -> ModelDeployAgentImagePruneResult:
    """Keep all tags/digests, all container references and recent images."""
    cutoff = datetime.fromtimestamp(
        int((now - timedelta(hours=policy.minimum_age_hours)).timestamp()),
        UTC,
    )
    remove: list[str] = []
    kept: dict[str, str] = {}
    nominal_bytes = 0
    for image in images:
        image_id = image["Id"]
        if not isinstance(image_id, str):
            raise ValueError("Image inventory has an invalid identity")
        config = image.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else {}
        if not isinstance(labels, dict):
            labels = {}
        reason = ""
        if labels.get(policy.source_label) != policy.source_value:
            reason = "outside deploy-agent runtime image scope"
        elif image.get("RepoTags") or image.get("RepoDigests"):
            reason = "tag or digest reference (including rollback)"
        elif image_id in inuse:
            reason = "referenced by a running or stopped container"
        else:
            try:
                raw_created = image["Created"]
                if not isinstance(raw_created, str):
                    raise ValueError("Image creation time is not a string")
                created = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
                if created.tzinfo is None or created >= cutoff:
                    reason = "within minimum age or unknown timezone"
            except (KeyError, TypeError, ValueError):
                reason = "unreadable creation time"
        if reason:
            kept[image_id] = reason
        else:
            remove.append(image_id)
            size = image["Size"]
            if not isinstance(size, int) or size < 0:
                raise ValueError("Image inventory has an invalid size")
            nominal_bytes += size
    return ModelDeployAgentImagePruneResult(
        success=True,
        remove_image_ids=tuple(sorted(set(remove))),
        kept_reasons=kept,
        remove_nominal_bytes=nominal_bytes,
        cutoff_epoch=int(cutoff.timestamp()),
    )


class HandlerDeployAgentImagePrune:
    """Stateless effect: configuration is supplied by the request's policy port."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelDeployAgentImagePruneRequest
    ) -> ModelDeployAgentImagePruneResult:
        # Import at dispatch time: the effect boundary uses the pure planner above.
        from omnibase_infra.nodes.node_deploy_agent_image_prune_effect.effect import (
            run_request,
        )

        return await asyncio.to_thread(run_request, request)
