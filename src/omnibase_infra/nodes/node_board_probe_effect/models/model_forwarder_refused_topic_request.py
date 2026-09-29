# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The input to the ``forwarder_refused_topic`` board check.

Ticket: OMN-19930
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelForwarderRefusedTopicRequest(BaseModel):
    """Which lane's forwarder to read, and what that lane declares.

    ``declared_cloud_broker`` is the cloud bootstrap the lane's overlay
    declares for the forwarder's cloud leg. The caller resolves it from the
    lane's declaration; an empty value means the lane declares none, and the
    check cannot attribute a refusal to a broker, so it grades INDETERMINATE.
    ``retry_interval_seconds`` is the forwarder contract's
    ``liveness.inbound_topic_retry_seconds``: a refused topic is logged again at
    every retry, so one interval of log is enough to see the current state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_lane: str = Field(min_length=1)
    forwarder_container: str = Field(min_length=1)
    declared_cloud_broker: str = ""
    cloud_broker_ref: str = Field(min_length=1)
    retry_interval_seconds: int = Field(ge=1)
    broker_ref_map_path: str = Field(
        default="/run/gateway/broker-ref-map.yaml", min_length=1
    )


__all__ = ["ModelForwarderRefusedTopicRequest"]
