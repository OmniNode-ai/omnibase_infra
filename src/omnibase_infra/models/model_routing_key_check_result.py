# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The key check's verdict and the command's routing decision (OMN-26485)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_routing_key_refusal import ModelRoutingKeyRefusal


class ModelRoutingKeyCheckResult(BaseModel):
    """``outcome`` is ``mismatch`` when any available instance fails the key check.

    ``mismatches`` lists every such instance, in table order. The routing decision is
    ``route_outcome``: ``not_requested`` when the request named no lane, ``routed`` when
    the routed instance passes, ``fell_back`` when it was refused and the default instance
    passes, and ``refused`` when no instance passes (nothing is sent: ``routed_instance``
    is empty). ``refusals`` holds one typed refusal per refused instance on the command's
    path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Literal["ok", "mismatch"]
    mismatches: tuple[ModelRoutingKeyRefusal, ...] = ()
    route_outcome: Literal["not_requested", "routed", "fell_back", "refused"] = (
        "not_requested"
    )
    routed_instance: str = ""
    refusals: tuple[ModelRoutingKeyRefusal, ...] = ()
