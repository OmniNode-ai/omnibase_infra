# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event bus topic wiring declarations from a contract (OMN-7653)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEventBusWiring(BaseModel):
    """Event bus topic declarations extracted from a contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    subscribe_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics this node subscribes to",
    )
    publish_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics this node publishes to",
    )
    consumer_purpose: str | None = Field(
        default=None,
        description="Optional contract-declared consumer purpose",
    )
    plugin_managed: bool = Field(
        default=False,
        description=(
            "When True, auto-wiring skips Kafka subscription for this contract's "
            "topics. A domain plugin owns the subscription with custom config "
            "(e.g. result_applier). Auto-wiring still registers dispatch routes."
        ),
    )
    tenant_scoped_ingress: bool = Field(
        default=False,
        description=(
            "OMN-14349 (OMN-14208 Path A): when True, this contract's "
            "subscribe_topics are tenant-<slug>. wire-prefixed variants. "
            "Auto-wiring derives tenant_id from the matched prefix and stamps "
            "it into the payload before dispatch -- overwriting any "
            "client-supplied value, never falling back to one. A topic with no "
            "tenant-<slug>. prefix is left unstamped (Stage-1 warn), never "
            "given a defaulted or guessed tenant. Off by default; zero "
            "behavior change for every non-opted-in contract."
        ),
    )
