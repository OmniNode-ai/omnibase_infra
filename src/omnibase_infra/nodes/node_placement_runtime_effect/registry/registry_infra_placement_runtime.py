# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Composition root of ``HandlerPlacementRuntime`` (OMN-22132).

The registry builds the membership session, one election per ``primary`` node,
the serving reader and the lock, and hands them to the handler by constructor
injection. It reads no literal: the timings come from the node contract's
``placement_runtime.profiles`` block, selected by profile name.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from omnibase_infra.handlers.placement.handler_placement_election import (
    HandlerPlacementElection,
)
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.handlers.placement.handler_placement_runtime import (
    HandlerPlacementRuntime,
    ProtocolPlacedNode,
)
from omnibase_infra.handlers.placement.handler_placement_serving import (
    read_serving_report,
)
from omnibase_infra.handlers.placement.handler_placement_session import (
    HandlerPlacementSession,
)
from omnibase_infra.handlers.placement.model_placement_policy import (
    ModelPlacementPolicy,
)
from omnibase_infra.handlers.placement.model_placement_serving_report import (
    ModelPlacementServingReport,
)
from omnibase_infra.models.model_placement_runtime_settings import (
    ModelPlacementRuntimeSettings,
)


class RegistryInfraPlacementRuntime:
    """Builds ``HandlerPlacementRuntime`` with its dependencies injected."""

    def __init__(
        self,
        *,
        adapter: HandlerPlacementEtcd,
        settings: ModelPlacementRuntimeSettings,
    ) -> None:
        self._adapter = adapter
        self._settings = settings

    @staticmethod
    def settings_from_contract(
        contract: Mapping[str, object], profile: str
    ) -> ModelPlacementRuntimeSettings:
        """Read one profile of the contract's ``placement_runtime.profiles`` block."""
        block = contract["placement_runtime"]
        if not isinstance(block, Mapping):
            raise TypeError("placement_runtime must be a mapping")
        profiles = block["profiles"]
        if not isinstance(profiles, Mapping) or profile not in profiles:
            raise KeyError(f"placement_runtime profile {profile!r} is not declared")
        return ModelPlacementRuntimeSettings.model_validate(profiles[profile])

    async def _read_serving(
        self, *, expected_nodes: Sequence[str], min_serving: int
    ) -> ModelPlacementServingReport:
        return await read_serving_report(
            self._adapter, expected_nodes=expected_nodes, min_serving=min_serving
        )

    def build(
        self,
        *,
        runtime_id: str,
        nodes: Mapping[str, tuple[ModelPlacementPolicy, ProtocolPlacedNode]],
    ) -> HandlerPlacementRuntime:
        """Wire one runtime over ``nodes`` with the profile's timings."""
        session = HandlerPlacementSession(
            adapter=self._adapter,
            runtime_id=runtime_id,
            node_names=sorted(nodes),
            ttl_s=self._settings.ttl_s,
            margin_s=self._settings.margin_s,
        )
        elections = {
            name: HandlerPlacementElection(
                adapter=self._adapter, session=session, node_name=name
            )
            for name, (policy, _) in nodes.items()
            if policy.mode == "primary"
        }
        return HandlerPlacementRuntime(
            runtime_id=runtime_id,
            session=session,
            elections=elections,
            nodes=dict(nodes),
            serving_reader=self._read_serving,
            lock=asyncio.Lock(),
            min_serving=self._settings.min_serving,
            keepalive_interval_s=self._settings.keepalive_interval_s,
            campaign_interval_s=self._settings.campaign_interval_s,
            fence_tick_s=self._settings.fence_tick_s,
        )


__all__ = ["RegistryInfraPlacementRuntime"]
