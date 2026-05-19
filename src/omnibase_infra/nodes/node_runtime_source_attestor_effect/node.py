# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node Runtime Source Attestor Effect — OMN-9139.

Subscribes to ``onex.evt.runtime.booted.v1`` and verifies the
``runtime_source_hash`` baked into each container against the current
``main`` HEAD for the corresponding repo.

On drift > N commits (configurable, default 5):
    * Emits a friction event to ``.onex_state/friction/``
    * Flags the container for the morning-report surface

On ``unknown`` or empty hash:
    * Treats as infinite drift — always emits friction

Integrates with runtime_sweep (OMN-9122), which fails on any container
whose ``RUNTIME_SOURCE_HASH`` is stale or unknown.

Architecture::

    onex.evt.runtime.booted.v1 (Kafka)
        -> NodeRuntimeSourceAttestorEffect (this declarative shell)
        -> HandlerSourceAttestation
        -> .onex_state/friction/runtime-source-drift-<container>.yaml

Related Tickets:
    - OMN-9139: Deployed-artifact source-hash attestation
    - OMN-9122: runtime_sweep (consumes this signal)
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeSourceAttestorEffect(NodeEffect):
    """Declarative effect node for runtime source-hash attestation.

    Pure declarative shell — all behaviour defined in ``contract.yaml``.
    Routing, retry, and error-handling policies live there.

    Supported Operations (defined in contract.yaml handler_routing):
        - attest_source_hash: Compare hash against main HEAD; emit friction on drift

    Example::

        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_runtime_source_attestor_effect import (
            NodeRuntimeSourceAttestorEffect,
        )
        from omnibase_infra.nodes.node_runtime_source_attestor_effect.handlers import (
            HandlerSourceAttestation,
        )
        from omnibase_infra.models.health.model_runtime_booted_event import (
            ModelRuntimeBootedEvent,
        )

        container = ModelONEXContainer()
        node = NodeRuntimeSourceAttestorEffect(container)

        handler = HandlerSourceAttestation()
        result = await handler.handle(booted_event)
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

    # Pure declarative shell — all behaviour defined in contract.yaml


__all__ = ["NodeRuntimeSourceAttestorEffect"]
