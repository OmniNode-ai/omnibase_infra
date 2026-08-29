# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative EFFECT node for headless Infisical secret seeding.

All behaviour lives in ``handlers/handler_secret_seed.py`` and
``contract.yaml``. This class exists only to bind them.

Ticket: OMN-16897
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSecretSeedEffect(NodeEffect):
    """Declarative effect node for headless secret seeding.

    Handlers:
        - ``HandlerSecretSeed``: upserts named secrets into a named
          Infisical instance from a local source file, with no UI and no
          interactive login. Values never enter the request, the result,
          the bus, or the log.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the secret-seeding effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeSecretSeedEffect"]
