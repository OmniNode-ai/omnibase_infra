# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node onex-api image build (OMN-23521).

Zero custom logic: the behaviour is the handler named in ``contract.yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeOnexApiImageBuildEffect(NodeEffect):
    """Builds the onex-api image the dev pin delivery advances to.

    Capability: onex_api.image_build
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeOnexApiImageBuildEffect"]
