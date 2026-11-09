# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node onex-api image import check (OMN-23039).

Zero custom logic: the behaviour is the handler named in ``contract.yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeOnexApiImageImportCheckEffect(NodeEffect):
    """Decides whether a candidate onex-api image may keep the advanced pin.

    Capability: onex_api.image_import_check
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeOnexApiImageImportCheckEffect"]
