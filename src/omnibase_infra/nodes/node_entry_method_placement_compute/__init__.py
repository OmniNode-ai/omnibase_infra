# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Entry method placement — declarative compute node.

All behavior is defined in contract.yaml - no custom logic here.
"""

from omnibase_infra.nodes.node_entry_method_placement_compute.node import (
    NodeEntryMethodPlacementCompute,
)

__all__: list[str] = ["NodeEntryMethodPlacementCompute"]
