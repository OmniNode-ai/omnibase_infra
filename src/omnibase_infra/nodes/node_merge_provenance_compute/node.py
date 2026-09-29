# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeMergeProvenanceCompute -- declarative compute node (OMN-19927).

Grades whether a commit was validated by a successful merge group. All
behavior is declared in ``contract.yaml``; no custom logic here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeMergeProvenanceCompute(NodeCompute):
    """Compute node for merge provenance.

    Capability: ci.merge_provenance.evaluate
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the merge-provenance compute node."""
        super().__init__(container)


__all__: list[str] = ["NodeMergeProvenanceCompute"]
