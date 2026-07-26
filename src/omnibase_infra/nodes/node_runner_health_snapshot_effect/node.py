# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet snapshot effect -- declarative effect node for runner-fleet gather.

Completes the OMN-6091 stub (OMN-13942): all behavior is defined in
contract.yaml, no custom logic here.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRunnerHealthSnapshotEffect(NodeEffect):
    """Declarative effect node for gathering runner-fleet health facts.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRunnerHealthSnapshotEffect"]
