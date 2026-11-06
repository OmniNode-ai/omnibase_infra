# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Release-notes apply - declarative effect node.

All behavior is defined in contract.yaml - no custom logic here.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeReleaseNotesApplyEffect(NodeEffect):
    """Declarative effect node for placing a release block under the repo's docs policy.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeReleaseNotesApplyEffect"]
