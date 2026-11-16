# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns the readings."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDevLanePinComposeGuardEffect(NodeEffect):
    """Report a code-ref pin behind dev and refuse a service the compose lacks, before a build."""
