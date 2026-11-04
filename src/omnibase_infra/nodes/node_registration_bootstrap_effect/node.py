# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect for registration's pre-dispatch resource acquisition."""

from omnibase_core.nodes import NodeEffect


class NodeRegistrationBootstrapEffect(NodeEffect):
    """Behavior is bound to the bootstrap handler in contract.yaml."""
