# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handlers for the release-identity compute node.

Provides the pure release-identity decision handler:
    - HandlerReleaseIdentity: version-ahead / src-change fitness gate.

Ticket: OMN-14471
"""

from omnibase_infra.nodes.node_release_identity_compute.handlers.handler_release_identity import (
    HandlerReleaseIdentity,
)

__all__: list[str] = [
    "HandlerReleaseIdentity",
]
