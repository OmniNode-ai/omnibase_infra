# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handlers for auth gate compute operations.

This package provides the authorization decision handler:
    - HandlerAuthGate: 10-step authorization cascade for tool invocations

Ticket: OMN-2125
"""

from omnibase_infra.nodes.node_auth_gate_compute.handlers.handler_auth_gate import (
    HandlerAuthGate,
)

__all__: list[str] = [
    "HandlerAuthGate",
]
