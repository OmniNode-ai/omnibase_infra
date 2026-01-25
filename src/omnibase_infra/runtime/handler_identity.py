# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Identity Utilities for HYBRID Mode Resolution.

This module provides the `handler_identity()` function used by both bootstrap
and contract sources to generate consistent handler IDs. This enables per-handler
identity matching in HYBRID mode.

The Problem:
    Prior to this module, contract-discovered handlers used a "bootstrap." prefix
    for handler_id to enable HYBRID mode identity matching. This was semantically
    confusing because "bootstrap" reads like "where it came from," not "what it is."

The Solution:
    A neutral "proto." prefix that indicates this is a **protocol identity namespace**,
    not a source indicator. Both HandlerBootstrapSource and PluginLoaderContractSource
    use this shared helper to generate consistent IDs.

Example:
    >>> from omnibase_infra.runtime.handler_identity import handler_identity
    >>> handler_identity("consul")
    'proto.consul'
    >>> handler_identity("http")
    'proto.http'

See Also:
    - HandlerSourceResolver._resolve_hybrid(): Resolution logic that compares handler_id
    - HandlerBootstrapSource: Uses this to generate bootstrap handler IDs
    - PluginLoaderContractSource: Uses this for contract-discovered handlers

Part of OMN-1095: Handler Source Mode Feature Flag / Bootstrap Contract Hybrid.

.. versionadded:: 0.7.0
    Introduced to fix handler ID namespace confusion.
"""

from __future__ import annotations

# Prefix used for handler identity in HYBRID mode resolution.
# This is a protocol namespace, NOT a source indicator.
# Both bootstrap and contract sources use this prefix.
HANDLER_IDENTITY_PREFIX = "proto"


def handler_identity(protocol_type: str) -> str:
    """Generate stable handler identity for HYBRID mode resolution.

    Both bootstrap and contract sources use this to generate consistent IDs,
    enabling per-handler identity matching in HYBRID mode. When both sources
    provide a handler with the same identity, the resolver applies precedence
    rules (contract wins by default, or bootstrap wins if allow_bootstrap_override=True).

    The "proto." prefix indicates this is a **protocol identity namespace**, not
    a source origin indicator. Contract-discovered handlers use this prefix
    specifically so they can be compared against bootstrap-discovered handlers
    with the same protocol_type.

    Args:
        protocol_type: The protocol type (e.g., "consul", "http", "db", "vault", "mcp").

    Returns:
        Stable handler identity string (e.g., "proto.consul", "proto.http").

    Example:
        >>> handler_identity("consul")
        'proto.consul'
        >>> handler_identity("http")
        'proto.http'

    See Also:
        HandlerSourceResolver._resolve_hybrid() for resolution logic that uses
        these identities to determine which handler wins when both sources
        provide handlers with the same identity.
    """
    return f"{HANDLER_IDENTITY_PREFIX}.{protocol_type}"


__all__ = [
    "HANDLER_IDENTITY_PREFIX",
    "handler_identity",
]
