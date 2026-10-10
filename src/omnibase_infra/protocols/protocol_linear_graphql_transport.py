# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Transport boundary for existing Linear GraphQL callers (OMN-17678)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class ProtocolLinearGraphQLTransport(Protocol):
    """One HTTP attempt; caller policies interpret status and GraphQL errors.

    Returning the response preserves Retry-After, OAuth identity and the
    existing sweeps' bounded retry and fail-closed policies. Domain callers
    remain responsible for their write authorization and receipt gates.
    """

    async def post_graphql(
        self, query: str, variables: dict[str, object] | None = None
    ) -> httpx.Response:
        """Post without a viewer probe, implicit retry or response unwrapping."""
        ...
