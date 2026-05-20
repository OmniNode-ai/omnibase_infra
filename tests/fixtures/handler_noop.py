# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Minimal no-op handler for integration tests.

Used by dynamic contract registration E2E tests (OMN-11248) as a
lightweight in-process handler that avoids network dependencies.
"""

from __future__ import annotations


class HandlerNoop:
    """No-op handler: accepts any input, returns None."""

    async def handle(self, envelope: object) -> None:
        return None
