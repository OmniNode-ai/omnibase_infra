# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structured reasons for handler quarantine during auto-wiring (OMN-9457).

Distinct from :class:`HandshakeFailureReason` (OMN-7657), which classifies
failures of the pre-wiring handshake lifecycle hooks. These values classify
handler construction failures that are deterministically contained by
auto-wiring so a single async-incompatible handler cannot poison runtime-effects
boot.
"""

from __future__ import annotations

from enum import Enum


class EnumQuarantineReason(str, Enum):
    """Structured reasons for auto-wiring handler quarantine.

    Each value maps to a distinct operational response:

    - ``ASYNC_INCOMPATIBLE``: handler construction raised
      ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
      The handler relies on ``asyncio.run()`` in its ``__init__`` or a
      container-resolved dependency, which is incompatible with runtime-managed
      async boot. Follow-up: convert the handler or its dependency to async-safe
      construction.
    """

    ASYNC_INCOMPATIBLE = "async_incompatible"


__all__ = ["EnumQuarantineReason"]
