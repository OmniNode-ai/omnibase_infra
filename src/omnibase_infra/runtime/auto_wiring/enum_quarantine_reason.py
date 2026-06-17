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
    - ``PROTOCOL_HANDLER_DECLARATION``: contract handler routing points at a
      ``typing.Protocol`` interface rather than a concrete handler class.
      Follow-up: change the contract to reference the concrete handler and keep
      Protocols in dependency/interface declarations.
    - ``UNRESOLVABLE_HANDLER``: per-handler resolution/construction failed with
      a deterministic, never-recoverable wiring bug — the resolver could not
      satisfy the handler's constructor (unsatisfiable-ctor / ctor-arg-mismatch
      ``TypeError`` from ``ServiceHandlerResolver`` Steps 2/6) or the handler
      entry was malformed (``ValueError`` — not-handle-shaped / blank-required).
      Before OMN-13203 this re-raised and crashed runtime-effects boot, taking
      down every healthy handler in the manifest. It is now contained so the
      runtime binds its health server and reports the one bad handler. This is
      a per-handler wiring bug, NEVER recoverable runtime state and NEVER an
      infra outage (broker/DB/secret failures surface as ModelOnexError /
      InfraConnectionError / ConnectionError / OSError, not a bare resolver
      ``TypeError``/``ValueError``, and still crash boot). Counts toward
      ``total_failed`` so it is reported as a failure, not silently dropped.
      Follow-up: fix the contract/handler constructor so the resolver can wire
      it. ``ONEX_WIRING_STRICT_MODE=1`` re-raises instead of quarantining.
    """

    ASYNC_INCOMPATIBLE = "async_incompatible"
    PROTOCOL_HANDLER_DECLARATION = "protocol_handler_declaration"
    UNRESOLVABLE_HANDLER = "unresolvable_handler"


__all__ = ["EnumQuarantineReason"]
