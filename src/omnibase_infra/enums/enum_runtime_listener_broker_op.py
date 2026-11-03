# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Operations of the listener broker (OMN-22379)."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class EnumRuntimeListenerBrokerOp(StrEnum):
    """Broker operations.

    The first four arrive from a runtime over the unix socket. The last four
    are raised only by the agent process itself (the connection closing, the
    handover flag and starting the socket), so the reader refuses them from a client.
    """

    GET = "get"
    ATTACH = "attach"
    CONFIRMED = "confirmed"
    DETACH_REQUEST = "detach_request"
    END = "end"
    HANDOVER_BEGIN = "handover_begin"
    HANDOVER_END = "handover_end"
    SERVE = "serve"

    @property
    def from_client(self) -> bool:
        """True for the operations a runtime may send on the socket."""
        return self in {
            EnumRuntimeListenerBrokerOp.GET,
            EnumRuntimeListenerBrokerOp.ATTACH,
            EnumRuntimeListenerBrokerOp.CONFIRMED,
            EnumRuntimeListenerBrokerOp.DETACH_REQUEST,
        }


__all__: list[str] = ["EnumRuntimeListenerBrokerOp"]
