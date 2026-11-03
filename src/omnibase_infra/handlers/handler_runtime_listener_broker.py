# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The listener broker's decisions (OMN-22379, design rev 4 section 3 and rule A6).

The broker hands the held listener to a runtime of a pool, keeps the pool's
attachment sessions and the agent's handover-in-progress flag, and answers detach
requests so the last confirmed acceptor never leaves while a handover runs. The
transport (a unix socket and SCM_RIGHTS) is ``HandlerRuntimeListenerBrokerServer``.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_runtime_listener_broker_op import (
    EnumRuntimeListenerBrokerOp,
)
from omnibase_infra.models.model_runtime_listener_broker_request import (
    ModelRuntimeListenerBrokerRequest,
)
from omnibase_infra.models.model_runtime_listener_broker_response import (
    ModelRuntimeListenerBrokerResponse,
)
from omnibase_infra.models.model_runtime_listener_fd_store import (
    ModelRuntimeListenerFdStore,
)
from omnibase_infra.models.model_runtime_listener_session_table import (
    ModelRuntimeListenerSessionTable,
)
from omnibase_infra.protocols.protocol_runtime_listener_lookup import (
    ProtocolRuntimeListenerLookup,
)


class HandlerRuntimeListenerBroker:
    """Serve one broker operation against the held listeners and the session table."""

    def __init__(self, container: ProtocolRuntimeListenerLookup) -> None:
        self._container = container

    @property
    def _fds(self) -> ModelRuntimeListenerFdStore:
        return self._container.get_service(ModelRuntimeListenerFdStore)

    @property
    def _sessions(self) -> ModelRuntimeListenerSessionTable:
        return self._container.get_service(ModelRuntimeListenerSessionTable)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @property
    def fd_store(self) -> ModelRuntimeListenerFdStore:
        return self._fds

    @property
    def sessions(self) -> ModelRuntimeListenerSessionTable:
        return self._sessions

    def handle(
        self, request: ModelRuntimeListenerBrokerRequest
    ) -> ModelRuntimeListenerBrokerResponse:
        pool = request.pool.value
        op = request.op

        def refused(error: str) -> ModelRuntimeListenerBrokerResponse:
            return ModelRuntimeListenerBrokerResponse(
                ok=False, op=op.value, pool=pool, error=error
            )

        def served(
            *, state: str = "", verdict: str = "", session: bool = False
        ) -> ModelRuntimeListenerBrokerResponse:
            return ModelRuntimeListenerBrokerResponse(
                ok=True,
                op=op.value,
                pool=pool,
                state=state,
                verdict=verdict,
                session_id=request.session_id if session else None,
            )

        match op:
            case EnumRuntimeListenerBrokerOp.GET:
                if self._fds.socket_for(pool) is None:
                    return refused("not_installed")
                return served()
            case EnumRuntimeListenerBrokerOp.ATTACH:
                if self._fds.socket_for(pool) is None:
                    return refused("not_installed")
                if request.session_id is None:
                    return refused("no_session")
                state = self._sessions.open_session(
                    str(request.session_id), pool, request.instance
                )
                return served(state=state, session=True)
            case EnumRuntimeListenerBrokerOp.CONFIRMED:
                state = self._sessions.confirm_session(str(request.session_id))
                if not state:
                    return refused("no_session")
                return served(state=state, session=True)
            case EnumRuntimeListenerBrokerOp.DETACH_REQUEST:
                verdict, error = self._sessions.decide_detach(str(request.session_id))
                if error:
                    return refused(error)
                return served(verdict=verdict, session=True)
            case EnumRuntimeListenerBrokerOp.END:
                self._sessions.end_session(str(request.session_id))
                return served()
            case EnumRuntimeListenerBrokerOp.HANDOVER_BEGIN:
                self._sessions.mark_handover(pool, True)
                return served()
            case EnumRuntimeListenerBrokerOp.HANDOVER_END:
                self._sessions.mark_handover(pool, False)
                return served()
            case EnumRuntimeListenerBrokerOp.SERVE:
                return refused("bad_request")  # the server handler serves this one
        raise ValueError(f"unknown broker operation {op!r}")


__all__: list[str] = ["HandlerRuntimeListenerBroker"]
