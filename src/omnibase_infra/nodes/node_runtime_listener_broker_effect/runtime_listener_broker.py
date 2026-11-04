# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The listener broker's unix socket, served from threads (OMN-22379).

Threads, not the agent's event loop: a rebuild keeps the agent busy for minutes
and the broker must keep answering detach requests and ending sessions while it
does. One thread accepts; one thread per connection reads JSON lines. A
connection is a session: its end, by close or by crash, removes the session.
The answer to ``get`` carries the listening socket as one SCM_RIGHTS descriptor.
"""

from __future__ import annotations

import array
import json
import logging
import os
import socket
import struct
import threading
import uuid
from uuid import UUID

from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_runtime_listener_broker_op import (
    EnumRuntimeListenerBrokerOp,
)
from omnibase_infra.enums.enum_runtime_listener_pool import EnumRuntimeListenerPool
from omnibase_infra.models.model_runtime_listener_broker_port import (
    ModelRuntimeListenerBrokerPort,
)
from omnibase_infra.models.model_runtime_listener_broker_request import (
    ModelRuntimeListenerBrokerRequest,
)
from omnibase_infra.models.model_runtime_listener_broker_response import (
    ModelRuntimeListenerBrokerResponse,
)
from omnibase_infra.models.model_runtime_listener_broker_server_io import (
    ModelRuntimeListenerBrokerServerIo,
)
from omnibase_infra.models.model_runtime_listener_endpoint import (
    ModelRuntimeListenerEndpoint,
)
from omnibase_infra.models.model_runtime_listener_fd_store import (
    ModelRuntimeListenerFdStore,
)
from omnibase_infra.models.model_runtime_listener_registry import (
    ModelRuntimeListenerRegistry,
)

logger = logging.getLogger(__name__)

#: A request line is a few dozen bytes; anything near this is not a client.
MAX_REQUEST_BYTES = 4096


def _bad_request(op: str = "", pool: str = "") -> ModelRuntimeListenerBrokerResponse:
    return ModelRuntimeListenerBrokerResponse(
        ok=False, op=op, pool=pool, error="bad_request"
    )


def _send(
    conn: socket.socket,
    response: ModelRuntimeListenerBrokerResponse,
    listener: socket.socket | None = None,
) -> None:
    payload = response.model_dump_json().encode() + b"\n"
    if listener is None:
        conn.sendall(payload)
        return
    sent = conn.sendmsg(
        [payload],
        [
            (
                socket.SOL_SOCKET,
                socket.SCM_RIGHTS,
                array.array("i", [listener.fileno()]).tobytes(),
            )
        ],
    )
    if sent < len(payload):
        conn.sendall(payload[sent:])


def _answer(
    conn: socket.socket,
    broker: ModelRuntimeListenerBrokerPort,
    fds: ModelRuntimeListenerFdStore,
    session_id: UUID,
    line: bytes,
) -> None:
    try:
        raw = json.loads(line)
        if not isinstance(raw, dict) or "session_id" in raw:
            raise ValueError("not a client request")
        request = ModelRuntimeListenerBrokerRequest(**raw, session_id=session_id)
        if not request.op.from_client:
            raise ValueError("operation is not for clients")
    except (ValueError, TypeError, ValidationError):
        _send(conn, _bad_request())
        return
    response = broker.serve(request)
    is_get = request.op is EnumRuntimeListenerBrokerOp.GET and response.ok
    _send(conn, response, fds.socket_for(request.pool.value) if is_get else None)


def broker_server_io(
    container: ModelRuntimeListenerRegistry,
) -> ModelRuntimeListenerBrokerServerIo:
    """Bind the EFFECT transport to the broker and endpoint registered for this agent."""
    broker = container.get_service(ModelRuntimeListenerBrokerPort)
    endpoint = container.get_service(ModelRuntimeListenerEndpoint)

    def start() -> None:
        """Bind the socket and begin accepting; a second call does nothing."""
        path = endpoint.socket_path
        with endpoint.guard:
            if endpoint.listening.current is not None:
                return
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Tighten an existing install directory before creating the socket:
            # bind initially uses the process umask, before chmod can run.
            path.parent.chmod(0o700)
            path.unlink(missing_ok=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            path.chmod(0o600)
            server.listen(32)
            endpoint.listening.hold(server)
            threading.Thread(
                target=_accept_forever,
                args=(server,),
                name="runtime-listener-broker",
                daemon=True,
            ).start()
        logger.info("runtime listener broker serving at %s", path)

    def stop() -> None:
        """Stop accepting and remove the socket file; open sessions end with the process."""
        with endpoint.guard:
            server = endpoint.listening.release()
        if server is not None:
            server.close()
            endpoint.socket_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ serving

    def _accept_forever(server: socket.socket) -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(
                target=_serve,
                args=(conn,),
                name="runtime-listener-session",
                daemon=True,
            ).start()

    def _serve(conn: socket.socket) -> None:
        # Runtime clients belong to the same service uid as this user agent.
        # Check the kernel's peer credentials before reading or passing any fd.
        try:
            _, uid, _ = struct.unpack(
                "3i",
                conn.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                ),
            )
            if uid != os.geteuid():
                conn.close()
                return
        except OSError:
            conn.close()
            return
        session_id = uuid.uuid4()
        reader = conn.makefile("rb")
        try:
            while True:
                line = reader.readline(MAX_REQUEST_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_REQUEST_BYTES:
                    _send(conn, _bad_request())
                    return
                _answer(
                    conn,
                    broker,
                    container.get_service(ModelRuntimeListenerFdStore),
                    session_id,
                    line,
                )
        except OSError:
            return
        finally:
            broker.serve(
                ModelRuntimeListenerBrokerRequest(
                    op=EnumRuntimeListenerBrokerOp.END,
                    pool=EnumRuntimeListenerPool.MAIN,
                    session_id=session_id,
                )
            )
            reader.close()
            conn.close()

    return ModelRuntimeListenerBrokerServerIo(start=start, stop=stop)


__all__: list[str] = ["broker_server_io"]
