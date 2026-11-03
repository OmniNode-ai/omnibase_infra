# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's holder of the runtime host listeners (OMN-22379, design rev 4 section 3).

The agent binds ``[::]:8085`` (pool ``runtime-main``) and ``[::]:8086`` (pool
``runtime-effects``) once, at the first install, and keeps them for good: in
its own descriptor table, in the systemd file-descriptor store (``sd_notify``
``FDSTORE=1 FDNAME=<pool>``), across its self-update re-exec (descriptors 3 and 4
in ``LISTEN_FDNAMES`` order) and across a systemd restart (``LISTEN_FDS``).
A listening socket is a kernel object: a connection queued on it while the agent
is between processes is accepted by whoever holds the socket next.

Everything here is inert until an install binds: with no ``LISTEN_FDS`` and no
``DEV_RUNTIME_HANDOVER_DIR`` the agent never imports this module
(``ModelRuntimeListenerRegistry.start`` is the one entry).
"""

from __future__ import annotations

import array
import errno
import fcntl
import json
import logging
import os
import socket
from collections.abc import MutableMapping
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_runtime_listener_pool import EnumRuntimeListenerPool
from omnibase_infra.models.model_runtime_listener_environment import (
    ModelRuntimeListenerEnvironment,
)
from omnibase_infra.models.model_runtime_listener_fd_store import (
    ModelRuntimeListenerFdStore,
)
from omnibase_infra.models.model_runtime_listener_holder_request import (
    ModelRuntimeListenerHolderRequest,
)
from omnibase_infra.models.model_runtime_listener_holder_result import (
    ModelRuntimeListenerHolderResult,
)
from omnibase_infra.models.model_runtime_listener_registry import (
    ModelRuntimeListenerRegistry,
)

logger = logging.getLogger(__name__)

#: The systemd activation protocol starts descriptors at 3.
SD_LISTEN_FDS_START = 3
#: Staging descriptors for the remap live above any descriptor it targets.
_STAGING_FD_FLOOR = 100
_POOL_ORDER = (EnumRuntimeListenerPool.MAIN, EnumRuntimeListenerPool.EFFECTS)
_LISTEN_VARIABLES = ("LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES")


def _sd_notify(notify_socket: str, message: str, fds: tuple[int, ...] = ()) -> None:
    """Send one datagram to the service manager, with descriptors when given."""
    address = (
        "\0" + notify_socket[1:] if notify_socket.startswith("@") else notify_socket
    )
    ancillary: list[tuple[int, int, bytes]] = []
    if fds:
        ancillary.append(
            (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds).tobytes())
        )
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.sendmsg([message.encode()], ancillary, 0, address)


def _is_listening_stream(sock: socket.socket) -> bool:
    return (
        sock.type == socket.SOCK_STREAM
        and sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    )


def _result(
    fds: ModelRuntimeListenerFdStore, outcome: str, *, ok: bool = True
) -> ModelRuntimeListenerHolderResult:
    held = [
        (pool.value, fds.socket_for(pool.value))
        for pool in _POOL_ORDER
        if fds.socket_for(pool.value) is not None
    ]
    return ModelRuntimeListenerHolderResult(
        ok=ok,
        outcome=outcome,
        pools=tuple(pool for pool, _ in held),
        ports=tuple(sock.getsockname()[1] for _, sock in held if sock is not None),
    )


# --------------------------------------------------------------------- bind


def _recover(
    fds: ModelRuntimeListenerFdStore, environ: MutableMapping[str, str]
) -> ModelRuntimeListenerHolderResult:
    count_text = environ.get("LISTEN_FDS", "")
    if not count_text:
        return _result(fds, "inert:no_listen_fds")
    if environ.get("LISTEN_PID", "") != str(os.getpid()):
        return _result(fds, "inert:listen_pid_mismatch")
    names = environ.get("LISTEN_FDNAMES", "").split(":")
    try:
        count = int(count_text)
    except ValueError:
        return _result(fds, "refused:listen_fds_unparsable", ok=False)
    if count != len(names):
        return _result(fds, "refused:listen_fdnames_count_mismatch", ok=False)
    pool_names = {pool.value for pool in _POOL_ORDER}
    if len({n for n in names if n in pool_names}) != len(
        [n for n in names if n in pool_names]
    ):
        return _result(fds, "refused:duplicate_listener_name", ok=False)
    for index, name in enumerate(names):
        if name not in pool_names:
            continue
        descriptor = SD_LISTEN_FDS_START + index
        try:
            sock = socket.socket(fileno=descriptor)
        except OSError:
            return _result(fds, f"refused:not_a_listener:{name}", ok=False)
        if not _is_listening_stream(sock):
            sock.detach()
            return _result(fds, f"refused:not_a_listener:{name}", ok=False)
        os.set_inheritable(descriptor, False)
        fds.keep(name, sock)
    for variable in _LISTEN_VARIABLES:
        environ.pop(variable, None)
    return _result(fds, "recovered")


# ------------------------------------------------------------------- re-exec


def _remap_for_reexec(
    fds: ModelRuntimeListenerFdStore, environ: MutableMapping[str, str]
) -> ModelRuntimeListenerHolderResult:
    held = [
        (pool.value, sock)
        for pool in _POOL_ORDER
        if (sock := fds.socket_for(pool.value)) is not None
    ]
    if not held:
        return _result(fds, "inert:nothing_held")
    # Stage every listener above the target range first, so that moving one
    # onto descriptor 3 or 4 can never overwrite another that still lives there.
    staged = [
        fcntl.fcntl(sock.fileno(), fcntl.F_DUPFD_CLOEXEC, _STAGING_FD_FLOOR)
        for _, sock in held
    ]
    old_descriptors = [sock.detach() for _, sock in held]
    targets = [SD_LISTEN_FDS_START + index for index in range(len(held))]
    for staging, target in zip(staged, targets, strict=True):
        os.dup2(staging, target, inheritable=True)
        os.close(staging)
    for old in old_descriptors:
        if old not in targets:
            os.close(old)
    for (name, _), target in zip(held, targets, strict=True):
        fds.keep(name, socket.socket(fileno=target))
    environ["LISTEN_FDS"] = str(len(held))
    environ["LISTEN_FDNAMES"] = ":".join(name for name, _ in held)
    environ["LISTEN_PID"] = str(os.getpid())
    return _result(fds, "remapped")


# --------------------------------------------------------------------- drop


def _drop(
    fds: ModelRuntimeListenerFdStore, environ: MutableMapping[str, str]
) -> ModelRuntimeListenerHolderResult:
    notify = environ.get("NOTIFY_SOCKET", "")
    for name, sock in fds.release_all():
        if notify:
            _sd_notify(notify, f"FDSTOREREMOVE=1\nFDNAME={name}")
        sock.close()
    for variable in _LISTEN_VARIABLES:
        environ.pop(variable, None)
    return _result(fds, "dropped")


class HandlerRuntimeListenerHolder:
    """Bind, store, recover and re-exec-remap the two runtime host listeners."""

    def __init__(self, container: ModelRuntimeListenerRegistry) -> None:
        self._container = container

    @property
    def _fds(self) -> ModelRuntimeListenerFdStore:
        return self._container.get_service(ModelRuntimeListenerFdStore)

    @property
    def _environ(self) -> MutableMapping[str, str]:
        return self._container.get_service(ModelRuntimeListenerEnvironment).values

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @property
    def fd_store(self) -> ModelRuntimeListenerFdStore:
        """The sockets this holder keeps; the broker serves from the same store."""
        return self._fds

    def handle(
        self, request: ModelRuntimeListenerHolderRequest
    ) -> ModelRuntimeListenerHolderResult:
        match request.operation:
            case "install_bind":
                return self._bind(request)
            case "recover":
                return _recover(self._fds, self._environ)
            case "reboot_rebind":
                return self._reboot_rebind(request)
            case "prepare_reexec":
                return _remap_for_reexec(self._fds, self._environ)
            case "drop":
                return _drop(self._fds, self._environ)
        raise ValueError(f"unknown holder operation {request.operation!r}")

    def prepare_for_reexec(self) -> ModelRuntimeListenerHolderResult:
        """Remap the listeners; the agent calls this immediately before ``os.execv``."""
        return self.handle(
            ModelRuntimeListenerHolderRequest(operation="prepare_reexec")
        )

    # ------------------------------------------------------------------ results

    def _bind(
        self, request: ModelRuntimeListenerHolderRequest
    ) -> ModelRuntimeListenerHolderResult:
        wanted = (
            (EnumRuntimeListenerPool.MAIN, request.main_port),
            (EnumRuntimeListenerPool.EFFECTS, request.effects_port),
        )
        bound: list[tuple[str, socket.socket]] = []
        try:
            for pool, port in wanted:
                if self._fds.socket_for(pool.value) is not None:
                    continue
                sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                try:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((request.host, port))
                    sock.listen(request.backlog)
                except OSError:
                    sock.close()
                    raise
                bound.append((pool.value, sock))
        except OSError as exc:
            for _, sock in bound:
                sock.close()
            reason = (
                "port_in_use"
                if exc.errno == errno.EADDRINUSE
                else errno.errorcode.get(exc.errno or 0, "os_error")
            )
            return _result(self._fds, f"refused:{reason}", ok=False)
        for pool_name, sock in bound:
            self._fds.keep(pool_name, sock)
        notify = self._environ.get("NOTIFY_SOCKET", "")
        if notify:
            try:
                for pool_name, sock in bound:
                    _sd_notify(
                        notify, f"FDSTORE=1\nFDNAME={pool_name}", (sock.fileno(),)
                    )
            except OSError:
                for _, held in self._fds.release_all():
                    held.close()
                return _result(self._fds, "refused:fd_store_notify_failed", ok=False)
        return _result(self._fds, "bound" if notify else "bound:no_fd_store")

    def _reboot_rebind(
        self, request: ModelRuntimeListenerHolderRequest
    ) -> ModelRuntimeListenerHolderResult:
        if self._fds.pools():
            return _result(self._fds, "inert:already_held")
        if not request.install_record_path:
            return _result(self._fds, "inert:no_install_record")
        try:
            record = json.loads(Path(request.install_record_path).read_text())
        except FileNotFoundError:
            return _result(self._fds, "inert:no_install_record")
        except (OSError, ValueError):
            return _result(self._fds, "refused:install_record_unreadable", ok=False)
        if not isinstance(record, dict) or record.get("state") != "completed":
            return _result(self._fds, "inert:install_not_completed")
        return self._bind(request)

    # ------------------------------------------------------------------ recover


__all__: list[str] = ["HandlerRuntimeListenerHolder"]
