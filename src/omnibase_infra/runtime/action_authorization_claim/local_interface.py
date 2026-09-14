# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Restricted local Unix-domain RPC for claim-before-action bootstrap."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import struct
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.runtime.action_authorization_claim.enum_action_authorization_claim_outcome import (
    EnumActionAuthorizationClaimOutcome,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_overlay import (
    ModelActionAuthorizationClaimOverlay,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_request import (
    ModelActionAuthorizationClaimRequest,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_result import (
    ModelActionAuthorizationClaimResult,
)
from omnibase_infra.runtime.action_authorization_claim.protocol import (
    ProtocolActionAuthorizationClaimPort,
)

AdapterFactory = Callable[
    [ModelActionAuthorizationClaimOverlay], ProtocolActionAuthorizationClaimPort
]
_MAX_REQUEST_BYTES = 16_384


class ActionAuthorizationClaimOverlayError(ValueError):
    """The bootstrap overlay is missing or does not name the restricted seam."""


def build_local_claim_interface(
    overlay: Mapping[str, object], *, adapter_factory: AdapterFactory
) -> ActionAuthorizationClaimUnixRpc:
    """Build a local claim-only server from explicit overlay references.

    The composition root supplies the adapter factory. This module deliberately
    performs no environment lookup, DSN construction, or configuration-file
    fallback.
    """
    try:
        resolved_overlay = ModelActionAuthorizationClaimOverlay.model_validate(overlay)
    except ValidationError as exc:
        raise ActionAuthorizationClaimOverlayError(
            "invalid claim bootstrap overlay"
        ) from exc
    return ActionAuthorizationClaimUnixRpc(
        socket_path=Path(resolved_overlay.unix_socket_path),
        claim_port=adapter_factory(resolved_overlay),
        socket_owner_uid=resolved_overlay.socket_owner_uid,
        authorized_unix_uid=resolved_overlay.authorized_unix_uid,
        restricted_principal=resolved_overlay.restricted_principal,
    )


class ActionAuthorizationClaimUnixRpc:
    """A mode-0600 Unix socket exposing only the durable ``claim`` operation."""

    def __init__(
        self,
        *,
        socket_path: Path,
        claim_port: ProtocolActionAuthorizationClaimPort,
        socket_owner_uid: int,
        authorized_unix_uid: int,
        restricted_principal: str,
    ) -> None:
        self._socket_path = socket_path
        self._claim_port = claim_port
        self._socket_owner_uid = socket_owner_uid
        self._authorized_unix_uid = authorized_unix_uid
        self._restricted_principal = restricted_principal
        self._server: asyncio.AbstractServer | None = None
        self._socket_identity: tuple[int, int] | None = None

    def _validate_socket_parent(self) -> None:
        """Require an overlay-owned, non-writable directory before binding."""
        if not self._socket_path.is_absolute():
            msg = "local claim socket path must be absolute"
            raise ValueError(msg)
        try:
            parent_stat = os.lstat(self._socket_path.parent)
        except FileNotFoundError as exc:
            msg = "local claim socket parent does not exist"
            raise FileNotFoundError(msg) from exc
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            msg = "local claim socket parent must be a non-symlink directory"
            raise PermissionError(msg)
        if parent_stat.st_uid != self._socket_owner_uid:
            msg = "local claim socket parent owner does not match the overlay"
            raise PermissionError(msg)
        group_writable = bool(parent_stat.st_mode & stat.S_IWGRP)
        other_writable = bool(parent_stat.st_mode & stat.S_IWOTH)
        if group_writable or other_writable:
            msg = "local claim socket parent must not be group or other writable"
            raise PermissionError(msg)

    def _assert_socket_path_absent(self) -> None:
        try:
            os.lstat(self._socket_path)
        except FileNotFoundError:
            return
        msg = "refusing to replace an existing local claim socket"
        raise FileExistsError(msg)

    def _remove_owned_socket(self, identity: tuple[int, int]) -> None:
        """Remove only the socket this server created after identity revalidation."""
        try:
            socket_stat = os.lstat(self._socket_path)
        except FileNotFoundError:
            return
        if (
            stat.S_ISSOCK(socket_stat.st_mode)
            and (socket_stat.st_dev, socket_stat.st_ino) == identity
        ):
            self._socket_path.unlink()

    def _record_bound_socket_identity(self) -> tuple[int, int]:
        """Verify the just-bound inode before it can start listening."""
        created_stat = os.lstat(self._socket_path)
        if (
            not stat.S_ISSOCK(created_stat.st_mode)
            or stat.S_IMODE(created_stat.st_mode) != 0o600
        ):
            msg = "local claim bind did not create an exact-mode Unix socket"
            raise PermissionError(msg)
        return (created_stat.st_dev, created_stat.st_ino)

    def _bind_restricted_listener(self) -> tuple[socket.socket, tuple[int, int]]:
        """Bind and verify the final socket before asyncio can accept a peer.

        AF_UNIX permissions are fixed at bind time. Creating the listener with
        a restrictive umask avoids an ambient-umask exposure window; the
        explicit chmod/chown and final lstat make that property auditable.
        """
        self._validate_socket_parent()
        self._assert_socket_path_absent()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        prior_umask = os.umask(0o177)
        identity: tuple[int, int] | None = None
        try:
            try:
                listener.bind(str(self._socket_path))
            except Exception:
                listener.close()
                raise
            try:
                identity = self._record_bound_socket_identity()
            except Exception as bind_identity_error:
                listener.close()
                msg = "local claim socket cleanup is uncertain after bind"
                raise RuntimeError(msg) from bind_identity_error
            try:
                listener.listen()
            except Exception:
                listener.close()
                self._remove_owned_socket(identity)
                raise
        finally:
            os.umask(prior_umask)

        try:
            if identity is None:
                msg = "local claim socket identity was not recorded"
                raise RuntimeError(msg)
            os.chmod(  # noqa: PTH101 - exact pathname syscall is security-critical.
                self._socket_path, stat.S_IRUSR + stat.S_IWUSR
            )
            os.chown(self._socket_path, self._socket_owner_uid, -1)
            socket_stat = os.lstat(self._socket_path)
            if (
                not stat.S_ISSOCK(socket_stat.st_mode)
                or (socket_stat.st_dev, socket_stat.st_ino) != identity
                or stat.S_IMODE(socket_stat.st_mode) != 0o600
                or socket_stat.st_uid != self._socket_owner_uid
            ):
                msg = "local claim socket failed final ownership or mode verification"
                raise PermissionError(msg)
        except Exception:
            listener.close()
            self._remove_owned_socket(identity)
            raise
        return listener, identity

    async def start(self) -> None:
        """Bind one verified local socket; never replace an existing entry."""
        listener, identity = self._bind_restricted_listener()
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                sock=listener,
                limit=_MAX_REQUEST_BYTES,
            )
        except Exception:
            listener.close()
            self._remove_owned_socket(identity)
            raise
        self._socket_identity = identity

    async def close(self) -> None:
        """Close only the socket instance this server created."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._socket_identity is None:
            return
        self._remove_owned_socket(self._socket_identity)
        self._socket_identity = None

    @staticmethod
    def _error_result() -> ModelActionAuthorizationClaimResult:
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.ERROR
        )

    def _peer_is_authorized(self, writer: asyncio.StreamWriter) -> bool:
        """Authenticate the local peer before any request bytes are read.

        The Unix UID is an overlay-pinned local identity. The PostgreSQL role is
        also overlay-pinned and never accepted from the wire; Linux supplies
        both the required socket credential API and trustworthy peer metadata.
        """
        if (
            sys.platform != "linux"
            or self._restricted_principal != "rsd_action_authorization_claim"
            or not hasattr(socket, "SO_PEERCRED")
        ):
            return False
        peer_socket = writer.get_extra_info("socket")
        if peer_socket is None:
            return False
        try:
            raw_credentials = peer_socket.getsockopt(  # type: ignore[union-attr]
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            _, peer_uid, _ = struct.unpack("3i", raw_credentials)
        except (AttributeError, OSError, struct.error):
            return False
        return peer_uid == self._authorized_unix_uid

    async def _claim_from_wire(
        self, message: object
    ) -> ModelActionAuthorizationClaimResult:
        if not isinstance(message, dict) or set(message) != {"operation", "request"}:
            return self._error_result()
        if message["operation"] != "claim" or not isinstance(message["request"], dict):
            return self._error_result()
        try:
            request = ModelActionAuthorizationClaimRequest.model_validate(
                message["request"]
            )
        except ValidationError:
            return self._error_result()
        return await self._claim_port.claim(request)

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            if not self._peer_is_authorized(writer):
                return
            try:
                raw = await reader.readuntil(b"\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                result = self._error_result()
            else:
                if not raw or len(raw) > _MAX_REQUEST_BYTES:
                    result = self._error_result()
                else:
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        result = self._error_result()
                    else:
                        result = await self._claim_from_wire(message)
            response = json.dumps(result.model_dump(mode="json"), separators=(",", ":"))
            writer.write(response.encode("utf-8") + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


__all__ = [
    "ActionAuthorizationClaimOverlayError",
    "ActionAuthorizationClaimUnixRpc",
    "build_local_claim_interface",
]
