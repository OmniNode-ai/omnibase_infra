# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline enforcement checks for the restricted durable-claim boundary."""

from __future__ import annotations

import asyncio
import socket
import struct
from pathlib import Path
from typing import cast

import pytest

from omnibase_infra.runtime.action_authorization_claim import (
    ActionAuthorizationClaimUnixRpc,
    EnumActionAuthorizationClaimOutcome,
    ModelActionAuthorizationClaimResult,
)
from omnibase_infra.runtime.action_authorization_claim import (
    local_interface as local_interface_module,
)

_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION = (
    _ROOT / "docker/migrations/forward/107_create_action_authorization_nonce_claim.sql"
)
_ROLE = "rsd_action_authorization_claim"


class _Port:
    async def claim(self, request: object) -> ModelActionAuthorizationClaimResult:
        del request
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.ERROR
        )


class _PeerSocket:
    def __init__(self, uid: int) -> None:
        self._uid = uid

    def getsockopt(self, level: int, option: int, size: int) -> bytes:
        assert level == socket.SOL_SOCKET
        assert option == socket.SO_PEERCRED
        assert size == struct.calcsize("3i")
        return struct.pack("3i", 731, self._uid, 732)


class _Writer:
    def __init__(self, peer_socket: _PeerSocket | None) -> None:
        self._peer_socket = peer_socket
        self.closed = False

    def get_extra_info(self, name: str) -> _PeerSocket | None:
        assert name == "socket"
        return self._peer_socket

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _UnreadableReader:
    async def readuntil(self, separator: bytes) -> bytes:
        del separator
        raise AssertionError("payload parsing must follow peer authentication")


def _server() -> ActionAuthorizationClaimUnixRpc:
    return ActionAuthorizationClaimUnixRpc(
        socket_path=Path("/private/tmp/claim.sock"),
        claim_port=_Port(),  # type: ignore[arg-type]
        socket_owner_uid=501,
        authorized_unix_uid=501,
        restricted_principal=_ROLE,
    )


@pytest.mark.unit
def test_restricted_principal_has_schema_and_function_only_not_public_or_table_dml() -> (
    None
):
    sql = _MIGRATION.read_text(encoding="utf-8")

    assert "CREATE ROLE rsd_action_authorization_claim" in sql
    assert "REVOKE ALL ON SCHEMA action_authorization_claim FROM PUBLIC" in sql
    assert (
        "REVOKE ALL ON TABLE action_authorization_claim.nonce_claims FROM PUBLIC" in sql
    )
    assert (
        "REVOKE ALL ON TABLE action_authorization_claim.nonce_claims FROM rsd_action_authorization_claim"
        in sql
    )
    assert (
        "REVOKE ALL ON FUNCTION action_authorization_claim.claim_action_authorization"
        in sql
    )
    assert (
        "GRANT USAGE ON SCHEMA action_authorization_claim TO rsd_action_authorization_claim"
        in sql
    )
    assert (
        "GRANT EXECUTE ON FUNCTION action_authorization_claim.claim_action_authorization"
        in sql
    )
    for privilege in (
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "TRUNCATE",
        "REFERENCES",
        "TRIGGER",
    ):
        assert (
            "GRANT "
            f"{privilege} ON TABLE action_authorization_claim.nonce_claims "
            f"TO {_ROLE}"
        ) not in sql


@pytest.mark.unit
def test_uds_peer_credential_is_exact_uid_and_principal_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_interface_module.sys, "platform", "linux")
    monkeypatch.setattr(local_interface_module.socket, "SO_PEERCRED", 17, raising=False)
    server = _server()

    authorized = _Writer(_PeerSocket(501))
    wrong_uid = _Writer(_PeerSocket(502))

    assert server._peer_is_authorized(cast("asyncio.StreamWriter", authorized))
    assert not server._peer_is_authorized(cast("asyncio.StreamWriter", wrong_uid))


@pytest.mark.unit
def test_uds_refuses_unsupported_peer_api_before_payload_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_interface_module.sys, "platform", "darwin")
    server = _server()
    reader = _UnreadableReader()
    writer = _Writer(None)

    asyncio.run(
        server._handle_connection(
            cast("asyncio.StreamReader", reader), cast("asyncio.StreamWriter", writer)
        )
    )

    assert writer.closed


@pytest.mark.unit
def test_uds_framing_is_bounded_before_json_decoding() -> None:
    source = Path(local_interface_module.__file__).read_text(encoding="utf-8")

    assert "limit=_MAX_REQUEST_BYTES" in source
    assert 'reader.readuntil(b"\\n")' in source
    assert "asyncio.LimitOverrunError" in source
    assert "if not self._peer_is_authorized(writer):" in source
