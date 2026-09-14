# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Closed local Unix RPC shape tests for the claim-before-action endpoint."""

from __future__ import annotations

import asyncio
import socket
import stat
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.runtime.action_authorization_claim import (
    ActionAuthorizationClaimUnixRpc,
    EnumActionAuthorizationClaimOutcome,
    ModelActionAuthorizationClaimResult,
)
from omnibase_infra.runtime.action_authorization_claim import (
    local_interface as local_interface_module,
)


def _wire_request() -> dict[str, object]:
    return {
        "authorization_id": "action-auth-12345678-1234-1234-1234-123456789abc",
        "ticket_id": "OMN-17462",
        "contract_path": "contracts/OMN-17462.yaml",
        "contract_commit_sha": "a" * 40,
        "contract_sha256": "sha256:" + "b" * 64,
        "action_id": "postgres-push-lane-bootstrap",
        "source_sha": "c" * 40,
        "artifact_sha256": "sha256:" + "d" * 64,
        "target_database": "rsd_push_lanes",
        "target_schema": "push_lanes",
        "target_service": "rsd_push_lane_broker",
        "target_principal": "rsd_push_lane_broker",
        "execute_enabled": False,
        "issuer": "operator-governance",
        "nonce": "e" * 64,
        "issued_at": datetime(2030, 1, 1, 10, tzinfo=UTC).isoformat(),
        "expires_at": datetime(2030, 1, 1, 11, tzinfo=UTC).isoformat(),
        "one_time_use": True,
        "reason": "bounded execute-disabled bootstrap verification",
    }


class _Port:
    def __init__(self) -> None:
        self.claimed = 0

    async def claim(self, request: object) -> ModelActionAuthorizationClaimResult:
        self.claimed += 1
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.CLAIMED,
            state="CLAIMED",
            version=1,
            redacted_receipt_digest="f" * 64,
        )


@pytest.mark.unit
def test_local_interface_only_accepts_the_claim_operation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    port = _Port()
    interface = ActionAuthorizationClaimUnixRpc(
        socket_path=tmp_path / "claim.sock",
        claim_port=port,
        socket_owner_uid=501,
        authorized_unix_uid=501,
        restricted_principal="rsd_action_authorization_claim",
    )

    claimed = asyncio.run(
        interface._claim_from_wire({"operation": "claim", "request": _wire_request()})
    )
    assert claimed.outcome is EnumActionAuthorizationClaimOutcome.CLAIMED
    assert port.claimed == 1

    rejected = asyncio.run(
        interface._claim_from_wire(
            {"operation": "register", "request": _wire_request()}
        )
    )
    assert rejected.outcome is EnumActionAuthorizationClaimOutcome.ERROR
    assert port.claimed == 1


@pytest.mark.unit
def test_local_socket_binds_under_restrictive_umask_before_asyncio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/claim-parent/claim.sock")
    events: list[str] = []
    state: dict[str, object] = {"bound": False, "mode": 0o600, "uid": 700}

    class _FakeServer:
        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    class _FakeListener:
        def bind(self, path: str) -> None:
            assert path == str(socket_path)
            assert events[-1] == "umask:177"
            state["bound"] = True
            events.append("bind")

        def listen(self) -> None:
            events.append("listen")

        def close(self) -> None:
            events.append("close")

    def fake_lstat(path: Path | str) -> SimpleNamespace:
        candidate = Path(path)
        if candidate == socket_path.parent:
            events.append("lstat-parent")
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o700,
                st_uid=501,
                st_dev=9,
                st_ino=10,
            )
        if candidate == socket_path and state["bound"]:
            events.append("lstat-socket")
            return SimpleNamespace(
                st_mode=stat.S_IFSOCK | int(state["mode"]),
                st_uid=int(state["uid"]),
                st_dev=9,
                st_ino=11,
            )
        raise FileNotFoundError(path)

    def fake_umask(value: int) -> int:
        events.append(f"umask:{value:o}")
        return 0o022 if value == 0o177 else 0o177

    def fake_chmod(path: Path | str, mode: int) -> None:
        assert Path(path) == socket_path
        assert mode == 0o600
        state["mode"] = mode
        events.append("chmod")

    def fake_chown(path: Path | str, uid: int, group: int) -> None:
        assert Path(path) == socket_path
        assert (uid, group) == (501, -1)
        state["uid"] = uid
        events.append("chown")

    async def fake_start_unix_server(
        callback: object, *, sock: _FakeListener, limit: int
    ) -> _FakeServer:
        del callback
        assert limit == 16_384
        assert sock is listener
        assert events[-1] == "lstat-socket"
        events.append("asyncio")
        return _FakeServer()

    listener = _FakeListener()
    monkeypatch.setattr(local_interface_module.os, "lstat", fake_lstat)
    monkeypatch.setattr(local_interface_module.os, "umask", fake_umask)
    monkeypatch.setattr(local_interface_module.os, "chmod", fake_chmod)
    monkeypatch.setattr(local_interface_module.os, "chown", fake_chown)
    monkeypatch.setattr(
        local_interface_module,
        "socket",
        SimpleNamespace(
            AF_UNIX=socket.AF_UNIX,
            SOCK_STREAM=socket.SOCK_STREAM,
            socket=lambda *_: listener,
        ),
    )
    monkeypatch.setattr(asyncio, "start_unix_server", fake_start_unix_server)

    interface = ActionAuthorizationClaimUnixRpc(
        socket_path=socket_path,
        claim_port=_Port(),
        socket_owner_uid=501,
        authorized_unix_uid=700,
        restricted_principal="rsd_action_authorization_claim",
    )
    asyncio.run(interface.start())

    assert events == [
        "lstat-parent",
        "umask:177",
        "bind",
        "lstat-socket",
        "listen",
        "umask:22",
        "chmod",
        "chown",
        "lstat-socket",
        "asyncio",
    ]
    assert state["mode"] == 0o600
    assert state["uid"] == 501


@pytest.mark.unit
def test_listen_failure_closes_and_unlinks_only_the_recorded_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/claim-parent/claim.sock")
    bound = False
    unlinked: list[Path] = []

    class _FailingListener:
        def __init__(self) -> None:
            self.close_calls = 0

        def bind(self, path: str) -> None:
            nonlocal bound
            assert path == str(socket_path)
            bound = True

        def listen(self) -> None:
            raise OSError("listen failed")

        def close(self) -> None:
            self.close_calls += 1

    def fake_lstat(path: Path | str) -> SimpleNamespace:
        candidate = Path(path)
        if candidate == socket_path.parent:
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=501)
        if candidate == socket_path and bound:
            return SimpleNamespace(
                st_mode=stat.S_IFSOCK | 0o600,
                st_uid=501,
                st_dev=9,
                st_ino=11,
            )
        raise FileNotFoundError(path)

    def fake_unlink(path: Path) -> None:
        unlinked.append(path)

    listener = _FailingListener()
    monkeypatch.setattr(local_interface_module.os, "lstat", fake_lstat)
    monkeypatch.setattr(local_interface_module.os, "umask", lambda _: 0o022)
    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(
        local_interface_module,
        "socket",
        SimpleNamespace(
            AF_UNIX=socket.AF_UNIX,
            SOCK_STREAM=socket.SOCK_STREAM,
            socket=lambda *_: listener,
        ),
    )
    interface = ActionAuthorizationClaimUnixRpc(
        socket_path=socket_path,
        claim_port=_Port(),
        socket_owner_uid=501,
        authorized_unix_uid=700,
        restricted_principal="rsd_action_authorization_claim",
    )

    with pytest.raises(OSError, match="listen failed"):
        interface._bind_restricted_listener()

    assert listener.close_calls == 1
    assert unlinked == [socket_path]


@pytest.mark.unit
def test_first_post_bind_lstat_failure_preserves_same_mode_owner_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/claim-parent/claim.sock")
    bound = False
    socket_lstats = 0
    unlinked: list[Path] = []

    class _Listener:
        def __init__(self) -> None:
            self.close_calls = 0
            self.listen_calls = 0

        def bind(self, path: str) -> None:
            nonlocal bound
            assert path == str(socket_path)
            bound = True

        def listen(self) -> None:
            self.listen_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    def fake_lstat(path: Path | str) -> SimpleNamespace:
        nonlocal socket_lstats
        candidate = Path(path)
        if candidate == socket_path.parent:
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=501)
        if candidate == socket_path and not bound:
            raise FileNotFoundError(path)
        if candidate == socket_path:
            socket_lstats += 1
            if socket_lstats == 1:
                raise OSError("first post-bind lstat failed")
            return SimpleNamespace(
                st_mode=stat.S_IFSOCK | 0o600,
                st_uid=501,
                st_dev=9,
                st_ino=11,
            )
        raise FileNotFoundError(path)

    def fake_unlink(path: Path) -> None:
        unlinked.append(path)

    listener = _Listener()
    monkeypatch.setattr(local_interface_module.os, "lstat", fake_lstat)
    monkeypatch.setattr(local_interface_module.os, "umask", lambda _: 0o022)
    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(
        local_interface_module,
        "socket",
        SimpleNamespace(
            AF_UNIX=socket.AF_UNIX,
            SOCK_STREAM=socket.SOCK_STREAM,
            socket=lambda *_: listener,
        ),
    )
    interface = ActionAuthorizationClaimUnixRpc(
        socket_path=socket_path,
        claim_port=_Port(),
        socket_owner_uid=501,
        authorized_unix_uid=700,
        restricted_principal="rsd_action_authorization_claim",
    )

    with pytest.raises(RuntimeError, match="cleanup is uncertain after bind"):
        interface._bind_restricted_listener()

    assert listener.close_calls == 1
    assert listener.listen_calls == 0
    assert unlinked == []
    # A hypothetical retry would report a plausible replacement, but no retry
    # or unlink is allowed because the original dev/inode was never proven.
    assert socket_lstats == 1


@pytest.mark.parametrize(
    ("parent_mode", "parent_uid"),
    [
        (stat.S_IFDIR | 0o770, 501),
        (stat.S_IFLNK | 0o700, 501),
        (stat.S_IFDIR | 0o700, 502),
    ],
    ids=("group-writable", "symlink", "wrong-owner"),
)
def test_local_socket_rejects_unsafe_parent_before_listener_or_handler(
    monkeypatch: pytest.MonkeyPatch, parent_mode: int, parent_uid: int
) -> None:
    socket_path = Path("/claim-parent/claim.sock")
    socket_calls = 0
    handler_calls = 0

    def fake_lstat(path: Path | str) -> SimpleNamespace:
        assert Path(path) == socket_path.parent
        return SimpleNamespace(st_mode=parent_mode, st_uid=parent_uid)

    def fake_socket(*_: object) -> object:
        nonlocal socket_calls
        socket_calls += 1
        return object()

    async def fake_start_unix_server(*_: object, **__: object) -> object:
        nonlocal handler_calls
        handler_calls += 1
        return object()

    monkeypatch.setattr(local_interface_module.os, "lstat", fake_lstat)
    monkeypatch.setattr(
        local_interface_module,
        "socket",
        SimpleNamespace(
            AF_UNIX=socket.AF_UNIX,
            SOCK_STREAM=socket.SOCK_STREAM,
            socket=fake_socket,
        ),
    )
    monkeypatch.setattr(asyncio, "start_unix_server", fake_start_unix_server)
    interface = ActionAuthorizationClaimUnixRpc(
        socket_path=socket_path,
        claim_port=_Port(),
        socket_owner_uid=501,
        authorized_unix_uid=700,
        restricted_principal="rsd_action_authorization_claim",
    )

    with pytest.raises(PermissionError):
        asyncio.run(interface.start())

    assert socket_calls == 0
    assert handler_calls == 0
