# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Safe-mock tests for the public Unix-socket claim client and CLI seam."""

from __future__ import annotations

import asyncio
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.runtime.action_authorization_claim import (
    EnumActionAuthorizationClaimOutcome,
    ModelActionAuthorizationClaimRequest,
    ModelActionAuthorizationClaimResult,
    claim_action_authorization_via_unix_socket,
)
from omnibase_infra.runtime.action_authorization_claim import cli as cli_module
from omnibase_infra.runtime.action_authorization_claim import client as client_module


def _request() -> ModelActionAuthorizationClaimRequest:
    return ModelActionAuthorizationClaimRequest(
        authorization_id="action-auth-12345678-1234-1234-1234-123456789abc",
        ticket_id="OMN-17462",
        contract_path="contracts/OMN-17462.yaml",
        contract_commit_sha="a" * 40,
        contract_sha256="sha256:" + "b" * 64,
        action_id="postgres-push-lane-bootstrap",
        source_sha="c" * 40,
        artifact_sha256="sha256:" + "d" * 64,
        target_database="rsd_push_lanes",
        target_schema="push_lanes",
        target_service="rsd_push_lane_broker",
        target_principal="rsd_push_lane_broker",
        execute_enabled=False,
        issuer="operator-governance",
        nonce="e" * 64,
        issued_at=datetime(2030, 1, 1, 10, tzinfo=UTC),
        expires_at=datetime(2030, 1, 1, 11, tzinfo=UTC),
        one_time_use=True,
        reason="bounded execute-disabled bootstrap verification",
    )


class _Reader:
    async def readuntil(self, separator: bytes) -> bytes:
        assert separator == b"\n"
        return (
            b'{"outcome":"CLAIMED","state":"CLAIMED","version":1,'
            b'"redacted_receipt_digest":"ffffffffffffffffffffffffffffffff'
            b'ffffffffffffffffffffffffffffffff"}\n'
        )


class _Writer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, value: bytes) -> None:
        self.writes.append(value)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


@pytest.mark.unit
def test_stable_client_uses_one_bounded_claim_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _Writer()

    async def fake_open_unix_connection(
        path: str, *, limit: int
    ) -> tuple[_Reader, _Writer]:
        assert path == "/private/tmp/claim.sock"
        assert limit == 16_384
        return _Reader(), writer

    monkeypatch.setattr(
        client_module.asyncio, "open_unix_connection", fake_open_unix_connection
    )

    result = asyncio.run(
        claim_action_authorization_via_unix_socket(
            socket_path=Path("/private/tmp/claim.sock"), request=_request()
        )
    )

    assert result.outcome is EnumActionAuthorizationClaimOutcome.CLAIMED
    assert writer.closed
    assert len(writer.writes) == 1
    assert writer.writes[0].startswith(b'{"operation":"claim","request":')


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "expected_exit"),
    [
        (EnumActionAuthorizationClaimOutcome.CLAIMED, 0),
        (EnumActionAuthorizationClaimOutcome.ALREADY_CONSUMED, 1),
        (EnumActionAuthorizationClaimOutcome.EXPIRED, 1),
        (EnumActionAuthorizationClaimOutcome.MISMATCH, 1),
        (EnumActionAuthorizationClaimOutcome.ERROR, 1),
    ],
)
def test_cli_returns_success_only_for_claimed_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: EnumActionAuthorizationClaimOutcome,
    expected_exit: int,
) -> None:
    seen: list[tuple[Path, ModelActionAuthorizationClaimRequest]] = []

    async def fake_claim(
        *, socket_path: Path, request: ModelActionAuthorizationClaimRequest
    ) -> ModelActionAuthorizationClaimResult:
        seen.append((socket_path, request))
        return ModelActionAuthorizationClaimResult(
            outcome=outcome,
            state="CLAIMED",
            version=1,
            redacted_receipt_digest="f" * 64,
        )

    monkeypatch.setattr(
        cli_module, "claim_action_authorization_via_unix_socket", fake_claim
    )
    payload = json.dumps(_request().model_dump(mode="json")).encode("utf-8")
    monkeypatch.setattr(cli_module.sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))

    assert cli_module.main(["--socket", "/private/tmp/claim.sock"]) == expected_exit
    assert seen[0][0] == Path("/private/tmp/claim.sock")
    assert seen[0][1] == _request()
    assert json.loads(capsys.readouterr().out)["outcome"] == outcome.value


@pytest.mark.unit
def test_cli_rejects_oversized_or_invalid_stdin_without_opening_a_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = b"x" * 16_385
    monkeypatch.setattr(
        cli_module.sys, "stdin", io.TextIOWrapper(io.BytesIO(oversized))
    )

    assert cli_module.main(["--socket", "/private/tmp/claim.sock"]) == 2
