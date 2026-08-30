# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex ledger read`` -- the literal command in GOAL row 0b (OMN-17205).

The whole point of this command is that ONE line, run on the operator's own
machine with no cluster access and no secret on the command line, answers the
question "is this correlation id in the cloud ledger?". These tests hold that
shape: the credential comes from the ``~/.onex`` store and nowhere else, the
route comes from the stored config and not from a literal, every outcome is a
distinct exit code, and nothing secret reaches stdout or stderr.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

# click >= 8.2 keeps stdout and stderr separate on ``Result`` by default and
# removed the ``mix_stderr`` constructor argument, so a plain runner already
# gives the split these assertions need.
from click.testing import CliRunner
from pydantic import SecretStr

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.cli_ledger import ledger_group
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.gateway.models.model_cloud_ledger_read import ModelCloudLedgerRead
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)

pytestmark = pytest.mark.unit

_CID = "01J8ZC9K7Q0000000000000001"
_SECRET = "super-secret-value"  # pragma: allowlist secret


def _credential() -> ModelGatewayCredential:
    return ModelGatewayCredential(
        tenant_slug="acme",
        client_id="t-acme-principal",
        client_secret=SecretStr(_SECRET),
        token_endpoint="https://keycloak.invalid/realms/acme/protocol/openid-connect/token",
        base_url="https://api.invalid",
        edge_instance_id="test-edge",
    )


def _result(verdict: EnumCloudLedgerVerdict, **over: Any) -> ModelCloudLedgerRead:
    fields: dict[str, Any] = {
        "verdict": verdict,
        "correlation_id": _CID,
        "projection": "hook_events",
        "url": f"https://api.invalid/v1/projections/hook-events/by-correlation?correlation_id={_CID}",
        "http_status": 200,
        "count": 0,
        "rows": [],
        "detail": "d",
    }
    fields.update(over)
    return ModelCloudLedgerRead(**fields)


def _patch(monkeypatch: pytest.MonkeyPatch, result: ModelCloudLedgerRead) -> list[Any]:
    """Replace the store and the reader; the CLI wiring is what is under test."""
    calls: list[Any] = []

    class _Store:
        def __init__(self, *, onex_home: Path) -> None:
            calls.append(("store", onex_home))

        def load(self) -> ModelGatewayCredential:
            return _credential()

    class _Reader:
        def __init__(self, *, transport: Any, credential: Any) -> None:
            calls.append(("reader", credential))

        async def read(self, **kwargs: Any) -> ModelCloudLedgerRead:
            calls.append(("read", kwargs))
            return result

    monkeypatch.setattr("omnibase_infra.cli.cli_ledger.StoreGatewayCredential", _Store)
    monkeypatch.setattr("omnibase_infra.cli.cli_ledger.CloudLedgerReader", _Reader)
    return calls


# ---------------------------------------------------------------------------
# AC1/AC3 — one command, a row or a typed answer, distinct exit codes
# ---------------------------------------------------------------------------


def test_found_prints_the_row_and_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"correlation_id": _CID, "event_type": "onex.evt.omniclaude.tool-executed.v1"}
    _patch(
        monkeypatch,
        _result(EnumCloudLedgerVerdict.FOUND, count=1, rows=[row]),
    )
    res = CliRunner().invoke(ledger_group, ["read", "--correlation-id", _CID])
    assert res.exit_code == 0
    document = json.loads(res.stdout)
    assert document["verdict"] == "found"
    assert document["count"] == 1
    assert document["rows"][0]["event_type"] == "onex.evt.omniclaude.tool-executed.v1"


@pytest.mark.parametrize(
    ("verdict", "expected_exit"),
    [
        (EnumCloudLedgerVerdict.NOT_FOUND, 1),
        (EnumCloudLedgerVerdict.PROJECTION_ABSENT, 2),
        (EnumCloudLedgerVerdict.UNAUTHENTICATED, 3),
        (EnumCloudLedgerVerdict.UNAVAILABLE, 4),
    ],
)
def test_every_non_found_verdict_has_its_own_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    verdict: EnumCloudLedgerVerdict,
    expected_exit: int,
) -> None:
    _patch(monkeypatch, _result(verdict))
    res = CliRunner().invoke(ledger_group, ["read", "--correlation-id", _CID])
    assert res.exit_code == expected_exit
    assert json.loads(res.stdout)["verdict"] == verdict.value


def test_stdout_is_only_json_so_the_probe_composes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(EnumCloudLedgerVerdict.NOT_FOUND))
    res = CliRunner().invoke(ledger_group, ["read", "--correlation-id", _CID])
    json.loads(res.stdout)  # parses cleanly, no banner or prose prepended


# ---------------------------------------------------------------------------
# AC2 — the credential path
# ---------------------------------------------------------------------------


def test_credential_comes_from_the_onex_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch(monkeypatch, _result(EnumCloudLedgerVerdict.NOT_FOUND))
    CliRunner().invoke(ledger_group, ["read", "--correlation-id", _CID])
    kinds = [c[0] for c in calls]
    assert "store" in kinds
    onex_home = next(c[1] for c in calls if c[0] == "store")
    assert onex_home.name == ".onex"


def test_no_secret_reaches_stdout_or_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        _result(EnumCloudLedgerVerdict.UNAUTHENTICATED, detail="refused"),
    )
    res = CliRunner().invoke(ledger_group, ["read", "--correlation-id", _CID])
    assert _SECRET not in res.stdout
    assert _SECRET not in (res.stderr or "")


def test_there_is_no_way_to_pass_a_secret_on_the_command_line() -> None:
    help_text = CliRunner().invoke(ledger_group, ["read", "--help"]).output
    for forbidden in ("--client-secret", "--token", "--api-key", "--password"):
        assert forbidden not in help_text


def test_missing_credential_is_a_named_remediation_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EmptyStore:
        def __init__(self, *, onex_home: Path) -> None:
            pass

        def load(self) -> ModelGatewayCredential:
            raise ModelOnexError(
                "no gateway credential; run 'onex auth login --tenant-slug ...'"
            )

    monkeypatch.setattr(
        "omnibase_infra.cli.cli_ledger.StoreGatewayCredential", _EmptyStore
    )
    res = CliRunner().invoke(ledger_group, ["read", "--correlation-id", _CID])
    assert res.exit_code == 3
    assert "onex auth login" in (res.stderr or "")
    assert "Traceback" not in (res.stderr or "")


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------


def test_options_reach_the_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, _result(EnumCloudLedgerVerdict.NOT_FOUND))
    CliRunner().invoke(
        ledger_group,
        [
            "read",
            "--correlation-id",
            _CID,
            "--limit",
            "3",
            "--include-payload",
        ],
    )
    kwargs = next(c[1] for c in calls if c[0] == "read")
    assert kwargs["correlation_id"] == _CID
    assert kwargs["limit"] == 3
    assert kwargs["include_payload"] is True
    assert isinstance(kwargs["now"], datetime)
    assert kwargs["now"].tzinfo is not None
    assert kwargs["now"].utcoffset() == UTC.utcoffset(None)


def test_correlation_id_is_required() -> None:
    res = CliRunner().invoke(ledger_group, ["read"])
    assert res.exit_code != 0
