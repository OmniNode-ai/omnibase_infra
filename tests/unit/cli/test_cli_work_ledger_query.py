# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane-facing reads are dev-only and preserve the query node's answers."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from pydantic import BaseModel, SecretStr

from omnibase_infra.cli import cli_ledger as cli
from omnibase_infra.cli.cli_ledger import ledger_group
from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bound_lane_client_transport,
)
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)
from omnibase_infra.handlers import handler_work_ledger_query as transport_handler
from omnibase_infra.handlers.handler_work_ledger_query import WorkLedgerQueryError


class Answer(BaseModel):
    rows: tuple[str, ...] = ("database row",)
    freshness: dict[str, object] = {"rows_read": 17}
    parity: dict[str, object] = {"status": "not-exact"}


@pytest.fixture
def client(monkeypatch):
    reader = AsyncMock()
    reader.query.return_value = Answer()

    @contextlib.asynccontextmanager
    async def opened(*, lane, workspace_root):
        assert lane == "dev"
        yield reader

    monkeypatch.setattr(cli, "open_work_ledger_query_client", opened)
    return reader


def test_ledger_ask_reads_database_rows_and_metadata(client):
    result = CliRunner().invoke(
        ledger_group,
        ["query", "--lane", "dev", "--input", "-"],
        input='{"query":"rows","term":"OMN-20740"}',
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == Answer().model_dump(mode="json")
    client.query.assert_awaited_once_with(
        {"query": "rows", "term": "OMN-20740"}, timeout_s=60.0
    )


def test_catchup_queries_all_sections_with_one_fixed_window(client):
    result = CliRunner().invoke(
        ledger_group,
        [
            "catchup",
            "--lane",
            "dev",
            "--since",
            "2026-10-25T00:00:00Z",
            "--to",
            "reader-lane",
        ],
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert set(output) == {"open-claims", "holds", "rulings", "rows", "inbox"}
    calls = [call.args[0] for call in client.query.await_args_list]
    assert len({call["until"] for call in calls}) == 1
    assert all(
        call["since"] == datetime.fromisoformat("2026-10-25T00:00:00+00:00")
        for call in calls
    )
    assert calls[-1]["lane"] == "reader-lane"
    assert calls[-2]["kinds"] == ("FRICTION", "CORRECTION", "STATUS")
    assert all(
        section["parity"] == {"status": "not-exact"} for section in output.values()
    )


@pytest.mark.parametrize("lane", ["prod", "another-lane", "customer"])
@pytest.mark.parametrize("command", ["query", "catchup"])
def test_cli_refuses_other_lanes_before_connecting(lane, command, client):
    args = [command, "--lane", lane]
    if command == "query":
        args.extend(["--input", "-"])
    result = CliRunner().invoke(ledger_group, args, input='{"query":"rows"}')
    assert result.exit_code != 0
    client.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_factory_refuses_prod_before_resolving_anything(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a non-dev selection reached lane resolution")

    monkeypatch.setattr(transport_handler, "resolve_lane_selection_at", unexpected)
    with pytest.raises(WorkLedgerQueryError, match="only for the dev lane"):
        async with cli.open_work_ledger_query_client(lane="prod", workspace_root=None):
            pytest.fail("prod client opened")


def test_node_failure_exits_nonzero_without_an_empty_answer(client):
    client.query.side_effect = WorkLedgerQueryError("database unavailable")
    result = CliRunner().invoke(
        ledger_group,
        ["query", "--lane", "dev", "--input", "-"],
        input='{"query":"rows"}',
    )
    assert result.exit_code != 0
    assert "database unavailable" in result.output
    assert not result.stdout.strip().startswith("{")


@pytest.mark.parametrize("payload", ["[]", "invalid"])
def test_invalid_input_is_refused_before_query(payload, client):
    result = CliRunner().invoke(
        ledger_group, ["query", "--lane", "dev", "--input", "-"], input=payload
    )
    assert result.exit_code != 0
    client.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_factory_binds_the_caller_credential_for_the_entire_bus_lifecycle(
    monkeypatch, tmp_path
):
    selection = ModelDelegateLaneSelection(
        lane="dev",
        bootstrap_servers="test-broker:9092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        declared_in=tmp_path / "lanes.yaml",
    )
    transport = ModelLaneClientTransport(
        lane="dev",
        bootstrap_servers=selection.bootstrap_servers,
        security_protocol=selection.security_protocol,
        sasl_mechanism=selection.sasl_mechanism,
        sasl_username="dev-lane-reader",
        sasl_password=SecretStr("synthetic-test-password"),
        declared_in=selection.declared_in,
    )
    monkeypatch.setattr(
        transport_handler,
        "resolve_lane_selection_at",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        transport_handler,
        "resolve_lane_client_transport_for",
        lambda **kwargs: transport,
    )
    bus = AsyncMock(spec_set=EventBusKafka)
    reader = AsyncMock()
    reader.__aenter__.return_value = reader

    async def check_binding():
        assert bound_lane_client_transport() is transport

    bus.start.side_effect = check_binding
    bus.close.side_effect = check_binding
    monkeypatch.setattr(
        transport_handler.EventBusKafka, "from_bootstrap", lambda address: bus
    )
    monkeypatch.setattr(
        transport_handler, "HandlerWorkLedgerQuery", lambda supplied: reader
    )
    async with cli.open_work_ledger_query_client(
        lane="dev", workspace_root=Path(tmp_path)
    ) as opened:
        assert opened is reader
        assert bound_lane_client_transport() is transport
    assert bound_lane_client_transport() is None
    bus.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_factory_refuses_a_dev_declaration_without_authentication(
    monkeypatch, tmp_path
):
    selection = ModelDelegateLaneSelection(
        lane="dev",
        bootstrap_servers="test:9092",
        security_protocol="PLAINTEXT",
        declared_in=tmp_path / "lanes.yaml",
    )
    monkeypatch.setattr(
        transport_handler,
        "resolve_lane_selection_at",
        lambda *args, **kwargs: selection,
    )
    with pytest.raises(WorkLedgerQueryError, match="authenticated dev lane"):
        async with cli.open_work_ledger_query_client(
            lane="dev", workspace_root=tmp_path
        ):
            pytest.fail("anonymous client opened")
