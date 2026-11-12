# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev ledger client accepts the node's open-rulings point answer (OMN-24004).

The omnimarket node answers open-rulings from its state table as a point query:
status ok, a page size and a watermark, no freshness or parity. The client's
declaration must name it in ``queries`` and in ``point_queries`` so the answer
is accepted and the CLI prints it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_ledger import ledger_group
from omnibase_infra.errors.error_infra import WorkLedgerQueryError
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_work_ledger_query import (
    HandlerWorkLedgerQuery,
    load_work_ledger_query_client_config,
)
from tests.unit.cli import test_work_ledger_query_client_omn20740 as client_tests

pytestmark = pytest.mark.unit

# The reply-wire fixture of the client's own tests, reused as it stands.
wire = client_tests.wire

OPEN_RULINGS_ANSWER = {
    "query": "open-rulings",
    "status": "ok",
    "watermark": "2026-11-02T02:59:40Z",
    "page_size": 500,
    "matched": 1,
    "lines": ["open-rulings ticket=OMN-1 rulings=1"],
    "rows": [],
    "freshness": None,
    "parity": None,
    "error": None,
    "refusal": None,
}


def _client() -> HandlerWorkLedgerQuery:
    return HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )


@pytest.fixture
def broker_less_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "omnibase_infra.cli.cli_ledger.resolve_work_ledger_query_transport",
        lambda lane: ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092"),
    )


# ---- golden ---------------------------------------------------------------


async def test_golden_the_client_accepts_open_rulings_and_returns_the_point_answer(
    wire: SimpleNamespace,
) -> None:
    wire.reply_override = dict(OPEN_RULINGS_ANSWER)
    result = await _client().query({"query": "open-rulings", "ticket": "OMN-1"})
    assert result["status"] == "ok"
    assert result["page_size"] == 500
    assert result["lines"] == ["open-rulings ticket=OMN-1 rulings=1"]
    sent = wire.sent[0][1]["payload"]
    assert sent["query"] == "open-rulings"
    assert sent["ticket"] == "OMN-1"


def test_the_point_queries_hold_every_state_backed_question_of_the_node() -> None:
    config = load_work_ledger_query_client_config()
    assert set(config.point_queries) == {
        "open-claims",
        "holds",
        "inbox",
        "open-rulings",
    }
    assert set(config.point_queries) <= set(config.queries)


def test_cli_open_rulings_with_a_ticket_prints_the_nodes_answer(
    broker_less_cli: None, wire: SimpleNamespace
) -> None:
    wire.reply_override = dict(OPEN_RULINGS_ANSWER)
    result = CliRunner().invoke(
        ledger_group, ["query", "--query", "open-rulings", "--ticket", "OMN-1"]
    )
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed["status"] == "ok"
    sent = wire.sent[0][1]["payload"]
    assert sent["ticket"] == "OMN-1"


# ---- error chain ----------------------------------------------------------


def test_cli_open_rulings_answer_without_its_watermark_is_refused(
    broker_less_cli: None, wire: SimpleNamespace
) -> None:
    wire.reply_override = dict(OPEN_RULINGS_ANSWER, watermark=None)
    result = CliRunner().invoke(
        ledger_group, ["query", "--query", "open-rulings", "--ticket", "OMN-1"]
    )
    assert result.exit_code == 1
    assert not result.stdout
    assert "watermark" in result.stderr


async def test_a_question_the_client_does_not_declare_is_still_refused(
    wire: SimpleNamespace,
) -> None:
    with pytest.raises(WorkLedgerQueryError, match="Unsupported work-ledger query"):
        await _client().query({"query": "no-such-question"})
