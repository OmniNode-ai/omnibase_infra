# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex ledger query --query inbox`` end to end over the reply wire (OMN-22478).

The command, the client, the reply consumer and the answer check run together;
only the broker is replaced, by the reply-wire stand-in the client's own tests use.
The reply is the node's inbox answer as the node serialises it (recorded on the lab
from the node's point-query fixtures): status ok, a page size and a watermark, no
freshness and no parity. Before the change this command exited 1 with
``Ledger reply lacks freshness or parity``, as it did against the dev lane on
2026-11-06.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_ledger import ledger_group
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.unit.cli import test_work_ledger_query_client_omn20740 as client_tests

RECORDED = json.loads(
    (
        Path(__file__).parents[2]
        / "unit"
        / "cli"
        / "fixtures"
        / "work_ledger_point_answers_omn22478.json"
    ).read_text()
)

# The reply-wire stand-in of the client's own tests, reused as it stands.
wire = client_tests.wire


@pytest.fixture
def broker_less_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "omnibase_infra.cli.cli_ledger.resolve_work_ledger_query_transport",
        lambda lane: ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092"),
    )


def _inbox_answer(**changes: object) -> dict[str, object]:
    answer = {k: v for k, v in RECORDED["inbox"].items() if k != "correlation_id"}
    answer.update(changes)
    return answer


def test_golden_the_inbox_command_prints_the_nodes_answer(
    broker_less_cli: None, wire: SimpleNamespace
) -> None:
    wire.reply_override = _inbox_answer()
    result = CliRunner().invoke(
        ledger_group, ["query", "--query", "inbox", "--lane", "lane-x"]
    )
    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed["status"] == "ok"
    assert printed["page_size"] == 500
    assert printed["lines"] == RECORDED["inbox"]["lines"]
    assert wire.sent[0][1]["payload"]["lanes"] == ["lane-x"]


def test_error_a_node_refusal_ends_the_command_with_its_bound_on_the_error_line(
    broker_less_cli: None, wire: SimpleNamespace
) -> None:
    wire.reply_override = {
        "status": "refused",
        "refusal": {
            "suggestion": (
                "the newest-per-lane answer would be 73182575 bytes against the "
                "bound of 917516 bytes, so it was not sent; set limit=10"
            )
        },
    }
    result = CliRunner().invoke(
        ledger_group, ["query", "--query", "newest-per-lane", "--limit", "1000"]
    )
    assert result.exit_code == 1
    assert not result.stdout
    last = result.stderr.strip().splitlines()[-1]
    assert "917516" in last
    assert "limit=10" in last


def test_error_an_inbox_answer_without_its_watermark_is_refused(
    broker_less_cli: None, wire: SimpleNamespace
) -> None:
    wire.reply_override = _inbox_answer(watermark=None)
    result = CliRunner().invoke(
        ledger_group, ["query", "--query", "inbox", "--lane", "lane-x"]
    )
    assert result.exit_code == 1
    assert not result.stdout
    assert "watermark" in result.stderr
