# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev ledger client reads the node's state-backed point answers (OMN-22478).

On 2026-11-06 ``onex ledger query --query inbox`` failed with WorkLedgerQueryError
and a bounded read printed the cause, "Ledger reply lacks freshness or parity":
the node answers inbox, holds and open-claims from ``work_ledger_state`` with a
status, a page size and a watermark and no freshness or parity, and the client
demanded both of every answer. ``fixtures/work_ledger_point_answers_omn22478.json``
holds the three answers as the node serialises them (recorded from the node's own
point-query fixtures on the lab, ``model_dump_json`` of the result).
"""

from __future__ import annotations

import json
from pathlib import Path
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
from omnibase_infra.models.model_work_ledger_query_client_config import (
    ModelWorkLedgerQueryClientConfig,
)
from tests.unit.cli import test_work_ledger_query_client_omn20740 as client_tests

pytestmark = pytest.mark.unit

# The reply-wire fixture of the client's own tests, reused as it stands.
wire = client_tests.wire

RECORDED = json.loads(
    (
        Path(__file__).parent / "fixtures" / "work_ledger_point_answers_omn22478.json"
    ).read_text()
)
REFUSAL = {
    "code": "answer_too_large",
    "retryable": False,
    "bound_bytes": 917516,
    "answer_bytes": 73182575,
    "matched": 3000,
    "suggested_limit": 10,
    "suggestion": (
        "the newest-per-lane answer would be 73182575 bytes against the bound of "
        "917516 bytes, so it was not sent; set limit=10 (the newest rows are "
        "answered), or add since, until or kinds to narrow the lanes. Repeating "
        "this request refuses again."
    ),
}


def _client() -> HandlerWorkLedgerQuery:
    return HandlerWorkLedgerQuery(
        ModelKafkaEventBusConfig(bootstrap_servers="broker.invalid:9092")
    )


def _recorded(kind: str, **changes: object) -> dict[str, object]:
    answer = {k: v for k, v in RECORDED[kind].items() if k != "correlation_id"}
    answer.update(changes)
    return answer


# ---- golden ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "request_fields"),
    [
        ("inbox", {"lanes": ["lane-x"]}),
        ("holds", {}),
        ("open-claims", {}),
    ],
)
async def test_golden_recorded_point_answer_is_returned(
    kind: str, request_fields: dict[str, object], wire: SimpleNamespace
) -> None:
    wire.reply_override = _recorded(kind)
    result = await _client().query({"query": kind, **request_fields})
    assert result["status"] == "ok"
    assert result["page_size"] == 500
    assert result["watermark"] == "2026-11-02T02:59:40Z"
    assert result["lines"]
    assert "freshness" in result and result["freshness"] is None


def test_golden_the_contract_names_the_point_queries() -> None:
    config = load_work_ledger_query_client_config()
    assert set(config.point_queries) == {"open-claims", "holds", "inbox"}
    assert set(config.point_queries) <= set(config.queries)


async def test_golden_a_node_refusal_is_named_with_its_bound_and_bounded_form(
    wire: SimpleNamespace,
) -> None:
    wire.reply_override = {
        "status": "refused",
        "refusal": REFUSAL,
        "freshness": None,
        "parity": None,
    }
    with pytest.raises(WorkLedgerQueryError, match="refused") as raised:
        await _client().query({"query": "newest-per-lane"})
    assert "917516" in raised.value.message
    assert "limit=10" in raised.value.message
    wire.consumer.stop.assert_awaited_once()


def test_golden_cli_prints_the_refusal_sentence_on_its_last_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refused(*args: object, **kwargs: object) -> dict[str, object]:
        raise WorkLedgerQueryError(f"Ledger node refused: {REFUSAL['suggestion']}")

    monkeypatch.setattr("omnibase_infra.cli.cli_ledger.run_work_ledger_read", refused)
    result = CliRunner().invoke(ledger_group, ["query", "--query", "newest-per-lane"])
    assert result.exit_code == 1
    assert not result.stdout
    last = result.stderr.strip().splitlines()[-1]
    assert "WorkLedgerQueryError" in last
    assert "limit=10" in last


# ---- error chain ----------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"watermark": None},
        {"status": "error"},
        {"page_size": None},
    ],
)
async def test_error_a_point_answer_missing_its_own_facts_is_refused(
    override: dict[str, object], wire: SimpleNamespace
) -> None:
    wire.reply_override = _recorded("inbox", **override)
    with pytest.raises(WorkLedgerQueryError, match="watermark"):
        await _client().query({"query": "inbox", "lanes": ["lane-x"]})
    wire.consumer.stop.assert_awaited_once()


async def test_error_a_non_point_answer_still_needs_freshness_and_parity(
    wire: SimpleNamespace,
) -> None:
    """The point rule must not loosen the other queries: a rows answer built like
    the recorded inbox answer (watermark and page size, no freshness) is refused."""
    wire.reply_override = _recorded("inbox", query="rows")
    with pytest.raises(WorkLedgerQueryError, match="freshness or parity"):
        await _client().query({"query": "rows"})


async def test_error_a_refusal_without_a_bound_is_still_a_refusal(
    wire: SimpleNamespace,
) -> None:
    wire.reply_override = {"status": "refused", "refusal": None}
    with pytest.raises(WorkLedgerQueryError, match="named no bound"):
        await _client().query({"query": "rows"})


def test_error_a_point_query_the_client_does_not_support_does_not_load() -> None:
    declared = load_work_ledger_query_client_config()
    with pytest.raises(ValueError, match="point_queries"):
        ModelWorkLedgerQueryClientConfig(
            lane="dev",
            command_topic=declared.command_topic,
            completed_topic=declared.completed_topic,
            consumer_group_prefix="local.client.consume.v1.",
            queries=("rows",),
            point_queries=("inbox",),
        )
