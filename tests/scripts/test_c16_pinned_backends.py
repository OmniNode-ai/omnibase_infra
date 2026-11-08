# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pinned-backend comparison of the C16 probe (OMN-23176, scenario R7).

Each case is built from the recorded shape of a pinned delegation (the receipt's
accepted attempt and the ``delegation_events`` row) with exactly one field
changed, named for what it changes. Nothing here touches a network or database.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers import handler_c16_pinned_backend as probe
from omnibase_infra.models.model_c16_pinned_backend_request import (
    ModelC16PinnedBackendRequest,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "c16_pinned_backends.yaml"
)
CONFIG = {"local": "backend-a", "free_cloud": "backend-b", "paid": "backend-c"}


def _receipt(backend: str, host: str, cost: float) -> dict[str, Any]:
    correlation = str(uuid.uuid4())
    return {
        "correlation_id": correlation,
        "terminal_payload": {
            "status": "completed",
            "attempts": [
                {
                    "acceptance_decision": "accept",
                    "backend_id": backend,
                    "host": host,
                    "cost_usd": cost,
                }
            ],
            "metrics": {"cost_usd": cost},
        },
    }


def _payload(mutate: Any = None) -> dict[str, Any]:
    """A passing replay; ``mutate`` receives (runs, rows) and changes one thing."""
    costs = {"local": 0.0, "free_cloud": 0.0, "paid": 0.0031}
    runs: list[dict[str, Any]] = []
    rows: dict[str, Any] = {}
    for role, backend in CONFIG.items():
        receipt = _receipt(backend, f"host-{role}", costs[role])
        runs.append(
            {"role": role, "backend_id": backend, "receipt": receipt, "error": None}
        )
        rows[receipt["correlation_id"]] = {
            "backend_id": backend,
            "host": f"host-{role}",
            "cost_usd": costs[role],
        }
    if mutate is not None:
        mutate(runs, rows)
    return {"pinned": {"runs": runs, "rows": rows}}


def _grade(payload: dict[str, Any]) -> probe.ProbeResult:
    return probe.grade_pinned_backends(
        CONFIG, probe.pinned_observations_from_replay(payload)
    )


def _row_of(runs: list[dict[str, Any]], role: str) -> str:
    return next(r for r in runs if r["role"] == role)["receipt"]["correlation_id"]


@pytest.mark.unit
def test_three_pinned_runs_that_agree_with_their_rows_pass() -> None:
    result = _grade(_payload())
    assert result.outcome == probe.PASS, result.reasons
    assert result.probe == probe.PINNED_PROBE


@pytest.mark.unit
def test_the_checked_in_config_names_each_role_once() -> None:
    assert probe.ROW_QUERY_PLACEHOLDER in probe.load_row_query(CONFIG_PATH)
    config = probe.load_pinned_config(CONFIG_PATH)
    assert tuple(config) == probe.PINNED_ROLES
    assert len(set(config.values())) == 3


@pytest.mark.unit
def test_a_config_missing_a_role_is_an_input_error_and_never_a_shrunk_set(
    tmp_path: Path,
) -> None:
    short = tmp_path / "short.yaml"
    short.write_text("pinned_backends:\n  local: backend-a\n", encoding="utf-8")
    with pytest.raises(probe.ProbeInputError, match="free_cloud"):
        probe.load_pinned_config(short)


@pytest.mark.unit
def test_a_host_that_differs_from_the_row_fails() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        rows[_row_of(runs, "free_cloud")]["host"] = "another-host"

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any("free_cloud" in r and "host" in r for r in result.reasons)


@pytest.mark.unit
def test_a_backend_that_differs_from_the_row_fails() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        rows[_row_of(runs, "local")]["backend_id"] = "backend-z"

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any("local" in r and "backend_id" in r for r in result.reasons)


@pytest.mark.unit
def test_a_cost_that_differs_from_the_row_fails() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        rows[_row_of(runs, "paid")]["cost_usd"] = 0.0

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any("paid" in r and "cost_usd" in r for r in result.reasons)


@pytest.mark.unit
def test_a_backend_missing_from_the_lane_is_red_and_the_set_is_not_shrunk() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        for run in runs:
            if run["role"] == "paid":
                run["receipt"] = None
                run["error"] = "delegate exited 1"

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any(
        "backend-c" in r and "missing from the lane" in r for r in result.reasons
    )
    # the two backends that did answer are still compared, not made the whole set
    assert "local.row.backend_id" in result.evidence


@pytest.mark.unit
def test_a_role_with_no_run_at_all_is_red() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        runs[:] = [r for r in runs if r["role"] != "free_cloud"]

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any(
        "free_cloud" in r and "no pinned delegation" in r for r in result.reasons
    )


@pytest.mark.unit
def test_an_answer_from_another_rung_than_the_pin_fails() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        run = next(r for r in runs if r["role"] == "local")
        run["receipt"]["terminal_payload"]["attempts"][0]["backend_id"] = "backend-z"

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any("pinned to 'backend-a'" in r for r in result.reasons)


@pytest.mark.unit
def test_an_unreadable_rows_table_is_skip_and_skip_is_red() -> None:
    payload = _payload()
    payload["pinned"]["rows"] = {}
    payload["pinned"]["rows_error"] = "no delegation_events reader is configured"
    result = _grade(payload)
    assert result.outcome == probe.SKIP


@pytest.mark.unit
def test_a_missing_row_for_a_correlation_id_fails() -> None:
    def mutate(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        rows.pop(_row_of(runs, "local"))

    result = _grade(_payload(mutate))
    assert result.outcome == probe.FAIL
    assert any("no delegation_events row" in r for r in result.reasons)


@pytest.mark.unit
def test_the_handler_succeeds_on_a_pass_and_fails_with_a_missing_backend(
    tmp_path: Path,
) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        "pinned_backends:\n  local: backend-a\n  free_cloud: backend-b\n  paid: backend-c\n",
        encoding="utf-8",
    )
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_payload()), encoding="utf-8")

    def drop_paid(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        runs[:] = [r for r in runs if r["role"] != "paid"]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_payload(drop_paid)), encoding="utf-8")
    record = tmp_path / "record.json"
    handler = probe.HandlerC16PinnedBackend(ModelONEXContainer())

    def run(replay: Path) -> Any:
        request = ModelC16PinnedBackendRequest(
            config_path=str(config),
            replay_path=str(replay),
            record_path=str(record),
            timeout_seconds=1.0,
        )
        return asyncio.run(handler.handle(request))

    assert run(good).success is True
    assert json.loads(record.read_text())["probes"][0]["outcome"] == probe.PASS
    failed = run(bad)
    assert failed.success is False
    assert failed.outcome == probe.FAIL
    assert failed.error_message and "backend-c" in failed.error_message
    assert json.loads(record.read_text())["probes"][0]["outcome"] == probe.FAIL


@pytest.mark.unit
def test_an_unreadable_config_is_could_not_run_and_never_success(
    tmp_path: Path,
) -> None:
    request = ModelC16PinnedBackendRequest(
        config_path=str(tmp_path / "absent.yaml"), timeout_seconds=1.0
    )
    result = asyncio.run(
        probe.HandlerC16PinnedBackend(ModelONEXContainer()).handle(request)
    )
    assert result.success is False
    assert result.outcome == "COULD_NOT_RUN"


@pytest.mark.unit
def test_the_live_half_pins_each_role_and_reads_each_row_through_its_one_runner(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    seen: dict[str, dict[str, Any]] = {}

    def run_command(argv: list[str], timeout: float) -> tuple[int, str]:
        calls.append(argv)
        if "--backend-id" in argv:
            backend = argv[argv.index("--backend-id") + 1]
            if backend == "backend-c":
                return 1, "unresolvable pin"
            receipt = _receipt(backend, "host-" + backend, 0.0)
            path = tmp_path / backend / "receipt.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    {
                        "correlation_id": receipt["correlation_id"],
                        "receipt": {
                            "result": {
                                "terminal_payload": {
                                    "payload": receipt["terminal_payload"]
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            seen[receipt["correlation_id"]] = receipt
            return 0, f"full receipt {path}"
        sql = argv[-1]
        correlation = next(c for c in seen if c in sql)
        attempt = seen[correlation]["terminal_payload"]["attempts"][0]
        return 0, f"{attempt['backend_id']}|{attempt['host']}|0.0\n"

    observations = probe.observe_pinned_live(
        CONFIG,
        delegate_argv=["delegate", "prompt"],
        psql_argv=["psql"],
        row_query=probe.ROW_QUERY_PLACEHOLDER,
        timeout=1.0,
        run_command=run_command,
    )
    result = probe.grade_pinned_backends(CONFIG, observations)
    assert [r.role for r in observations.runs] == list(probe.PINNED_ROLES)
    assert result.outcome == probe.FAIL
    assert any(
        "backend-c" in r and "missing from the lane" in r for r in result.reasons
    )
    assert not any("backend-a" in r or "backend-b" in r for r in result.reasons), (
        result.reasons
    )
    assert sum("--backend-id" in c for c in calls) == 3
