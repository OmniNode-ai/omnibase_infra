# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue dev merge build decision runs through the node contract's own route and settings (OMN-26316).

The deploy job reading node's contract binds ``request_overdue_dev_merge_build`` to a handler
and sets the window that merge waits for a dev build. Nothing here is stubbed: the handler is
resolved from the contract's route by name and module, the window is the contract's setting,
and the jobs are the deploy agent's real job records of omnibase_infra#6191 (merged
17:46:52Z on 2026-11-23, next dev job accepted 19:00:23Z).
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_build_request import (
    ModelOverdueDevMergeBuildRequest,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _ROOT / (
    "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_FIXTURE = (
    _ROOT
    / "tests/fixtures/overdue_dev_merge_build_omn26316/merge_6191_job_records.json"
)
_OPERATION = "request_overdue_dev_merge_build"


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _handler() -> Any:
    routes = _contract()["handler_routing"]["handlers"]
    declared = next(r for r in routes if r["operation"] == _OPERATION)["handler"]
    module = importlib.import_module(declared["module"])
    return getattr(module, declared["name"])()


def _request(now: str, *, drop: tuple[str, ...] = ()) -> Any:
    record = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    merge = record["merge"]
    jobs = [
        {k: v for k, v in job.items() if k != "completed_at"}
        for job in record["jobs"]
        if job["job_id"][:8] not in drop
    ]
    return ModelOverdueDevMergeBuildRequest.model_validate(
        {
            "repository": merge["repository"],
            "merge_sha": merge["merge_sha"],
            "merged_at": merge["merged_at"],
            "runtime_class": merge["runtime_class"],
            "now": datetime.fromisoformat(now.replace("Z", "+00:00")).astimezone(UTC),
            "window_seconds": _contract()["settings"]["dev_merge_build_window_seconds"],
            "jobs": jobs,
            "descendant_shas": merge["descendant_shas"],
        }
    )


def test_contract_routes_the_operation_to_the_overdue_handler_and_names_its_caller() -> (
    None
):
    contract = _contract()
    handler = _handler()
    assert type(handler).__name__ == "HandlerDevMergeBuildOverdue"
    assert (
        contract["settings"]["operation_callers"][_OPERATION]
        == "check_overdue_dev_merges"
    )
    assert any(
        c["name"] == "deploy.dev_merge_build_overdue" for c in contract["capabilities"]
    )


def test_contract_window_is_one_hour_and_the_merge_waits_until_it_ends() -> None:
    assert _contract()["settings"]["dev_merge_build_window_seconds"] == 3600
    decision = _handler().handle(_request("2026-11-23T18:46:51Z", drop=("2098b18c",)))
    assert isinstance(decision, ModelOverdueDevMergeBuildDecision)
    assert (decision.outcome, decision.reason) == ("WAIT", "within_window")


def test_contract_window_requests_one_build_when_it_ends_with_no_dev_build() -> None:
    decision = _handler().handle(_request("2026-11-23T18:46:52Z", drop=("2098b18c",)))
    assert (decision.outcome, decision.reason) == ("REQUEST", "overdue_no_build")
    assert decision.request_id is not None
    again = _handler().handle(_request("2026-11-23T19:30:00Z", drop=("2098b18c",)))
    assert again.request_id == decision.request_id


def test_the_dev_job_accepted_at_1900_satisfies_the_merge() -> None:
    decision = _handler().handle(_request("2026-11-23T20:00:00Z"))
    assert (decision.outcome, decision.reason) == ("SATISFIED", "built_by_job")
    assert str(decision.job_id).startswith("2098b18c")
