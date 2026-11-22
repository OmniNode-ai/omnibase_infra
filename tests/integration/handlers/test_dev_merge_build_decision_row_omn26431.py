# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The build-decision handler writes its row through the node contract's real settings (OMN-26431)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_dev_merge_build_decision import (
    HandlerDevMergeBuildDecision,
)
from omnibase_infra.models.model_dev_merge_build_decision_request import (
    ModelDevMergeBuildDecisionRequest,
)

pytestmark = pytest.mark.integration

_CONTRACT = Path(__file__).resolve().parents[3] / (
    "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_SHA = "a" * 40
_NOW = datetime(2026, 11, 22, 2, 0, tzinfo=UTC)


def _handler(
    taken: list[tuple[str, str]], error: str = ""
) -> HandlerDevMergeBuildDecision:
    settings = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["settings"]

    def append(row: str, request_id: str) -> str:
        taken.append((row, request_id))
        return error

    return HandlerDevMergeBuildDecision(settings=settings, append=append)


def _request(**fields: object) -> ModelDevMergeBuildDecisionRequest:
    base: dict[str, object] = {
        "repository": "omnibase_infra",
        "trigger_sha": _SHA,
        "runtime_class": "PUBLISHED",
        "decided_at": _NOW,
    }
    return ModelDevMergeBuildDecisionRequest(**{**base, **fields})


def test_queued_decision_row_carries_sha_class_and_job() -> None:
    taken: list[tuple[str, str]] = []
    result = _handler(taken).handle(_request(result="queued", job="job-1"))
    assert result.outcome == "appended"
    row = taken[0][0]
    assert " | STATUS | " in row
    for cell in (
        "src=build-decision",
        f"trigger_sha={_SHA}",
        "class=PUBLISHED",
        "result=queued",
        "job=job-1",
    ):
        assert cell in row


def test_same_decision_has_the_same_request_id() -> None:
    taken: list[tuple[str, str]] = []
    handler = _handler(taken)
    handler.handle(_request(result="refused", reason="gate"))
    handler.handle(_request(result="refused", reason="gate"))
    assert taken[0][1] == taken[1][1]


def test_failed_append_is_reported_not_raised() -> None:
    result = _handler([], error="no route").handle(_request(result="queued", job="j"))
    assert result.outcome == "unappended"
    assert result.detail == "no route"


def test_non_runtime_row_has_no_result_cell() -> None:
    taken: list[tuple[str, str]] = []
    _handler(taken).handle(_request(runtime_class="NON_RUNTIME"))
    assert "result=" not in taken[0][0]
