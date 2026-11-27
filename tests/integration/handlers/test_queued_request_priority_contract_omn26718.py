# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The on-disk node contract routes to the handler that runs an operator request (OMN-26718)."""

from __future__ import annotations

import importlib
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.models.model_queued_request_entry import ModelQueuedRequestEntry
from omnibase_infra.models.model_queued_request_priority_request import (
    ModelQueuedRequestPriorityRequest,
)

pytestmark = pytest.mark.integration

CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_queued_request_priority_compute/contract.yaml"
)


def _entry(requested_by: str, sha: str) -> ModelQueuedRequestEntry:
    return ModelQueuedRequestEntry(
        correlation_id=uuid4(),
        runtime_lane="dev",
        requested_by=requested_by,
        git_ref=sha,
        scope="runtime",
        build_source="workspace",
    )


def test_contract_routed_handler_admits_the_operator_request() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    spec = contract["handler_routing"]["handlers"][0]["handler"]
    handler_cls = getattr(importlib.import_module(spec["module"]), spec["name"])
    auto = _entry("agent/overdue-dev-merge-build", "a" * 40)
    operator = _entry("operator-manual", "b" * 40)
    decision = handler_cls().handle(
        ModelQueuedRequestPriorityRequest(
            entries=(auto, operator), supersede_queue=True
        )
    )
    assert decision.operator_path is True
    assert decision.runner_index == 1
    assert str(operator.correlation_id) in decision.reasons[0]
