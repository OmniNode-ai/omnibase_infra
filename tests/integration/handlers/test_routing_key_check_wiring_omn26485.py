# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The routing key check is wired through the real node contract (OMN-26485)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_routing_key_check import HandlerRoutingKeyCheck
from omnibase_infra.models.model_deploy_lane_routing_table import (
    ModelDeployLaneRoutingTable,
)
from omnibase_infra.models.model_routing_key_check_request import (
    ModelRoutingKeyCheckRequest,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_NODE = _ROOT / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
_OPERATIONS = {
    "check_routing_key_checks": "HandlerRoutingKeyCheck",
    "alarm_routed_commands_without_job_record": "HandlerRoutedCommandAlarm",
}


def test_node_contract_binds_the_routing_handlers_and_the_committed_table_passes() -> (
    None
):
    contract = yaml.safe_load(_NODE.read_text(encoding="utf-8"))
    bound = {
        entry["operation"]: entry["handler"]
        for entry in contract["handler_routing"]["handlers"]
        if entry["operation"] in _OPERATIONS
    }
    assert set(bound) == set(_OPERATIONS)
    for operation, name in _OPERATIONS.items():
        handler_class = getattr(
            importlib.import_module(bound[operation]["module"]), name
        )
        assert bound[operation]["name"] == name
        assert handler_class is not None

    table = ModelDeployLaneRoutingTable.model_validate(
        yaml.safe_load(
            (_ROOT / "config/deploy_lane_routing.yaml").read_text(encoding="utf-8")
        )
    )
    handler = HandlerRoutingKeyCheck()
    keys = dict.fromkeys(table.instances, "k")
    result = handler.handle(
        ModelRoutingKeyCheckRequest(
            table=table, runtime_key_check="k", instance_key_checks=keys
        )
    )
    assert result.outcome == "ok"
    keys[table.default_instance] = "other"
    assert (
        handler.handle(
            ModelRoutingKeyCheckRequest(
                table=table, runtime_key_check="k", instance_key_checks=keys
            )
        ).outcome
        == "mismatch"
    )
