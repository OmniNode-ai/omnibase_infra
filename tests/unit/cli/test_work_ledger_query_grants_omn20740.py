# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind the client declaration to the six existing dev query grants."""

import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_broker_grant_derive import HandlerBrokerGrantDerive
from omnibase_infra.handlers.handler_work_ledger_query import (
    load_work_ledger_query_client_config,
)
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant_bindings import (
    ModelBrokerGrantBindings,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[3]


def test_client_operations_are_exactly_the_derived_dev_grant() -> None:
    config = load_work_ledger_query_client_config()
    bindings = ModelBrokerGrantBindings.model_validate(
        yaml.safe_load(
            (ROOT / "deploy/lab/developer-principal-bindings.yaml").read_text()
        )
    )
    derived = HandlerBrokerGrantDerive().handle(bindings)
    expected = (
        {
            ("dev", "TOPIC", config.command_topic, "LITERAL", op)
            for op in ("WRITE", "DESCRIBE")
        }
        | {
            ("dev", "TOPIC", config.completed_topic, "LITERAL", op)
            for op in ("READ", "DESCRIBE")
        }
        | {
            ("dev", "GROUP", config.consumer_group_prefix, "PREFIXED", op)
            for op in ("READ", "DESCRIBE")
        }
    )
    resources = {row[2] for row in expected}
    actual = {
        (g.broker, g.resource_type, g.resource, g.pattern, g.operation)
        for g in derived.grants
        if g.resource in resources
    }
    assert actual == expected
    committed = json.loads(
        (ROOT / "deploy/lab/developer-principal-derived-grants.json").read_text()
    )
    assert {
        (g["broker"], g["resource_type"], g["resource"], g["pattern"], g["operation"])
        for g in committed["grants"]
        if g["resource"] in resources
    } == expected
    assert not any(
        g.resource == "*" and g.operation != "DESCRIBE" for g in derived.grants
    )


def test_unrelated_principal_gets_no_query_grants() -> None:
    bindings = ModelBrokerGrantBindings(principal="unrelated", bindings=())
    assert HandlerBrokerGrantDerive().handle(bindings).grants == ()
