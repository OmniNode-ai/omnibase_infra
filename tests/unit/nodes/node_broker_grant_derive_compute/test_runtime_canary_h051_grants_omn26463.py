# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26463: the declared grant of the lab-only principal runtime-canary-h051.

OPERATOR-CONSENT 2026-11-27T14:16:02Z (rolling ledger line 4024): read of the dev
candidate topics the dev deploy agent publishes and consumes for a build, publish of
the runtime canary receipt topic, nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_broker_grant_derive.handler import (
    HandlerBrokerGrantDerive,
)
from omnibase_infra.models.security.model_broker_grant_bindings import (
    ModelBrokerGrantBindings,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[4]
BINDINGS = ROOT / "deploy/lab/runtime-canary-h051-bindings.yaml"
DERIVED = ROOT / "deploy/lab/runtime-canary-h051-derived-grants.json"
EVENTS = ROOT / "scripts/deploy-agent/deploy_agent/events.py"
REBUILD_TOPICS = {
    "onex.cmd.deploy.rebuild-requested.v1",
    "onex.evt.deploy.rebuild-completed.v1",
}
RECEIPT_TOPIC = "onex.evt.omnibase-infra.runtime-canary-receipt.v1"


def _bindings() -> ModelBrokerGrantBindings:
    return ModelBrokerGrantBindings.model_validate(yaml.safe_load(BINDINGS.read_text()))


def test_derived_file_is_exactly_the_nodes_output() -> None:
    bindings = _bindings()
    assert bindings.principal == "runtime-canary-h051"
    assert (
        DERIVED.read_text()
        == HandlerBrokerGrantDerive().handle(bindings).model_dump_json(indent=2) + "\n"
    )


def test_grants_are_the_agent_candidate_topics_read_and_the_receipt_write_only() -> (
    None
):
    derived = HandlerBrokerGrantDerive().handle(_bindings())
    topic_ops: dict[str, set[str]] = {}
    for grant in derived.grants:
        assert grant.broker == "dev"
        if grant.resource_type == "TOPIC":
            topic_ops.setdefault(grant.resource, set()).add(grant.operation)
    assert topic_ops == {
        **{topic: {"READ", "DESCRIBE"} for topic in REBUILD_TOPICS},
        RECEIPT_TOPIC: {"WRITE", "DESCRIBE"},
    }


def test_the_read_topics_are_the_ones_the_deploy_agent_declares() -> None:
    source = EVENTS.read_text()
    for topic in REBUILD_TOPICS:
        assert f'"{topic}"' in source
