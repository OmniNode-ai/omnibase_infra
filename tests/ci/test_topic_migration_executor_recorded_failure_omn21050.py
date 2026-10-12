# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay of the lab dev-lane refusal of a topic-migration execute command (OMN-21050).

The recorded lifecycle event is what the runtime emitted for a real
topic-migration-execute command on the lab dev lane: the command was terminalized
because the executor handler exposed no ``handle()`` / ``handle_async()``
dispatch entrypoint. The replay asserts the recorded refusal is the one this
change removes, using the same resolver the runtime uses.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from omnibase_infra.nodes.node_topic_migration_executor_effect.handlers.handler_topic_migration_executor import (
    HandlerTopicMigrationExecutor,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_effective_handle_method,
)

pytestmark = pytest.mark.unit


@pytest.mark.live_contact(
    "tests/ci/fixtures/topic_migration_boundary_failure_omn21050.json"
)
def test_recorded_dispatch_refusal_no_longer_applies(
    recorded_response: dict[str, Any],
) -> None:
    event = recorded_response["recorded_lifecycle_event"]
    payload = event["payload"]
    assert payload["status"] == "failed"
    assert (
        payload["origin_topic"] == "onex.cmd.omnibase-infra.topic-migration-execute.v1"
    )
    assert (
        "HandlerTopicMigrationExecutor does not expose a callable handle() or "
        "handle_async() dispatch entrypoint"
    ) in payload["failure_reason"]
    UUID(payload["correlation_id"])

    handler = HandlerTopicMigrationExecutor.__new__(HandlerTopicMigrationExecutor)
    assert _resolve_effective_handle_method(handler) is not None, (
        "the recorded refusal still applies: no handle()/handle_async() entrypoint"
    )
