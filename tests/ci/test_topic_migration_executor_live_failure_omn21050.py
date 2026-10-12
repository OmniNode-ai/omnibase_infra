# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21050: the failure the lab dev lane recorded no longer has its cause.

The artifact is the failure terminal the dev lane runtime wrote to the lifecycle
topic for one DUAL_WRITE execute command: the executor handler exposed no
``handle()`` entrypoint, so the command never became a lifecycle event.
"""

from __future__ import annotations

import inspect
from typing import cast

import pytest

from omnibase_infra.nodes.node_topic_migration_executor_effect.handlers.handler_topic_migration_executor import (
    HandlerTopicMigrationExecutor,
)

_CONTRACT_COMMAND_TOPIC = "onex.cmd.omnibase-infra.topic-migration-execute.v1"


@pytest.mark.live_contact(
    "tests/ci/fixtures/topic_migration_executor_failure_terminal_omn21050.json"
)
def test_recorded_missing_entrypoint_failure_is_closed_by_a_handle_method(
    recorded_response: dict[str, object],
) -> None:
    terminal = cast("dict[str, object]", recorded_response["value"])
    payload = cast("dict[str, object]", terminal["payload"])
    assert payload["status"] == "failed"
    assert payload["origin_topic"] == _CONTRACT_COMMAND_TOPIC
    reason = cast("str", payload["failure_reason"])
    assert "does not expose a callable handle() or handle_async()" in reason
    assert HandlerTopicMigrationExecutor.__name__ in reason

    # The named cause is gone: the runtime's entrypoint exists and is awaitable.
    handle = getattr(HandlerTopicMigrationExecutor, "handle", None)
    assert callable(handle)
    assert inspect.iscoroutinefunction(handle)
