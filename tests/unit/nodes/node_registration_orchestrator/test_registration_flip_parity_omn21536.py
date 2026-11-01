# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The registration orchestrator's handlers take a typed request (OMN-21536).

These tests state the shape of the flip and import nothing the flip introduced,
so they run, and fail on their own assertion, against the pre-flip tree.
"""

from __future__ import annotations

import inspect

import pytest

from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
    handler_node_introspected,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)


@pytest.mark.unit
def test_introspected_handle_takes_a_request_not_an_envelope() -> None:
    parameters = list(inspect.signature(HandlerNodeIntrospected.handle).parameters)
    assert parameters == ["self", "request"]


@pytest.mark.unit
def test_introspected_core_does_not_reference_the_event_envelope() -> None:
    mentions = inspect.getsource(handler_node_introspected).count("ModelEventEnvelope")
    assert mentions == 0, "the handler core still references ModelEventEnvelope"
