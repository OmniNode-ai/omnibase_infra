# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The registration orchestrator's handlers take a typed request (OMN-21536).

These tests state the shape of the flip and import nothing the flip introduced,
so they run, and fail on their own assertion, against the pre-flip tree.
"""

from __future__ import annotations

import importlib
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


_OTHER_HANDLERS = [
    ("handler_catalog_request", "HandlerCatalogRequest"),
    ("handler_node_heartbeat", "HandlerNodeHeartbeat"),
    ("handler_node_registration_acked", "HandlerNodeRegistrationAcked"),
    ("handler_runtime_tick", "HandlerRuntimeTick"),
    ("handler_topic_catalog_query", "HandlerTopicCatalogQuery"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("module_name", "class_name"), _OTHER_HANDLERS)
def test_other_handlers_take_a_request_not_an_envelope(
    module_name: str, class_name: str
) -> None:
    module = importlib.import_module(
        f"omnibase_infra.nodes.node_registration_orchestrator.handlers.{module_name}"
    )
    parameters = list(inspect.signature(getattr(module, class_name).handle).parameters)
    assert parameters == ["self", "request"]


@pytest.mark.unit
@pytest.mark.parametrize(("module_name", "class_name"), _OTHER_HANDLERS)
def test_other_handler_cores_do_not_reference_the_event_envelope(
    module_name: str, class_name: str
) -> None:
    module = importlib.import_module(
        f"omnibase_infra.nodes.node_registration_orchestrator.handlers.{module_name}"
    )
    mentions = inspect.getsource(module).count("ModelEventEnvelope")
    assert mentions == 0, f"{class_name} still references ModelEventEnvelope"
