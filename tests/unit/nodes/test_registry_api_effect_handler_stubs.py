# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for node_registry_api_effect stub handlers.

Verifies that all 10 handler modules declared in contract.yaml are importable,
have the correct handler_type and handler_category properties, and raise
NotImplementedError when handle() is called.

Ticket: OMN-2909
"""

from __future__ import annotations

import importlib
import uuid

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

HANDLER_CASES = [
    (
        "handler_registry_api_list_nodes",
        "HandlerRegistryApiListNodes",
        "list_nodes",
    ),
    (
        "handler_registry_api_get_node",
        "HandlerRegistryApiGetNode",
        "get_node",
    ),
    (
        "handler_registry_api_list_instances",
        "HandlerRegistryApiListInstances",
        "list_instances",
    ),
    (
        "handler_registry_api_get_widget_mapping",
        "HandlerRegistryApiGetWidgetMapping",
        "get_widget_mapping",
    ),
    (
        "handler_registry_api_get_discovery",
        "HandlerRegistryApiGetDiscovery",
        "get_discovery",
    ),
    (
        "handler_registry_api_get_health",
        "HandlerRegistryApiGetHealth",
        "get_health",
    ),
    (
        "handler_registry_api_list_contracts",
        "HandlerRegistryApiListContracts",
        "list_contracts",
    ),
    (
        "handler_registry_api_get_contract",
        "HandlerRegistryApiGetContract",
        "get_contract",
    ),
    (
        "handler_registry_api_list_topics",
        "HandlerRegistryApiListTopics",
        "list_topics",
    ),
    (
        "handler_registry_api_get_topic",
        "HandlerRegistryApiGetTopic",
        "get_topic",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("module_suffix", "class_name", "operation"), HANDLER_CASES)
def test_handler_module_is_importable(
    module_suffix: str, class_name: str, operation: str
) -> None:
    """Each handler module must be importable and contain the declared class."""
    module_path = (
        f"omnibase_infra.nodes.node_registry_api_effect.handlers.{module_suffix}"
    )
    module = importlib.import_module(module_path)
    assert hasattr(module, class_name), (
        f"Module {module_path} does not export {class_name}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("module_suffix", "class_name", "operation"), HANDLER_CASES)
def test_handler_has_correct_type_properties(
    module_suffix: str, class_name: str, operation: str
) -> None:
    """Each handler must expose handler_type and handler_category properties."""
    module_path = (
        f"omnibase_infra.nodes.node_registry_api_effect.handlers.{module_suffix}"
    )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()

    assert hasattr(instance, "handler_type"), (
        f"{class_name} missing handler_type property"
    )
    assert hasattr(instance, "handler_category"), (
        f"{class_name} missing handler_category property"
    )
    assert instance.handler_type == EnumHandlerType.INFRA_HANDLER, (
        f"{class_name}.handler_type must be INFRA_HANDLER"
    )
    assert instance.handler_category == EnumHandlerTypeCategory.EFFECT, (
        f"{class_name}.handler_category must be EFFECT"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("module_suffix", "class_name", "operation"), HANDLER_CASES)
@pytest.mark.asyncio
async def test_handler_raises_not_implemented(
    module_suffix: str, class_name: str, operation: str
) -> None:
    """Each stub handler must raise NotImplementedError when handle() is called."""
    module_path = (
        f"omnibase_infra.nodes.node_registry_api_effect.handlers.{module_suffix}"
    )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()
    correlation_id = uuid.uuid4()

    with pytest.raises(NotImplementedError, match=class_name):
        await instance.handle(request=object(), correlation_id=correlation_id)


@pytest.mark.unit
def test_handlers_init_exports_all_classes() -> None:
    """The handlers __init__.py must export all 10 handler classes."""
    import omnibase_infra.nodes.node_registry_api_effect.handlers as handlers_pkg

    expected_classes = {class_name for _, class_name, _ in HANDLER_CASES}
    exported = set(handlers_pkg.__all__)
    missing = expected_classes - exported
    assert not missing, f"handlers/__init__.py missing exports: {missing}"
