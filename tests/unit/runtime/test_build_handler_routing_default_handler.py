# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for KafkaContractSource._build_handler_routing default_handler shorthand (OMN-11884).

The ``default_handler`` key allows a contract to declare a single handler using
a ``module_ref:ClassName`` shorthand instead of a full ``handlers:`` list.

Covered cases:
- default_handler shorthand (``handler:ClassName``) with no handlers list
- default_handler with fully-qualified module path
- handlers list present — handlers list takes priority, default_handler ignored
- both absent — returns ModelHandlerRouting with empty handlers tuple
- handler_routing absent — returns None
- default_handler with no colon — ignored (not a valid shorthand)
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

pytestmark = pytest.mark.unit


class TestBuildHandlerRoutingDefaultHandler:
    """Tests for _build_handler_routing default_handler shorthand (OMN-11884)."""

    def test_default_handler_bare_module_colon_class(self) -> None:
        """default_handler: handler:ClassName is parsed into a single-entry handlers tuple."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "default_handler": "handler:NodeFooCompute",
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        assert len(result.handlers) == 1
        entry = result.handlers[0]
        assert entry.handler.name == "NodeFooCompute"
        assert entry.handler.module == "handler"
        assert entry.event_model is None
        assert entry.operation is None

    def test_default_handler_fully_qualified_module(self) -> None:
        """default_handler: full.module.path:ClassName is parsed correctly."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "default_handler": (
                    "omnimarket.nodes.node_foo.handlers.handler_foo:HandlerFoo"
                ),
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        assert len(result.handlers) == 1
        entry = result.handlers[0]
        assert entry.handler.name == "HandlerFoo"
        assert entry.handler.module == (
            "omnimarket.nodes.node_foo.handlers.handler_foo"
        )

    def test_handlers_list_takes_priority_over_default_handler(self) -> None:
        """When handlers list is present, default_handler is ignored."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "operation_match",
                "default_handler": "handler:ShouldBeIgnored",
                "handlers": [
                    {
                        "operation": "do_something",
                        "handler": {
                            "name": "HandlerReal",
                            "module": "some.module",
                        },
                    }
                ],
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        assert len(result.handlers) == 1
        assert result.handlers[0].handler.name == "HandlerReal"
        assert result.handlers[0].handler.module == "some.module"
        assert result.routing_strategy == "operation_match"

    def test_neither_handlers_nor_default_handler_returns_empty(self) -> None:
        """When neither handlers nor default_handler is present, returns empty handlers."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "payload_type_match",
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        assert result.handlers == ()

    def test_handler_routing_absent_returns_none(self) -> None:
        """When handler_routing key is missing entirely, returns None."""
        config: dict[str, object] = {"event_bus": {"subscribe_topics": []}}

        result = KafkaContractSource._build_handler_routing(config)

        assert result is None

    def test_default_handler_without_colon_is_ignored(self) -> None:
        """default_handler without a colon separator is not a valid shorthand; ignored."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "default_handler": "postgresql",
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        # No colon means it cannot be parsed as module:ClassName — skip silently
        assert result.handlers == ()

    def test_default_handler_null_is_ignored(self) -> None:
        """default_handler: null (None) is treated as absent."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "default_handler": None,
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        assert result.handlers == ()

    def test_routing_strategy_preserved_with_default_handler(self) -> None:
        """routing_strategy from YAML is forwarded even when using default_handler."""
        config: dict[str, object] = {
            "handler_routing": {
                "routing_strategy": "operation_match",
                "default_handler": "handler:HandlerFoo",
            }
        }

        result = KafkaContractSource._build_handler_routing(config)

        assert result is not None
        assert result.routing_strategy == "operation_match"
