# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler entry key guards for auto-wiring runtime stability."""

from __future__ import annotations

import hashlib

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_dispatcher_id,
    _derive_handler_entry_key,
    _derive_route_id,
    _required_handler_init_params,
    _should_skip_sync_container_resolution,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelHandlerRef,
    ModelHandlerRoutingEntry,
)


def _entry(operation: str | None = None) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name="HandlerShared", module="fake.handlers"),
        operation=operation,
    )


class TestHandlerEntryKeyDerivation:
    @pytest.mark.unit
    def test_key_without_operation_preserves_handler_name(self) -> None:
        assert _derive_handler_entry_key(_entry()) == "HandlerShared"

    @pytest.mark.unit
    def test_key_with_operation_includes_sanitized_operation_and_digest(self) -> None:
        operation = "inference.gemini-cli"
        digest = hashlib.sha1(operation.encode()).hexdigest()[:8]

        assert (
            _derive_handler_entry_key(_entry(operation))
            == f"HandlerShared.inference_gemini_cli_{digest}"
        )

    @pytest.mark.unit
    def test_key_with_symbol_only_operation_falls_back_to_digest(self) -> None:
        operation = "!!!"
        digest = hashlib.sha1(operation.encode()).hexdigest()[:8]

        assert _derive_handler_entry_key(_entry(operation)) == f"HandlerShared.{digest}"

    @pytest.mark.unit
    def test_dispatcher_id_uses_handler_entry_key(self) -> None:
        handler_key = _derive_handler_entry_key(_entry("inference.opencode-cli"))

        assert (
            _derive_dispatcher_id("node_model_router", handler_key)
            == f"dispatcher.auto.node_model_router.{handler_key}"
        )

    @pytest.mark.unit
    def test_dispatcher_ids_are_distinct_for_same_handler_different_operations(
        self,
    ) -> None:
        gemini_key = _derive_handler_entry_key(_entry("inference.gemini-cli"))
        opencode_key = _derive_handler_entry_key(_entry("inference.opencode-cli"))

        assert _derive_dispatcher_id("node_model_router", gemini_key) != (
            _derive_dispatcher_id("node_model_router", opencode_key)
        )

    @pytest.mark.unit
    def test_route_ids_are_distinct_for_same_handler_same_topic_different_operations(
        self,
    ) -> None:
        topic = "onex.cmd.omnimarket.model-route.v1"
        gemini_key = _derive_handler_entry_key(_entry("inference.gemini-cli"))
        opencode_key = _derive_handler_entry_key(_entry("inference.opencode-cli"))

        assert _derive_route_id("node_model_router", gemini_key, topic) != (
            _derive_route_id("node_model_router", opencode_key, topic)
        )


class TestRequiredHandlerInitParams:
    @pytest.mark.unit
    def test_required_params_empty_for_zero_arg_handler(self) -> None:
        class HandlerZeroArg:
            pass

        assert _required_handler_init_params(HandlerZeroArg) == frozenset()

    @pytest.mark.unit
    def test_required_params_include_required_positional_dependency(self) -> None:
        class HandlerWithDependency:
            def __init__(self, service: object) -> None:
                self.service = service

        assert _required_handler_init_params(HandlerWithDependency) == frozenset(
            {"service"}
        )

    @pytest.mark.unit
    def test_required_params_include_required_keyword_only_dependency(self) -> None:
        class HandlerWithKeywordDependency:
            def __init__(self, *, service: object) -> None:
                self.service = service

        assert _required_handler_init_params(HandlerWithKeywordDependency) == (
            frozenset({"service"})
        )

    @pytest.mark.unit
    def test_required_params_ignore_optional_dependencies(self) -> None:
        class HandlerWithOptionalDependency:
            def __init__(self, service: object | None = None) -> None:
                self.service = service

        assert _required_handler_init_params(HandlerWithOptionalDependency) == (
            frozenset()
        )


class TestSyncContainerResolutionSkip:
    @pytest.mark.unit
    def test_skip_sync_container_resolution_for_zero_arg_handler(self) -> None:
        class HandlerZeroArg:
            pass

        assert _should_skip_sync_container_resolution(HandlerZeroArg) is True

    @pytest.mark.unit
    def test_skip_sync_container_resolution_for_event_bus_only_handler(self) -> None:
        class HandlerEventBusOnly:
            def __init__(self, event_bus: object) -> None:
                self.event_bus = event_bus

        assert _should_skip_sync_container_resolution(HandlerEventBusOnly) is True

    @pytest.mark.unit
    def test_do_not_skip_sync_container_resolution_for_complex_handler(self) -> None:
        class HandlerWithDependency:
            def __init__(self, service: object) -> None:
                self.service = service

        assert _should_skip_sync_container_resolution(HandlerWithDependency) is False
