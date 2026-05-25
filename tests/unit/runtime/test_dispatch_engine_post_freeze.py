# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for MessageDispatchEngine post-freeze dynamic registration (OMN-11246)."""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PreparedWiring,
    _commit_handler_wiring,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
)


@pytest.mark.unit
class TestPostFreezeRegistration:
    """Tests for _register_dispatcher_dynamic and _register_route_dynamic."""

    def test__register_dispatcher_dynamic_succeeds_after_freeze(self) -> None:
        engine = MessageDispatchEngine()
        engine.freeze()
        assert engine.is_frozen

        async def handler(envelope: object) -> None:
            return None

        # Standard registration must still fail after freeze
        with pytest.raises(ModelOnexError) as exc_info:
            engine.register_dispatcher(
                dispatcher_id="standard-dispatcher",
                dispatcher=handler,
                category=EnumMessageCategory.EVENT,
            )
        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE

        # Dynamic registration succeeds after freeze
        engine._register_dispatcher_dynamic(
            dispatcher_id="dynamic-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        assert "dynamic-dispatcher" in engine._dispatchers

    def test__register_route_dynamic_succeeds_after_freeze(self) -> None:
        engine = MessageDispatchEngine()

        async def handler(envelope: object) -> None:
            return None

        engine.register_dispatcher(
            dispatcher_id="pre-freeze-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        engine.freeze()

        # Register dynamic dispatcher first (needed as route target)
        engine._register_dispatcher_dynamic(
            dispatcher_id="dynamic-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )

        route = ModelDispatchRoute(
            route_id="dynamic-route",
            topic_pattern="*.evt.test.dynamic-event.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="dynamic-dispatcher",
        )
        engine._register_route_dynamic(route)
        assert "dynamic-route" in engine._routes

    def test__register_dispatcher_dynamic_rejects_duplicates(self) -> None:
        engine = MessageDispatchEngine()
        engine.freeze()

        async def handler(envelope: object) -> None:
            return None

        engine._register_dispatcher_dynamic(
            dispatcher_id="dup-test",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )

        with pytest.raises(ModelOnexError) as exc_info:
            engine._register_dispatcher_dynamic(
                dispatcher_id="dup-test",
                dispatcher=handler,
                category=EnumMessageCategory.EVENT,
            )
        assert exc_info.value.error_code == EnumCoreErrorCode.DUPLICATE_REGISTRATION

    def test__register_route_dynamic_rejects_duplicates(self) -> None:
        engine = MessageDispatchEngine()
        engine.freeze()

        async def handler(envelope: object) -> None:
            return None

        engine._register_dispatcher_dynamic(
            dispatcher_id="dup-route-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )

        route = ModelDispatchRoute(
            route_id="dup-route",
            topic_pattern="*.evt.test.dup.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="dup-route-dispatcher",
        )
        engine._register_route_dynamic(route)

        with pytest.raises(ModelOnexError) as exc_info:
            engine._register_route_dynamic(route)
        assert exc_info.value.error_code == EnumCoreErrorCode.DUPLICATE_REGISTRATION

    def test__register_route_dynamic_rejects_unregistered_dispatcher(self) -> None:
        engine = MessageDispatchEngine()
        engine.freeze()

        route = ModelDispatchRoute(
            route_id="orphan-route",
            topic_pattern="*.evt.test.orphan.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="nonexistent-dispatcher",
        )

        with pytest.raises(ModelOnexError) as exc_info:
            engine._register_route_dynamic(route)
        assert exc_info.value.error_code == EnumCoreErrorCode.ITEM_NOT_REGISTERED

    def test_standard_register_dispatcher_still_raises_after_freeze(self) -> None:
        engine = MessageDispatchEngine()
        engine.freeze()

        async def handler(envelope: object) -> None:
            return None

        with pytest.raises(ModelOnexError) as exc_info:
            engine.register_dispatcher(
                dispatcher_id="blocked",
                dispatcher=handler,
                category=EnumMessageCategory.EVENT,
            )
        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE

    def test_standard_register_route_still_raises_after_freeze(self) -> None:
        engine = MessageDispatchEngine()

        async def handler(envelope: object) -> None:
            return None

        engine.register_dispatcher(
            dispatcher_id="pre",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
        )
        engine.freeze()

        route = ModelDispatchRoute(
            route_id="blocked-route",
            topic_pattern="*.evt.test.blocked.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="pre",
        )
        with pytest.raises(ModelOnexError) as exc_info:
            engine.register_route(route)
        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE

    @pytest.mark.asyncio
    async def test_dispatch_routes_to_dynamic_handler(self) -> None:
        engine = MessageDispatchEngine()
        engine.freeze()

        handler_called = False

        async def handler(envelope: object) -> None:
            nonlocal handler_called
            handler_called = True

        engine._register_dispatcher_dynamic(
            dispatcher_id="live-handler",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types={"platform.dynamic-test"},
        )
        route = ModelDispatchRoute(
            route_id="live-route",
            topic_pattern="onex.evt.platform.dynamic-test.v1",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="live-handler",
        )
        engine._register_route_dynamic(route)

        envelope = ModelEventEnvelope(
            payload={"test": True},
            event_type="platform.dynamic-test",
        )
        await engine.dispatch("onex.evt.platform.dynamic-test.v1", envelope)
        assert handler_called is True

    def test__commit_handler_wiring_raises_on_frozen_without_auth(self) -> None:
        """Post-freeze _commit_handler_wiring without auth raises ModelOnexError."""
        engine = MessageDispatchEngine()
        engine.freeze()

        async def handler(envelope: object) -> None:
            return None

        route = ModelDispatchRoute(
            route_id="test-route",
            topic_pattern="*.evt.test.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="test-dispatcher",
        )

        prepared = PreparedWiring(
            dispatcher_id="test-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            routes=[route],
            route_ids=["test-route"],
            resolution_outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
        )

        with pytest.raises(ModelOnexError) as exc_info:
            _commit_handler_wiring(prepared, engine)
        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE

    def test__commit_handler_wiring_succeeds_frozen_with_auth(self) -> None:
        """Post-freeze _commit_handler_wiring with auth uses dynamic methods."""
        engine = MessageDispatchEngine()
        engine.freeze()

        async def handler(envelope: object) -> None:
            return None

        route = ModelDispatchRoute(
            route_id="auth-route",
            topic_pattern="*.evt.test.authorized.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="auth-dispatcher",
        )

        prepared = PreparedWiring(
            dispatcher_id="auth-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.EVENT,
            message_types=None,
            routes=[route],
            route_ids=["auth-route"],
            resolution_outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
        )

        dispatcher_id, route_ids = _commit_handler_wiring(
            prepared, engine, dynamic_materialization_authorized=True
        )
        assert dispatcher_id == "auth-dispatcher"
        assert "auth-route" in route_ids
        assert "auth-dispatcher" in engine._dispatchers
        assert "auth-route" in engine._routes
