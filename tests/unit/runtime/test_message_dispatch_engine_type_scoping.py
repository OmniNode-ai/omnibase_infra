# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Type-scoped dispatch routing tests for MessageDispatchEngine (OMN-12416).

A node contract with MULTIPLE handler entries (each with its own event_model)
must route a consumed message ONLY to the handler whose declared event_model
matches the payload type — not fan the message out to every sibling handler.

These tests exercise the ``payload_type_matcher`` selection path in
``_find_matching_dispatchers`` directly against the engine: two dispatchers
share one topic + message_type alias, and a payload of one type must reach only
its matching dispatcher. They also confirm that a successful handler's output is
returned even when a sibling handler errored on a non-matching payload (the
poisoning regression), and that untyped (no matcher) dispatchers keep legacy
behaviour.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_payload_type_matcher,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine


class _PayloadA:
    """First payload type for a shared-topic multi-handler contract."""


class _PayloadB:
    """Second payload type for a shared-topic multi-handler contract."""


_TOPIC = "onex.cmd.test-service.shared-command.v1"
_EVENT_TYPE_ALIAS = "test-service.shared-command"


def _make_envelope(payload: object) -> ModelEventEnvelope[object]:
    return ModelEventEnvelope(
        payload=payload,
        correlation_id=uuid4(),
        event_type=_EVENT_TYPE_ALIAS,
    )


def _register_shared_topic_route(
    engine: MessageDispatchEngine,
    dispatcher_id: str,
) -> None:
    """Register a wildcard route on the shared topic for a dispatcher."""
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"route.{dispatcher_id}",
            topic_pattern="*.cmd.test-service.shared-command.*",
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id=dispatcher_id,
        )
    )


@pytest.mark.unit
class TestTypeScopedRouting:
    """Multi-handler contract routes each message by event_model type."""

    @pytest.mark.asyncio
    async def test_only_matching_typed_dispatcher_is_invoked(self) -> None:
        """A _PayloadA message reaches only handler-A, never sibling handler-B.

        Reproduces OMN-12416: both handlers share one topic + message_type
        alias, so without type-scoping both would fire. With the matcher, only
        the dispatcher whose event_model matches _PayloadA runs.
        """
        engine = MessageDispatchEngine()
        invoked: list[str] = []

        async def handler_a(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("a")

        async def handler_b(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("b")

        engine.register_dispatcher(
            dispatcher_id="dispatcher-a",
            dispatcher=handler_a,
            category=EnumMessageCategory.COMMAND,
            message_types={"_PayloadA", _EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadA),
        )
        engine.register_dispatcher(
            dispatcher_id="dispatcher-b",
            dispatcher=handler_b,
            category=EnumMessageCategory.COMMAND,
            message_types={"_PayloadB", _EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadB),
        )
        _register_shared_topic_route(engine, "dispatcher-a")
        _register_shared_topic_route(engine, "dispatcher-b")
        engine.freeze()

        result = await engine.dispatch(_TOPIC, _make_envelope(_PayloadA()))

        assert invoked == ["a"]
        assert result.status == EnumDispatchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_sibling_is_not_invoked_for_b_payload(self) -> None:
        """The mirror case: a _PayloadB message reaches only handler-B."""
        engine = MessageDispatchEngine()
        invoked: list[str] = []

        async def handler_a(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("a")

        async def handler_b(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("b")

        engine.register_dispatcher(
            dispatcher_id="dispatcher-a",
            dispatcher=handler_a,
            category=EnumMessageCategory.COMMAND,
            message_types={"_PayloadA", _EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadA),
        )
        engine.register_dispatcher(
            dispatcher_id="dispatcher-b",
            dispatcher=handler_b,
            category=EnumMessageCategory.COMMAND,
            message_types={"_PayloadB", _EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadB),
        )
        _register_shared_topic_route(engine, "dispatcher-a")
        _register_shared_topic_route(engine, "dispatcher-b")
        engine.freeze()

        result = await engine.dispatch(_TOPIC, _make_envelope(_PayloadB()))

        assert invoked == ["b"]
        assert result.status == EnumDispatchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_dict_envelope_uses_payload_for_type_scoping(self) -> None:
        """A dict envelope must route by its payload, not by the envelope dict."""
        engine = MessageDispatchEngine()
        invoked: list[str] = []

        async def handler_a(envelope: object) -> None:
            invoked.append("a")

        async def handler_b(envelope: object) -> None:
            invoked.append("b")

        engine.register_dispatcher(
            dispatcher_id="dispatcher-a",
            dispatcher=handler_a,
            category=EnumMessageCategory.COMMAND,
            message_types={"_PayloadA", _EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadA),
        )
        engine.register_dispatcher(
            dispatcher_id="dispatcher-b",
            dispatcher=handler_b,
            category=EnumMessageCategory.COMMAND,
            message_types={"_PayloadB", _EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadB),
        )
        _register_shared_topic_route(engine, "dispatcher-a")
        _register_shared_topic_route(engine, "dispatcher-b")
        engine.freeze()

        result = await engine.dispatch(
            _TOPIC,
            {
                "payload": _PayloadA(),
                "event_type": _EVENT_TYPE_ALIAS,
                "correlation_id": str(uuid4()),
            },
        )

        assert invoked == ["a"]
        assert result.status == EnumDispatchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_failing_sibling_handler_never_receives_nonmatching_payload(
        self,
    ) -> None:
        """A handler that would raise on a non-matching payload is never called.

        This is the exact §20 defect: HandlerLlmDelegationCall raised
        ValidationError because it received a ModelInferenceIntent. With
        type-scoping, the non-matching handler is not selected, so it never
        raises and the dispatch is SUCCESS — no poisoning.
        """
        engine = MessageDispatchEngine()
        invoked: list[str] = []

        async def matching_handler(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("matching")

        async def strict_sibling(envelope: ModelEventEnvelope[object]) -> None:
            # Mirrors a handler whose own model validation would fail for a
            # foreign payload. It must never run for a _PayloadA message.
            invoked.append("sibling")
            raise ValueError("sibling received a payload it cannot handle")

        engine.register_dispatcher(
            dispatcher_id="matching",
            dispatcher=matching_handler,
            category=EnumMessageCategory.COMMAND,
            message_types={_EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadA),
        )
        engine.register_dispatcher(
            dispatcher_id="sibling",
            dispatcher=strict_sibling,
            category=EnumMessageCategory.COMMAND,
            message_types={_EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadB),
        )
        _register_shared_topic_route(engine, "matching")
        _register_shared_topic_route(engine, "sibling")
        engine.freeze()

        result = await engine.dispatch(_TOPIC, _make_envelope(_PayloadA()))

        assert invoked == ["matching"]
        assert result.status == EnumDispatchStatus.SUCCESS
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_untyped_dispatcher_keeps_legacy_string_matching(self) -> None:
        """A dispatcher without a payload_type_matcher matches by string only.

        Single-handler / operation-only contracts must be unaffected by the
        type-scoping change.
        """
        engine = MessageDispatchEngine()
        invoked: list[str] = []

        async def untyped_handler(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("untyped")

        engine.register_dispatcher(
            dispatcher_id="untyped",
            dispatcher=untyped_handler,
            category=EnumMessageCategory.COMMAND,
            message_types={_EVENT_TYPE_ALIAS},
        )
        _register_shared_topic_route(engine, "untyped")
        engine.freeze()

        # Any payload type matches because there is no event_model scoping.
        result = await engine.dispatch(_TOPIC, _make_envelope(_PayloadB()))

        assert invoked == ["untyped"]
        assert result.status == EnumDispatchStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_no_dispatcher_when_no_typed_handler_matches(self) -> None:
        """When every candidate is type-scoped and none match, no handler runs.

        A payload of a third type must not be force-routed to a typed handler
        that does not accept it.
        """
        engine = MessageDispatchEngine()
        invoked: list[str] = []

        async def handler_a(envelope: ModelEventEnvelope[object]) -> None:
            invoked.append("a")

        engine.register_dispatcher(
            dispatcher_id="dispatcher-a",
            dispatcher=handler_a,
            category=EnumMessageCategory.COMMAND,
            message_types={_EVENT_TYPE_ALIAS},
            payload_type_matcher=lambda p: isinstance(p, _PayloadA),
        )
        _register_shared_topic_route(engine, "dispatcher-a")
        engine.freeze()

        result = await engine.dispatch(_TOPIC, _make_envelope(_PayloadB()))

        assert invoked == []
        assert result.status == EnumDispatchStatus.NO_DISPATCHER


@pytest.mark.unit
class TestDlqFailureClassification:
    """OMN-14492: classify WHY no dispatcher matched, with real ValidationError detail.

    Before this fix, both a genuinely-unregistered dispatcher and a payload
    that failed its type-scoped ``event_model.model_validate`` collapsed into
    the same ``EnumDispatchStatus.NO_DISPATCHER`` with a generic
    "No dispatcher registered" message — no ``ValidationError`` detail
    anywhere. These tests drive the REAL production matcher
    (``_make_payload_type_matcher``, not a synthetic test lambda) against a
    REAL core model (``ModelDelegationRequest``) so the assertion is against
    an actual malformed-payload dispatch, not a stub.
    """

    @pytest.mark.asyncio
    async def test_publisher_malformed_carries_real_validation_detail(self) -> None:
        """A registered type-scoped dispatcher rejecting a malformed payload
        must classify as publisher_malformed with the real pydantic detail —
        not the generic, unclassifiable "No dispatcher found" message."""
        engine = MessageDispatchEngine()

        async def handler(envelope: ModelEventEnvelope[object]) -> None:
            raise AssertionError("a malformed payload must never reach the handler")

        event_model_ref = ModelHandlerRef(
            name="ModelDelegationRequest",
            module="omnibase_core.models.delegation.wire",
        )
        engine.register_dispatcher(
            dispatcher_id="delegation-dispatcher",
            dispatcher=handler,
            category=EnumMessageCategory.COMMAND,
            message_types={_EVENT_TYPE_ALIAS},
            payload_type_matcher=_make_payload_type_matcher(event_model_ref),
        )
        _register_shared_topic_route(engine, "delegation-dispatcher")
        engine.freeze()

        # Malformed: missing every required field of ModelDelegationRequest
        # (prompt, task_type, correlation_id, emitted_at) and carries an
        # extra_forbidden field — mirrors the live OMN-14484 harness defect.
        malformed_payload = {"not_a_real_field": "x"}
        result = await engine.dispatch(_TOPIC, _make_envelope(malformed_payload))

        assert result.status == EnumDispatchStatus.NO_DISPATCHER
        assert result.error_code == EnumCoreErrorCode.ENVELOPE_VALIDATION_FAILED
        assert result.error_details.get("failure_class") == "publisher_malformed"
        validation_detail = result.error_details.get("validation_detail")
        assert isinstance(validation_detail, str)
        # Real pydantic field errors, not a generic placeholder.
        assert "prompt" in validation_detail
        assert "task_type" in validation_detail
        assert validation_detail != "validation detail unavailable"
        assert "publisher-malformed" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_no_dispatcher_when_zero_candidates_ever_matched(self) -> None:
        """A topic with NO registered route at all classifies as no_dispatcher
        (true wiring gap) — never publisher_malformed, and carries no
        validation_detail since no type-scoped candidate was ever rejected."""
        engine = MessageDispatchEngine()
        engine.freeze()

        result = await engine.dispatch(
            "onex.cmd.test-service.nobody-subscribed.v1",
            _make_envelope(_PayloadA()),
        )

        assert result.status == EnumDispatchStatus.NO_DISPATCHER
        assert result.error_code == EnumCoreErrorCode.ITEM_NOT_REGISTERED
        assert result.error_details.get("failure_class") == "no_dispatcher"
        assert "validation_detail" not in result.error_details
