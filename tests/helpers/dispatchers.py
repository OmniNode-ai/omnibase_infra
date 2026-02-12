# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared test dispatchers for integration tests.

Provides ``ContextCapturingDispatcher``, a test dispatcher that captures the
dispatch context and envelope it receives, enabling assertions on context
injection rules (time injection, correlation ID propagation, etc.).

Usage::

    from tests.helpers.dispatchers import ContextCapturingDispatcher

    dispatcher = ContextCapturingDispatcher(
        dispatcher_id="test-reducer",
        node_kind=EnumNodeKind.REDUCER,
        category=EnumMessageCategory.EVENT,
        message_types={"UserCreatedEvent"},
    )

    # After dispatch:
    assert dispatcher.captured_context is not None
    assert dispatcher.captured_context.now is None  # Reducers get no time
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult


class ContextCapturingDispatcher:
    """Test dispatcher that captures context and envelope for assertions.

    This dispatcher implements the ProtocolMessageDispatcher interface and
    stores the dispatch context it receives, allowing tests to verify context
    injection rules.

    Attributes:
        captured_context: The ModelDispatchContext received on last handle() call.
        captured_envelope: The envelope received on last handle() call.
        invocation_count: Number of times handle() has been called.
    """

    def __init__(
        self,
        dispatcher_id: str,
        node_kind: EnumNodeKind,
        category: EnumMessageCategory = EnumMessageCategory.EVENT,
        message_types: set[str] | None = None,
    ) -> None:
        self._dispatcher_id = dispatcher_id
        self._node_kind = node_kind
        self._category = category
        self._message_types = message_types or set()

        # Captured data for assertions
        self.captured_context: ModelDispatchContext | None = None
        self.captured_envelope: object | None = None
        self.invocation_count: int = 0

    @property
    def dispatcher_id(self) -> str:
        return self._dispatcher_id

    @property
    def category(self) -> EnumMessageCategory:
        return self._category

    @property
    def message_types(self) -> set[str]:
        return self._message_types

    @property
    def node_kind(self) -> EnumNodeKind:
        return self._node_kind

    async def handle(
        self,
        envelope: object,
        context: ModelDispatchContext | None = None,
        *,
        started_at: datetime | None = None,
    ) -> ModelDispatchResult:
        """Handle the message and capture the context for assertions.

        Args:
            envelope: The message envelope to dispatch.
            context: Optional dispatch context injected by the engine.
            started_at: Optional deterministic timestamp for the result.
                Defaults to ``datetime.now(UTC)`` when not provided, which
                suits most tests.  Pass an explicit value when the test
                requires a deterministic, reproducible timestamp.
        """
        self.captured_envelope = envelope
        self.captured_context = context
        self.invocation_count += 1

        return ModelDispatchResult(
            dispatch_id=uuid4(),
            status=EnumDispatchStatus.SUCCESS,
            topic="test.events.v1",
            dispatcher_id=self._dispatcher_id,
            message_type=type(envelope).__name__ if envelope else None,
            started_at=started_at if started_at is not None else datetime.now(UTC),
        )

    def reset(self) -> None:
        """Reset captured state for reuse between tests."""
        self.captured_context = None
        self.captured_envelope = None
        self.invocation_count = 0
