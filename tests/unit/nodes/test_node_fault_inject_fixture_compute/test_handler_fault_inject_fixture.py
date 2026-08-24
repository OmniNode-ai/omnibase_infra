# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerFaultInjectFixture.

Ticket: OMN-16265

Covers the mechanism this fixture exists to prove deterministically: a
command's inflate_result_bytes field controls the exact serialized size of
the returned result, reproducing the OMN-14498 live-probe amplification
technique (small input, handler-amplified output) without needing a live
broker for the unit-level guarantee.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_fault_inject_fixture_compute.handlers.handler_fault_inject_fixture import (
    HandlerFaultInjectFixture,
)
from omnibase_infra.nodes.node_fault_inject_fixture_compute.models.model_fault_inject_fixture_command import (
    MAX_INFLATE_RESULT_BYTES,
    ModelFaultInjectFixtureCommand,
)


@pytest.fixture
def handler() -> HandlerFaultInjectFixture:
    return HandlerFaultInjectFixture()


class TestHandlerFaultInjectFixtureClassification:
    def test_handler_type_is_compute(self, handler: HandlerFaultInjectFixture) -> None:
        assert handler.handler_type == EnumHandlerType.COMPUTE_HANDLER

    def test_handler_category_is_compute(
        self, handler: HandlerFaultInjectFixture
    ) -> None:
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE


class TestHandlerFaultInjectFixtureHandle:
    async def test_zero_inflate_produces_empty_padding(
        self, handler: HandlerFaultInjectFixture
    ) -> None:
        corr_id = uuid4()
        request = ModelFaultInjectFixtureCommand(
            correlation_id=corr_id, inflate_result_bytes=0
        )

        result = await handler.handle(request)

        assert result.correlation_id == corr_id
        assert result.inflate_result_bytes == 0
        assert result.padding == ""
        assert result.padding_byte_length == 0

    @pytest.mark.parametrize(
        "size",
        [1, 17, 1_000, 100_000, 1_048_600],
    )
    async def test_padding_byte_length_matches_requested_size_exactly(
        self, handler: HandlerFaultInjectFixture, size: int
    ) -> None:
        """The core proof-of-mechanism: requested size == actual serialized size.

        1_048_600 is deliberately just past the live message.max.bytes
        (~1,048,588) measured on onex-dev by the OMN-14498 probe — the exact
        class of value a deployed run would use to trigger the primary
        publish failure.
        """
        request = ModelFaultInjectFixtureCommand(
            correlation_id=uuid4(), inflate_result_bytes=size
        )

        result = await handler.handle(request)

        assert len(result.padding) == size
        assert result.padding_byte_length == size
        # Single-byte ASCII filler: char count and UTF-8 byte count agree,
        # so a caller can size-target without an encoding-width surprise.
        assert len(result.padding.encode("utf-8")) == result.padding_byte_length

    async def test_correlation_id_is_propagated_not_regenerated(
        self, handler: HandlerFaultInjectFixture
    ) -> None:
        corr_id = uuid4()
        request = ModelFaultInjectFixtureCommand(
            correlation_id=corr_id, inflate_result_bytes=5
        )

        result = await handler.handle(request)

        assert result.correlation_id == corr_id

    async def test_marker_field_does_not_affect_result(
        self, handler: HandlerFaultInjectFixture
    ) -> None:
        corr_id = uuid4()
        request = ModelFaultInjectFixtureCommand(
            correlation_id=corr_id,
            inflate_result_bytes=10,
            marker="OMN-16265-manual-run",
        )

        result = await handler.handle(request)

        assert result.padding_byte_length == 10

    async def test_handle_is_deterministic_across_calls(
        self, handler: HandlerFaultInjectFixture
    ) -> None:
        corr_id = uuid4()
        request = ModelFaultInjectFixtureCommand(
            correlation_id=corr_id, inflate_result_bytes=256
        )

        first = await handler.handle(request)
        second = await handler.handle(request)

        assert first.padding == second.padding
        assert first.padding_byte_length == second.padding_byte_length


class TestModelFaultInjectFixtureCommandValidation:
    def test_negative_inflate_result_bytes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelFaultInjectFixtureCommand(
                correlation_id=uuid4(), inflate_result_bytes=-1
            )

    def test_inflate_result_bytes_above_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelFaultInjectFixtureCommand(
                correlation_id=uuid4(),
                inflate_result_bytes=MAX_INFLATE_RESULT_BYTES + 1,
            )

    def test_inflate_result_bytes_at_cap_accepted(self) -> None:
        command = ModelFaultInjectFixtureCommand(
            correlation_id=uuid4(),
            inflate_result_bytes=MAX_INFLATE_RESULT_BYTES,
        )
        assert command.inflate_result_bytes == MAX_INFLATE_RESULT_BYTES

    def test_default_inflate_result_bytes_is_zero(self) -> None:
        command = ModelFaultInjectFixtureCommand(correlation_id=uuid4())
        assert command.inflate_result_bytes == 0

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelFaultInjectFixtureCommand(
                correlation_id=uuid4(),
                inflate_result_bytes=0,
                unexpected_field="nope",  # type: ignore[call-arg]
            )

    def test_command_is_frozen(self) -> None:
        command = ModelFaultInjectFixtureCommand(correlation_id=uuid4())
        with pytest.raises(ValidationError):
            command.inflate_result_bytes = 5  # type: ignore[misc]
