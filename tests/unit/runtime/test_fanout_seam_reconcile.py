# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka-half fan-out seam reconciled to the shared core resolver (OMN-14403 §6ii).

Proves the infra ``fanout_seam`` + ``_normalize_handler_result`` fan-out branch
delegate their pure logic to ``omnibase_core.runtime.runtime_fanout_resolver`` (the
same resolver ``LocalRuntimeBusAdapter`` calls) and honor the default-OFF seam:
warn-drop when OFF, validated batch when ON. §6ii ONLY — the §8.1 causation/tenant
carry is a separate lane and is intentionally NOT wired here.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.runtime.auto_wiring.fanout_seam import (
    is_fanout_sequence,
    normalize_fanout_sequence,
)

_ENV = "ONEX_MULTI_EVENT_PUBLISH_SEAM"


class ModelAlpha(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 0


class ModelBeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 0


# A stand-in for the routing carrier the shared resolver rejects (keyed by name).
class ModelEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: int = 0


class TestSharedResolverDelegation:
    def test_is_fanout_sequence_is_the_core_symbol(self) -> None:
        # Imported from core via fanout_seam's re-export — one definition, not two.
        from omnibase_core.runtime import runtime_fanout_resolver as core

        assert is_fanout_sequence is core.is_fanout_sequence

    def test_seam_off_warn_drops(self) -> None:
        out = normalize_fanout_sequence(
            [ModelAlpha(value=1), ModelBeta(value=2)],
            "handler",
            seam_enabled=False,
            env_flag=_ENV,
        )
        assert out == []

    def test_seam_on_validates_batch(self) -> None:
        a, b = ModelAlpha(value=1), ModelBeta(value=2)
        out = normalize_fanout_sequence(
            [a, b], "handler", seam_enabled=True, env_flag=_ENV
        )
        assert out == [a, b]

    def test_seam_on_rejects_carrier_via_core(self) -> None:
        with pytest.raises(ModelOnexError, match="routing carrier"):
            normalize_fanout_sequence(
                [ModelEventEnvelope()], "handler", seam_enabled=True, env_flag=_ENV
            )

    def test_seam_on_rejects_non_model_via_core(self) -> None:
        with pytest.raises(ModelOnexError, match="not a BaseModel"):
            normalize_fanout_sequence(
                [ModelAlpha(), "x"], "handler", seam_enabled=True, env_flag=_ENV
            )


class TestNormalizeHandlerResultFanoutBranch:
    def test_bare_sequence_return_becomes_published_batch_when_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "1")
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _normalize_handler_result,
        )

        envelope = {"correlation_id": "c", "topic": "onex.evt.x.v1"}
        result = _normalize_handler_result(
            [ModelAlpha(value=1), ModelBeta(value=2)], envelope, "handler"
        )
        assert result is not None
        assert len(result.output_events) == 2

    def test_bare_sequence_return_drops_when_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _normalize_handler_result,
        )

        envelope = {"correlation_id": "c", "topic": "onex.evt.x.v1"}
        result = _normalize_handler_result(
            [ModelAlpha(value=1), ModelBeta(value=2)], envelope, "handler"
        )
        assert result is not None
        assert result.output_events == []  # seam OFF: byte-for-byte today's drop


class TestLegacyNoBusSequenceReturns:
    def test_event_model_free_callback_preserves_empty_sequence_noop(self) -> None:
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _make_dispatch_callback,
        )

        class Handler:
            def handle(self, _envelope: object) -> list[str]:
                return []

        callback = _make_dispatch_callback(Handler(), None)

        assert asyncio.run(callback({"topic": "onex.evt.x.v1"})) == []

    def test_event_model_free_callback_preserves_string_sequence_intent_marker(
        self,
    ) -> None:
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _make_dispatch_callback,
        )

        class Handler:
            def handle(self, _envelope: object) -> list[str]:
                return ["inference-intent"]

        callback = _make_dispatch_callback(Handler(), None)

        assert asyncio.run(callback({"topic": "onex.evt.x.v1"})) == ["inference-intent"]
