# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for double-wrapped transport-envelope unwrap (OMN-12940).

The runtime auto-wiring dispatch layer constructs the contract ``input_model``
itself, BEFORE the handler runs, via
``model_cls.model_validate(_extract_dispatch_payload(envelope))``. The runtime
delivers a DOUBLE-wrapped envelope::

    {"payload": {"payload": {domain}, ...markers}, "partition_key": None, ...}

``_extract_dispatch_payload`` previously unwrapped exactly ONE level, leaving an
inner envelope, so ``model_validate`` raised ``N validation errors ... Field
required`` before the omnimarket handler's own ``coerce_command``/
``_unwrap_envelope`` (OMN-12935/12936) could run. A second site,
``_normalize_handler_result``, re-read ``correlation_id`` off the still-wrapped
inner envelope and called bare ``UUID(...)`` with no guard, crashing dispatch
with ``ValueError: badly formed hexadecimal UUID string``.

These tests pin both fixes:
  - ``_extract_dispatch_payload`` recursively reaches the domain payload through
    arbitrarily nested transport envelopes, while leaving single-wrapped,
    domain-only, and legitimate ``payload``-field payloads untouched;
  - ``_normalize_handler_result`` tolerates a non-hex correlation candidate by
    falling back to ``uuid4()`` instead of raising.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _extract_dispatch_payload,
    _make_dispatch_callback,
    _normalize_handler_result,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef


class RuntimeDomainCommand(BaseModel):
    """Importable domain model the auto-wiring kernel constructs via model_validate."""

    correlation_id: UUID
    source_commit_sha: str


_THIS_MODULE = (
    "tests.unit.runtime.auto_wiring.test_handler_wiring_double_wrapped_envelope"
)


def _domain() -> dict[str, object]:
    return {
        "correlation_id": str(uuid4()),
        "source_commit_sha": "abcdef1",
    }


def _envelope(inner: dict[str, object]) -> dict[str, object]:
    """Wrap ``inner`` in one transport-envelope layer with marker keys."""
    return {
        "payload": inner,
        "partition_key": None,
        "correlation_id": "not-a-hex-uuid",
        "event_type": "onex.cmd.test.command.v1",
        "envelope_id": "not-a-hex-uuid",
    }


@pytest.mark.unit
class TestExtractDispatchPayloadRecursiveUnwrap:
    def test_double_wrapped_reaches_domain(self) -> None:
        domain = _domain()
        double_wrapped = _envelope(_envelope(domain))
        # Must reach the domain mapping, not the still-wrapped inner envelope.
        assert _extract_dispatch_payload(double_wrapped) == domain

    def test_single_wrapped_reaches_domain(self) -> None:
        domain = _domain()
        assert _extract_dispatch_payload(_envelope(domain)) == domain

    def test_domain_only_is_unchanged(self) -> None:
        domain = _domain()
        # No transport markers + no nested ``payload`` mapping → returned as-is.
        assert _extract_dispatch_payload(domain) == domain

    def test_legitimate_payload_domain_field_not_over_unwrapped(self) -> None:
        # A domain payload that legitimately has a ``payload`` mapping field but
        # NO transport marker keys must not be unwrapped.
        domain = {"payload": {"nested": "value"}, "name": "real-domain"}
        assert _extract_dispatch_payload(domain) == domain

    def test_triple_wrapped_reaches_domain(self) -> None:
        domain = _domain()
        triple = _envelope(_envelope(_envelope(domain)))
        assert _extract_dispatch_payload(triple) == domain


@pytest.mark.unit
class TestDoubleWrappedTypedModelConstruction:
    async def test_double_wrapped_envelope_constructs_typed_model(self) -> None:
        """The kernel-side model_validate must succeed on a double-wrapped delivery.

        Reproduces the live evidence-pipeline / readiness-gate failure: a
        contract-typed auto-wired callback fed a double-wrapped envelope must
        reach the domain payload and construct the typed model without a
        ValidationError.
        """
        captured: dict[str, object] = {}

        class _Handler:
            async def handle(self, payload: object) -> None:
                captured["payload"] = payload

        callback = _make_dispatch_callback(
            _Handler(),
            event_model=ModelHandlerRef(
                name="RuntimeDomainCommand", module=_THIS_MODULE
            ),
        )

        domain = _domain()
        double_wrapped = _envelope(_envelope(domain))

        # Before the fix this raised pydantic.ValidationError (N missing fields).
        await callback(double_wrapped)

        constructed = captured["payload"]
        assert isinstance(constructed, RuntimeDomainCommand)
        assert constructed.source_commit_sha == domain["source_commit_sha"]


@pytest.mark.unit
class TestNormalizeHandlerResultUuidGuard:
    def test_non_hex_correlation_falls_back_to_uuid4(self) -> None:
        """A non-hex correlation candidate must not crash dispatch (line 785)."""

        class _Event(BaseModel):
            value: str

        # Envelope carries a non-hex correlation_id; the handler returned a
        # plain BaseModel so _normalize_handler_result runs the correlation read.
        envelope = _envelope(_domain())
        result = _normalize_handler_result(_Event(value="x"), envelope, "test")
        assert result is not None
        # No ValueError raised; a valid UUID is synthesized.
        assert isinstance(result.correlation_id, UUID)
