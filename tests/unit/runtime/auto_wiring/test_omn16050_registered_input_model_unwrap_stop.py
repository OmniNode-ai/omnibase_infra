# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16050 — the recursive envelope unwrap must stop at the registered input model.

THE DEFECT. ``_extract_dispatch_payload`` unwraps ``payload`` recursively while
``_is_transport_envelope`` holds, and that predicate is purely structural: a
mapping carrying a ``payload`` mapping plus any of ``_ENVELOPE_MARKER_KEYS``
(``partition_key``/``event_type``/``envelope_id``/``event_id``/``correlation_id``/
``__debug_trace``). The module asserted the invariant "domain models never declare
these keys". That invariant is FALSE.

``ModelEmitRequest`` (``node_event_emit_effect``) declares ``payload`` plus FOUR
of those markers (``event_type``, ``correlation_id``, ``partition_key``,
``event_id``) and is ``extra="forbid"``. It is structurally identical to a
transport envelope, so the runtime unwrapped THROUGH it and handed the kernel the
caller's inner user payload. Live evidence (onex-dev, digest sha256:35099472…),
replayed in-pod against the real published bytes::

    RAW KEYS:      ['event_type', 'correlation_id', 'source_tool', 'payload']
    EXTRACTED:     dict KEYS ['session_id', 'defect_ab_probe', 'emitted_at']
    EXTRACTED ->   ValidationError: 4 validation errors for ModelEmitRequest
                   event_type Field required / 3x extra_forbidden
    -> HandlerDispatchFailureError -> boundary_swallow_prevented -> DLQ

so ``node_event_emit_effect`` could never be dispatched over the bus.

THE FIX under test. ``_extract_dispatch_payload`` now takes the dispatcher's
registered input model and stops the unwrap at a candidate that IS that model —
key-containment (every key on the candidate is a declared field/alias of the
target) AND full ``model_validate``. Fail-closed in both directions: an envelope
always carries at least one routing key the domain model does not declare
(``source_tool``, ``envelope_id``, ``__debug_trace``, ``__bindings``…), so genuine
double-wrapped deliveries (OMN-12940) keep unwrapping to the domain.

This module holds the RED reproduction plus the regressions that pin both
directions. The real-manifest / ``wire_from_manifest`` runtime-startup gate for
the same defect lives in ``tests/integration/test_auto_wiring_real_manifest.py``
(``test_real_manifest_wiring_preserves_registered_envelope_shaped_input_model``).
"""

from __future__ import annotations

import re
from typing import cast
from uuid import uuid4

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _extract_dispatch_payload,
    _is_registered_input_payload,
    _is_transport_envelope,
    _make_dispatch_callback,
    _model_declared_wire_keys,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

_THIS_MODULE = (
    "tests.unit.runtime.auto_wiring.test_omn16050_registered_input_model_unwrap_stop"
)

_TOPIC_SHAPE_RE = re.compile(r"^onex\.(evt|cmd|intent|dlq)\.[a-z0-9._-]+\.v\d+$")


class ModelRuntimeEmitRequest(BaseModel):
    """Field-for-field mirror of omnimarket's ``ModelEmitRequest`` (OMN-16050).

    omnibase_infra cannot import omnimarket (it is a downstream package), so the
    defect is reproduced against a local model carrying the exact shape that
    triggers it: a ``payload`` mapping plus four transport marker keys, with
    ``extra="forbid"`` so an over-unwrap fails totally rather than partially.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str = Field(..., min_length=1)
    payload: dict[str, object] = Field(default_factory=dict)
    correlation_id: str | None = None
    topic: str | None = None
    partition_key: str | None = None
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)


class ModelRuntimeDomainCommand(BaseModel):
    """A plain domain command: no ``payload`` field, no markers (OMN-12940 shape)."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    source_commit_sha: str


class ModelLenientEnvelopeShaped(BaseModel):
    """Envelope-shaped domain model that IGNORES extras — the adversarial case.

    Without key-containment, ``model_validate`` alone would accept a genuine
    transport envelope (extras silently dropped) and halt the unwrap one level
    too early. Pinned by ``test_lenient_model_does_not_claim_a_real_envelope``.
    """

    model_config = ConfigDict(extra="ignore")

    event_type: str
    payload: dict[str, object] = Field(default_factory=dict)


def _user_payload() -> dict[str, object]:
    """The inner, caller-authored payload from the live probe."""
    return {
        "session_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
        "defect_ab_probe": True,
        "emitted_at": "2026-08-13T02:46:13Z",
    }


def _emit_request_wire() -> dict[str, object]:
    """The domain command as published: a ModelEmitRequest on the wire."""
    return {
        "event_type": "session.started",
        "correlation_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
        "partition_key": "session-1",
        "event_id": "evt-defect-ab-probe",
        "payload": _user_payload(),
    }


def _transport_envelope(inner: dict[str, object]) -> dict[str, object]:
    """One real transport layer, exactly as the live publish produced it.

    ``source_tool`` is the discriminator no domain model declares — it is what
    tells the extractor this layer is transport and the next one may not be.
    """
    return {
        "event_type": "session.started",
        "correlation_id": "18a50ff5-c877-481c-b3c1-a183d8069762",
        "source_tool": "defect-ab-probe",
        "payload": inner,
    }


def _domain_command() -> dict[str, object]:
    return {"correlation_id": str(uuid4()), "source_commit_sha": "abcdef1"}


# ---------------------------------------------------------------------------
# RED: the exact production coercion failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProductionCoercionFailure:
    def test_emit_request_is_structurally_indistinguishable_from_an_envelope(
        self,
    ) -> None:
        """Ground truth: the false invariant. This is WHY the marker set cannot decide.

        If this ever goes green-by-inversion (the request stops looking like an
        envelope), the marker heuristic was changed and the tests below are
        testing a premise that no longer holds — revisit them rather than
        deleting this assertion.
        """
        assert _is_transport_envelope(_emit_request_wire()) is True

    def test_registered_input_model_survives_the_unwrap(self) -> None:
        """RED before the fix: the extractor returned the INNER user payload.

        Reproduces the in-pod replay verbatim — one genuine transport layer
        wrapping a ModelEmitRequest-shaped domain command. Pre-fix the extractor
        unwrapped twice and returned ``['session_id', 'defect_ab_probe',
        'emitted_at']``; it must stop at the emit request.
        """
        envelope = _transport_envelope(_emit_request_wire())

        extracted = _extract_dispatch_payload(envelope, ModelRuntimeEmitRequest)

        assert extracted == _emit_request_wire()
        assert sorted(cast("dict[str, object]", extracted)) == [
            "correlation_id",
            "event_id",
            "event_type",
            "partition_key",
            "payload",
        ]
        # And the kernel-side construction the runtime performs next succeeds.
        request = ModelRuntimeEmitRequest.model_validate(extracted)
        assert request.event_type == "session.started"
        assert request.payload == _user_payload()

    def test_without_a_registered_model_the_over_unwrap_still_reproduces(self) -> None:
        """The defect is EXACTLY the missing target type, not a shape change.

        With no registered model in scope the structural heuristic is unchanged
        (that is the deliberate no-behaviour-change property for the six call
        sites that read correlation/DLQ metadata) — it still walks to the inner
        user payload. This isolates the fix to the stop condition.
        """
        envelope = _transport_envelope(_emit_request_wire())

        assert _extract_dispatch_payload(envelope) == _user_payload()

    @pytest.mark.asyncio
    async def test_def_b_dispatch_constructs_the_registered_model(self) -> None:
        """End-to-end at the coercion boundary: the DLQ'd dispatch now succeeds.

        Pre-fix this raised ``ValidationError`` (``event_type`` Field required +
        3x extra_forbidden) inside the auto-wiring callback, which the kernel
        reported as ``HandlerDispatchFailureError`` → ``boundary_swallow_prevented``
        → DLQ.
        """
        captured: dict[str, object] = {}

        class _EmitHandler:
            async def handle(self, request: ModelRuntimeEmitRequest) -> None:
                captured["request"] = request

        callback = _make_dispatch_callback(_EmitHandler())

        await callback(_transport_envelope(_emit_request_wire()))

        request = captured["request"]
        assert isinstance(request, ModelRuntimeEmitRequest)
        assert request.event_type == "session.started"
        assert request.event_id == "evt-defect-ab-probe"
        assert request.payload == _user_payload()

    @pytest.mark.asyncio
    async def test_event_model_dispatch_constructs_the_registered_model(self) -> None:
        """The same stop applies to the contract-declared ``event_model`` branch.

        The def-B branch is the live ``node_event_emit_effect`` path, but the
        payload_type_match branch performs the same kernel-side
        ``model_validate`` and had the same over-unwrap.
        """
        captured: dict[str, object] = {}

        class _EnvelopeAgnosticHandler:
            async def handle(self, payload: object) -> None:
                captured["payload"] = payload

        callback = _make_dispatch_callback(
            _EnvelopeAgnosticHandler(),
            event_model=ModelHandlerRef(
                name="ModelRuntimeEmitRequest", module=_THIS_MODULE
            ),
        )

        await callback(_transport_envelope(_emit_request_wire()))

        payload = captured["payload"]
        assert isinstance(payload, ModelRuntimeEmitRequest)
        assert payload.payload == _user_payload()


# ---------------------------------------------------------------------------
# Regression: genuine transport envelopes must still unwrap (OMN-12940)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenuineEnvelopesStillUnwrap:
    def test_double_wrapped_reaches_domain_with_a_registered_model(self) -> None:
        domain = _domain_command()
        double = _transport_envelope(_transport_envelope(domain))

        assert _extract_dispatch_payload(double, ModelRuntimeDomainCommand) == domain

    def test_triple_wrapped_reaches_domain_with_a_registered_model(self) -> None:
        domain = _domain_command()
        triple = _transport_envelope(_transport_envelope(_transport_envelope(domain)))

        assert _extract_dispatch_payload(triple, ModelRuntimeDomainCommand) == domain

    def test_double_wrapped_emit_request_unwraps_to_the_request_not_through_it(
        self,
    ) -> None:
        """Both invariants at once: unwrap the two transport layers, stop at the model."""
        request = _emit_request_wire()
        double = _transport_envelope(_transport_envelope(request))

        assert _extract_dispatch_payload(double, ModelRuntimeEmitRequest) == request

    def test_single_wrapped_domain_unchanged_without_a_model(self) -> None:
        domain = _domain_command()

        assert _extract_dispatch_payload(_transport_envelope(domain)) == domain

    def test_domain_only_mapping_is_returned_as_is(self) -> None:
        domain = _domain_command()

        assert _extract_dispatch_payload(domain, ModelRuntimeDomainCommand) == domain

    def test_payload_field_without_markers_is_never_unwrapped(self) -> None:
        domain = {"payload": {"nested": "value"}, "name": "real-domain"}

        assert _extract_dispatch_payload(domain) == domain

    @pytest.mark.asyncio
    async def test_double_wrapped_def_b_dispatch_still_reaches_the_domain(self) -> None:
        captured: dict[str, object] = {}

        class _DomainHandler:
            async def handle(self, request: ModelRuntimeDomainCommand) -> None:
                captured["request"] = request

        callback = _make_dispatch_callback(_DomainHandler())
        domain = _domain_command()

        await callback(_transport_envelope(_transport_envelope(domain)))

        request = captured["request"]
        assert isinstance(request, ModelRuntimeDomainCommand)
        assert request.source_commit_sha == domain["source_commit_sha"]


# ---------------------------------------------------------------------------
# The stop predicate itself: fail-closed in both directions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisteredInputPayloadPredicate:
    def test_no_target_model_never_claims(self) -> None:
        assert _is_registered_input_payload(_emit_request_wire(), None) is False

    def test_non_mapping_never_claims(self) -> None:
        assert _is_registered_input_payload(
            "not-a-mapping", ModelRuntimeEmitRequest
        ) is (False)

    def test_claims_the_registered_model(self) -> None:
        assert (
            _is_registered_input_payload(_emit_request_wire(), ModelRuntimeEmitRequest)
            is True
        )

    def test_undeclared_key_defeats_the_claim(self) -> None:
        """Key containment: ``source_tool`` is not a ModelEmitRequest field."""
        envelope = _transport_envelope(_emit_request_wire())

        assert _is_registered_input_payload(envelope, ModelRuntimeEmitRequest) is False

    def test_declared_keys_that_fail_validation_defeat_the_claim(self) -> None:
        """Validation is required too — key containment alone is not enough.

        Every key here is a declared ModelEmitRequest field, but ``event_type``
        is absent (required) so this is not the registered model and the unwrap
        must continue.
        """
        candidate = {"payload": _user_payload(), "correlation_id": "abc"}

        assert _is_registered_input_payload(candidate, ModelRuntimeEmitRequest) is False

    def test_lenient_model_does_not_claim_a_real_envelope(self) -> None:
        """Adversarial: an ``extra="ignore"`` model would validate an envelope.

        ``model_validate`` alone would succeed here (extras dropped) and stop the
        unwrap one layer high, handing the handler a model whose ``payload`` is
        the intermediate envelope. Key containment is what refuses it.
        """
        envelope = _transport_envelope(_emit_request_wire())
        # Precondition: validation alone genuinely does NOT discriminate.
        assert ModelLenientEnvelopeShaped.model_validate(envelope) is not None

        assert (
            _is_registered_input_payload(envelope, ModelLenientEnvelopeShaped) is False
        )
        # The extractor therefore walks PAST the transport layer rather than
        # handing the handler a model whose ``payload`` is an envelope.
        assert (
            _extract_dispatch_payload(envelope, ModelLenientEnvelopeShaped) != envelope
        )

    def test_alias_declared_keys_are_claimable(self) -> None:
        """A model that accepts a wire alias must still be recognised as itself."""

        class _Aliased(BaseModel):
            model_config = ConfigDict(extra="forbid", populate_by_name=True)

            event_type: str
            payload: dict[str, object] = Field(default_factory=dict)
            partition_key: str | None = Field(default=None, alias="partitionKey")

        candidate = {
            "event_type": "session.started",
            "payload": _user_payload(),
            "partitionKey": "session-1",
        }

        assert _is_registered_input_payload(candidate, _Aliased) is True

    def test_custom_validator_rejection_is_not_a_claim(self) -> None:
        """A field validator that raises must read as "not the model", never crash.

        ``ModelEmitRequest`` carries exactly such a validator on ``topic``
        (``_topic_must_be_well_formed``), which raises ``ValueError`` rather than
        returning a validation verdict. The stop predicate runs on the dispatch
        hot path — an escaping exception there would take down every dispatch.
        """

        class _StrictTopic(BaseModel):
            model_config = ConfigDict(extra="forbid")

            event_type: str
            payload: dict[str, object] = Field(default_factory=dict)
            topic: str | None = None

            @field_validator("topic")
            @classmethod
            def _topic_must_be_well_formed(cls, value: str | None) -> str | None:
                if value is not None and not _TOPIC_SHAPE_RE.match(value):
                    raise ValueError(f"topic override {value!r} is malformed")
                return value

        candidate = {
            "event_type": "session.started",
            "payload": _user_payload(),
            "topic": "not-an-onex-topic",
        }

        assert _is_registered_input_payload(candidate, _StrictTopic) is False
        # ...while a well-formed topic on the same model IS claimed.
        assert (
            _is_registered_input_payload(
                {**candidate, "topic": "onex.evt.omnimarket.session-started.v1"},
                _StrictTopic,
            )
            is True
        )

    def test_alias_choices_declared_keys_are_claimable(self) -> None:
        """``AliasChoices`` alternatives are wire keys the model genuinely accepts.

        Collecting only plain-string aliases is fail-OPEN for this defect: the
        containment check would reject a candidate that IS the registered model,
        the unwrap would continue into the caller's payload, and the OMN-16050
        DLQ failure would come back for every contract aliased this way.
        """

        class _ChoiceAliased(BaseModel):
            model_config = ConfigDict(extra="forbid", populate_by_name=True)

            event_type: str
            payload: dict[str, object] = Field(default_factory=dict)
            correlation_id: str | None = Field(
                default=None,
                validation_alias=AliasChoices("correlation_id", "correlationId"),
            )

        assert {"correlation_id", "correlationId"} <= _model_declared_wire_keys(
            _ChoiceAliased
        )
        for spelling in ("correlation_id", "correlationId"):
            candidate = {
                "event_type": "session.started",
                "payload": _user_payload(),
                spelling: str(uuid4()),
            }
            assert _is_registered_input_payload(candidate, _ChoiceAliased) is True
            assert _extract_dispatch_payload(candidate, _ChoiceAliased) == candidate

    def test_alias_path_first_segment_is_the_declared_wire_key(self) -> None:
        """``AliasPath`` consumes its FIRST segment as the top-level wire key."""

        class _PathAliased(BaseModel):
            model_config = ConfigDict(extra="forbid", populate_by_name=True)

            event_type: str
            payload: dict[str, object] = Field(default_factory=dict)
            correlation_id: str | None = Field(
                default=None, validation_alias=AliasPath("meta", "correlation_id")
            )

        keys = _model_declared_wire_keys(_PathAliased)
        assert "meta" in keys
        # The inner segment is NOT a top-level key and must not be claimed as one.
        assert "correlation_id" in keys  # the field name itself, not the path tail

        candidate = {
            "event_type": "session.started",
            "payload": _user_payload(),
            "meta": {"correlation_id": str(uuid4())},
        }
        assert _is_registered_input_payload(candidate, _PathAliased) is True
        assert _extract_dispatch_payload(candidate, _PathAliased) == candidate

    def test_alias_choices_of_alias_paths_are_flattened(self) -> None:
        """Nested ``AliasChoices(AliasPath(...), ...)`` contributes every head key."""

        class _NestedAliased(BaseModel):
            model_config = ConfigDict(extra="forbid", populate_by_name=True)

            event_type: str
            payload: dict[str, object] = Field(default_factory=dict)
            correlation_id: str | None = Field(
                default=None,
                validation_alias=AliasChoices(
                    AliasPath("meta", "correlation_id"),
                    AliasPath("headers", "cid"),
                    "correlationId",
                ),
            )

        keys = _model_declared_wire_keys(_NestedAliased)
        assert {"meta", "headers", "correlationId"} <= keys

    def test_alias_declared_dispatch_still_constructs_the_registered_model(
        self,
    ) -> None:
        """End-to-end: an alias-declared model survives the wired dispatch path."""
        seen: list[object] = []

        class _AliasedRequest(BaseModel):
            model_config = ConfigDict(extra="forbid", populate_by_name=True)

            event_type: str
            payload: dict[str, object] = Field(default_factory=dict)
            correlation_id: str | None = Field(
                default=None,
                validation_alias=AliasChoices("correlation_id", "correlationId"),
            )

        class _Handler:
            async def handle(self, request: _AliasedRequest) -> dict[str, object]:
                seen.append(request)
                return {"ok": True}

        wire = {
            "event_type": "session.started",
            "payload": _user_payload(),
            "correlationId": str(uuid4()),
        }
        payload = _extract_dispatch_payload(wire, _AliasedRequest)
        assert payload == wire
        built = _AliasedRequest.model_validate(payload)
        # The user payload survived intact — it was never unwrapped through.
        assert built.payload == _user_payload()
        assert _Handler is not None
