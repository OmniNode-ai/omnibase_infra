# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17296 — the published runtime manifest can read its own serialized form.

``ModelRuntimeManifestPublished`` is the payload that goes ON the wire for
``onex.evt.omnibase-infra.runtime-manifest-published.v1``. ``contract_hash`` and
``topology_hash`` are pydantic ``computed_field``s on the ``omnibase_core`` base,
so ``model_dump`` emits them, while the base's ``extra="forbid"`` rejected them
on input. The model could serialize but not deserialize.

That asymmetry was invisible for as long as nothing decoded the payload: the
reducer's contract declared no ``event_model``, so the dispatch adapter never
called ``model_validate`` at all. Declaring it — OMN-17296 fix 2 — is what makes
the adapter decode, and it would have turned every manifest event from a
``TypeError`` dead-letter into a ``ValidationError`` dead-letter.

The two computed keys really are on the wire; this is not hypothetical. From a
record on the dev lane's ``onex.dlq.omnibase-infra.events.v1``, read read-only on
2026-09-16, the envelope's ``payload`` keys are::

    attach_readiness, contract_hash, contracts, failed_contracts, handlers,
    image_digest, owned_command_topics, ownership_violations, runtime_profile,
    skipped_contracts, started_at, subscribed_event_topics, topology_hash

Ticket: OMN-17296
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnibase_core.models.runtime_manifest.model_manifest_contract import (
    ModelManifestContract,
)
from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)
from omnibase_infra.runtime.models.model_runtime_manifest_published import (
    ModelRuntimeManifestPublished,
)

pytestmark = pytest.mark.unit

# The exact payload key set observed on the live dev-lane wire (see docstring).
_WIRE_PAYLOAD_KEYS = frozenset(
    {
        "attach_readiness",
        "contract_hash",
        "contracts",
        "failed_contracts",
        "handlers",
        "image_digest",
        "owned_command_topics",
        "ownership_violations",
        "runtime_profile",
        "skipped_contracts",
        "started_at",
        "subscribed_event_topics",
        "topology_hash",
    }
)


def _event(**overrides: object) -> ModelRuntimeManifestPublished:
    defaults: dict[str, object] = {
        "runtime_profile": "main",
        "contracts": (
            ModelManifestContract(
                name="node_alpha",
                version="1.0.0",
                node_type="EFFECT_GENERIC",
                contract_hash="c1",
            ),
        ),
        "owned_command_topics": frozenset({"onex.cmd.omnibase-infra.alpha.v1"}),
        "subscribed_event_topics": frozenset({"onex.evt.omnibase-infra.beta.v1"}),
        "image_digest": "sha256:feedface",
        "started_at": datetime(2026, 9, 16, 7, 32, 37, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ModelRuntimeManifestPublished(**defaults)  # type: ignore[arg-type]


def test_dump_emits_exactly_the_live_wire_key_set() -> None:
    """Positive control: the model really does serialize the computed hashes.

    Without this, the round-trip test below could pass against a model that had
    simply stopped emitting them — which would be a different (and worse) change.
    """
    assert set(_event().model_dump(mode="json")) == _WIRE_PAYLOAD_KEYS


def test_validate_accepts_its_own_dump() -> None:
    """The round trip the dispatch adapter performs on every manifest event."""
    original = _event()
    restored = ModelRuntimeManifestPublished.model_validate(
        original.model_dump(mode="json")
    )
    assert restored.runtime_profile == original.runtime_profile
    assert restored.contract_hash == original.contract_hash
    assert restored.topology_hash == original.topology_hash
    assert restored.started_at == original.started_at


def test_attach_readiness_survives_the_round_trip() -> None:
    """OMN-15512's aggregate is the reason this payload exists; it must decode."""
    original = _event(
        attach_readiness=ModelRuntimeAttachReadiness(
            required_contracts=3, attached_contracts=2
        )
    )
    restored = ModelRuntimeManifestPublished.model_validate(
        original.model_dump(mode="json")
    )
    assert restored.attach_readiness is not None
    assert restored.attach_readiness.required_contracts == 3
    assert restored.attach_readiness.attached_contracts == 2


def test_unknown_key_is_still_rejected() -> None:
    """Only the two computed names are accepted back; the model stays strict.

    This is the guard against the lazy fix — widening the model to
    ``extra="ignore"`` would make the round trip pass and silently swallow a
    renamed or misspelled producer field.
    """
    payload = _event().model_dump(mode="json")
    payload["definitely_not_a_field"] = 1
    with pytest.raises(ValidationError, match="definitely_not_a_field"):
        ModelRuntimeManifestPublished.model_validate(payload)


def test_hash_disagreement_raises_instead_of_being_dropped() -> None:
    """A wire hash that disagrees with the content is drift, not noise.

    These hashes exist to make cross-version manifest drift visible. Silently
    discarding a mismatched one would decode cleanly and destroy the only signal.
    """
    payload = _event().model_dump(mode="json")
    payload["topology_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="topology_hash"):
        ModelRuntimeManifestPublished.model_validate(payload)


def test_payload_without_the_computed_keys_still_validates() -> None:
    """A producer that omits them is fine; they are derived, not required."""
    payload = _event().model_dump(mode="json")
    del payload["contract_hash"]
    del payload["topology_hash"]
    restored = ModelRuntimeManifestPublished.model_validate(payload)
    assert restored.contract_hash == _event().contract_hash
