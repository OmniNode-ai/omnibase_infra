# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23024: an event whose tenant cannot be resolved is dead-lettered once.

## The defect

Since the delegation projection writer was folded into the one runtime
(omnibase_infra#5743) it is dispatched by this module's auto-wiring, not by the
standalone ``BaseProjectionRunner``. That runner classified
``TenantRegistryResolutionError`` POISON (omnimarket's
``projection/error_classification.py``, OMN-17985). This module's
``_is_projection_content_failure`` is a closed allowlist of validation types and
did not know the class, so the refusal fell into the "write path is the
defect" arm: the offset was withheld and the identical event was redelivered
forever.

Measured on the dev runtime 2026-11-07: the delegate-skill-completed event
produced at 17:45:07Z for a tenant the registry mirror holds no row for was
re-refused about once per second for hours (group lag 159 on
delegate-skill-completed, 3 on delegate-skill-failed behind it).

The classifier cannot import omnimarket (this package does not depend on it),
so the refusal is matched by its defining module and class name along the MRO,
never by name alone.

Every assertion that the event is dead-lettered has a paired control showing the
withhold still holds for an unconfirmed quarantine and for every failure this
change does not name.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _is_projection_content_failure,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"
_TEST_DSN = "postgresql://user:***REDACTED***@host:5432/omnidash_analytics"

#: The live topic and tenant of the wedged record on the dev runtime.
_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_UNKNOWN_TENANT = "99ffb290-5fdc-4b8a-9c6b-e70239214928"
#: The refusal text as the runtime logged it, 2026-11-07.
_LIVE_REFUSAL = (
    f"OMN-22478: tenant_registry_mirror holds no row for '{_UNKNOWN_TENANT}' and "
    "the event is 6862s old, past the 20s catch-up window, so it was not waited "
    "for: a mint in flight when it was produced has landed by now."
)

_RESOLUTION_MODULE = "omnimarket.projection.tenant_registry_resolution"


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


def _stand_in(name: str, module: str, base: type[Exception] = ValueError) -> type:
    """A class that reports ``module``/``name`` the way the real one does."""
    return type(name, (base,), {"__module__": module})


#: Stands in for omnimarket's real ``TenantRegistryResolutionError(ValueError)``.
TenantRegistryResolutionError = _stand_in(
    "TenantRegistryResolutionError", _RESOLUTION_MODULE
)


class _UnknownTenantHandler:
    """Raises what the delegation writer raised for the recorded event."""

    def __init__(self) -> None:
        self.calls = 0

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        raise TenantRegistryResolutionError(_LIVE_REFUSAL)


def _receipt(topic: str) -> ModelPublishReceipt:
    return ModelPublishReceipt(
        topic=topic,
        partition=0,
        offset=1,
        cluster="test-cluster",
        produced_at=datetime.now(UTC),
        transport=EnumInfraTransportType.INMEMORY,
    )


class _AcceptingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object, bytes]] = []

    async def publish(
        self, topic: str, key: object, value: bytes
    ) -> ModelPublishReceipt:
        self.published.append((topic, key, value))
        return _receipt(topic)


class _RaisingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object, bytes]] = []

    async def publish(
        self, topic: str, key: object, value: bytes
    ) -> ModelPublishReceipt:
        raise ConnectionError("broker unavailable")


def _recorded_envelope() -> MagicMock:
    """The recorded delegate-skill-completed event for the unknown tenant."""
    envelope = MagicMock()
    envelope.topic = _TOPIC
    envelope.payload = {
        "tenant_id": _UNKNOWN_TENANT,
        "skill": "delegate-skill",
        "status": "completed",
    }
    envelope.correlation_id = "d2fddb35-a014-4068-bc40-4c57330c6b58"
    return envelope


def _replay(handler: object, bus: object) -> object:
    callback = _make_projection_dispatch_callback(
        handler,
        # The write target is irrelevant to the classification under test; this
        # is the declared target the sibling poison tests dispatch against.
        projection_database_target(
            "session_replay_snapshots", schema="omninode_internal"
        ),
        (_TOPIC,),
        sinks=ProjectionDispatchSinks(event_bus=bus),
    )
    with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            return asyncio.run(callback(_recorded_envelope()))


# ---------------------------------------------------------------------------
# AC1: the recorded event is dead-lettered once and the offset advances.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recorded_unknown_tenant_event_is_dead_lettered_once_and_acked() -> None:
    """RED on origin/dev: it raises ProjectionNotMaterializedError (withhold)."""
    handler = _UnknownTenantHandler()
    bus = _AcceptingBus()

    assert _replay(handler, bus) is None, (
        "an unresolvable tenant is the event's defect: the offset must advance"
    )
    assert handler.calls == 1, "the handler ran once; nothing retried it"
    assert len(bus.published) == 1, "exactly one dead-letter record"
    assert "dlq" in bus.published[0][0]


@pytest.mark.unit
def test_unknown_tenant_dead_letter_carries_the_typed_refusal() -> None:
    """Quarantine is not a drop: the record names the refusal and the tenant."""
    bus = _AcceptingBus()

    _replay(_UnknownTenantHandler(), bus)

    body = bus.published[0][2].decode("utf-8", "replace")
    assert "TenantRegistryResolutionError" in body
    assert _UNKNOWN_TENANT in body


# ---------------------------------------------------------------------------
# Controls: the withhold still holds where it must.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unconfirmed_dead_letter_still_withholds_the_offset() -> None:
    """The OMN-17862 fence is not relaxed: no durable quarantine, no advance."""
    bus = _RaisingBus()

    with pytest.raises(ProjectionNotMaterializedError):
        _replay(_UnknownTenantHandler(), bus)

    assert bus.published == []


@pytest.mark.unit
def test_tenant_refusal_from_an_unrelated_module_is_not_poison() -> None:
    """Matching on the class name alone would widen the allowlist to any type."""
    lookalike = _stand_in("TenantRegistryResolutionError", "some_other_package.errors")

    assert _is_projection_content_failure(lookalike("refused")) is False


@pytest.mark.unit
def test_plain_value_error_is_not_poison() -> None:
    assert _is_projection_content_failure(ValueError("refused")) is False


@pytest.mark.unit
def test_a_write_path_failure_is_not_poison() -> None:
    assert _is_projection_content_failure(OSError("connection reset")) is False
    assert _is_projection_content_failure(TypeError("missing _db")) is False


@pytest.mark.unit
def test_subclass_of_the_tenant_refusal_is_poison() -> None:
    subclass = type("MirrorMiss", (TenantRegistryResolutionError,), {})

    assert _is_projection_content_failure(subclass("refused")) is True
    assert _is_projection_content_failure(TenantRegistryResolutionError("x")) is True
