# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17296 fix 2 — the manifest reducer is REACHABLE through the real dispatch path.

RED-against-EXISTS-but-WRONG. OMN-17296's fix 1 (omnibase_infra#3067) made the
publisher stamp the topic-derived ``event_type`` alias, so the envelope now
RESOLVES to ``HandlerPostgresRuntimeManifestInsert`` instead of being classified
``no_dispatcher``. Section 2 of that ticket predicted what would remain::

    fixing the event_type alone does not fix this ticket — it converts a silent
    NO_DISPATCHER into a real dispatch that then fails inside an intent executor
    that cannot accept the payload.

That is exactly what the .201 dev lane shows. Measured read-only 2026-09-16,
455 of the newest 600 records on ``onex.dlq.omnibase-infra.events.v1``::

    HandlerDispatchFailureError: dispatch to
    topic=onex.evt.omnibase-infra.runtime-manifest-published.v1 returned
    status=handler_error with no terminal output (dispatcher_id=): Dispatcher
    'dispatcher.auto.node_runtime_manifest_reducer.HandlerPostgresRuntimeManifestInsert
    .postgres_insert_runtime_manifest_21269662' failed: TypeError:
    HandlerPostgresRuntimeManifestInsert.handle() missing 1 required positional
    argument: 'correlation_id'

Two independent defects produce that line, and both are asserted here:

1. **Arity.** ``_make_dispatch_callback`` only ever calls ``handle_method`` with
   ONE positional argument (``handler_wiring.py`` — ``handle_method(dispatch_arg)``,
   ``handle_method(handler_envelope)``, ``handle_method(typed_payload)``). The
   handler declared ``handle(self, payload, correlation_id)``.
2. **Model.** The handler consumed the INTENT payload
   ``ModelPayloadInsertRuntimeManifest``, never the EVENT
   ``ModelRuntimeManifestPublished``, and nothing folds one into the other:
   ``intent_emission`` is an inert contract declaration with zero runtime
   consumers. So fixing arity alone would still fail on the payload.

These tests drive the REAL production dispatch callback over the REAL handler
class and the REAL contract-declared ``event_model`` — no fake handler, no
patched entrypoint — so they fail against the pre-fix tree and pass only once
the handler consumes the event envelope and the contract declares the event
model that lets the adapter coerce it.

Ticket: OMN-17296
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime_manifest.model_manifest_contract import (
    ModelManifestContract,
)
from omnibase_core.models.runtime_manifest.model_manifest_handler import (
    ModelManifestHandler,
)
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_runtime_manifest_reducer.handlers.handler_postgres_runtime_manifest_insert import (
    SQL_INSERT_RUNTIME_MANIFEST,
    HandlerPostgresRuntimeManifestInsert,
)
from omnibase_infra.runtime.auto_wiring.discovery import _parse_handler_routing
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.models.model_runtime_manifest_published import (
    ModelRuntimeManifestPublished,
)

if TYPE_CHECKING:
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult

pytestmark = pytest.mark.unit

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_runtime_manifest_reducer"
    / "contract.yaml"
)

_MANIFEST_TOPIC = "onex.evt.omnibase-infra.runtime-manifest-published.v1"
_EVENT_TYPE_ALIAS = "omnibase-infra.runtime-manifest-published"


# ---------------------------------------------------------------------------
# Helpers — deterministic, no real broker and no real postgres
# ---------------------------------------------------------------------------


def _make_pool(row_id: int | None = 7) -> MagicMock:
    """Mock asyncpg.Pool whose fetchrow records the exact positional args."""
    pool = MagicMock()
    conn = AsyncMock()
    record = None
    if row_id is not None:
        record = MagicMock()
        record.__getitem__ = MagicMock(
            side_effect=lambda k: row_id if k == "id" else None
        )
    conn.fetchrow = AsyncMock(return_value=record)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool._test_conn = conn
    return pool


def _make_event(runtime_profile: str = "main") -> ModelRuntimeManifestPublished:
    """A real published manifest, shaped the way service_kernel publishes one."""
    return ModelRuntimeManifestPublished(
        runtime_profile=runtime_profile,
        contracts=(
            ModelManifestContract(
                name="node_runtime_manifest_reducer",
                version="1.1.0",
                node_type="REDUCER_GENERIC",
                contract_hash="c0ffee",
            ),
        ),
        owned_command_topics=frozenset({"onex.cmd.omnibase-infra.alpha.v1"}),
        subscribed_event_topics=frozenset({_MANIFEST_TOPIC}),
        handlers=(
            ModelManifestHandler(
                name="HandlerPostgresRuntimeManifestInsert",
                module_path=(
                    "omnibase_infra.nodes.node_runtime_manifest_reducer.handlers"
                    ".handler_postgres_runtime_manifest_insert"
                ),
                routing_strategy="operation_match",
            ),
        ),
        image_digest="sha256:deadbeef",
        started_at=datetime(2026, 9, 16, 7, 32, 37, tzinfo=UTC),
    )


def _contract_handler_entry() -> ModelHandlerRoutingEntry:
    """The single handler_routing entry, parsed by the RUNTIME's own parser.

    Deliberately ``_parse_handler_routing`` and not a hand-rolled read of the
    YAML: the entry under test must be the one auto-wiring would build at boot,
    including its fail-closed schema checks, or the test proves something about
    a shape the runtime never sees.
    """
    contract = yaml.safe_load(_CONTRACT_PATH.read_text())
    routing = _parse_handler_routing(
        contract["handler_routing"], contract_path=_CONTRACT_PATH
    )
    assert len(routing.handlers) == 1, (
        "node_runtime_manifest_reducer is a single-handler contract; "
        f"found {len(routing.handlers)} entries. _topics_for_handler_entry's "
        "sole-handler branch is what assigns it the subscribe topic."
    )
    return routing.handlers[0]


def _event_model_ref() -> ModelHandlerRef:
    """Build the adapter's event_model ref from the CONTRACT, not from a literal.

    Reading it from contract.yaml is what makes this a contract-first test: the
    dispatch path under test is driven by the same declaration the runtime reads
    at boot, so a contract that stops declaring the event model fails here.
    """
    event_model = _contract_handler_entry().event_model
    assert event_model is not None, (
        "contract.yaml handler entry declares no event_model, so "
        "_make_dispatch_callback takes the event_model-is-None branch, never "
        "coerces the wire payload into ModelRuntimeManifestPublished, and hands "
        "the handler a raw envelope it cannot fold (OMN-17296 fix 2)."
    )
    return event_model


def _wire_envelope(event: ModelRuntimeManifestPublished) -> ModelEventEnvelope[object]:
    """An envelope shaped like the wire: payload is a plain dict, not a model.

    MessageDispatchEngine hands the callback a materialized wire payload, so a
    test that passes an already-typed model would not exercise the coercion the
    adapter is responsible for.
    """
    return ModelEventEnvelope(
        payload=event.model_dump(mode="json"),
        correlation_id=uuid4(),
        event_type=_EVENT_TYPE_ALIAS,
        source_tool="service_kernel",
    )


# ---------------------------------------------------------------------------
# Defect 1 — arity
# ---------------------------------------------------------------------------


def test_handle_takes_exactly_one_dispatch_argument() -> None:
    """Auto-wiring only ever passes ONE positional arg to handle().

    RED before the fix: ``handle(self, payload, correlation_id)`` has two, so the
    first real dispatch raises TypeError and every manifest event dead-letters.
    """
    params = [
        p
        for name, p in inspect.signature(
            HandlerPostgresRuntimeManifestInsert.handle
        ).parameters.items()
        if name != "self"
    ]
    required = [p for p in params if p.default is inspect.Parameter.empty]
    assert len(required) == 1, (
        "handle() declares "
        f"{[p.name for p in required]} as required parameters. "
        "_make_dispatch_callback calls handle_method(<one arg>) on every path, "
        "so anything but exactly one required parameter raises TypeError on the "
        "first dispatch (OMN-17296)."
    )


def test_handle_consumes_the_published_event_not_the_intent_payload() -> None:
    """The dispatch entrypoint is annotated for the EVENT, not the INTENT payload.

    ``intent_emission`` has zero runtime consumers, so nothing ever builds a
    ``ModelPayloadInsertRuntimeManifest`` from the event. A handler annotated for
    the intent payload can only be reached by a code path that does not exist.
    """
    params = [
        p
        for name, p in inspect.signature(
            HandlerPostgresRuntimeManifestInsert.handle
        ).parameters.items()
        if name != "self"
    ]
    annotation = str(params[0].annotation)
    assert "ModelEventEnvelope" in annotation, (
        f"handle()'s dispatch parameter is annotated {annotation!r}. It must "
        "accept ModelEventEnvelope so the adapter's typed-envelope branch runs "
        "and the handler can read envelope.correlation_id for tracing."
    )
    assert "ModelRuntimeManifestPublished" in annotation, (
        f"handle()'s dispatch parameter is annotated {annotation!r}, which does "
        "not name the published event model. The reducer subscribes to "
        f"{_MANIFEST_TOPIC} and must consume what that topic carries."
    )


# ---------------------------------------------------------------------------
# Defect 2 — the contract must declare the event model
# ---------------------------------------------------------------------------


def test_contract_declares_the_event_model_for_the_subscribed_topic() -> None:
    """The contract entry carries an event_model naming the published event."""
    ref = _event_model_ref()
    assert ref.name == "ModelRuntimeManifestPublished", (
        f"contract event_model.name is {ref.name!r}; the topic "
        f"{_MANIFEST_TOPIC} carries ModelRuntimeManifestPublished."
    )
    assert (
        ref.module == "omnibase_infra.runtime.models.model_runtime_manifest_published"
    ), f"contract event_model.module is {ref.module!r} and will not import."


def test_contract_still_subscribes_exactly_this_topic() -> None:
    """Positive control on the contract read itself.

    Guards the two tests above from passing vacuously against a contract whose
    subscription was silently dropped instead of fixed — dropping the
    subscription is the OTHER disposition OMN-17296 AC2 allows, and it must not
    be reachable by accident.
    """
    contract = yaml.safe_load(_CONTRACT_PATH.read_text())
    assert contract["event_bus"]["subscribe_topics"] == [_MANIFEST_TOPIC]


# ---------------------------------------------------------------------------
# End to end over the REAL dispatch callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_dispatch_callback_inserts_the_manifest() -> None:
    """The whole seam: wire dict -> adapter coercion -> handler -> INSERT.

    This is the test that reproduces the live dev-lane failure. Against the
    pre-fix tree it raises
    ``TypeError: handle() missing 1 required positional argument:
    'correlation_id'`` — verbatim the DLQ failure_reason.
    """
    pool = _make_pool(row_id=42)
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    callback = _make_dispatch_callback(handler, _event_model_ref())

    event = _make_event()
    envelope = _wire_envelope(event)

    result: ModelDispatchResult | None = await callback(envelope)

    assert result is not None, "dispatch produced no result at all"
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"dispatch status {result.status}: {result.error_message}"
    )

    conn = pool._test_conn
    conn.fetchrow.assert_called_once()
    args = conn.fetchrow.call_args[0]
    assert args[0] == SQL_INSERT_RUNTIME_MANIFEST

    # The values must come from THIS event, not from model defaults — otherwise
    # a handler that inserted an empty manifest would pass.
    assert args[1] == "main"
    assert args[2] == event.contract_hash
    assert args[3] == event.topology_hash
    assert args[4], "manifest_hash was empty; the payload model requires min_length=1"
    assert args[13] == event.started_at


@pytest.mark.asyncio
async def test_real_dispatch_callback_propagates_the_envelope_correlation_id() -> None:
    """correlation_id survives the fold, so the INSERT is traceable to its boot.

    The pre-fix signature took correlation_id as a second positional argument
    precisely because the SQL op logs it. Reading it off the envelope is what
    replaces that argument; if it were dropped, the op would be untraceable.
    """
    pool = _make_pool()
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    callback = _make_dispatch_callback(handler, _event_model_ref())

    correlation_id = uuid4()
    envelope = _wire_envelope(_make_event())
    envelope = envelope.model_copy(update={"correlation_id": correlation_id})

    logged: dict[str, object] = {}
    original = handler._execute_postgres_op

    async def _spy(**kwargs: object) -> object:
        logged.update(kwargs)
        return await original(**kwargs)  # type: ignore[arg-type]

    handler._execute_postgres_op = _spy  # type: ignore[method-assign]
    await callback(envelope)

    assert logged["correlation_id"] == correlation_id, (
        "the INSERT ran under "
        f"{logged.get('correlation_id')!r} instead of the envelope's "
        f"{correlation_id!r}; the boot trace is broken."
    )


@pytest.mark.asyncio
async def test_dispatch_fails_closed_when_the_envelope_carries_no_correlation_id() -> (
    None
):
    """Fail-closed, not a synthesised id.

    ``publish_runtime_manifest`` declares ``correlation_id: UUID`` (required), so
    an envelope without one did not come from the sanctioned publisher. Inventing
    a UUID here would write an untraceable row and hide the real producer defect,
    so the dispatch must raise and dead-letter instead (rule 8, fail fast on a
    missing value rather than substituting a default).
    """
    pool = _make_pool()
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    callback = _make_dispatch_callback(handler, _event_model_ref())

    envelope = _wire_envelope(_make_event())
    envelope = envelope.model_copy(update={"correlation_id": None})

    with pytest.raises(Exception) as excinfo:
        await callback(envelope)

    assert "correlation_id" in str(excinfo.value)
    pool._test_conn.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_attach_readiness_absent_writes_unknown_not_ready() -> None:
    """Positive control for the OMN-15512 columns through the new fold.

    ``attach_state`` must never read ``ready`` for a manifest that carried no
    aggregate — that distinction is the whole point of the OMN-15512 columns, and
    the fold is a new place it could be lost.
    """
    pool = _make_pool()
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    callback = _make_dispatch_callback(handler, _event_model_ref())

    event = _make_event()
    assert event.attach_readiness is None
    await callback(_wire_envelope(event))

    args = pool._test_conn.fetchrow.call_args[0]
    assert args[14] == "unknown"


@pytest.mark.asyncio
async def test_dispatch_id_matches_the_live_dlq_dispatcher_id() -> None:
    """Pins the operation the live failures name, so this test tracks that defect.

    The dev-lane DLQ names dispatcher
    ``...HandlerPostgresRuntimeManifestInsert.postgres_insert_runtime_manifest_21269662``.
    The suffix is ``sha1("postgres.insert_runtime_manifest")[:8]``. If the
    contract's operation is renamed, this test's subject is no longer the defect
    that was measured, and that should be an explicit decision.
    """
    import hashlib

    operation = _contract_handler_entry().operation
    assert operation == "postgres.insert_runtime_manifest"
    digest = hashlib.sha1(str(operation).encode()).hexdigest()[:8]
    assert digest == "21269662", (
        f"operation {operation!r} hashes to {digest}, but the measured live "
        "dispatcher id ends 21269662."
    )


def test_handler_is_registered_for_pool_injection() -> None:
    """The handler only ever gets its pool from the kernel's name-keyed map.

    ``ServiceHandlerResolver`` looks the handler class NAME up in
    ``materialized_explicit_dependencies``; a ``pool`` parameter satisfies none of
    the fallback injectables, so an unregistered name fails construction with a
    TypeError at wiring time rather than at dispatch.
    """
    kernel_src = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "service_kernel.py"
    ).read_text()
    assert '"HandlerPostgresRuntimeManifestInsert": {"pool": postgres_pool}' in (
        kernel_src
    ), (
        "service_kernel._build_runtime_handler_dependencies no longer registers "
        "HandlerPostgresRuntimeManifestInsert for pool injection."
    )


def test_correlation_id_is_a_uuid_on_the_publisher_side() -> None:
    """The producer-side half of the fail-closed contract above."""
    from omnibase_infra.runtime import manifest_builder

    sig = inspect.signature(manifest_builder.publish_runtime_manifest)
    assert sig.parameters["correlation_id"].annotation in (UUID, "UUID"), (
        "publish_runtime_manifest no longer requires a UUID correlation_id, so "
        "the consumer's fail-closed branch would start dead-lettering real "
        "manifests."
    )
