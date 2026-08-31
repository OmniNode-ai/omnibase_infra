# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17379: a projection that writes no row must not silently ACK the event.

Seeded from a live .201 dev-lane forensic, not from a hypothesis. On 2026-08-31
``public.pr_merged_events`` held 28 rows whose newest was **2026-08-03** while its
consumer group reported::

    GROUP  local.omnimarket.pr_merged_projection.consume.1.0.0...
    STATE  Stable    MEMBERS 1    TOTAL-LAG 0
    onex.evt.github.pr-merged.v1  0  CURRENT-OFFSET 97  LOG-END 97  LAG 0

The topic still retained 2026-08-27 events at offsets 94-96. Rewinding the group
to 94 (``rpk group seek --to-file``) and restarting the runtime made the REAL
wired path re-consume 94->96, which produced exactly three::

    [ERROR] handler_wiring: Projection handler error:
        handler=HandlerPrMergedProjection topic=onex.evt.github.pr-merged.v1
        error_type=InsufficientPrivilege
        error=permission denied for sequence pr_merged_events_projection_cursor_seq

three quarantine records, **zero** rows, and a committed offset back at 97.

That ruled the mechanism, rather than inferring it. Of the three candidates the
external signature could not distinguish:

* the OMN-14936 no-dispatcher drop -- RULED OUT. The dispatcher exists, is wired
  (``Auto-wired projection handler with DB injection:
  handler=HandlerPrMergedProjection db_tables=['pr_merged_events']``) and RAN.
* a seek-to-end on restart -- RULED OUT. The group was rewound and re-consumed the
  same offsets, and still wrote nothing.
* a swallowed write-path failure -- PROVEN, with its SQLSTATE.

The defect these tests pin is the SWALLOW, not the grant. ``permission denied``
is one instance of an unbounded class (a dead connection, a dropped relation, a
revoked role); what made it cost 230 merged PRs is that the runtime acknowledged
every one of those events into nothing while reporting healthy.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
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

# The verbatim psycopg2 message the .201 dev lane produced on offsets 94/95/96.
_LIVE_FAILURE = "permission denied for sequence pr_merged_events_projection_cursor_seq"


class InsufficientPrivilegeError(Exception):
    """Stand-in for ``psycopg2.errors.InsufficientPrivilege``.

    Declared locally so the test pins the runtime's CLASSIFICATION rule (an error
    it cannot positively identify as the event's own defect is the runtime's)
    rather than a psycopg2 type. A driver-specific class would let the fix pass
    by special-casing one library instead of closing the class.
    """


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


def _pr_merged_envelope() -> MagicMock:
    """Offset 96 from the live dev-lane topic, verbatim."""
    envelope = MagicMock()
    envelope.topic = "onex.evt.github.pr-merged.v1"
    envelope.payload = {
        "event_id": "f46ce1e5-8611-40fa-a389-75e372a9fc2b",
        "topic": "onex.evt.github.pr-merged.v1",
        "repo": "OmniNode-ai/omnimarket",
        "branch": "jonah/omn-16589-kbgate-caller-repin",
        "pr_number": 2159,
        "ticket": "OMN-16589",
        "merged_at": "2026-08-27T04:27:37Z",
    }
    envelope.correlation_id = "omn-17379-offset-96"
    return envelope


def _run_projection(handler: object, published: list[tuple]) -> object:
    """Drive the real projection dispatch callback over the live envelope."""

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        handler,
        projection_database_target("pr_merged_events", schema="omninode_internal"),
        ("onex.evt.github.pr-merged.v1",),
        sinks=ProjectionDispatchSinks(event_bus=FakeEventBus()),
    )
    with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            return asyncio.run(callback(_pr_merged_envelope()))


@pytest.mark.unit
def test_write_path_failure_raises_instead_of_acking() -> None:
    """The live drop: sequence privilege denied, zero rows, offset advanced.

    Returning normally from the dispatch callback IS an ACK -- the consume
    boundary reads "no exception" as success and the offset moves. Before the
    fix this call returned ``None`` and the event was gone.
    """

    class SequencePrivilegeDeniedHandler:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            raise InsufficientPrivilegeError(_LIVE_FAILURE)

    published: list[tuple] = []
    with pytest.raises(ProjectionNotMaterializedError) as raised:
        _run_projection(SequencePrivilegeDeniedHandler(), published)

    assert isinstance(raised.value.__cause__, InsufficientPrivilegeError)
    assert "pr_merged_events" in str(raised.value)


@pytest.mark.unit
def test_write_path_failure_is_not_traded_for_a_dead_letter_copy() -> None:
    """A valid event owed a row is preserved by the offset, not by a DLQ copy.

    The pre-fix code routed this to the platform quarantine sink and returned --
    which is how three well-formed pr-merged events reached a topic holding 8.9M
    records that nothing consumes, while the projection they were owed stayed
    24 days stale. The record's home is its own topic, uncommitted.
    """

    class DeadConnectionHandler:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            raise OSError("server closed the connection unexpectedly")

    published: list[tuple] = []
    with pytest.raises(ProjectionNotMaterializedError):
        _run_projection(DeadConnectionHandler(), published)

    assert published == []


@pytest.mark.unit
def test_missing_adapter_is_a_wiring_defect_and_withholds_the_offset() -> None:
    """A ``TypeError`` here means the runtime denied the handler its own contract.

    ``HandlerPrMergedProjection.handle`` raises ``TypeError`` when ``_db`` is
    absent. That is never the event's defect, so it takes the same withhold path.
    """

    class AdapterStarvedHandler:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            raise TypeError("handle() requires a DatabaseAdapter in input_data['_db']")

    published: list[tuple] = []
    with pytest.raises(ProjectionNotMaterializedError):
        _run_projection(AdapterStarvedHandler(), published)

    assert published == []


@pytest.mark.unit
def test_malformed_event_still_dlqs_and_advances() -> None:
    """The counterweight: a content failure must NOT withhold the offset.

    Redelivering identical malformed bytes reproduces the identical failure, so
    withholding here would wedge the partition forever on one poison record. The
    OMN-13548 DLQ-and-advance contract is deliberately preserved -- the fix
    separates the two classes, it does not make every failure fatal.
    """

    class Strict(BaseModel):
        pr_number: int

    class MalformedEventHandler:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            Strict.model_validate({"pr_number": "not-an-int"})
            raise AssertionError("unreachable -- validation must fail first")

    published: list[tuple] = []
    result = _run_projection(MalformedEventHandler(), published)

    assert result is None, "a content failure ACKs; the offset must advance"
    assert len(published) == 1, "the malformed record must still be captured"
    assert "dlq" in published[0][0]


@pytest.mark.unit
def test_successful_projection_is_unchanged() -> None:
    """A handler that writes a row still ACKs and publishes no dead letter."""

    class WritingHandler:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            return {"rows_upserted": 1, "table": "pr_merged_events"}

    published: list[tuple] = []
    assert _run_projection(WritingHandler(), published) is None
    assert published == []


@pytest.mark.unit
def test_validation_error_is_the_only_content_class() -> None:
    """Pin the classifier directly: the allowlist is closed, not a denylist.

    An error the runtime cannot positively identify as the event's own defect is
    treated as the runtime's. That direction stalls loudly; the other direction
    discards a fact silently, which is the failure being closed here.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _is_projection_content_failure,
    )

    class Strict(BaseModel):
        pr_number: int

    try:
        Strict.model_validate({"pr_number": "not-an-int"})
    except ValidationError as exc:
        assert _is_projection_content_failure(exc) is True
    else:  # pragma: no cover - the model above cannot validate
        raise AssertionError("expected a ValidationError")

    assert (
        _is_projection_content_failure(InsufficientPrivilegeError(_LIVE_FAILURE))
        is False
    )
    assert _is_projection_content_failure(OSError("connection reset")) is False
    assert _is_projection_content_failure(TypeError("missing _db")) is False
