# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The in-row outbox publishes for a row keyed on a domain key (OMN-20127).

OMN-16924 let a contract key its state_io row on a domain field, and the PR
landing orchestrator keys ``pr_landing_workflow_state`` on ``landing_key``
(``owner/repo#123``). The outbox records the row key as each entry's
``correlation_id``, and ``_publish_outbox_batch`` parsed that value with
``UUID(...)``. For a landing key that raises ``ValueError``, surfaced as
``BoundaryPublishError: state_io outbox publish-from-row failed``: on the .201
dev lane (2026-09-30T03:19:41Z) the leg for omnimarket#3095 committed its
GitHub read into ``pending_emissions`` and then published nothing, so the row
sat in flight at UNSEEN and the request topic stayed empty.

A row key that is not a UUID now publishes under the emitted event's own
correlation id when it carries one, else under an id derived from the key, and
the causation edge is recorded only when it names a real envelope.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_stateful_dispatch_callback,
)
from tests.integration.test_state_io_domain_key_omn16924 import (
    CID,
    DELEGATION_STATE_IO,
    INPUT_ENVELOPE_ID,
    LANDING_STATE_IO,
    _derived_key_route,
    _envelope,
    _FakeStateStoreAdapter,
    _FoldingHandler,
    _SeamCodec,
)

LANDING_KEY = "OmniNode-ai/omnimarket#3095"
EFFECT_CORRELATION = UUID("20127001-1111-4111-8111-142012700001")
REQUEST_TOPIC = (
    "onex.cmd.test-seam.landing-request.v1"  # onex-topic-allow: test fixture
)

_PATCH_IMPORT = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_PATCH_ADAPTER = "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter"


class ModelSeamLandingRequest(BaseModel):
    """Stand-in for ModelPrLandingGithubRequest: carries its own correlation id."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID
    pr_number: int


class ModelSeamUncorrelatedRequest(BaseModel):
    """An emission with no correlation id of its own."""

    model_config = ConfigDict(extra="forbid")

    pr_number: int


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish_envelope(
        self, *, envelope: Any, topic: str, key: bytes | None = None
    ) -> None:
        self.published.append((topic, envelope))


def _callback(
    handler: _FoldingHandler,
    adapter: _FakeStateStoreAdapter,
    bus: _RecordingBus,
    state_io: dict[str, object],
    topic_map: dict[str, str],
    *,
    event_model: Any = None,
) -> Any:
    with (
        patch.dict(
            "os.environ",
            {"OMNIBASE_INFRA_DB_URL": "postgresql://user:***REDACTED***@host:5432/db"},
        ),
        patch(_PATCH_IMPORT, return_value=_SeamCodec),
        patch(_PATCH_ADAPTER, return_value=adapter),
    ):
        return _make_stateful_dispatch_callback(
            cast("Any", handler),
            event_model,
            dict(state_io),
            event_bus=bus,
            output_topic_map=topic_map,
        )


def _landing_envelope() -> Any:
    return _envelope(
        {
            "repository": "OmniNode-ai/omnimarket",
            "pr_number": 3095,
            "correlation_id": str(CID),
        }
    )


@pytest.mark.integration
def test_a_landing_keyed_row_publishes_its_captured_request() -> None:
    """RED pre-fix: ValueError from UUID(landing_key), BoundaryPublishError."""
    adapter = _FakeStateStoreAdapter()
    bus = _RecordingBus()
    handler = _FoldingHandler(
        state="UNSEEN",
        events=(
            ModelSeamLandingRequest(correlation_id=EFFECT_CORRELATION, pr_number=3095),
        ),
    )
    callback = _callback(
        handler,
        adapter,
        bus,
        LANDING_STATE_IO,
        {"SeamLandingRequest": REQUEST_TOPIC},
        event_model=_derived_key_route(),
    )

    asyncio.run(callback(_landing_envelope()))

    assert [topic for topic, _ in bus.published] == [REQUEST_TOPIC]
    _, envelope = bus.published[0]
    assert envelope.correlation_id == EFFECT_CORRELATION
    assert envelope.parent_envelope_id == INPUT_ENVELOPE_ID
    row = adapter.rows[LANDING_KEY]
    assert row["in_flight"] is False, "the published batch was finalized"
    assert not row["pending_emissions"]


@pytest.mark.integration
def test_an_uncorrelated_emission_publishes_under_a_key_derived_id() -> None:
    def publish_once() -> Any:
        adapter = _FakeStateStoreAdapter()
        bus = _RecordingBus()
        handler = _FoldingHandler(
            state="UNSEEN", events=(ModelSeamUncorrelatedRequest(pr_number=3095),)
        )
        callback = _callback(
            handler,
            adapter,
            bus,
            LANDING_STATE_IO,
            {"SeamUncorrelatedRequest": REQUEST_TOPIC},
            event_model=_derived_key_route(),
        )
        asyncio.run(callback(_landing_envelope()))
        assert len(bus.published) == 1
        return bus.published[0][1]

    first, second = publish_once(), publish_once()

    assert isinstance(first.correlation_id, UUID)
    assert first.correlation_id == second.correlation_id, "derived, not random"
    assert first.envelope_id == second.envelope_id, "a re-publish deduplicates"


@pytest.mark.integration
def test_a_correlation_keyed_row_publishes_under_its_key_as_before() -> None:
    """Regression control: the delegation shape's wire correlation is unchanged."""
    adapter = _FakeStateStoreAdapter()
    bus = _RecordingBus()
    handler = _FoldingHandler(
        state="RECEIVED",
        events=(
            ModelSeamLandingRequest(correlation_id=EFFECT_CORRELATION, pr_number=1),
        ),
    )
    callback = _callback(
        handler,
        adapter,
        bus,
        DELEGATION_STATE_IO,
        {"SeamLandingRequest": REQUEST_TOPIC},
    )

    asyncio.run(callback(_envelope({"correlation_id": str(CID)})))

    assert len(bus.published) == 1
    assert bus.published[0][1].correlation_id == CID
