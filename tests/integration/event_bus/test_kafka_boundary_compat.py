# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Cross-repo Kafka boundary compatibility tests (OMN-3256).

Verifies that producer and consumer Pydantic models on the same Kafka topic
can successfully round-trip: the JSON serialised by the producer must be
deserializable by the consumer without validation errors.

No Kafka broker is required — these are pure Pydantic round-trip tests.

Background
----------
OMN-3248 was caused by a schema drift between the ``omniintelligence``
producer model and the ``omnimemory`` consumer model for the same
``onex.evt.intent.classified.v1`` topic.  The producer serialised
``intent_class`` (EnumIntentClass); the consumer expected ``intent_category``
(str).  This test would have caught that failure at PR review time.

Pairs marked ``xfail``
----------------------
Some pairs are marked ``@pytest.mark.xfail(strict=False)`` when the upstream
fix is tracked in a separate open ticket.  Once that ticket is merged the
xfail becomes an unexpected pass (XPASS), which is still a passing state.
When the upstream fix lands:

1. Remove the ``xfail`` marker from the affected pair.
2. Confirm the test passes clean (no XPASS noise in output).
3. Reference the fix ticket in the pair's inline comment.

Setup requirements
------------------
The test requires ``omniintelligence`` and ``omnimemory`` to be installed in
the active environment.  Both are available as editable installs from the
sibling repos::

    uv pip install -e ../omniintelligence -e ../omnimemory

If the packages are not installed the test collection is **skipped** via
``pytest.importorskip`` — the test suite will not fail in environments where
the sibling repos are absent.

Adding new pairs
----------------
To extend boundary coverage, add a new entry to ``BOUNDARY_PAIRS``::

    BoundaryPair(
        producer_cls=ProducerClass,
        consumer_cls=ConsumerClass,
        sample=sample_dict,
        topic="onex.evt.<producer>.<event>.v1",
        xfail_reason=None,  # or "OMN-XXXX: description of known drift"
    )

- ``producer_cls``:  the Pydantic model used by the **publishing** service.
- ``consumer_cls``:  the Pydantic model used by the **consuming** service.
- ``sample``:        a minimal valid dict accepted by ``producer_cls``.
- ``topic``:         the Kafka topic name (for diagnostic messages only).
- ``xfail_reason``:  set to an OMN ticket reference if the boundary is known
                     broken and tracked for a fix; ``None`` means PASS expected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Optional imports — skip entire module if sibling packages are not installed
# ---------------------------------------------------------------------------
omniintelligence = pytest.importorskip(
    "omniintelligence",
    reason=(
        "omniintelligence not installed. Run: uv pip install -e ../omniintelligence"
    ),
)
omnimemory = pytest.importorskip(
    "omnimemory",
    reason=("omnimemory not installed. Run: uv pip install -e ../omnimemory"),
)

from omniintelligence.nodes.node_intent_classifier_compute.models.enum_intent_class import (
    EnumIntentClass,
)
from omniintelligence.nodes.node_intent_classifier_compute.models.model_intent_classified_event import (
    ModelIntentClassifiedEvent as OmniIntelligenceIntentClassifiedEvent,
)
from omnimemory.models.events.model_intent_classified_event import (
    ModelIntentClassifiedEvent as OmniMemoryIntentClassifiedEvent,
)

# ---------------------------------------------------------------------------
# BoundaryPair descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryPair:
    """Descriptor for a single producer→consumer boundary pair.

    Attributes:
        producer_cls: Pydantic model used by the publishing service.
        consumer_cls: Pydantic model used by the consuming service.
        sample: Minimal valid dict for the producer model.
        topic: Kafka topic name (used in test IDs and error messages).
        xfail_reason: If set, the pair is expected to fail (known drift).
            Value should be an OMN ticket reference and short description.
            Remove once the upstream fix is merged.
    """

    producer_cls: type[Any]
    consumer_cls: type[Any]
    sample: dict[str, Any]
    topic: str
    xfail_reason: str | None = field(default=None)

    def __str__(self) -> str:
        drift = " [XFAIL]" if self.xfail_reason else ""
        return f"{self.topic}{drift}"


# ---------------------------------------------------------------------------
# Boundary pairs registry
# ---------------------------------------------------------------------------

_EMITTED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_CORRELATION_ID = "00000000-0000-0000-0000-000000000001"

BOUNDARY_PAIRS: list[BoundaryPair] = [
    # -------------------------------------------------------------------------
    # onex.evt.intent.classified.v1
    # Producer: omniintelligence (ModelIntentClassifiedEvent)
    # Consumer: omnimemory (ModelIntentClassifiedEvent)
    #
    # Status: XFAIL — OMN-3248 open
    # The producer emits `intent_class: EnumIntentClass`; the consumer requires
    # `intent_category: str`.  These are different field names for the same
    # concept.  Remove xfail_reason once OMN-3248 is merged.
    # -------------------------------------------------------------------------
    BoundaryPair(
        producer_cls=OmniIntelligenceIntentClassifiedEvent,
        consumer_cls=OmniMemoryIntentClassifiedEvent,
        sample={
            "session_id": "test-session-omn-3256",
            "correlation_id": _CORRELATION_ID,
            "intent_class": EnumIntentClass.FEATURE,
            "confidence": 0.95,
            "model_hint": "qwen3-coder-30b",
            "temperature": 0.3,
            "emitted_at": _EMITTED_AT,
        },
        topic="onex.evt.intent.classified.v1",
        xfail_reason=(
            "OMN-3248 open: producer emits intent_class (EnumIntentClass) "
            "but consumer requires intent_category (str). "
            "Remove this xfail_reason once OMN-3248 is merged."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Parametrize helper — injects xfail marker per pair
# ---------------------------------------------------------------------------


def _pytest_params() -> list[pytest.param]:
    params = []
    for pair in BOUNDARY_PAIRS:
        marks: list[Any] = []
        if pair.xfail_reason:
            marks.append(
                pytest.mark.xfail(
                    strict=False,
                    reason=pair.xfail_reason,
                )
            )
        params.append(pytest.param(pair, id=str(pair), marks=marks))
    return params


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.unit  # no broker needed — pure Pydantic round-trip
@pytest.mark.parametrize("pair", _pytest_params())
def test_kafka_boundary_compat(pair: BoundaryPair) -> None:
    """Verify that a producer's JSON output can be parsed by the consumer.

    Args:
        pair: BoundaryPair descriptor with producer/consumer classes and sample.

    The test:
    1. Validates ``pair.sample`` against ``pair.producer_cls``
       (catches producer-side schema issues early).
    2. Serialises the validated model to JSON
       (simulates the wire format written to Kafka).
    3. Deserialises the JSON using ``pair.consumer_cls``
       (must not raise — this is the boundary contract check).

    A ``pydantic.ValidationError`` at step 3 means the producer and consumer
    have drifted and the Kafka boundary is broken.  The failing test output
    will include the topic name and the exact validation error.

    Pairs with known drift are marked ``xfail`` (see ``BoundaryPair.xfail_reason``).
    Once the upstream fix is merged, remove the ``xfail_reason`` from the pair.
    """
    # Step 1: Validate sample against producer schema
    # Catches bad sample data or producer-side schema regressions.
    produced_model = pair.producer_cls.model_validate(pair.sample)

    # Step 2: Serialise to JSON dict (simulates Kafka wire format)
    produced_json = produced_model.model_dump(mode="json")

    # Step 3: Deserialise using consumer schema
    # ValidationError here = Kafka boundary contract broken for topic:
    #   pair.topic
    pair.consumer_cls.model_validate(produced_json)
