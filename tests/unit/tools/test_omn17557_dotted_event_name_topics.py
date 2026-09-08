# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dotted event names parse instead of being skipped as malformed (OMN-17557).

The topic grammar is ``onex.<kind>.<producer>.<event_name>.<version>`` and
``_RE_EVENT_NAME`` has always been ``^[a-z0-9._-]+$`` -- a dot inside the event
name is legal by the extractor's own declared rule. Under a fixed five-way
unpack that branch was unreachable by construction: any topic whose event name
contained a dot split into six parts and was rejected as "expected 5 segments,
got 6".

Measured on the live onex-dev ``omnimarket-tenant-projection-writer`` pod
(2026-09-08, ``RUNTIME_PROFILE=tenant-projection``, read-only SSM readback):
EIGHTEEN distinct topics were being skipped this way, not one. A skipped topic
never reaches the contract-first provisioner's create-set, so on a managed
broker with auto-create off it is never created. This is the same failure the
``snapshot`` KIND had before OMN-15832, recurring on ARITY.

Renaming those topics to five segments was considered and rejected: they are
live wire names consumed by the projection API and omnidash widgets, and a
rename would break every existing consumer in order to work around a parser
that was already documented to accept them.

These tests fail for four different regressions: the arity rule reverting; the
first three and last positions drifting; a genuinely malformed topic being
LET THROUGH by the looser rule (the risk this change actually carries); and the
DLQ four-segment form, which is a separate grammar, being collateral damage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import _parse_topic

pytestmark = pytest.mark.unit

_SOURCE = Path("tests/unit/tools/test_omn17557_dotted_event_name_topics.py")

# Verbatim from the live pod log, 2026-09-08. Every one of these was skipped.
_LIVE_SKIPPED_TOPICS: tuple[str, ...] = (
    "onex.snapshot.projection.coding-agent.correlation-trace.v1",
    "onex.snapshot.projection.cost.savings-overview.v1",
    "onex.snapshot.projection.delegation.budget-state.v1",
    "onex.snapshot.projection.delegation.correlation-trace.v1",
    "onex.snapshot.projection.delegation.decisions.v1",
    "onex.snapshot.projection.delegation.inference-response-text.v1",
    "onex.snapshot.projection.delegation.judge-verdicts.v1",
    "onex.snapshot.projection.delegation.model-routing.v1",
    "onex.snapshot.projection.delegation.quality-gate.v1",
    "onex.snapshot.projection.delegation.summary.v1",
    "onex.snapshot.projection.delegation.token-usage.v1",
    "onex.snapshot.projection.gate.activity.v1",
    "onex.snapshot.projection.gate.metrics.v1",
    "onex.snapshot.projection.intent-classification.distribution.v1",
    "onex.snapshot.projection.sandbox.decisions.v1",
    "onex.snapshot.projection.session.replay.v1",
    "onex.snapshot.projection.work.events.v1",
)


def test_the_inference_response_text_topic_parses_with_a_dotted_event_name() -> None:
    """The topic named on OMN-17557 parses, and into the right four fields."""
    entry = _parse_topic(
        "onex.snapshot.projection.delegation.inference-response-text.v1", _SOURCE
    )

    assert entry is not None, (
        "a dotted event name is legal per _RE_EVENT_NAME; skipping it is the "
        "OMN-17557 defect"
    )
    assert entry.kind == "snapshot"
    assert entry.producer == "projection"
    assert entry.event_name == "delegation.inference-response-text"
    assert entry.version == "v1"
    assert (
        entry.topic == "onex.snapshot.projection.delegation.inference-response-text.v1"
    )


@pytest.mark.parametrize("topic", _LIVE_SKIPPED_TOPICS)
def test_every_topic_the_live_pod_skipped_now_parses(topic: str) -> None:
    """The whole measured set, not just the one the ticket happened to name."""
    entry = _parse_topic(topic, _SOURCE)
    assert entry is not None, f"{topic} still skipped"
    assert entry.topic == topic
    assert entry.producer == "projection"


def test_five_segment_topics_are_byte_identical_under_the_new_rule() -> None:
    """The common case must not move. ``parts[3:-1]`` is a single element here."""
    entry = _parse_topic("onex.evt.omnibase-infra.quality-gate-result.v1", _SOURCE)

    assert entry is not None
    assert entry.kind == "evt"
    assert entry.producer == "omnibase-infra"
    assert entry.event_name == "quality-gate-result"
    assert entry.version == "v1"


def test_genuinely_malformed_topics_are_still_refused() -> None:
    """The looser arity must not become a looser grammar.

    This is the risk the change carries, so it is asserted directly rather
    than assumed: too few segments, a bad prefix, an unknown kind, a
    non-version tail and an underscored producer all still refuse.
    """
    assert _parse_topic("onex.tenant.events", _SOURCE) is None
    assert _parse_topic("onex.contract.resolve.completed", _SOURCE) is None
    assert _parse_topic("notonex.evt.producer.name.v1", _SOURCE) is None
    assert _parse_topic("onex.bogus.producer.name.v1", _SOURCE) is None
    assert _parse_topic("onex.evt.producer.name.notaversion", _SOURCE) is None
    assert _parse_topic("onex.evt.bad_producer.name.v1", _SOURCE) is None
    # A dot is legal in the event name; an UPPERCASE character is not.
    assert _parse_topic("onex.evt.producer.Bad.Name.v1", _SOURCE) is None


def test_the_four_segment_dlq_grammar_is_untouched() -> None:
    """DLQ topics are a separate four-part form and must not be collateral."""
    entry = _parse_topic("onex.dlq.events.v1", _SOURCE)

    assert entry is not None
    assert entry.kind == "dlq"
    assert entry.producer == "dlq"
    assert entry.event_name == "events"
    assert entry.version == "v1"
    # A four-segment topic that is NOT a dlq is still refused.
    assert _parse_topic("onex.evt.events.v1", _SOURCE) is None
