# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Build a golden-chain input from the PUBLISHER'S contract (OMN-18013).

OPERATOR RULING (2026-09-06), ITEM 4
------------------------------------
A golden chain must build its input from the contract that declares the topic,
so a test cannot pass on a shape the bus never carries.

THE SHAPE THE BUS NEVER CARRIES
-------------------------------
At the time of the ruling, 108 test sites across 44 files passed a full ONEX
TOPIC string as an envelope ``event_type``::

    ModelEventEnvelope(event_type="onex.cmd.omnimarket.redeploy-start.v1", ...)

The runtime never stamps that. ``derive_event_type_alias_for_topic`` — the
single source for the alias on both sides of the wire (OMN-17296) — stamps
``<producer>.<event-name>``; a live read of the dev-lane bus on 2026-09-06
confirmed ``onex.evt.omnimarket.delegate-skill-completed.v1`` arriving with
``event_type`` = ``omnimarket.delegate-skill-completed``. Those tests were green
only because ``derive_entry_message_types`` registers BOTH the literal topic and
the alias as dispatcher index keys, so the wrong spelling still resolved in the
test while the real one was what production used.

WHAT THIS FIXTURE GUARANTEES
----------------------------
``publisher_envelope`` resolves the topic against the corpus of contracts, and:

* REFUSES a topic that no contract declares as a publisher — there is nothing to
  build the input from, and a chain whose entry topic has no producer is not a
  chain (this is the same closure requirement gate 3 enforces);
* stamps ``event_type`` with the DERIVED ALIAS, never the topic string, so the
  envelope is byte-for-byte the shape the consume boundary produces;
* refuses a topic whose name derives no message category, because
  ``MessageDispatchEngine`` rejects such a message before any route is consulted.

Usage::

    from omnibase_infra.testing.publisher_contract_fixture import (
        PublisherContractCorpus,
    )

    corpus = PublisherContractCorpus.from_repo_root(Path.cwd())
    envelope = corpus.publisher_envelope(
        "onex.cmd.omnibase-infra.coding-agent-invoke.v1",
        payload=command,
    )
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.validators.contract_topic_graph import parse_contract

__all__ = ["PublisherContractCorpus", "PublisherTopicError"]


class PublisherTopicError(RuntimeError):
    """A topic a golden chain wants to publish on is not declared by any contract."""


def _publisher_topics(path: Path) -> tuple[str, set[str]] | None:
    """The contract's name and every topic it declares itself a PUBLISHER of.

    Reads the same union of declaration shapes the contract topic graph reads
    (``parse_contract``), so a topic the graph sees as published is one the fixture
    resolves. ``runtime_dispatch.command_topic`` also counts: the CLI publishes there.
    """
    try:
        node = parse_contract(path, path.parent.name)
    except (OSError, RuntimeError):
        return None
    if node is None:
        return None
    topics = set(node.publish_topics)
    if node.command_topic is not None:
        topics.add(node.command_topic)
    return node.name, topics


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: test-fixture index
class PublisherContractCorpus:
    """An index of which contract publishes which topic."""

    publishers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_paths(cls, contract_paths: Iterable[Path]) -> PublisherContractCorpus:
        """Index the given ``contract.yaml`` paths by the topics they publish."""
        index: dict[str, list[str]] = {}
        for path in contract_paths:
            parsed = _publisher_topics(path)
            if parsed is None:
                continue
            name, topics = parsed
            for topic in topics:
                index.setdefault(topic, []).append(name)
        return cls(publishers={t: tuple(sorted(set(v))) for t, v in index.items()})

    @classmethod
    def from_repo_root(cls, *roots: Path) -> PublisherContractCorpus:
        """Index every ``contract.yaml`` under each of ``roots``."""
        paths: list[Path] = []
        for root in roots:
            paths.extend(
                p
                for p in root.rglob("contract.yaml")
                if ".venv" not in p.parts and "site-packages" not in p.parts
            )
        return cls.from_paths(sorted(paths))

    def publishers_of(self, topic: str) -> tuple[str, ...]:
        """Contracts declaring themselves publishers of ``topic`` (possibly empty)."""
        return self.publishers.get(topic, ())

    def event_type_for(self, topic: str) -> str:
        """The alias the bus actually carries for ``topic``, refusing an orphan topic."""
        producers = self.publishers_of(topic)
        if not producers:
            raise PublisherTopicError(
                f"No contract in the indexed corpus declares itself a publisher of "
                f"{topic!r}. A golden chain cannot build an input for a topic nothing "
                "produces — that is not a chain, it is a test asserting a shape the bus "
                "never carries. Declare the publisher (or an explicit external producer) "
                "before writing the chain."
            )
        if EnumMessageCategory.from_topic(topic) is None:
            raise PublisherTopicError(
                f"{topic!r} carries no message category in its name, so "
                "MessageDispatchEngine rejects every message on it as an invalid topic "
                "category before any route is consulted. A chain over it can never pass "
                "in production."
            )
        alias = derive_event_type_alias_for_topic(topic)
        if alias is None:
            raise PublisherTopicError(
                f"derive_event_type_alias_for_topic({topic!r}) is None, so the runtime "
                "stamps no event_type for this topic."
            )
        return alias

    def publisher_envelope(
        self,
        topic: str,
        *,
        payload: object,
        correlation_id: UUID | None = None,
        source_tool: str = "auto-wiring",
    ) -> ModelEventEnvelope[object]:
        """Build the envelope the auto-wired consume boundary produces for ``topic``.

        ``event_type`` is the DERIVED ALIAS, never the topic string, so a test
        cannot pass on the topic spelling the bus does not carry.
        """
        return ModelEventEnvelope[object](
            payload=payload,
            correlation_id=correlation_id or uuid4(),
            event_type=self.event_type_for(topic),
            source_tool=source_tool,
            tenant_id=None,
        )
