# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-key references for the topics recorded in the dispatch-parity oracle (OMN-22002).

The committed ``selection-oracle-v2.json`` must carry no topic literal: topics are
declared in contract YAML, and the hardcoded-topic hook refuses a quoted topic
anywhere else. The oracle therefore stores each topic as a reference token,
``${topic:<ref>}``, and the harness resolves the token to the topic string when the
oracle is loaded, so the parity test compares resolved values.

Reference forms:

* ``contract:<producer>.<MEMBER>`` - a topic declared by a discovered contract or a
  generated topic enum. ``<MEMBER>`` is the enum member key
  (:func:`contract_topic_member_key`), for example ``contract:rsd.CMD_SCORE_V1``.
  Resolution is a lookup in the live contract corpus plus the generated enums, so a
  topic that stops being declared fails the load instead of silently matching.
* ``dlq:<category>`` - the dead-letter topic the engine derives, built by
  ``build_dlq_topic`` (the engine's own builder).
* ``synthetic:<family>`` - the deliberately undeclared topics the P2 probes feed the
  engine; they are named once, in :data:`SYNTHETIC_PROBE_TOPICS`.

Encoding is lossless: :func:`encode_topics` refuses a topic it cannot reference, and
the round trip ``decode(encode(text)) == text`` is the property the regeneration
step relies on.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Iterable
from enum import Enum
from typing import Any

from omnibase_infra.enums.contract_topic_value import contract_topic_member_key
from omnibase_infra.event_bus.topic_constants import build_dlq_topic

#: Topics the P2 probes synthesize on purpose; they exist in no contract.
SYNTHETIC_PROBE_TOPICS: dict[str, str] = {
    "P2_unparseable_category": "onex.evt.nomatch",
    "P2_projections_suffix": "onex.proj.omnibase-infra.some-projection.v1",
    "P2_case_variant": "onex.EVT.omnibase-infra.SOME-THING.v1",
}

_TOKEN = re.compile(r"\$\{topic:([^}]+)\}")
# Anything shaped like a topic: the encoder must reference every match.
_CANDIDATE = re.compile(r"onex\.[A-Za-z0-9\-]+(?:\.[A-Za-z0-9_\-]+)+")
_DLQ_KIND = "dlq"


def _member_ref(topic: str) -> str | None:
    """``<producer>.<MEMBER>`` for a conforming ``onex.<kind>.<producer>.<event>.<vN>`` topic."""
    parts = topic.split(".")
    if len(parts) < 5 or parts[0] != "onex" or not re.fullmatch(r"v\d+", parts[-1]):
        return None
    kind, producer, version = parts[1], parts[2], parts[-1]
    event_name = ".".join(parts[3:-1])
    try:
        member = contract_topic_member_key(kind, event_name, version)
    except KeyError:
        return None
    return f"{producer}.{member}"


def _generated_enum_topics() -> set[str]:
    package = importlib.import_module("omnibase_infra.enums.generated")
    topics: set[str] = set()
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("enum_") or not info.name.endswith("_topic"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, Enum) and value is not Enum:
                topics.update(str(member.value) for member in value)
    return topics


def contract_topic_refs(contracts: Iterable[Any]) -> dict[str, str]:
    """Map ``contract:<producer>.<MEMBER>`` -> topic for the corpus and generated enums."""
    declared = _generated_enum_topics()
    for contract in contracts:
        bus = contract.event_bus
        if bus is None:
            continue
        declared.update(bus.subscribe_topics or ())
        declared.update(bus.publish_topics or ())
    refs: dict[str, str] = {}
    for topic in sorted(declared):
        member = _member_ref(topic)
        if member is None:
            continue
        ref = f"contract:{member}"
        if refs.setdefault(ref, topic) != topic:
            raise ValueError(f"{ref} binds two topics: {refs[ref]} and {topic}")
    return refs


def _dlq_prefix() -> str:
    return build_dlq_topic("x").removesuffix(".x.v1")


def _reference_for(topic: str, reverse: dict[str, str]) -> str | None:
    if topic in reverse:
        return reverse[topic]
    prefix = _dlq_prefix() + "."
    if topic.startswith(prefix) and topic.endswith(".v1"):
        category = topic[len(prefix) : -len(".v1")]
        if build_dlq_topic(category) == topic:
            return f"dlq:{category}"
    return None


def encode_topics(text: str, contracts: Iterable[Any]) -> str:
    """Replace every topic in ``text`` with its ``${topic:<ref>}`` token."""
    reverse = {topic: ref for ref, topic in contract_topic_refs(contracts).items()}
    reverse.update(
        {topic: f"synthetic:{family}" for family, topic in SYNTHETIC_PROBE_TOPICS.items()}
    )
    unreferenced: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        topic = match.group(0)
        ref = _reference_for(topic, reverse)
        if ref is None:
            unreferenced.add(topic)
            return topic
        return f"${{topic:{ref}}}"

    encoded = _CANDIDATE.sub(_sub, text)
    if unreferenced:
        raise ValueError(
            "topics with no contract, dlq or synthetic reference: "
            + ", ".join(sorted(unreferenced))
        )
    return encoded


def decode_topics(text: str, contracts: Iterable[Any]) -> str:
    """Resolve every ``${topic:<ref>}`` token in ``text`` to its topic string."""
    refs = contract_topic_refs(contracts)

    def _sub(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ref.startswith("dlq:"):
            return build_dlq_topic(ref.removeprefix("dlq:"))
        if ref.startswith("synthetic:"):
            return SYNTHETIC_PROBE_TOPICS[ref.removeprefix("synthetic:")]
        try:
            return refs[ref]
        except KeyError:
            raise ValueError(
                f"oracle topic reference {ref!r} is declared by no contract or "
                "generated topic enum"
            ) from None

    return _TOKEN.sub(_sub, text)
