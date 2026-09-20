# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Optional physical topic namespace for lane isolation (OMN-18891).

A second runtime sharing a broker with the dev lane must consume nothing from
and publish nothing onto the unprefixed topics. The consumer-group half of
that isolation already exists (``KAFKA_ENVIRONMENT`` feeds
``compute_consumer_group_id``); the topic half did not, and an isolated group
over shared topic names is the worst of both worlds — the second runtime
receives a full copy of every event the dev lane produces and acts on it.

This module is the one configuration surface for the topic half. It reads
``KAFKA_TOPIC_NAMESPACE`` and turns it into a prefix applied at the transport
seam, never inside a contract.

Three properties are load-bearing:

**Unset is byte-identical pass-through.** An absent, empty or whitespace-only
variable yields ``""``, :func:`apply_topic_namespace` and
:func:`strip_topic_namespace` become the identity, and
:func:`create_topic_resolver` returns a resolver indistinguishable from the
bare ``TopicResolver()`` every existing call site constructs today. This is
asserted by test, because it is what makes the change safe to land ahead of
any slot existing.

**The prefix is applied AFTER validation and cannot reach a contract.** The
ONEX suffix grammar anchors on five dot-separated segments beginning with the
literal ``onex``, so a prefixed physical name is refused as a declared suffix.
That refusal is the mechanism keeping the prefix a deployment fact: it is
legal on the wire and illegal in a ``contract.yaml``.

**Every consume boundary can recover the canonical name.** Handlers compare
topic names and some index by them, so a physical name must be mapped back
before any comparison or lookup. :func:`strip_topic_namespace` is the inverse
and is deliberately tolerant: a canonical name handed to it is returned
unchanged, so a seam that strips is correct whether or not the message it is
handed carries the prefix.

Why an environment variable rather than the config store: this is a
deployment fact about which broker namespace a process owns, resolved before
any store client exists, and it sits in the same family as
``KAFKA_ENVIRONMENT`` and ``KAFKA_BOOTSTRAP_SERVERS`` which the kernel already
reads the same way. The empty default is the semantic identity rather than a
silent fallback to a wrong value: there is no "correct" namespace this code
could guess, and a malformed one fails closed rather than being coerced.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from uuid import UUID

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.topics.model_bus_descriptor import ModelBusDescriptor
from omnibase_infra.topics.topic_resolver import TopicResolutionError, TopicResolver

__all__ = [
    "TOPIC_NAMESPACE_ENV_VAR",
    "TopicNamespaceError",
    "apply_topic_namespace",
    "apply_topic_namespace_all",
    "create_topic_resolver",
    "namespace_consumer_group_id",
    "resolve_topic_namespace",
    "strip_topic_namespace",
]

#: The single environment variable that carries the physical topic namespace.
#: Unset or blank means pass-through, which is what every existing lane runs.
TOPIC_NAMESPACE_ENV_VAR = "KAFKA_TOPIC_NAMESPACE"

#: A namespace token must be a safe single topic segment: lowercase, starting
#: with a letter, no dots. The dot separator is added by this module so that
#: a prefix boundary is unambiguous and ``prepr1`` cannot match ``prepr10``.
_NAMESPACE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class TopicNamespaceError(TopicResolutionError):
    """Raised when the configured namespace token is not a usable segment.

    Fails closed rather than coercing. A typo in a slot token would otherwise
    silently produce a namespace nobody is watching, which reads exactly like
    isolation working while the runtime talks to topics no one provisioned.
    """


#: Memoised normalisation, keyed on the RAW value read from the environment.
#: ``strip_topic_namespace`` runs once per consumed record on the transport's
#: hot path, and re-validating the token there would put a regex match on
#: every message for a value that changes only at process start. Keyed on the
#: raw string rather than cached outright so a test that monkeypatches the
#: variable still observes the new value.
_NORMALISED: dict[str, str] = {}


def resolve_topic_namespace(env: Mapping[str, str] | None = None) -> str:
    """Return the physical namespace prefix, including its trailing dot.

    Args:
        env: Environment mapping to read. Defaults to ``os.environ``. Passing
            one explicitly is how a caller resolves the namespace of a
            subprocess it is about to launch.

    Returns:
        ``""`` when the variable is absent, empty or whitespace-only, which is
        every lane running today. Otherwise ``"<token>."``.

    Raises:
        TopicNamespaceError: When the token is not a safe topic segment.
    """
    source = os.environ if env is None else env
    raw = source.get(TOPIC_NAMESPACE_ENV_VAR, "")
    cached = _NORMALISED.get(raw)
    if cached is not None:
        return cached
    token = raw.strip()
    if not token:
        _NORMALISED[raw] = ""
        return ""
    if token.endswith("."):
        token = token[:-1]
    if not _NAMESPACE_TOKEN_RE.match(token):
        raise TopicNamespaceError(
            f"Invalid {TOPIC_NAMESPACE_ENV_VAR} value {raw!r}: a topic namespace "
            "must be a single lowercase segment matching "
            f"{_NAMESPACE_TOKEN_RE.pattern!r} (an optional trailing dot is "
            "accepted and normalised away).",
            infra_context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="resolve_topic_namespace",
            ),
        )
    normalised = f"{token}."
    _NORMALISED[raw] = normalised
    return normalised


def apply_topic_namespace(
    topic: str,
    *,
    namespace: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Map a canonical topic name to its physical name on this broker.

    Idempotent: a name already carrying the namespace is returned unchanged,
    so a seam cannot double-prefix by being called twice on the same path.

    Args:
        topic: The canonical topic name.
        namespace: A pre-resolved prefix, to avoid re-reading the environment
            in a loop. Resolved from the environment when omitted.
        env: Environment mapping, used only when ``namespace`` is omitted.
    """
    prefix = resolve_topic_namespace(env) if namespace is None else namespace
    if not prefix or topic.startswith(prefix):
        return topic
    return f"{prefix}{topic}"


def apply_topic_namespace_all(
    topics: Iterable[str],
    *,
    namespace: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Map a list of canonical topic names to physical names, order preserved.

    Resolves the namespace once for the whole list.
    """
    prefix = resolve_topic_namespace(env) if namespace is None else namespace
    return [apply_topic_namespace(t, namespace=prefix) for t in topics]


def strip_topic_namespace(
    topic: str,
    *,
    namespace: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Map a physical topic name back to its canonical name.

    The inverse of :func:`apply_topic_namespace`, and the function every
    consume boundary calls BEFORE it compares or indexes by topic name. It is
    deliberately tolerant of a name that does not carry the prefix: during a
    roll-out a namespaced consumer may legitimately be handed either form, and
    a strip that refused the canonical one would convert a benign case into an
    exception on the message path.
    """
    prefix = resolve_topic_namespace(env) if namespace is None else namespace
    if prefix and topic.startswith(prefix):
        return topic[len(prefix) :]
    return topic


def namespace_consumer_group_id(
    group_id: str,
    *,
    namespace: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Prefix a LITERAL consumer group id with the slot namespace.

    Groups composed by ``compute_consumer_group_id`` already carry the
    environment token and need nothing here. This exists for the group ids
    spelled as literals — the projection writers' compose values — which
    otherwise collide with the dev lane's groups on a shared broker.

    Idempotent, for the same reason topic application is.
    """
    prefix = resolve_topic_namespace(env) if namespace is None else namespace
    if not prefix or group_id.startswith(prefix):
        return group_id
    return f"{prefix}{group_id}"


def create_topic_resolver(
    bus_descriptors: list[ModelBusDescriptor] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    correlation_id: UUID | None = None,
) -> TopicResolver:
    """Build the canonical resolver, honouring the configured namespace.

    This is the constructor every call site should use in place of a bare
    ``TopicResolver()``. With the variable unset the returned resolver is
    behaviourally identical to the bare one; with it set, ``resolve()``
    prefixes even though no call site supplies a trust domain, which is the
    whole reason a factory is needed rather than a descriptor argument.

    Args:
        bus_descriptors: Multi-bus descriptors, passed through unchanged. The
            slot namespace applies when trust-domain routing does not.
        env: Environment mapping, for tests and for resolving on behalf of a
            subprocess.
        correlation_id: Propagated into a namespace validation failure.

    Raises:
        TopicNamespaceError: When the configured token is malformed.
    """
    del correlation_id  # Reserved for error traceability; validation carries its own.
    prefix = resolve_topic_namespace(env)
    return TopicResolver(
        bus_descriptors=bus_descriptors,
        default_namespace_prefix=prefix,
    )
