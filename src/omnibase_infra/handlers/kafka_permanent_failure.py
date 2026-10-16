# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one declaration of which Kafka failures retrying cannot fix (OMN-19092).

Before this module the question "is this Kafka failure permanent" was answered
in several places that could not see each other, and nothing failed when one of
them learned a new class and the others did not. A surface that misses a
permanent class does not fail loudly: a retry ladder outlasts the caller's
timeout, the cancellation replaces the real cause with a ``TimeoutError``, and
the precise error becomes the one shape that means "try again shortly".

**What this module decides, and what it does not.** It classifies. It never says
what to do with a verdict. Connect raises at once, publish fails the attempt and
raises a typed error, the gateway quarantines the record; each caller keeps its
own decision and acts only on the kinds it owns (see
:class:`~omnibase_infra.enums.enum_kafka_permanent_failure.EnumKafkaPermanentFailure`).

**Type and chain, never text.** The classification keys on the exception TYPE
and walks ``__cause__`` / ``__context__`` with a visited set, because a
transport wraps the broker's verdict (``raise X(...) from last_exception``) and
the chain can form a cycle. A string match is a second definition that diverges
the first time a broker rewords a response.

**Why this lives here.** ``omnibase_infra`` sits below ``omnimarket`` in the
dependency order, so a definition kept in the upper layer could not be consumed
by the lower one; the lower layer owns it and the upper layer reads it.

This is a classifier, not an ONEX handler: it defines no ``handler_type`` or
``handler_category`` and implements no handler protocol, so the module name does
not carry the ``handler_`` prefix that ``tests/ci/test_handler_architecture_invariants.py``
reserves for handler classes (the same arrangement as ``done_write_receipt_guard``).

``aiokafka`` is imported lazily and the table is cached: callers import this
module from a DI provider that defers its own ``aiokafka`` import so that merely
materializing a contract does not pay for the client library.
"""

from __future__ import annotations

from functools import lru_cache

from omnibase_infra.enums.enum_kafka_permanent_failure import (
    EnumKafkaPermanentFailure,
)

__all__ = ["classify_permanent_kafka_failure", "kafka_failure_types"]


@lru_cache(maxsize=1)
def _declaration() -> dict[EnumKafkaPermanentFailure, tuple[type[Exception], ...]]:
    """Each kind and the aiokafka types that carry it.

    Membership is deliberately narrow. A connection refused, a timeout and a
    reset are all absent on purpose: each is the case a retry exists for.
    ``InvalidConfigurationError`` is absent too, a judgement rather than an
    oversight: aiokafka raises it for some conditions a rejoining cluster
    clears on its own.
    """
    from aiokafka.errors import (
        AuthenticationFailedError,
        AuthenticationMethodNotSupported,
        ClusterAuthorizationFailedError,
        IllegalSaslStateError,
        TopicAuthorizationFailedError,
        UnsupportedSaslMechanismError,
        UnsupportedVersionError,
    )

    return {
        EnumKafkaPermanentFailure.CONNECT_AUTHENTICATION_REJECTED: (
            AuthenticationFailedError,
            AuthenticationMethodNotSupported,
            IllegalSaslStateError,
            UnsupportedSaslMechanismError,
            UnsupportedVersionError,
        ),
        EnumKafkaPermanentFailure.TOPIC_AUTHORIZATION_REFUSED: (
            TopicAuthorizationFailedError,
        ),
        EnumKafkaPermanentFailure.CLUSTER_AUTHORIZATION_REFUSED: (
            ClusterAuthorizationFailedError,
        ),
    }


def kafka_failure_types(
    *kinds: EnumKafkaPermanentFailure,
) -> tuple[type[Exception], ...]:
    """The exception types declared for ``kinds``, for an ``except`` clause.

    ``except`` cannot call a predicate, and the fail-fast branches in a publish
    path must sit ahead of the generic ``KafkaError`` handler (all of them
    subclass it). Reading the types from the same table keeps those branches
    from becoming a second list.
    """
    table = _declaration()
    return tuple(exc_type for kind in kinds for exc_type in table[kind])


def classify_permanent_kafka_failure(
    exc: BaseException,
) -> EnumKafkaPermanentFailure | None:
    """The permanent kind ``exc`` carries anywhere in its cause chain, or ``None``.

    The outermost match wins. ``None`` means "not known to be permanent", which
    is the answer for every transient class and for anything this module has
    not been told about, so an unknown failure keeps its retry.
    """
    table = _declaration()
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        for kind, exc_types in table.items():
            if isinstance(current, exc_types):
                return kind
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None
