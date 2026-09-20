# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for the OMN-18891 topic namespace seam.

The namespace prefix is a DEPLOYMENT fact applied at the transport seam so a
second runtime can share a broker without consuming from or publishing onto
the unprefixed topics. Three properties are load-bearing and each is asserted
here rather than left to review:

1. **Unset is byte-identical pass-through.** With the variable absent or
   blank, every seam returns exactly what it returns today. A slot that does
   not opt in cannot be changed by this code.
2. **The prefix never reaches a contract.** The suffix grammar still refuses a
   prefixed string, so the physical name cannot be declared in a
   ``contract.yaml``. The prefix is applied AFTER validation.
3. **Every consume boundary can recover the canonical name.** Handlers compare
   and index by topic; a physical name must map back to the canonical one
   before any comparison or lookup.
"""

from __future__ import annotations

import pytest

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.topics import TopicResolver
from omnibase_infra.topics.topic_namespace import (
    TOPIC_NAMESPACE_ENV_VAR,
    TopicNamespaceError,
    apply_topic_namespace,
    create_topic_resolver,
    namespace_consumer_group_id,
    resolve_topic_namespace,
    strip_topic_namespace,
)

pytestmark = [pytest.mark.unit]

VALID_SUFFIX = "onex.evt.platform.node-registration.v1"
VALID_CMD_SUFFIX = "onex.cmd.platform.request-introspection.v1"
SLOT = "prepr1"


# ---------------------------------------------------------------------------
# 1. Unset is byte-identical pass-through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_namespace_resolves_to_empty(
    raw: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent, empty or whitespace-only variable yields no prefix."""
    if raw is None:
        monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, raw)
    assert resolve_topic_namespace() == ""


def test_unset_apply_and_strip_are_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no namespace configured both directions are the identity."""
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    for topic in (VALID_SUFFIX, VALID_CMD_SUFFIX, "tenant-acme.onex.evt.x.y.v1"):
        assert apply_topic_namespace(topic) == topic
        assert strip_topic_namespace(topic) == topic


def test_unset_factory_is_byte_identical_to_bare_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory's resolver returns exactly what ``TopicResolver()`` returns.

    This is the assertion that an unset variable cannot change any existing
    lane: the whole point of routing the nine bare constructions through a
    factory is that the default is unchanged.
    """
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    bare = TopicResolver()
    made = create_topic_resolver()
    for suffix in (VALID_SUFFIX, VALID_CMD_SUFFIX):
        assert made.resolve(suffix) == bare.resolve(suffix)
    assert made.namespace_prefix == ""


def test_unset_consumer_group_id_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    assert namespace_consumer_group_id("projection-writer-main") == (
        "projection-writer-main"
    )


# ---------------------------------------------------------------------------
# 2. Set: the prefix is applied at the seam, and only at the seam
# ---------------------------------------------------------------------------


def test_set_namespace_normalises_to_dotted_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    assert resolve_topic_namespace() == f"{SLOT}."


def test_trailing_dot_is_accepted_and_not_doubled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, f"{SLOT}.")
    assert resolve_topic_namespace() == f"{SLOT}."


def test_set_factory_prefixes_with_no_trust_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No live call site passes a trust domain, so the default must prefix."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    resolver = create_topic_resolver()
    assert resolver.resolve(VALID_SUFFIX) == f"{SLOT}.{VALID_SUFFIX}"
    assert resolver.namespace_prefix == f"{SLOT}."


def test_set_resolver_still_refuses_an_invalid_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation happens BEFORE prefixing and is not relaxed by a namespace."""
    from omnibase_infra.topics import TopicResolutionError

    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    with pytest.raises(TopicResolutionError):
        create_topic_resolver().resolve("not.a.valid.topic")


def test_double_application_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying the namespace to an already-physical name must not double it."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    physical = apply_topic_namespace(VALID_SUFFIX)
    assert physical == f"{SLOT}.{VALID_SUFFIX}"
    assert apply_topic_namespace(physical) == physical


@pytest.mark.parametrize(
    "bad", ["Prepr1", "pre pr", "1slot", "slot!", "-slot", "a" * 40]
)
def test_malformed_namespace_token_fails_closed(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token that is not a safe topic segment refuses rather than guessing."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, bad)
    with pytest.raises(TopicNamespaceError):
        resolve_topic_namespace()


def test_set_consumer_group_id_is_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Literal compose group ids get the same isolation as resolved ones."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    assert namespace_consumer_group_id("projection-writer-main") == (
        f"{SLOT}.projection-writer-main"
    )
    # Idempotent for the same reason topics are.
    assert namespace_consumer_group_id(f"{SLOT}.projection-writer-main") == (
        f"{SLOT}.projection-writer-main"
    )


# ---------------------------------------------------------------------------
# 3. The consume boundary recovers the canonical name
# ---------------------------------------------------------------------------


def test_strip_recovers_the_canonical_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    physical = apply_topic_namespace(VALID_SUFFIX)
    assert strip_topic_namespace(physical) == VALID_SUFFIX


def test_strip_leaves_an_unprefixed_name_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical name arriving at a namespaced consumer is returned as-is.

    Roll-out safety: a seam that strips before comparing is correct whether or
    not the message it is handed carries the prefix.
    """
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    assert strip_topic_namespace(VALID_SUFFIX) == VALID_SUFFIX


def test_strip_does_not_eat_a_similar_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``prepr1`` must not strip from ``prepr10.…`` — the dot is part of it."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    other = f"{SLOT}0.{VALID_SUFFIX}"
    assert strip_topic_namespace(other) == other


def test_round_trip_over_the_tenant_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The slot namespace composes with, and does not disturb, tenant naming."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    tenant_topic = f"tenant-acme.{VALID_SUFFIX}"
    physical = apply_topic_namespace(tenant_topic)
    assert physical == f"{SLOT}.tenant-acme.{VALID_SUFFIX}"
    assert strip_topic_namespace(physical) == tenant_topic


# ---------------------------------------------------------------------------
# Negative control: the prefix provably cannot be declared in a contract
# ---------------------------------------------------------------------------


def test_suffix_grammar_still_refuses_a_prefixed_string() -> None:
    """The physical name is illegal as a contract-declared suffix, by design."""
    assert validate_topic_suffix(VALID_SUFFIX).is_valid is True
    assert validate_topic_suffix(f"{SLOT}.{VALID_SUFFIX}").is_valid is False
