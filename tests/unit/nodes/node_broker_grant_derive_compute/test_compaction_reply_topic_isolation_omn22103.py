# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A developer principal reads only its own compaction reply topics (OMN-22103).

The compaction hook edge awaits three replies. Declared as shared topics, every
developer principal could read every machine's package. Each reply is therefore a
`{principal}` topic: the issuer resolves the placeholder to the exact principal it
grants, and an ACL on a literal topic name matches that name only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_broker_grant_derive.handler import (
    HandlerBrokerGrantDerive,
)
from omnibase_infra.models.security.model_broker_grant_bindings import (
    ModelBrokerGrantBindings,
)
from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[4]
BINDINGS = ROOT / "deploy/lab/developer-principal-bindings.yaml"
GRANTS = ROOT / "deploy/lab/developer-principal-grants.yaml"

REPLY_TOPIC_TEMPLATES: tuple[str, ...] = (
    "onex.cmd.omniintelligence.context-compaction-package-for-{principal}.v1",
    "onex.evt.omnimarket.context-compaction-prepared-for-{principal}.v1",
    "onex.evt.omnimarket.context-compaction-fallback-for-{principal}.v1",
)
ALICE = "dev-alice-laptop"
BOB = "dev-bob-laptop"
# A principal whose name is the leading part of another's: a prefix ACL would let
# the shorter read the longer's topics; a literal ACL does not.
SHORT = "dev-alice"


def reply_topics(principal: str) -> set[str]:
    return {t.replace("{principal}", principal) for t in REPLY_TOPIC_TEMPLATES}


def topic_rows(principal: str) -> set[tuple[str, str, str]]:
    """(name, pattern, operation) of every TOPIC ACL row the file derives for one principal."""
    declaration = lpi.load_declaration(GRANTS)
    return {
        (row.name, row.pattern, row.operation)
        for row in lpi.expected_acl_rows(declaration, principal)
        if row.resource == "topic"
    }


def reads(principal: str) -> set[str]:
    return {name for name, _, op in topic_rows(principal) if op == "read"}


def test_compaction_reply_topic_isolation_a_principal_reads_its_own_three_replies() -> (
    None
):
    own = reply_topics(ALICE)
    assert len(own) == 3
    assert own <= reads(ALICE)
    for name in own:
        assert (name, "literal", "describe") in topic_rows(ALICE)
        assert (name, "literal", "write") not in topic_rows(ALICE)


def test_compaction_reply_topic_isolation_no_principal_reads_another_principals_replies() -> (
    None
):
    assert reads(ALICE).isdisjoint(reply_topics(BOB))
    assert reads(BOB).isdisjoint(reply_topics(ALICE))
    # Positive control: each side does read its own, so the disjointness above is not
    # the product of an empty derivation.
    assert reply_topics(ALICE) <= reads(ALICE)
    assert reply_topics(BOB) <= reads(BOB)


def test_compaction_reply_topic_isolation_a_name_prefix_is_not_a_reach() -> None:
    assert reply_topics(SHORT) <= reads(SHORT)
    assert reads(SHORT).isdisjoint(reply_topics(ALICE))
    assert reads(ALICE).isdisjoint(reply_topics(SHORT))
    assert all(
        pattern == "literal"
        for name, pattern, _ in topic_rows(ALICE)
        if name in reply_topics(ALICE)
    )


def test_compaction_reply_topic_isolation_the_derivation_has_read_describe_only() -> (
    None
):
    bindings = ModelBrokerGrantBindings.model_validate(
        yaml.safe_load(BINDINGS.read_text())
    )
    derived = {
        (g.resource, g.operation)
        for g in HandlerBrokerGrantDerive().handle(bindings).grants
        if g.resource in REPLY_TOPIC_TEMPLATES
    }
    assert derived == {
        (template, op)
        for template in REPLY_TOPIC_TEMPLATES
        for op in ("READ", "DESCRIBE")
    }
