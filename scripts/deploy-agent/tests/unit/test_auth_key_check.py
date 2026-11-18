# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24397 -- the check value of the agent's signing key, and its golden vector.

The agent discards a command signed with a different key and says nothing a sender
can read, so a sender that holds the wrong key loses a deploy before it learns the
keys differ. The agent therefore serves a one-way check value of the key it was
started with, and the signer compares it with its own before it publishes.

The vector below is held twice, once on each side of that comparison. The signer's
copy is omnimarket ``tests/nodes/node_deploy_agent_key_check_effect/key_check_vector.json``
(``secret`` fixture-key, ``label`` deploy-agent-key-check-v1, ``check`` 6dce9bc99db04417).
Each copy is asserted against its own implementation, so a change to the recipe, the
label or the width on either side turns that side's test red. This file reads nothing
from the other repository: a check that needs the other clone skips where it is absent.
"""

from __future__ import annotations

import hashlib
import hmac
import re

import pytest
from deploy_agent.auth import hmac_key_check

pytestmark = pytest.mark.unit

GOLDEN_SECRET = "fixture-key"
GOLDEN_LABEL = b"deploy-agent-key-check-v1"
GOLDEN_CHECK = "6dce9bc99db04417"


def test_the_golden_vector_is_the_value_for_the_key_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", GOLDEN_SECRET)
    assert hmac_key_check() == GOLDEN_CHECK


def test_the_golden_vector_is_the_recipe_spelled_out_here() -> None:
    derived = hmac.new(GOLDEN_SECRET.encode(), GOLDEN_LABEL, hashlib.sha256)
    assert derived.hexdigest()[:16] == GOLDEN_CHECK


def test_the_check_is_sixteen_lowercase_hex_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "another-key-entirely")
    check = hmac_key_check()
    assert check is not None
    assert re.fullmatch(r"[0-9a-f]{16}", check)
    assert check != GOLDEN_CHECK


def test_two_keys_have_two_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", GOLDEN_SECRET + "x")
    assert hmac_key_check() != GOLDEN_CHECK


@pytest.mark.parametrize("blank", [None, ""])
def test_no_key_has_no_check(
    monkeypatch: pytest.MonkeyPatch, blank: str | None
) -> None:
    if blank is None:
        monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
    else:
        monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", blank)
    assert hmac_key_check() is None
