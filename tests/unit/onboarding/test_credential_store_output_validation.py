# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``credential_store_output`` is validated at policy LOAD time (OMN-17028).

WHY THE VALIDATION IS UP FRONT AND FAIL-CLOSED
    Every failure below, left to run time, lands AFTER the operator has typed a
    live credential into an interactive prompt: a misspelled store field would
    surface as a ``TypeError`` inside the store call, a missing one as a
    half-written credential, and two credential blocks on one step as two
    writers racing for the same 0600 file. All four are refusals at load, so a
    policy that cannot store what it collects never gets to collect it.

WHY THE FIELD SET IS CLOSED
    The accepted fields are ``ModelCredentialStoreWrite``'s, which are
    ``StoreGatewayCredential.save_api_key``'s. Keeping them mechanically
    identical is what makes policy/store drift a load error instead of a
    credential that writes cleanly and reads back as nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy

pytestmark = pytest.mark.unit


def _policy(**overrides: Any) -> dict[str, Any]:
    """A minimal, valid one-prompt policy the overrides mutate."""
    document: dict[str, Any] = {
        "policy_name": "test_policy",
        "description": "d",
        "version": {"major": 1, "minor": 0, "patch": 0},
        "policy_type": "interactive",
        "target_capabilities": ["c"],
        "max_estimated_minutes": 1,
        "steps": [
            {"id": "ask", "prompt": "p", "type": "text", "required": True},
            {"id": "done", "prompt": "p", "type": "action", "action": "write"},
        ],
        "transitions": [
            {
                "from": "ask",
                "on_submit": [{"next": "done", "set_state": {"slug": "{response}"}}],
            },
            {"from": "done", "terminal": True},
        ],
        "env_output": {"done": {"SLUG": "{state.slug}"}},
    }
    document.update(overrides)
    return document


_VALID_BLOCK = {
    "tenant_slug": "{state.slug}",
    "base_url": "https://gw.invalid",
    "api_key": "{state.slug}-key",
}


def test_a_well_formed_block_validates() -> None:
    policy = ModelInteractivePolicy.model_validate(
        _policy(credential_store_output={"done": dict(_VALID_BLOCK)})
    )

    assert set(policy.credential_store_output["done"]) == set(_VALID_BLOCK)


def test_an_unknown_store_field_is_refused() -> None:
    block = dict(_VALID_BLOCK) | {"edge_instance_id": "{state.slug}"}

    with pytest.raises(ValidationError, match="unknown store field"):
        ModelInteractivePolicy.model_validate(
            _policy(credential_store_output={"done": block})
        )


def test_a_missing_store_field_is_refused() -> None:
    """A partially specified credential is the state the store itself refuses."""
    block = {k: v for k, v in _VALID_BLOCK.items() if k != "base_url"}

    with pytest.raises(ValidationError, match="missing required store field"):
        ModelInteractivePolicy.model_validate(
            _policy(credential_store_output={"done": block})
        )


def test_a_non_terminal_step_is_refused() -> None:
    with pytest.raises(ValidationError, match="not a terminal step"):
        ModelInteractivePolicy.model_validate(
            _policy(credential_store_output={"ask": dict(_VALID_BLOCK)})
        )


def test_declaring_both_credential_blocks_on_one_step_is_refused() -> None:
    """Two writers for credentials.json is the drift this block ends."""
    with pytest.raises(ValidationError, match="exactly one writer may own it"):
        ModelInteractivePolicy.model_validate(
            _policy(
                credential_store_output={"done": dict(_VALID_BLOCK)},
                credentials_output={"done": {"{state.slug}-key": "{state.slug}"}},
            )
        )


def test_policies_that_declare_no_block_are_unaffected() -> None:
    policy = ModelInteractivePolicy.model_validate(_policy())

    assert policy.credential_store_output == {}
