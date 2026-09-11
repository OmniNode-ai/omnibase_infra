#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the declarative User Profile reconciler extension to
seed-keycloak-clients.py (OMN-16195, applied under OMN-18165).

Uses the identical module-loading / _request-mock harness established in
test_seed_keycloak_clients.py so every seeder test exercises the same script
instance.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_mod: types.ModuleType  # assigned by _ensure_mod() on first use


def _ensure_mod() -> types.ModuleType:
    global _mod  # noqa: PLW0603
    try:
        return _mod
    except NameError:
        pass
    spec = importlib.util.spec_from_file_location("seed_keycloak_clients", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seed_keycloak_clients"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _mod = mod
    return _mod


@pytest.fixture(autouse=True, scope="session")
def _load_module() -> None:
    _ensure_mod()


_REALM = "omninode"
_KC_URL = "http://localhost:28080"
_TOKEN = "test-access-token"
_PROFILE_URL = f"{_KC_URL}/admin/realms/{_REALM}/users/profile"

_LIVE_NAMES_REQUIRED: dict[str, Any] = {
    "attributes": [
        {"name": "email", "required": {"roles": ["user"]}},
        {"name": "firstName", "required": {"roles": ["user"]}},
        {"name": "lastName", "required": {"roles": ["user"]}},
    ],
    "groups": [{"name": "user-metadata"}],
}

_DESIRED_NAMES_OPTIONAL: dict[str, Any] = {
    "_comment": "documentation key -- must never be sent to Keycloak",
    "attributes": [
        {"name": "email", "required": {"roles": ["user"]}},
        {"name": "firstName"},
        {"name": "lastName"},
    ],
    "groups": [{"name": "user-metadata"}],
}


def _reconcile(
    desired: dict[str, Any], responses: list[tuple[int, Any]]
) -> list[tuple[Any, ...]]:
    """Run the reconciler against a scripted _request sequence; return calls."""
    mod = _ensure_mod()
    calls: list[tuple[Any, ...]] = []

    def fake_request(
        method: str,
        url: str,
        token: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        calls.append((method, url, payload))
        return responses.pop(0)

    with patch.object(mod, "_request", side_effect=fake_request):
        mod._reconcile_user_profile(_KC_URL, _REALM, _TOKEN, desired)
    return calls


def test_drift_is_put_back_to_keycloak() -> None:
    """A live profile that requires firstName/lastName is corrected by a PUT."""
    calls = _reconcile(
        _DESIRED_NAMES_OPTIONAL,
        [(200, dict(_LIVE_NAMES_REQUIRED)), (204, None)],
    )
    assert [c[0] for c in calls] == ["GET", "PUT"]
    put_payload = calls[1][2]
    assert put_payload is not None
    by_name = {a["name"]: a for a in put_payload["attributes"]}
    assert "required" not in by_name["firstName"]
    assert "required" not in by_name["lastName"]
    assert by_name["email"]["required"] == {"roles": ["user"]}


def test_documentation_keys_are_never_sent() -> None:
    """``_``-prefixed keys are documentation, not Keycloak user-profile schema."""
    calls = _reconcile(
        _DESIRED_NAMES_OPTIONAL,
        [(200, dict(_LIVE_NAMES_REQUIRED)), (204, None)],
    )
    put_payload = calls[1][2]
    assert put_payload is not None
    assert not [k for k in put_payload if k.startswith("_")]


def test_already_correct_profile_is_left_alone() -> None:
    """Idempotent: no PUT when the live profile already matches desired."""
    live = {k: v for k, v in _DESIRED_NAMES_OPTIONAL.items() if not k.startswith("_")}
    calls = _reconcile(_DESIRED_NAMES_OPTIONAL, [(200, live)])
    assert [c[0] for c in calls] == ["GET"]


def test_undeclared_live_sections_are_preserved() -> None:
    """Sections the desired document does not declare survive the PUT."""
    live = dict(_LIVE_NAMES_REQUIRED)
    live["unmanagedAttributePolicy"] = "ADMIN_EDIT"
    calls = _reconcile(_DESIRED_NAMES_OPTIONAL, [(200, live), (204, None)])
    put_payload = calls[1][2]
    assert put_payload is not None
    assert put_payload["unmanagedAttributePolicy"] == "ADMIN_EDIT"


def test_failed_get_is_fatal() -> None:
    mod = _ensure_mod()
    with patch.object(mod, "_request", return_value=(404, None)):
        with pytest.raises(SystemExit):
            mod._reconcile_user_profile(
                _KC_URL, _REALM, _TOKEN, _DESIRED_NAMES_OPTIONAL
            )


def test_failed_put_is_fatal() -> None:
    with pytest.raises(SystemExit):
        _reconcile(
            _DESIRED_NAMES_OPTIONAL,
            [(200, dict(_LIVE_NAMES_REQUIRED)), (500, None)],
        )
