#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract tests for the canonical Keycloak user-profile configuration.

Static tests only -- no network, no mocks. Parses the desired-user-profile.json
file from disk and asserts the three tenant attributes are admin-only and that
built-in attributes are preserved.

Related: OMN-17812 (cross-tenant claim via self-editable Keycloak user attributes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_PROFILE_PATH = Path(__file__).parents[2] / "docker/keycloak/desired-user-profile.json"
_TENANT_ATTRS = ("tenant_id", "tenant_slug", "principal_id")
_BUILTIN_ATTRS = ("username", "email", "firstName", "lastName")


def _attr(profile: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [a for a in profile["attributes"] if a["name"] == name]
    assert len(matches) == 1, f"expected exactly one attribute named {name!r}, got {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def profile() -> dict[str, Any]:
    return json.loads(_PROFILE_PATH.read_text())


def test_tenant_attrs_view_is_admin_only(profile: dict[str, Any]) -> None:
    for name in _TENANT_ATTRS:
        attr = _attr(profile, name)
        assert attr["permissions"]["view"] == ["admin"], (
            f"{name}: expected view=[\"admin\"], got {attr['permissions']['view']}"
        )


def test_tenant_attrs_edit_is_admin_only(profile: dict[str, Any]) -> None:
    for name in _TENANT_ATTRS:
        attr = _attr(profile, name)
        assert attr["permissions"]["edit"] == ["admin"], (
            f"{name}: expected edit=[\"admin\"], got {attr['permissions']['edit']}"
        )


def test_tenant_attrs_have_no_required_roles(profile: dict[str, Any]) -> None:
    for name in _TENANT_ATTRS:
        attr = _attr(profile, name)
        assert "required" not in attr, (
            f"{name}: tenant attribute must not have a 'required' entry (would force user to supply it)"
        )


def test_tenant_attrs_are_in_user_metadata_group(profile: dict[str, Any]) -> None:
    for name in _TENANT_ATTRS:
        attr = _attr(profile, name)
        assert attr.get("group") == "user-metadata", (
            f"{name}: expected group='user-metadata', got {attr.get('group')!r}"
        )


def test_builtin_attrs_are_present(profile: dict[str, Any]) -> None:
    attr_names = {a["name"] for a in profile["attributes"]}
    for name in _BUILTIN_ATTRS:
        assert name in attr_names, f"built-in attribute {name!r} missing from desired-user-profile.json"


def test_user_metadata_group_is_declared(profile: dict[str, Any]) -> None:
    group_names = {g["name"] for g in profile.get("groups", [])}
    assert "user-metadata" in group_names, "groups must contain an entry with name='user-metadata'"


def test_all_three_tenant_attrs_are_present(profile: dict[str, Any]) -> None:
    attr_names = {a["name"] for a in profile["attributes"]}
    missing = [n for n in _TENANT_ATTRS if n not in attr_names]
    assert not missing, f"tenant attributes missing from desired-user-profile.json: {missing}"
