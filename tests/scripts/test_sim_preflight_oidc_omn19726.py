# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard the private authorization-code probe's credential submission target."""

import json
from pathlib import Path

import pytest

from scripts.runtime_build.prove_sim_preflight_oidc import (
    LoginForm,
    read_seeded_identity,
)


def test_identity_receipt_requires_private_canonical_subject(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    identity = {
        "tenant_id": "00000000-0000-4000-8000-000000000001",
        "subject": "00000000-0000-4000-8000-000000000002",
        "principal_id": "expected-principal",
    }
    path.write_text(json.dumps(identity))
    path.chmod(0o600)
    assert read_seeded_identity(path) == identity


@pytest.mark.parametrize("mode", [0o644, 0o666])
def test_identity_receipt_refuses_public_permissions(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "identity.json"
    path.write_text("{}")
    path.chmod(mode)
    with pytest.raises(ValueError, match="private regular"):
        read_seeded_identity(path)


def test_identity_receipt_refuses_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    source.chmod(0o600)
    path = tmp_path / "identity.json"
    path.symlink_to(source)
    with pytest.raises(ValueError, match="private regular"):
        read_seeded_identity(path)


def test_identity_receipt_refuses_non_uuid_subject(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps(
            {
                "tenant_id": "00000000-0000-4000-8000-000000000001",
                "subject": "",
                "principal_id": "principal",
            }
        )
    )
    path.chmod(0o600)
    with pytest.raises(ValueError):
        read_seeded_identity(path)


def test_login_form_preserves_only_the_disposable_keycloak_action() -> None:
    form = LoginForm()
    form.feed(
        '<form id="kc-form-login" action="http://auth.localhost:28080/realms/omninode/login-actions/authenticate?a=1&amp;b=2"></form>'
    )
    assert form.action().endswith("authenticate?a=1&b=2")


@pytest.mark.parametrize(
    "action",
    [
        "https://auth.omninode.ai/realms/omninode/login-actions/authenticate",
        "http://auth.localhost:28080/realms/master/login-actions/authenticate",
        "http://auth.localhost:28081/realms/omninode/login-actions/authenticate",
        "http://auth.localhost:28080.evil.invalid/realms/omninode/login-actions/authenticate",
        "http://example.invalid/realms/omninode/login-actions/authenticate",
    ],
)
def test_refuses_other_credential_targets(action: str) -> None:
    form = LoginForm()
    form.feed(f'<form id="kc-form-login" action="{action}"></form>')
    with pytest.raises(ValueError):
        form.action()


@pytest.mark.parametrize("count", [0, 2])
def test_refuses_missing_or_ambiguous_login_forms(count: int) -> None:
    form = LoginForm()
    form.feed(
        '<form id="kc-form-login" action="http://auth.localhost:28080/realms/omninode/login-actions/authenticate"></form>'
        * count
    )
    with pytest.raises(ValueError):
        form.action()
