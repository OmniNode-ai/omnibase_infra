#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the realm-settings (verify-email + SMTP) reconciler extension
to seed-keycloak-clients.py (OMN-14115).

Uses the identical module-loading / _request-mock harness established in
test_seed_keycloak_clients.py so both files exercise the same script instance.
"""

from __future__ import annotations

import importlib.util
import json
import os
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

# Matches the realmSettings.smtpServer shape from docker/keycloak/desired-clients.json
_SMTP_SPEC = {
    "fromDisplayNameEnv": "SMTP_FROM_DISPLAY_NAME",
    "fromEnv": "SMTP_FROM",
    "hostEnv": "SMTP_HOST",
    "portEnv": "SMTP_PORT",
    "starttlsEnv": "SMTP_STARTTLS",
    "authEnv": "SMTP_AUTH",
    "userEnv": "SMTP_USER",
    "passwordEnv": "SMTP_PASSWORD",
}

_SMTP_ENV = {
    "SMTP_FROM_DISPLAY_NAME": "OmniNode",
    "SMTP_FROM": "noreply@omninode.ai",
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_STARTTLS": "true",
    "SMTP_AUTH": "true",
    "SMTP_USER": "smtp-user",
    "SMTP_PASSWORD": "smtp-pass",
}

_REALM_SETTINGS_SPEC = {
    "verifyEmail": True,
    "registrationAllowed": True,
    "attributes": {"actionTokenGeneratedByUserLifespan.verify-email": "900"},
    "smtpServer": _SMTP_SPEC,
}


def _fresh_realm_body() -> dict[str, Any]:
    """Matches the real current omninode-realm.json shape: verifyEmail unset,
    smtpServer={}, no verify-email lifespan attribute."""
    return {
        "realm": _REALM,
        "registrationAllowed": True,
        "smtpServer": {},
        "attributes": {},
    }


def _already_correct_realm_body() -> dict[str, Any]:
    body = _fresh_realm_body()
    body["verifyEmail"] = True
    body["attributes"] = {"actionTokenGeneratedByUserLifespan.verify-email": "900"}
    body["smtpServer"] = {
        "host": _SMTP_ENV["SMTP_HOST"],
        "port": _SMTP_ENV["SMTP_PORT"],
        "from": _SMTP_ENV["SMTP_FROM"],
        "fromDisplayName": _SMTP_ENV["SMTP_FROM_DISPLAY_NAME"],
        "starttls": _SMTP_ENV["SMTP_STARTTLS"],
        "auth": _SMTP_ENV["SMTP_AUTH"],
        "user": _SMTP_ENV["SMTP_USER"],
        "password": _SMTP_ENV["SMTP_PASSWORD"],
    }
    return body


class TestReconcileRealmSettingsAppliesVerifyEmailSmtpAndLifespan:
    def test_reconcile_realm_settings_applies_verify_email_smtp_and_lifespan(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        existing = _fresh_realm_body()
        put_payloads: list[dict[str, Any]] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            if method == "PUT" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                put_payloads.append(kwargs.get("payload", {}))
                return (204, None)
            raise AssertionError(f"unexpected request: {method} {url}")

        with patch.dict(os.environ, _SMTP_ENV):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_realm_settings(
                    _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                )

        assert len(put_payloads) == 1
        payload = put_payloads[0]
        assert payload["verifyEmail"] is True
        assert (
            payload["attributes"]["actionTokenGeneratedByUserLifespan.verify-email"]
            == "900"
        )
        assert payload["smtpServer"]["host"] == "smtp.example.com"
        assert payload["smtpServer"]["port"] == "587"
        assert payload["smtpServer"]["from"] == "noreply@omninode.ai"
        assert payload["smtpServer"]["fromDisplayName"] == "OmniNode"
        assert payload["smtpServer"]["starttls"] == "true"
        assert payload["smtpServer"]["auth"] == "true"
        assert payload["smtpServer"]["user"] == "smtp-user"
        assert payload["smtpServer"]["password"] == "smtp-pass"

        record = json.loads(capsys.readouterr().out.strip())
        assert record["op"] == "updated"
        assert record["clientId"] == f"realm:{_REALM}"


class TestReconcileRealmSettingsIdempotent:
    def test_reconcile_realm_settings_idempotent_on_already_correct_realm(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        existing = _already_correct_realm_body()

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            if method == "PUT":
                raise AssertionError(
                    "PUT must not be called when realm already correct"
                )
            raise AssertionError(f"unexpected request: {method} {url}")

        with patch.dict(os.environ, _SMTP_ENV):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_realm_settings(
                    _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                )

        record = json.loads(capsys.readouterr().out.strip())
        assert record["op"] == "unchanged"
        assert record["clientId"] == f"realm:{_REALM}"


class TestReconcileRealmSettingsFailsClosedOnMissingSmtpEnv:
    def test_reconcile_realm_settings_missing_required_smtp_env_fails_closed(
        self,
    ) -> None:
        existing = _fresh_realm_body()
        put_called = {"n": 0}
        env = {k: v for k, v in os.environ.items() if k != "SMTP_PASSWORD"}

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "PUT":
                put_called["n"] += 1
                return (204, None)
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            return (200, None)

        smtp_env_missing_password = {
            k: v for k, v in _SMTP_ENV.items() if k != "SMTP_PASSWORD"
        }
        with patch.dict(os.environ, {**env, **smtp_env_missing_password}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                with pytest.raises(SystemExit) as exc_info:
                    _ensure_mod()._reconcile_realm_settings(
                        _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                    )
        assert exc_info.value.code != 0
        assert put_called["n"] == 0


class TestReconcileRealmSettingsPreservesExistingTrueFlags:
    def test_registration_allowed_never_flipped_off_by_partial_drift_put(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Regression: PUT must carry forward existing registrationAllowed=True
        even when only verifyEmail/smtp drifted (full-representation PUT, not
        a partial patch — Keycloak's realm PUT replaces the whole object)."""
        existing = _fresh_realm_body()
        assert existing["registrationAllowed"] is True
        put_payloads: list[dict[str, Any]] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            if method == "PUT" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                put_payloads.append(kwargs.get("payload", {}))
                return (204, None)
            raise AssertionError(f"unexpected request: {method} {url}")

        with patch.dict(os.environ, _SMTP_ENV):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_realm_settings(
                    _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                )

        assert len(put_payloads) == 1
        assert put_payloads[0]["registrationAllowed"] is True


class TestResolveRealmSmtpSettings:
    def test_resolves_all_fields_from_env(self) -> None:
        with patch.dict(os.environ, _SMTP_ENV):
            resolved = _ensure_mod()._resolve_realm_smtp_settings(_SMTP_SPEC)
        assert resolved == {
            "host": "smtp.example.com",
            "port": "587",
            "from": "noreply@omninode.ai",
            "fromDisplayName": "OmniNode",
            "starttls": "true",
            "auth": "true",
            "user": "smtp-user",
            "password": "smtp-pass",
        }

    def test_dies_on_missing_env_var(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "SMTP_HOST"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod()._resolve_realm_smtp_settings(_SMTP_SPEC)
        assert exc_info.value.code != 0
