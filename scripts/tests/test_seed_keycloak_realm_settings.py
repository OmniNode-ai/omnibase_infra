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


def _last_record(out: str) -> dict[str, Any]:
    """Parse the LAST JSON-lines record from the reconcile's stdout.

    OMN-18170 adds an additive `op=realm_smtp` record ahead of the op record,
    so the stream is now more than one line. It always was a JSON-LINES stream
    (the client loop emits one record per client); these two realm tests only
    ever saw one line because they drive the realm reconciler in isolation.
    Nothing parses this stdout in production -- seed-keycloak.sh execs the
    script and the k8s Job runs it directly -- so an added record is safe.
    """
    record: dict[str, Any] = json.loads(out.strip().splitlines()[-1])
    return record


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

        record = _last_record(capsys.readouterr().out)
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

        record = _last_record(capsys.readouterr().out)
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
        """A PARTIAL mail configuration is refused (OMN-18170).

        This test used to clear only ``SMTP_HOST`` and expect a refusal. On a
        host with no ``SMTP_*`` variables at all that cleared *every* declared
        key, so it was really asserting "absent means die" -- the behaviour
        OMN-18170 changes. It now asserts the case that genuinely must still
        refuse: some declared keys resolve and some do not.
        """
        partial = {"SMTP_USER": _SMTP_ENV["SMTP_USER"]}
        with patch.dict(os.environ, partial, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod()._resolve_realm_smtp_settings(_SMTP_SPEC)
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# OMN-18170: realm SMTP is conditional on a declared mail provider.
#
# The `.201` compose dev lane sends no email and declares none of the eight
# SMTP_* variables desired-clients.json indirects through. Before this, the
# reconcile died there -- so a lane that legitimately has no mail provider
# could not reconcile its Keycloak clients at all. Absent now means SKIP;
# PARTIAL still means refuse, naming the keys that did not resolve.
# ---------------------------------------------------------------------------


class TestRealmSmtpSkippedWhenNoMailProviderDeclared:
    def test_resolve_returns_none_when_no_declared_smtp_var_is_set(self) -> None:
        """Absent is a skip signal, not a failure -- and never a default."""
        with patch.dict(os.environ, {}, clear=True):
            assert _ensure_mod()._resolve_realm_smtp_settings(_SMTP_SPEC) is None

    def test_resolve_returns_none_when_the_roster_declares_no_smtp_block(self) -> None:
        with patch.dict(os.environ, _SMTP_ENV, clear=True):
            assert _ensure_mod()._resolve_realm_smtp_settings({}) is None

    def test_reconcile_skips_smtp_and_records_not_configured(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The reconcile completes, and says in its own output that it skipped."""
        existing = _fresh_realm_body()
        put_payloads: list[dict[str, Any]] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            if method == "PUT" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                put_payloads.append(kwargs.get("payload", {}))
                return (204, None)
            raise AssertionError(f"unexpected request: {method} {url}")

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_realm_settings(
                    _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                )

        records = [
            json.loads(line)
            for line in capsys.readouterr().out.strip().splitlines()
            if line.strip()
        ]
        smtp_records = [r for r in records if "smtp" in r]
        assert len(smtp_records) == 1, records
        assert smtp_records[0]["smtp"] == "not_configured"
        assert smtp_records[0]["clientId"] == f"realm:{_REALM}"

        # The non-SMTP half of the realm still reconciles.
        assert len(put_payloads) == 1
        assert put_payloads[0]["verifyEmail"] is True

    def test_existing_realm_smtp_is_carried_forward_never_blanked(self) -> None:
        """Skipping must leave Keycloak's own SMTP state exactly as it was.

        The hazard this pins: a lane with no mail provider reconciling a realm
        that DOES have SMTP configured (a shared realm, or a previously
        configured one) must not erase it as a side effect of some other
        field drifting.
        """
        existing = _fresh_realm_body()
        existing["smtpServer"] = {
            "host": "smtp.pre-existing.example",
            "from": "someone@pre-existing.example",
            "auth": "true",
        }
        pre_image = dict(existing["smtpServer"])
        put_payloads: list[dict[str, Any]] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            if method == "PUT" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                put_payloads.append(kwargs.get("payload", {}))
                return (204, None)
            raise AssertionError(f"unexpected request: {method} {url}")

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_realm_settings(
                    _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                )

        assert len(put_payloads) == 1
        assert put_payloads[0]["smtpServer"] == pre_image
        assert "smtpServer" not in put_payloads[0].get("fields_changed", [])


class TestRealmSmtpRefusedWhenPartiallyConfigured:
    def test_partial_smtp_refuses_and_names_every_missing_key(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Half a mail configuration is a misconfiguration, not a skip.

        Positive control for the skip test above: the same code path that
        returns None on a fully-absent lane must still die here, and the
        refusal must name the keys an operator has to provision.
        """
        partial = {
            "SMTP_HOST": _SMTP_ENV["SMTP_HOST"],
            "SMTP_PORT": _SMTP_ENV["SMTP_PORT"],
        }
        with patch.dict(os.environ, partial, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod()._resolve_realm_smtp_settings(_SMTP_SPEC)
        assert exc_info.value.code != 0

        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert set(record["keys"]) == {
            "SMTP_FROM",
            "SMTP_FROM_DISPLAY_NAME",
            "SMTP_STARTTLS",
            "SMTP_AUTH",
            "SMTP_USER",
            "SMTP_PASSWORD",
        }
        assert record["site"] == "_resolve_realm_smtp_settings"

    def test_host_absent_while_other_fields_present_is_refused_not_skipped(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No mail host, but leftover credentials -- the dangerous shape.

        Treating this as "not configured" would silently ignore a real,
        half-applied configuration. It is refused, and SMTP_HOST is named.
        """
        leftovers = {
            "SMTP_USER": _SMTP_ENV["SMTP_USER"],
            "SMTP_PASSWORD": _SMTP_ENV["SMTP_PASSWORD"],
        }
        with patch.dict(os.environ, leftovers, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod()._resolve_realm_smtp_settings(_SMTP_SPEC)
        assert exc_info.value.code != 0

        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert "SMTP_HOST" in record["keys"]

    def test_reconcile_refuses_partial_without_issuing_a_put(self) -> None:
        existing = _fresh_realm_body()
        put_called = {"n": 0}

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "PUT":
                put_called["n"] += 1
                return (204, None)
            if method == "GET" and url == f"{_KC_URL}/admin/realms/{_REALM}":
                return (200, existing)
            return (200, None)

        with patch.dict(os.environ, {"SMTP_HOST": "smtp.example.com"}, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                with pytest.raises(SystemExit) as exc_info:
                    _ensure_mod()._reconcile_realm_settings(
                        _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                    )
        assert exc_info.value.code != 0
        assert put_called["n"] == 0


class TestRealmSmtpStagingBehaviourUnchanged:
    """Golden: where a mail provider IS declared, nothing about this changes."""

    def test_fully_declared_smtp_resolves_exactly_as_before(self) -> None:
        with patch.dict(os.environ, _SMTP_ENV, clear=True):
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

    def test_reconcile_records_smtp_configured_and_applies_it(
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

        with patch.dict(os.environ, _SMTP_ENV, clear=True):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_realm_settings(
                    _KC_URL, _REALM, _TOKEN, _REALM_SETTINGS_SPEC
                )

        records = [
            json.loads(line)
            for line in capsys.readouterr().out.strip().splitlines()
            if line.strip()
        ]
        smtp_records = [r for r in records if "smtp" in r]
        assert len(smtp_records) == 1
        assert smtp_records[0]["smtp"] == "configured"

        assert len(put_payloads) == 1
        assert put_payloads[0]["smtpServer"]["host"] == "smtp.example.com"
        assert put_payloads[0]["smtpServer"]["password"] == "smtp-pass"


# ---------------------------------------------------------------------------
# OMN-18170 residual 2: the failure helper discarded its failure site.
#
# Every failure printed one identical redacted line, so an operator could not
# tell a provisioning gap (an unset env var) from a credential failure (a
# wrong password) without re-deriving the failure by hand.
# ---------------------------------------------------------------------------


class TestDieReportsTheFailureSite:
    def test_die_names_the_calling_function(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mod = _ensure_mod()

        def a_caller_with_a_distinctive_name() -> None:
            mod._die("free text that must never be printed")

        with pytest.raises(SystemExit):
            a_caller_with_a_distinctive_name()

        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert record["site"] == "a_caller_with_a_distinctive_name"
        assert isinstance(record["line"], int)

    def test_die_names_the_keys_but_never_the_free_text_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The redaction guarantee is unchanged: the message is still dropped.

        Only the SITE and the KEY NAMES are added. A key name is an env var
        name, never its value -- that is what makes this safe to print.
        """
        mod = _ensure_mod()
        with pytest.raises(SystemExit):
            mod._die(
                "a-secret-looking-string-that-must-not-appear",
                keys=["SMTP_PASSWORD", "KEYCLOAK_SIGNUP_CLIENT_SECRET"],
                client_id="omniweb-signup-admin",
            )

        err = capsys.readouterr().err
        assert "a-secret-looking-string-that-must-not-appear" not in err
        record = json.loads(err.strip().splitlines()[-1])
        assert record["message"] == "seed-keycloak-clients failed; details redacted"
        assert record["keys"] == ["SMTP_PASSWORD", "KEYCLOAK_SIGNUP_CLIENT_SECRET"]
        assert record["clientId"] == "omniweb-signup-admin"

    def test_missing_client_secret_names_the_client_and_the_env_var(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The exact gap this lane hit on `.201`, made legible.

        `omniweb-signup-admin` declares secretEnv KEYCLOAK_SIGNUP_CLIENT_SECRET
        and that variable is not provisioned on the dev lane. Before this the
        run printed "details redacted" and nothing else.
        """
        spec = {
            "clientId": "omniweb-signup-admin",
            "secretEnv": "KEYCLOAK_SIGNUP_CLIENT_SECRET",
        }
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                _ensure_mod()._resolve_client_secret(spec)

        record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert record["site"] == "_resolve_client_secret"
        assert record["clientId"] == "omniweb-signup-admin"
        assert record["keys"] == ["KEYCLOAK_SIGNUP_CLIENT_SECRET"]
