#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for seed-keycloak-clients.py reconciler."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Load script as module (it lives in scripts/, not in src/)
# The module is loaded lazily (first test access) so that coverage.py has
# already started tracing before exec_module runs.
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "seed-keycloak-clients.py"
_mod: types.ModuleType  # assigned by _ensure_mod() on first use


def _ensure_mod() -> types.ModuleType:
    global _mod
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
def _load_module() -> None:  # noqa: PT004
    _ensure_mod()


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_REALM = "omninode"
_KC_URL = "http://localhost:28080"
_TOKEN = "test-access-token"


def _client_list_response(clients: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    return (200, clients)


def _empty_client_list() -> tuple[int, list[dict[str, Any]]]:
    return (200, [])


def _ok(body: Any = None) -> tuple[int, Any]:
    return (201, body)


def _no_content() -> tuple[int, None]:
    return (204, None)


def _scopes_response(names: list[str]) -> tuple[int, list[dict[str, Any]]]:
    return (200, [{"id": f"scope-id-{n}", "name": n} for n in names])


def _mappers_response(names: list[str]) -> tuple[int, list[dict[str, Any]]]:
    return (200, [{"name": n} for n in names])


DESIRED_SCOPES = ["basic", "web-origins", "roles", "profile", "email"]


def _make_existing_client(client_id: str, **kwargs: Any) -> dict[str, Any]:
    base = {
        "id": f"internal-{client_id}",
        "clientId": client_id,
        "enabled": True,
        "publicClient": False,
        "serviceAccountsEnabled": False,
        "standardFlowEnabled": True,
        "directAccessGrantsEnabled": False,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Tests: idempotency (all unchanged)
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_client_already_correct_reports_unchanged(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        existing = _make_existing_client(
            "omniweb",
            publicClient=True,
            standardFlowEnabled=True,
            directAccessGrantsEnabled=False,
            serviceAccountsEnabled=False,
            redirectUris=["https://app.omninode.ai/*", "http://localhost:3000/*"],
        )
        spec = {
            "clientId": "omniweb",
            "publicClient": True,
            "standardFlowEnabled": True,
            "directAccessGrantsEnabled": False,
            "serviceAccountsEnabled": False,
            "redirectUris": ["https://app.omninode.ai/*", "http://localhost:3000/*"],
            "defaultClientScopes": DESIRED_SCOPES,
        }

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId" in url:
                return _client_list_response([existing])
            if method == "GET" and "default-client-scopes" in url:
                # all scopes already present
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            _ensure_mod()._reconcile_client(_KC_URL, _REALM, _TOKEN, spec)

        out = capsys.readouterr().out
        record = json.loads(out.strip())
        assert record["op"] == "unchanged"
        assert record["clientId"] == "omniweb"

    def test_second_run_all_unchanged(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Running reconciler twice in a row: second run is all unchanged."""
        existing = _make_existing_client("onex-api", bearerOnly=True)
        spec = {
            "clientId": "onex-api",
            "publicClient": False,
            "bearerOnly": True,
            "defaultClientScopes": DESIRED_SCOPES,
        }

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId" in url:
                return _client_list_response([existing])
            if method == "GET" and "default-client-scopes" in url:
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            return (200, None)

        for _ in range(2):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_client(_KC_URL, _REALM, _TOKEN, spec)

        lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
        assert all(rec["op"] == "unchanged" for rec in lines)


# ---------------------------------------------------------------------------
# Tests: missing secret env error
# ---------------------------------------------------------------------------


class TestMissingSecretEnv:
    def test_exits_nonzero_when_secret_env_missing(self) -> None:
        spec = {
            "clientId": "onex-admin",
            "secretEnv": "KEYCLOAK_ADMIN_CLIENT_SECRET_MISSING_IN_TEST",
            "serviceAccountsEnabled": True,
        }
        env = {k: v for k, v in os.environ.items() if k != "KEYCLOAK_ADMIN_CLIENT_SECRET_MISSING_IN_TEST"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod()._resolve_client_secret(spec)
        assert exc_info.value.code != 0

    def test_secret_resolved_when_env_set(self) -> None:
        spec = {
            "clientId": "onex-admin",
            "secretEnv": "KEYCLOAK_ADMIN_CLIENT_SECRET_TEST_VAR",
        }
        with patch.dict(os.environ, {"KEYCLOAK_ADMIN_CLIENT_SECRET_TEST_VAR": "mysecret"}):
            result = _ensure_mod()._resolve_client_secret(spec)
        assert result == "mysecret"

    def test_no_secret_env_returns_none(self) -> None:
        spec = {"clientId": "omniweb"}
        result = _ensure_mod()._resolve_client_secret(spec)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: scope binding only when missing
# ---------------------------------------------------------------------------


class TestScopeBinding:
    def test_missing_scope_is_added(self) -> None:
        all_scope_names = DESIRED_SCOPES
        # client currently has all except "basic"
        current_scope_names = [s for s in DESIRED_SCOPES if s != "basic"]
        calls_made: list[str] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [{"name": n} for n in current_scope_names])
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(all_scope_names)
            if method == "PUT" and "default-client-scopes" in url:
                calls_made.append(url)
                return _no_content()
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            changed = _ensure_mod()._ensure_default_scopes(
                _KC_URL, _REALM, _TOKEN, "internal-omniweb", DESIRED_SCOPES
            )

        assert "defaultClientScope:basic" in changed
        assert len(calls_made) == 1  # only "basic" was PUT

    def test_already_bound_scopes_not_re_added(self) -> None:
        calls_made: list[str] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            if method == "PUT" and "default-client-scopes" in url:
                calls_made.append(url)
                return _no_content()
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            changed = _ensure_mod()._ensure_default_scopes(
                _KC_URL, _REALM, _TOKEN, "internal-omniweb", DESIRED_SCOPES
            )

        assert changed == []
        assert calls_made == []


# ---------------------------------------------------------------------------
# Tests: audience mapper creation
# ---------------------------------------------------------------------------


class TestAudienceMapper:
    def test_mapper_created_when_absent(self) -> None:
        mapper_spec = [
            {
                "name": "onex-api-audience",
                "protocolMapper": "oidc-audience-mapper",
                "config": {
                    "included.client.audience": "onex-api",
                    "id.token.claim": "false",
                    "access.token.claim": "true",
                },
            }
        ]
        post_calls: list[dict[str, Any]] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "protocol-mappers" in url:
                return _mappers_response([])  # no mappers exist yet
            if method == "POST" and "protocol-mappers" in url:
                post_calls.append(kwargs.get("payload", {}))
                return _ok()
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            changed = _ensure_mod()._ensure_protocol_mappers(
                _KC_URL, _REALM, _TOKEN, "internal-omniweb", mapper_spec
            )

        assert "protocolMapper:onex-api-audience" in changed
        assert len(post_calls) == 1
        assert post_calls[0]["name"] == "onex-api-audience"
        assert post_calls[0]["protocolMapper"] == "oidc-audience-mapper"

    def test_mapper_not_recreated_when_present(self) -> None:
        mapper_spec = [
            {
                "name": "onex-api-audience",
                "protocolMapper": "oidc-audience-mapper",
                "config": {},
            }
        ]

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "protocol-mappers" in url:
                return _mappers_response(["onex-api-audience"])
            return (200, None)

        post_called = False
        original_request = _ensure_mod()._request

        def tracking_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            nonlocal post_called
            if method == "POST" and "protocol-mappers" in url:
                post_called = True
            return fake_request(method, url, **kwargs)

        with patch.object(_ensure_mod(), "_request", side_effect=tracking_request):
            changed = _ensure_mod()._ensure_protocol_mappers(
                _KC_URL, _REALM, _TOKEN, "internal-omniweb", mapper_spec
            )

        assert changed == []
        assert not post_called


# ---------------------------------------------------------------------------
# Tests: role mapping for onex-admin
# ---------------------------------------------------------------------------


class TestRoleMapping:
    def test_realm_roles_assigned_to_service_account(self) -> None:
        realm_mgmt_client = {"id": "realm-mgmt-id", "clientId": "realm-management"}
        sa_user = {"id": "sa-user-id"}
        role_obj = {"id": "role-id-manage-users", "name": "manage-users"}
        post_calls: list[Any] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId=realm-management" in url:
                return (200, [realm_mgmt_client])
            if method == "GET" and "service-account-user" in url:
                return (200, sa_user)
            if method == "GET" and "role-mappings/clients" in url:
                return (200, [])  # no roles assigned yet
            if method == "GET" and "roles/manage-users" in url:
                return (200, role_obj)
            if method == "POST" and "role-mappings" in url:
                post_calls.append(kwargs.get("payload", []))
                return _no_content()
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            changed = _ensure_mod()._ensure_realm_roles(
                _KC_URL,
                _REALM,
                _TOKEN,
                "internal-onex-admin",
                ["realm-management:manage-users"],
            )

        assert "realmRole:manage-users" in changed
        assert len(post_calls) == 1
        assert post_calls[0][0]["name"] == "manage-users"

    def test_already_assigned_roles_not_reassigned(self) -> None:
        realm_mgmt_client = {"id": "realm-mgmt-id", "clientId": "realm-management"}
        sa_user = {"id": "sa-user-id"}
        post_calls: list[Any] = []

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId=realm-management" in url:
                return (200, [realm_mgmt_client])
            if method == "GET" and "service-account-user" in url:
                return (200, sa_user)
            if method == "GET" and "role-mappings/clients" in url:
                return (200, [{"name": "manage-users"}])  # already assigned
            if method == "POST" and "role-mappings" in url:
                post_calls.append(kwargs.get("payload", []))
                return _no_content()
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            changed = _ensure_mod()._ensure_realm_roles(
                _KC_URL,
                _REALM,
                _TOKEN,
                "internal-onex-admin",
                ["realm-management:manage-users"],
            )

        assert changed == []
        assert post_calls == []


# ---------------------------------------------------------------------------
# Tests: client creation flow
# ---------------------------------------------------------------------------


class TestClientCreation:
    def test_missing_client_is_created(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        spec = {
            "clientId": "onex-service",
            "publicClient": False,
            "serviceAccountsEnabled": True,
            "standardFlowEnabled": False,
            "directAccessGrantsEnabled": False,
            "secretEnv": "ONEX_SERVICE_CLIENT_SECRET_TEST",
            "defaultClientScopes": DESIRED_SCOPES,
            "protocolMappers": [
                {
                    "name": "onex-api-audience",
                    "protocolMapper": "oidc-audience-mapper",
                    "config": {"included.client.audience": "onex-api"},
                }
            ],
        }
        # Re-fetched client matches spec exactly — no drift, so op=created not updated
        existing_client = _make_existing_client(
            "onex-service",
            publicClient=False,
            serviceAccountsEnabled=True,
            standardFlowEnabled=False,
            directAccessGrantsEnabled=False,
            redirectUris=[],
        )
        fetch_count = {"n": 0}

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId" in url:
                fetch_count["n"] += 1
                if fetch_count["n"] == 1:
                    return _empty_client_list()  # not yet created
                return _client_list_response([existing_client])
            if method == "POST" and "/clients" in url and "protocol-mappers" not in url:
                return _ok()
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            if method == "GET" and "protocol-mappers" in url:
                return _mappers_response([])
            if method == "POST" and "protocol-mappers" in url:
                return _ok()
            return (200, None)

        with patch.dict(os.environ, {"ONEX_SERVICE_CLIENT_SECRET_TEST": "svc-secret"}):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                _ensure_mod()._reconcile_client(_KC_URL, _REALM, _TOKEN, spec)

        out = capsys.readouterr().out
        record = json.loads(out.strip())
        assert record["op"] == "created"
        assert record["clientId"] == "onex-service"

    def test_full_reconcile_loop_all_clients(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Smoke: all 5 desired clients reconcile without error."""
        clients_spec = [
            {"clientId": "omniweb", "publicClient": True, "standardFlowEnabled": True, "directAccessGrantsEnabled": False, "serviceAccountsEnabled": False, "defaultClientScopes": DESIRED_SCOPES},
            {"clientId": "onex-api", "publicClient": False, "bearerOnly": True, "standardFlowEnabled": True, "directAccessGrantsEnabled": False, "serviceAccountsEnabled": False, "defaultClientScopes": DESIRED_SCOPES},
            {"clientId": "redpanda-events", "publicClient": False, "serviceAccountsEnabled": True, "standardFlowEnabled": True, "directAccessGrantsEnabled": False, "defaultClientScopes": DESIRED_SCOPES},
            {"clientId": "onex-admin", "publicClient": False, "serviceAccountsEnabled": True, "standardFlowEnabled": False, "directAccessGrantsEnabled": False, "secretEnv": "KC_ADMIN_CLIENT_SECRET_T", "realmRoles": ["realm-management:manage-users"], "defaultClientScopes": DESIRED_SCOPES},
            {"clientId": "onex-service", "publicClient": False, "serviceAccountsEnabled": True, "standardFlowEnabled": False, "directAccessGrantsEnabled": False, "secretEnv": "ONEX_SVC_SECRET_T", "defaultClientScopes": DESIRED_SCOPES},
        ]
        # Build existing-client map that exactly matches each spec so no drift fires
        existing_by_id = {
            spec["clientId"]: _make_existing_client(
                spec["clientId"],
                publicClient=spec.get("publicClient", False),
                bearerOnly=spec.get("bearerOnly", False),
                serviceAccountsEnabled=spec.get("serviceAccountsEnabled", False),
                standardFlowEnabled=spec.get("standardFlowEnabled", True),
                directAccessGrantsEnabled=spec.get("directAccessGrantsEnabled", False),
            )
            for spec in clients_spec
        }
        realm_mgmt = {"id": "rm-id", "clientId": "realm-management"}
        sa_user = {"id": "sa-id"}

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId=realm-management" in url:
                return (200, [realm_mgmt])
            if method == "GET" and "clients?clientId" in url:
                client_id = url.split("clientId=")[-1]
                existing = existing_by_id.get(client_id)
                if existing:
                    return _client_list_response([existing])
                return _empty_client_list()
            if method == "GET" and "service-account-user" in url:
                return (200, sa_user)
            if method == "GET" and "role-mappings/clients" in url:
                return (200, [{"name": "manage-users"}])  # already assigned
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            if method == "GET" and "protocol-mappers" in url:
                return _mappers_response([])
            return (200, None)

        env_patch = {
            "KC_ADMIN_CLIENT_SECRET_T": "admin-secret",
            "ONEX_SVC_SECRET_T": "svc-secret",
        }
        with patch.dict(os.environ, env_patch):
            with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                for spec in clients_spec:
                    _ensure_mod()._reconcile_client(_KC_URL, _REALM, _TOKEN, spec)

        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 5
        ops = {json.loads(l)["op"] for l in lines}
        # All should be unchanged (existing clients with correct state)
        assert ops == {"unchanged"}


# ---------------------------------------------------------------------------
# Tests: _get_token
# ---------------------------------------------------------------------------


class TestGetToken:
    def test_returns_access_token_on_success(self) -> None:
        token_response = json.dumps({"access_token": "my-token-abc"}).encode()

        class FakeResponse:
            status = 200

            def read(self) -> bytes:
                return token_response

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = _ensure_mod()._get_token(_KC_URL, "admin", "password")
        assert result == "my-token-abc"

    def test_dies_on_http_error(self) -> None:
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        exc.read = lambda: b'{"error": "unauthorized"}'  # type: ignore[method-assign]

        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod()._get_token(_KC_URL, "admin", "wrong")
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Tests: _die and _log
# ---------------------------------------------------------------------------


class TestLogAndDie:
    def test_die_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            _ensure_mod()._die("something went wrong")
        assert exc_info.value.code != 0

    def test_log_unchanged(self, capsys: pytest.CaptureFixture[str]) -> None:
        _ensure_mod()._log("unchanged", "my-client")
        out = json.loads(capsys.readouterr().out.strip())
        assert out == {"op": "unchanged", "clientId": "my-client"}

    def test_log_updated_with_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        _ensure_mod()._log("updated", "my-client", ["defaultClientScopes"])
        out = json.loads(capsys.readouterr().out.strip())
        assert out["op"] == "updated"
        assert "defaultClientScopes" in out["fields_changed"]


# ---------------------------------------------------------------------------
# Tests: main() argument validation
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_dies_without_kc_url(self) -> None:
        with patch("sys.argv", ["seed-keycloak-clients.py"]):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SystemExit) as exc_info:
                    _ensure_mod().main()
        assert exc_info.value.code != 0

    def test_main_dies_without_admin_password(self) -> None:
        with patch("sys.argv", ["seed-keycloak-clients.py", "--kc-url", "http://localhost:28080"]):
            with patch.dict(os.environ, {"KC_URL": "http://localhost:28080"}, clear=False):
                with pytest.raises(SystemExit) as exc_info:
                    _ensure_mod().main()
        assert exc_info.value.code != 0

    def test_main_dies_without_config(self) -> None:
        with patch("sys.argv", [
            "seed-keycloak-clients.py",
            "--kc-url", "http://localhost:28080",
            "--admin-password", "secret",
        ]):
            with patch.dict(os.environ, {}, clear=False):
                with pytest.raises(SystemExit) as exc_info:
                    _ensure_mod().main()
        assert exc_info.value.code != 0

    def test_main_dies_on_missing_config_file(self, tmp_path: pytest.TempPathFactory) -> None:
        with patch("sys.argv", [
            "seed-keycloak-clients.py",
            "--kc-url", "http://localhost:28080",
            "--admin-password", "secret",
            "--config", "/nonexistent/path/desired-clients.json",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod().main()
        assert exc_info.value.code != 0

    def test_main_runs_reconciler_from_config_file(
        self, tmp_path: "pytest.TempPathFactory", capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = {
            "realm": "omninode",
            "clients": [
                {"clientId": "omniweb", "publicClient": True, "defaultClientScopes": DESIRED_SCOPES},
            ],
        }
        config_file = tmp_path / "desired-clients.json"  # type: ignore[operator]
        config_file.write_text(json.dumps(config))

        existing = _make_existing_client("omniweb", publicClient=True)

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if "token" in url:
                return (200, {"access_token": "tok"})
            if method == "GET" and "clients?clientId" in url:
                return _client_list_response([existing])
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            return (200, None)

        with patch("sys.argv", [
            "seed-keycloak-clients.py",
            "--kc-url", _KC_URL,
            "--admin-username", "admin",
            "--admin-password", "secret",
            "--config", str(config_file),
        ]):
            with patch.object(_ensure_mod(), "_get_token", return_value="tok"):
                with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                    _ensure_mod().main()

        out = capsys.readouterr().out
        record = json.loads(out.strip())
        assert record["clientId"] == "omniweb"
        assert record["op"] == "unchanged"


# ---------------------------------------------------------------------------
# Tests: error paths in helper functions
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_ensure_protocol_mappers_fails_on_http_error(self) -> None:
        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET":
                return (200, [])  # no existing mappers
            return (500, {"error": "server error"})  # POST fails

        mapper_spec = [{"name": "bad-mapper", "protocolMapper": "oidc-audience-mapper", "config": {}}]
        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            with pytest.raises(SystemExit):
                _ensure_mod()._ensure_protocol_mappers(
                    _KC_URL, _REALM, _TOKEN, "internal-id", mapper_spec
                )

    def test_ensure_default_scopes_fails_when_scope_not_found(self) -> None:
        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [])  # no scopes bound
            if method == "GET" and "/client-scopes" in url:
                return (200, [])  # scope doesn't exist in realm
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            with pytest.raises(SystemExit):
                _ensure_mod()._ensure_default_scopes(
                    _KC_URL, _REALM, _TOKEN, "internal-id", ["nonexistent-scope"]
                )

    def test_ensure_realm_roles_fails_when_realm_mgmt_missing(self) -> None:
        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId=realm-management" in url:
                return (200, [])  # realm-management not found
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            with pytest.raises(SystemExit):
                _ensure_mod()._ensure_realm_roles(
                    _KC_URL, _REALM, _TOKEN, "internal-id", ["realm-management:manage-users"]
                )

    def test_ensure_default_scopes_fails_on_put_error(self) -> None:
        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [])  # no scopes bound yet
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(["basic"])
            if method == "PUT" and "default-client-scopes" in url:
                return (500, {"error": "server error"})
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            with pytest.raises(SystemExit):
                _ensure_mod()._ensure_default_scopes(
                    _KC_URL, _REALM, _TOKEN, "internal-id", ["basic"]
                )

    def test_ensure_realm_roles_fails_on_post_error(self) -> None:
        realm_mgmt = {"id": "rm-id", "clientId": "realm-management"}
        sa_user = {"id": "sa-id"}
        role_obj = {"id": "r-id", "name": "manage-users"}

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId=realm-management" in url:
                return (200, [realm_mgmt])
            if method == "GET" and "service-account-user" in url:
                return (200, sa_user)
            if method == "GET" and "role-mappings/clients" in url:
                return (200, [])  # nothing assigned yet
            if method == "GET" and "roles/manage-users" in url:
                return (200, role_obj)
            if method == "POST" and "role-mappings" in url:
                return (500, {"error": "server error"})
            return (200, None)

        with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
            with pytest.raises(SystemExit):
                _ensure_mod()._ensure_realm_roles(
                    _KC_URL, _REALM, _TOKEN, "internal-id",
                    ["realm-management:manage-users"]
                )


# ---------------------------------------------------------------------------
# Tests: _request function (urllib path)
# ---------------------------------------------------------------------------


class TestRequest:
    def test_request_success(self) -> None:
        response_body = json.dumps({"id": "abc"}).encode()

        class FakeResp:
            status = 200

            def read(self) -> bytes:
                return response_body

            def __enter__(self) -> "FakeResp":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            status, body = _ensure_mod()._request("GET", "http://localhost/test", token="tok")
        assert status == 200
        assert body == {"id": "abc"}

    def test_request_http_error_with_json_body(self) -> None:
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        exc.read = lambda: b'{"error": "not found"}'  # type: ignore[method-assign]

        with patch("urllib.request.urlopen", side_effect=exc):
            status, body = _ensure_mod()._request("GET", "http://localhost/test", token="tok")
        assert status == 404
        assert body == {"error": "not found"}

    def test_request_http_error_with_non_json_body(self) -> None:
        import urllib.error

        exc = urllib.error.HTTPError(
            url="http://x", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        exc.read = lambda: b"Internal Server Error"  # type: ignore[method-assign]

        with patch("urllib.request.urlopen", side_effect=exc):
            status, body = _ensure_mod()._request("GET", "http://localhost/test")
        assert status == 500
        assert body == "Internal Server Error"


# ---------------------------------------------------------------------------
# Tests: _reset_bootstrap_admin
# ---------------------------------------------------------------------------


class TestResetBootstrapAdmin:
    def test_skips_non_localhost(self) -> None:
        with patch("subprocess.run") as mock_run:
            _ensure_mod()._reset_bootstrap_admin("https://keycloak.prod.example.com")
        mock_run.assert_not_called()

    def test_runs_docker_exec_on_localhost(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _ensure_mod()._reset_bootstrap_admin("http://localhost:28080")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "docker" in cmd
        assert "bootstrap-admin" in cmd

    def test_prints_warning_on_nonzero_exit(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some unexpected error"

        with patch("subprocess.run", return_value=mock_result):
            _ensure_mod()._reset_bootstrap_admin("http://localhost:28080")

        out = capsys.readouterr().out
        record = json.loads(out.strip())
        assert record["op"] == "warning"

    def test_no_warning_when_already_exists(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "User already exists"

        with patch("subprocess.run", return_value=mock_result):
            _ensure_mod()._reset_bootstrap_admin("http://localhost:28080")

        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# Tests: main() with --reset-bootstrap-admin and empty-clients guard
# ---------------------------------------------------------------------------


class TestMainExtended:
    def test_main_dies_on_empty_clients_list(self, tmp_path: "pytest.TempPathFactory") -> None:
        config = {"realm": "omninode", "clients": []}
        config_file = tmp_path / "empty-clients.json"  # type: ignore[operator]
        config_file.write_text(json.dumps(config))

        with patch("sys.argv", [
            "seed-keycloak-clients.py",
            "--kc-url", _KC_URL,
            "--admin-password", "secret",
            "--config", str(config_file),
        ]):
            with pytest.raises(SystemExit) as exc_info:
                _ensure_mod().main()
        assert exc_info.value.code != 0

    def test_main_calls_reset_bootstrap_admin_when_flag_set(
        self, tmp_path: "pytest.TempPathFactory", capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = {
            "realm": "omninode",
            "clients": [
                {"clientId": "omniweb", "publicClient": True, "defaultClientScopes": DESIRED_SCOPES},
            ],
        }
        config_file = tmp_path / "desired-clients.json"  # type: ignore[operator]
        config_file.write_text(json.dumps(config))
        existing = _make_existing_client("omniweb", publicClient=True)

        def fake_request(method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
            if method == "GET" and "clients?clientId" in url:
                return _client_list_response([existing])
            if method == "GET" and "default-client-scopes" in url and "clients/" in url:
                return (200, [{"name": n} for n in DESIRED_SCOPES])
            if method == "GET" and "/client-scopes" in url:
                return _scopes_response(DESIRED_SCOPES)
            return (200, None)

        with patch("sys.argv", [
            "seed-keycloak-clients.py",
            "--kc-url", _KC_URL,
            "--admin-password", "secret",
            "--config", str(config_file),
            "--reset-bootstrap-admin",
        ]):
            with patch.object(_ensure_mod(), "_get_token", return_value="tok"):
                with patch.object(_ensure_mod(), "_request", side_effect=fake_request):
                    with patch.object(_ensure_mod(), "_reset_bootstrap_admin") as mock_reset:
                        _ensure_mod().main()
        mock_reset.assert_called_once_with(_KC_URL)
