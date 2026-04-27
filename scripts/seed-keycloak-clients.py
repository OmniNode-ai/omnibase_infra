#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reconcile Keycloak clients from desired-clients.json against a running realm.

Usage:
    python seed-keycloak-clients.py \\
        --kc-url http://localhost:28080 \\
        --realm omninode \\
        --admin-username admin \\
        --admin-password "$KEYCLOAK_ADMIN_PASSWORD" \\
        --config docker/keycloak/desired-clients.json

Env-var equivalents (all CLI flags can be omitted when the env vars are set):
    KC_URL, KC_REALM, KC_ADMIN_USERNAME, KC_ADMIN_PASSWORD, KC_CONFIG

Idempotent: re-running against an already-correct realm produces all
op=unchanged lines and exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, NoReturn


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def _request(
    method: str,
    url: str,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Perform an HTTP request; return (status_code, parsed_json_or_None)."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            body = resp.read()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body.decode(errors="replace")
        return exc.code, parsed


def _get_token(kc_url: str, username: str, password: str) -> str:
    token_url = f"{kc_url}/realms/master/protocol/openid-connect/token"
    data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": username,
            "password": password,
        }
    ).encode()
    req = urllib.request.Request(
        token_url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            body = json.loads(resp.read())
            return str(body["access_token"])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        _die(f"Failed to obtain admin token from {token_url}: {exc.code} {body}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(op: str, client_id: str, fields_changed: list[str] | None = None) -> None:
    record: dict[str, Any] = {"op": op, "clientId": client_id}
    if fields_changed is not None:
        record["fields_changed"] = fields_changed
    print(json.dumps(record), flush=True)


def _die(msg: str) -> NoReturn:
    print(json.dumps({"op": "error", "message": msg}), file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Keycloak admin helpers
# ---------------------------------------------------------------------------

BASE_FIELDS = {
    "publicClient",
    "bearerOnly",
    "serviceAccountsEnabled",
    "standardFlowEnabled",
    "directAccessGrantsEnabled",
    "redirectUris",
    "defaultClientScopes",
}


def _resolve_client_secret(client_spec: dict[str, Any]) -> str | None:
    secret_env = client_spec.get("secretEnv")
    if not secret_env:
        return None
    val = os.environ.get(secret_env)
    if not val:
        _die(
            f"Client '{client_spec['clientId']}' requires env var '{secret_env}' "
            f"but it is not set or empty."
        )
    return val


def _get_existing_client(
    kc_url: str, realm: str, token: str, client_id: str
) -> dict[str, Any] | None:
    url = f"{kc_url}/admin/realms/{realm}/clients?clientId={urllib.parse.quote(client_id)}"
    status, body = _request("GET", url, token=token)
    if status != 200 or not body:
        return None
    matches = [c for c in body if c.get("clientId") == client_id]
    return matches[0] if matches else None


def _build_create_payload(
    spec: dict[str, Any], secret: str | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "clientId": spec["clientId"],
        "enabled": True,
    }
    for field in (
        "publicClient",
        "bearerOnly",
        "serviceAccountsEnabled",
        "standardFlowEnabled",
        "directAccessGrantsEnabled",
        "redirectUris",
    ):
        if field in spec:
            payload[field] = spec[field]
    # defaultClientScopes handled post-create
    if secret is not None:
        payload["secret"] = secret
    return payload


def _ensure_protocol_mappers(
    kc_url: str,
    realm: str,
    token: str,
    internal_id: str,
    mappers_spec: list[dict[str, Any]],
) -> list[str]:
    url = f"{kc_url}/admin/realms/{realm}/clients/{internal_id}/protocol-mappers/models"
    status, existing = _request("GET", url, token=token)
    existing_names = {m["name"] for m in (existing or [])}
    changed = []
    for mapper in mappers_spec:
        if mapper["name"] in existing_names:
            continue
        full_mapper = {
            "name": mapper["name"],
            "protocol": "openid-connect",
            "protocolMapper": mapper["protocolMapper"],
            "config": mapper.get("config", {}),
        }
        status, _ = _request("POST", url, token=token, payload=full_mapper)
        if status not in (200, 201):
            _die(
                f"Failed to create protocol mapper '{mapper['name']}': HTTP {status}"
            )
        changed.append(f"protocolMapper:{mapper['name']}")
    return changed


def _resolve_scope_id(
    kc_url: str, realm: str, token: str, scope_name: str
) -> str | None:
    url = f"{kc_url}/admin/realms/{realm}/client-scopes"
    status, scopes = _request("GET", url, token=token)
    if status != 200 or not scopes:
        return None
    for s in scopes:
        if s.get("name") == scope_name:
            return str(s["id"])
    return None


def _ensure_default_scopes(
    kc_url: str,
    realm: str,
    token: str,
    internal_id: str,
    desired_scope_names: list[str],
) -> list[str]:
    current_url = (
        f"{kc_url}/admin/realms/{realm}/clients/{internal_id}/default-client-scopes"
    )
    status, current_scopes = _request("GET", current_url, token=token)
    current_names = {s["name"] for s in (current_scopes or [])}
    changed = []
    for scope_name in desired_scope_names:
        if scope_name in current_names:
            continue
        scope_id = _resolve_scope_id(kc_url, realm, token, scope_name)
        if scope_id is None:
            _die(f"Client scope '{scope_name}' not found in realm '{realm}'")
        put_url = (
            f"{kc_url}/admin/realms/{realm}/clients/{internal_id}"
            f"/default-client-scopes/{scope_id}"
        )
        status, _ = _request("PUT", put_url, token=token)
        if status not in (200, 201, 204):
            _die(
                f"Failed to bind scope '{scope_name}' to client: HTTP {status}"
            )
        changed.append(f"defaultClientScope:{scope_name}")
    return changed


def _get_realm_mgmt_client_id(
    kc_url: str, realm: str, token: str
) -> str | None:
    existing = _get_existing_client(kc_url, realm, token, "realm-management")
    return str(existing["id"]) if existing else None


def _ensure_realm_roles(
    kc_url: str,
    realm: str,
    token: str,
    internal_id: str,
    role_specs: list[str],
) -> list[str]:
    """Assign realm-management roles to the service account user of a client."""
    if not role_specs:
        return []

    realm_mgmt_id = _get_realm_mgmt_client_id(kc_url, realm, token)
    if not realm_mgmt_id:
        _die(f"realm-management client not found in realm '{realm}'")

    # Get service account user
    sa_url = (
        f"{kc_url}/admin/realms/{realm}/clients/{internal_id}/service-account-user"
    )
    status, sa_user = _request("GET", sa_url, token=token)
    if status != 200 or not sa_user:
        _die(f"Could not retrieve service account user for client '{internal_id}'")
    sa_user_id = sa_user["id"]

    # Get already-assigned roles
    assigned_url = (
        f"{kc_url}/admin/realms/{realm}/users/{sa_user_id}"
        f"/role-mappings/clients/{realm_mgmt_id}"
    )
    status, already_assigned = _request("GET", assigned_url, token=token)
    assigned_names = {r["name"] for r in (already_assigned or [])}

    changed = []
    roles_to_add = []
    for role_spec in role_specs:
        # role_spec format: "realm-management:role-name"
        parts = role_spec.split(":", 1)
        role_name = parts[1] if len(parts) == 2 else parts[0]
        if role_name in assigned_names:
            continue
        # Resolve role ID
        role_url = (
            f"{kc_url}/admin/realms/{realm}/clients/{realm_mgmt_id}/roles/{role_name}"
        )
        status, role_obj = _request("GET", role_url, token=token)
        if status != 200 or not role_obj:
            _die(
                f"Role '{role_name}' not found in realm-management client"
            )
        roles_to_add.append(role_obj)
        changed.append(f"realmRole:{role_name}")

    if roles_to_add:
        status, _ = _request(
            "POST", assigned_url, token=token, payload=roles_to_add
        )
        if status not in (200, 201, 204):
            _die(f"Failed to assign realm roles: HTTP {status}")

    return changed


# ---------------------------------------------------------------------------
# Core reconcile loop
# ---------------------------------------------------------------------------


def _reconcile_client(
    kc_url: str,
    realm: str,
    token: str,
    spec: dict[str, Any],
) -> None:
    client_id = spec["clientId"]
    secret = _resolve_client_secret(spec)
    existing = _get_existing_client(kc_url, realm, token, client_id)
    all_changed: list[str] = []

    if existing is None:
        payload = _build_create_payload(spec, secret)
        url = f"{kc_url}/admin/realms/{realm}/clients"
        status, body = _request("POST", url, token=token, payload=payload)
        if status not in (200, 201):
            _die(f"Failed to create client '{client_id}': HTTP {status} {body}")
        # Re-fetch to get internal ID
        existing = _get_existing_client(kc_url, realm, token, client_id)
        if existing is None:
            _die(f"Client '{client_id}' created but could not be re-fetched")
        all_changed.append("created")
    else:
        # Drift detection on base fields
        drift_fields: list[str] = []
        for field in BASE_FIELDS - {"defaultClientScopes"}:
            if field in spec and existing.get(field) != spec[field]:
                drift_fields.append(field)
        if drift_fields:
            update_payload = {**existing}
            for field in drift_fields:
                update_payload[field] = spec[field]
            if secret is not None:
                update_payload["secret"] = secret
            url = f"{kc_url}/admin/realms/{realm}/clients/{existing['id']}"
            status, _ = _request("PUT", url, token=token, payload=update_payload)
            if status not in (200, 201, 204):
                _die(f"Failed to update client '{client_id}': HTTP {status}")
            all_changed.extend(drift_fields)

    internal_id = existing["id"]

    # Protocol mappers
    if "protocolMappers" in spec:
        mapper_changes = _ensure_protocol_mappers(
            kc_url, realm, token, internal_id, spec["protocolMappers"]
        )
        all_changed.extend(mapper_changes)

    # Default client scopes
    if "defaultClientScopes" in spec:
        scope_changes = _ensure_default_scopes(
            kc_url, realm, token, internal_id, spec["defaultClientScopes"]
        )
        all_changed.extend(scope_changes)

    # Realm roles (service account role assignments)
    if "realmRoles" in spec:
        role_changes = _ensure_realm_roles(
            kc_url, realm, token, internal_id, spec["realmRoles"]
        )
        all_changed.extend(role_changes)

    if "created" in all_changed:
        _log("created", client_id, [f for f in all_changed if f != "created"])
    elif all_changed:
        _log("updated", client_id, all_changed)
    else:
        _log("unchanged", client_id)


# ---------------------------------------------------------------------------
# Bootstrap admin reset
# ---------------------------------------------------------------------------


def _reset_bootstrap_admin(kc_url: str) -> None:
    if "localhost" not in kc_url and "127.0.0.1" not in kc_url:
        return
    result = subprocess.run(  # noqa: S603
        [
            "docker",
            "exec",
            "omnibase-infra-keycloak",
            "/opt/keycloak/bin/kc.sh",
            "bootstrap-admin",
            "user",
            "--no-prompt",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(
            json.dumps(
                {"op": "warning", "message": f"bootstrap-admin: {result.stderr.strip()}"}
            ),
            flush=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconcile Keycloak clients")
    p.add_argument("--kc-url", default=os.environ.get("KC_URL", ""))
    p.add_argument("--realm", default=os.environ.get("KC_REALM", "omninode"))
    p.add_argument(
        "--admin-username",
        default=os.environ.get("KC_ADMIN_USERNAME", "admin"),
    )
    p.add_argument("--admin-password", default=os.environ.get("KC_ADMIN_PASSWORD", ""))
    p.add_argument("--config", default=os.environ.get("KC_CONFIG", ""))
    p.add_argument(
        "--reset-bootstrap-admin",
        action="store_true",
        default=False,
        help="Run kc.sh bootstrap-admin (local only)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.kc_url:
        _die("--kc-url (or KC_URL env var) is required")
    if not args.admin_password:
        _die("--admin-password (or KC_ADMIN_PASSWORD env var) is required")
    if not args.config:
        _die("--config (or KC_CONFIG env var) is required")

    if args.reset_bootstrap_admin:
        _reset_bootstrap_admin(args.kc_url)

    config_path = args.config
    if not os.path.isfile(config_path):
        _die(f"Config file not found: {config_path}")

    with open(config_path) as f:  # noqa: PTH123
        config = json.load(f)

    clients = config.get("clients", [])
    if not clients:
        _die("No clients found in config file")

    token = _get_token(args.kc_url, args.admin_username, args.admin_password)

    for client_spec in clients:
        _reconcile_client(args.kc_url, args.realm, token, client_spec)


if __name__ == "__main__":
    main()
