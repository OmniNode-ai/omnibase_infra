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
    KC_URL, KC_REALM, KC_ADMIN_USERNAME, KC_ADMIN_PASSWORD, KC_CONFIG,
    KC_USER_PROFILE_CONFIG (optional -- also reconcile the realm's declarative
    User Profile from desired-user-profile.json)

Idempotent: re-running against an already-correct realm produces all
op=unchanged lines and exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
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
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            body = resp.read()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
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
    req = urllib.request.Request(  # noqa: S310
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
        _die(f"Failed to obtain admin token from {token_url}: HTTP {exc.code}")
        raise RuntimeError("unreachable after _die")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(op: str, client_id: str, fields_changed: list[str] | None = None) -> None:
    record: dict[str, Any] = {"op": op, "clientId": client_id}
    if fields_changed is not None:
        record["fields_changed"] = fields_changed
    print(json.dumps(record), flush=True)


def _log_realm_smtp(realm: str, status: str) -> None:
    """Record which realm-SMTP branch the reconcile took (OMN-18170).

    ``status`` is ``"configured"`` or ``"not_configured"`` -- a decision, not a
    value. No mail setting, host included, is ever emitted here.
    """
    print(
        json.dumps({"op": "realm_smtp", "clientId": f"realm:{realm}", "smtp": status}),
        flush=True,
    )


def _die(
    _msg: str,
    *,
    keys: Sequence[str] | None = None,
    client_id: str | None = None,
) -> NoReturn:
    """Abort, reporting WHERE the run failed without reporting WHY in prose.

    The free-text ``_msg`` is still discarded, deliberately and unconditionally:
    several call sites interpolate a Keycloak error body into it, and nothing
    guarantees such a body carries no secret material. That redaction is the
    whole reason this helper exists.

    OMN-18170: discarding the message also discarded the failure SITE, so every
    one of the ~30 failure paths printed the same anonymous line. An operator
    facing it could not tell a provisioning gap (an env var nobody set) from a
    credential failure (a password that is wrong) without re-deriving the
    failure by hand -- which is exactly what the `.201` dev-lane reconcile
    forced someone to do.

    Three non-secret identifiers are now reported:

    * ``site`` / ``line`` -- read off the CALLER'S frame, so every call site is
      covered with no annotation and the value cannot go stale as the file is
      edited;
    * ``keys`` -- env var NAMES, never their values. Passing a value here would
      defeat the redaction above, so callers pass names only;
    * ``clientId`` -- a roster identifier, the same non-secret field ``_log``
      already emits.
    """
    record: dict[str, Any] = {
        "op": "error",
        "message": "seed-keycloak-clients failed; details redacted",
    }
    caller = inspect.currentframe()
    caller = caller.f_back if caller is not None else None
    if caller is not None:
        record["site"] = caller.f_code.co_name
        record["line"] = caller.f_lineno
    if client_id is not None:
        record["clientId"] = client_id
    if keys:
        record["keys"] = list(keys)
    print(json.dumps(record), file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Secret fingerprinting (OMN-18391)
# ---------------------------------------------------------------------------


def _secret_fingerprint(value: str) -> str:
    """Return a comparison-safe fingerprint: sha256-12 plus length.

    Never returns, logs, or otherwise exposes the underlying secret value.
    Two equal secrets always fingerprint equal; a fingerprint collision
    between two different secrets is not a concern this guard needs to
    defend against -- it exists to catch an accidental mismatch (a
    Keycloak-minted secret no consumer holds), not an adversarial one.
    """
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{digest}/{len(value)}"


# ---------------------------------------------------------------------------
# Keycloak admin helpers
# ---------------------------------------------------------------------------

BASE_FIELDS = {
    "attributes",
    "publicClient",
    "bearerOnly",
    "fullScopeAllowed",
    "serviceAccountsEnabled",
    "standardFlowEnabled",
    "directAccessGrantsEnabled",
    "redirectUris",
    "webOrigins",
    "defaultClientScopes",
}

_SECRET_CLEARING_FLAGS = ("bearerOnly", "publicClient")
_SERVER_MANAGED_CLIENT_FIELDS = frozenset({"access", "id"})


def _resolve_client_secret(client_spec: dict[str, Any]) -> str | None:
    secret_env = client_spec.get("secretEnv")
    if not secret_env:
        return None
    val = os.environ.get(secret_env)
    if not val:
        _die(
            f"Client '{client_spec['clientId']}' requires env var '{secret_env}' "
            f"but it is not set or empty.",
            keys=[secret_env],
            client_id=client_spec["clientId"],
        )
    return val


def _resolve_consumer_secret(client_spec: dict[str, Any]) -> str | None:
    """Resolve the env var naming the consuming k8s Secret's copy of this
    client's secret (OMN-18391).

    A client that declares ``consumerSecretEnv`` is one whose Keycloak secret
    is not itself set by this reconciler (no ``secretEnv``) -- ``onex-api``
    is the case this exists for: Keycloak mints its own secret when the
    client becomes confidential, and the value ~15 runtime workloads actually
    authenticate with lives in a separate k8s Secret this Job also mounts.

    Fail-closed like _resolve_client_secret: a declared consumerSecretEnv
    that is unset or empty aborts the Job rather than silently skipping the
    fingerprint check.
    """
    consumer_env = client_spec.get("consumerSecretEnv")
    if not consumer_env:
        return None
    val = os.environ.get(consumer_env)
    if not val:
        _die(
            f"Client '{client_spec['clientId']}' declares consumerSecretEnv "
            f"'{consumer_env}' but it is not set or empty.",
            keys=[consumer_env],
            client_id=client_spec["clientId"],
        )
    return val


def _build_update_payload(
    existing: dict[str, Any],
    spec: dict[str, Any],
    drift_fields: list[str],
    secret: str | None,
) -> dict[str, Any]:
    update_payload: dict[str, Any] = {
        key: value
        for key, value in existing.items()
        if key not in _SERVER_MANAGED_CLIENT_FIELDS and key != "secret"
    }
    for field in drift_fields:
        update_payload[field] = spec[field]
    for secret_clearing_flag in _SECRET_CLEARING_FLAGS:
        if secret_clearing_flag not in drift_fields:
            update_payload.pop(secret_clearing_flag, None)
    if secret is not None:
        update_payload["secret"] = secret
    return update_payload


def _assert_update_preserved_undeclared_access_type(
    client_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    drift_fields: list[str],
) -> None:
    changed = [
        field
        for field in _SECRET_CLEARING_FLAGS
        if field not in drift_fields and before.get(field) != after.get(field)
    ]
    if changed:
        _die(
            f"Client '{client_id}' update changed non-drifted access-type fields: "
            f"{', '.join(changed)}"
        )


# Maps each realmSettings.smtpServer.<key>Env field to the Keycloak realm
# representation's smtpServer.<key> field it resolves into.
_SMTP_FIELD_MAP = {
    "hostEnv": "host",
    "portEnv": "port",
    "fromEnv": "from",
    "fromDisplayNameEnv": "fromDisplayName",
    "starttlsEnv": "starttls",
    "authEnv": "auth",
    "userEnv": "user",
    "passwordEnv": "password",
}


def _resolve_realm_smtp_settings(smtp_spec: dict[str, Any]) -> dict[str, str] | None:
    """Resolve realmSettings.smtpServer's *Env indirections against the live
    environment.

    Returns ``None`` when this lane declares NO mail provider -- the roster
    carries no ``smtpServer`` block, or it does and not one of the env vars it
    indirects through resolves. The caller then skips realm SMTP entirely and
    leaves Keycloak's own SMTP state exactly as it found it.

    OMN-18170: this used to be unconditionally fail-closed, which is correct
    where mail IS configured and wrong where it is not. The `.201` compose dev
    lane sends no email and declares none of the eight SMTP_* variables, so the
    reconcile died there before it reached a single client -- a lane with no
    mail provider could not reconcile its Keycloak clients at all.

    Three outcomes, and the middle one is the whole point:

    * every declared key resolves  -> configure, byte-identically to before
      (this is the staging / onex-dev path, where the eight vars are seeded);
    * NO declared key resolves     -> return None, skip, touch nothing;
    * SOME resolve and some do not -> refuse, naming every key that did not.

    No default is ever substituted for any mail value (rule 8, and the direct
    guard against the OMN-14938 optional:true-on-a-required-key anti-pattern).
    Absent is a skip; it is never a stand-in value. The partial case stays
    fail-closed deliberately: half a mail configuration is a misconfiguration,
    and silently treating it as "not configured" would ignore a real,
    half-applied setup -- including the shape where a mail host is gone but its
    credentials are still sitting in the environment.
    """
    declared = {
        spec_field: env_var
        for spec_field in _SMTP_FIELD_MAP
        if (env_var := smtp_spec.get(spec_field))
    }
    if not declared:
        return None

    missing = [env_var for env_var in declared.values() if not os.environ.get(env_var)]
    if len(missing) == len(declared):
        return None
    if missing:
        _die(
            "Realm SMTP is partially configured: "
            f"{len(missing)} of {len(declared)} declared env vars are unset or empty.",
            keys=missing,
        )

    return {
        _SMTP_FIELD_MAP[spec_field]: os.environ[env_var]
        for spec_field, env_var in declared.items()
    }


def _get_existing_client(
    kc_url: str, realm: str, token: str, client_id: str
) -> dict[str, Any] | None:
    url = f"{kc_url}/admin/realms/{realm}/clients?clientId={urllib.parse.quote(client_id)}"
    status, body = _request("GET", url, token=token)
    if status != 200 or not body:
        return None
    matches = [c for c in body if c.get("clientId") == client_id]
    return matches[0] if matches else None


def _build_create_payload(spec: dict[str, Any], secret: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "clientId": spec["clientId"],
        "enabled": True,
    }
    for field in (
        "publicClient",
        "bearerOnly",
        "fullScopeAllowed",
        "serviceAccountsEnabled",
        "standardFlowEnabled",
        "directAccessGrantsEnabled",
        "redirectUris",
        "webOrigins",
        "attributes",
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
            _die(f"Failed to create protocol mapper '{mapper['name']}': HTTP {status}")
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
            _die(f"Failed to bind scope '{scope_name}' to client: HTTP {status}")
        changed.append(f"defaultClientScope:{scope_name}")
    return changed


def _get_realm_mgmt_client_id(kc_url: str, realm: str, token: str) -> str | None:
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
    sa_url = f"{kc_url}/admin/realms/{realm}/clients/{internal_id}/service-account-user"
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
            _die(f"Role '{role_name}' not found in realm-management client")
        roles_to_add.append(role_obj)
        changed.append(f"realmRole:{role_name}")

    if roles_to_add:
        status, _ = _request("POST", assigned_url, token=token, payload=roles_to_add)
        if status not in (200, 201, 204):
            _die(f"Failed to assign realm roles: HTTP {status}")

    return changed


def _ensure_client_scope_mappings(
    kc_url: str,
    realm: str,
    token: str,
    internal_id: str,
    role_specs: list[str],
) -> list[str]:
    """Add client-level scope mappings so roles remain reachable when fullScopeAllowed=false.

    Without explicit scope mappings, Keycloak's scope filter drops service-account roles
    from client_credentials tokens when fullScopeAllowed=false, even if those roles are
    assigned to the service account user.
    """
    if not role_specs:
        return []

    realm_mgmt_id = _get_realm_mgmt_client_id(kc_url, realm, token)
    if not realm_mgmt_id:
        _die(f"realm-management client not found in realm '{realm}'")

    current_url = (
        f"{kc_url}/admin/realms/{realm}/clients/{internal_id}"
        f"/scope-mappings/clients/{realm_mgmt_id}"
    )
    status, existing = _request("GET", current_url, token=token)
    if status != 200:
        _die(f"Failed to fetch client scope mappings: HTTP {status}")
    existing_names = {r["name"] for r in (existing or [])}

    changed = []
    roles_to_add = []
    for role_spec in role_specs:
        parts = role_spec.split(":", 1)
        role_name = parts[1] if len(parts) == 2 else parts[0]
        if role_name in existing_names:
            continue
        role_url = (
            f"{kc_url}/admin/realms/{realm}/clients/{realm_mgmt_id}/roles/{role_name}"
        )
        status, role_obj = _request("GET", role_url, token=token)
        if status != 200 or not role_obj:
            _die(f"Role '{role_name}' not found in realm-management client")
        roles_to_add.append(role_obj)
        changed.append(f"clientScopeMapping:{role_name}")

    if roles_to_add:
        status, _ = _request("POST", current_url, token=token, payload=roles_to_add)
        if status not in (200, 201, 204):
            _die(f"Failed to add client scope mappings: HTTP {status}")

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
            # OMN-16504: send ONLY the drifted fields, never the full live
            # representation.
            #
            # This used to be `{**existing}` with the drifted fields written
            # over it. Keycloak's client update applies every non-null field of
            # the representation it is given, and when that representation
            # marks the client `bearerOnly` (or `publicClient`) it clears the
            # stored client secret as a side effect. So a full-representation
            # PUT re-asserted `bearerOnly: true` on every reconcile, and the
            # first reconcile that found ANY unrelated drift destroyed the
            # secret of a client the reconciler was not even asked to change.
            #
            # That is not hypothetical. omninode_infra b88ae5c1 (2026-09-10)
            # added `"fullScopeAllowed": false` to the `onex-api` entry in
            # desired-clients.json. Keycloak's default is `true`, so the next
            # reconcile necessarily saw drift on that one declared field,
            # carried `bearerOnly: true` forward on the PUT, and Keycloak
            # emptied the secret. onex-api is the realm's introspection caller;
            # its introspection POSTs have returned 401 ever since, and the 23
            # workloads holding the now-orphaned secret were all left correct
            # and all left unable to authenticate.
            #
            # THE FIX: keep sending the writable live representation, strip
            # server-managed/read-only fields and any returned secret, and drop
            # exactly the two flags that make Keycloak clear the secret unless
            # those flags are themselves the declared drift.
            #
            # An earlier revision of this change sent only `{clientId} +
            # drifted` instead. Review rejected that, correctly: Keycloak's
            # client update is documented as a merge over non-null fields, but
            # that has not held uniformly across versions for the
            # collection-valued fields -- `redirectUris`, `webOrigins`,
            # `defaultClientScopes`, `protocolMappers`. A narrow PUT that
            # omitted them would, on any version that replaces rather than
            # merges, silently empty omniweb's redirect URIs and break login.
            # That trades this bug for a worse one the guard below cannot see,
            # because the guard only reads secrets.
            #
            # Dropping the two flags is a strictly smaller change than a narrow
            # drift-only PUT. Every writable collection field is still carried
            # exactly as the previous, long-working code carried it, so no
            # collection semantics change at all; the only differences from the
            # code that caused the outage are the two keys that trigger secret
            # clearing and the server-managed fields that the admin API may
            # reject if echoed back.
            #
            # When the roster genuinely DOES declare a change to `bearerOnly`
            # or `publicClient`, the flag is sent, Keycloak clears the secret,
            # and _assert_confidential_clients_have_secrets() fails the Job red
            # rather than letting it pass silently. A post-PUT readback also
            # verifies that omitting a non-drifted access-type flag did not let
            # a replace-style server reset it. The layering is deliberate:
            # this block stops the INCIDENTAL destruction, the readback catches
            # replace-vs-merge ambiguity, and the secret guard catches the
            # deliberate access-type change.
            before_update = dict(existing)
            update_payload = _build_update_payload(existing, spec, drift_fields, secret)
            url = f"{kc_url}/admin/realms/{realm}/clients/{existing['id']}"
            status, _ = _request("PUT", url, token=token, payload=update_payload)
            if status not in (200, 201, 204):
                _die(f"Failed to update client '{client_id}': HTTP {status}")
            refreshed = _get_existing_client(kc_url, realm, token, client_id)
            if refreshed is None:
                _die(f"Client '{client_id}' updated but could not be re-fetched")
            _assert_update_preserved_undeclared_access_type(
                client_id, before_update, refreshed, drift_fields
            )
            existing = refreshed
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

    # Client scope mappings (makes roles reachable in scope when fullScopeAllowed=false)
    if "clientScopeMappings" in spec:
        scope_mapping_changes = _ensure_client_scope_mappings(
            kc_url, realm, token, internal_id, spec["clientScopeMappings"]
        )
        all_changed.extend(scope_mapping_changes)

    if "created" in all_changed:
        _log("created", client_id, [f for f in all_changed if f != "created"])
    elif all_changed:
        _log("updated", client_id, all_changed)
    else:
        _log("unchanged", client_id)


# ---------------------------------------------------------------------------
# Post-reconcile assertion: no confidential client may end with no secret
# ---------------------------------------------------------------------------


def _die_naming_clients(check: str, reason: str, client_ids: list[str]) -> NoReturn:
    """Fail the Job non-zero, naming the offending clients.

    Deliberately does NOT go through _die(), which redacts its message. What
    is printed here is a check name, a reason and a list of clientIds --
    clientIds are already emitted in the clear by _log() on every single
    reconcile line, so naming them costs nothing that the normal run does not
    already disclose. No secret, and no property of a secret other than "it is
    empty", is printed.

    The alternative -- a redacted failure -- would tell an operator that the
    realm is broken without telling them which client, which is the same
    debugging position this check exists to remove.
    """
    print(
        json.dumps(
            {
                "op": "error",
                "check": check,
                "reason": reason,
                "clients": sorted(client_ids),
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def _die_fingerprint_mismatch(mismatches: list[dict[str, str]]) -> NoReturn:
    """Fail the Job non-zero, naming each client by fingerprint only (OMN-18391).

    ``mismatches`` entries are ``{"clientId": ..., "live": <fingerprint>,
    "consumer": <fingerprint>}``. Neither secret value is ever read into this
    function's arguments or printed -- only their fingerprints (sha256-12 +
    length), computed by the caller via _secret_fingerprint().
    """
    print(
        json.dumps(
            {
                "op": "error",
                "check": "client_secret_fingerprint_mismatch",
                "reason": (
                    "one or more clients hold a live Keycloak secret that does "
                    "not match the consuming k8s Secret's copy; every consumer "
                    "authenticating with the stale copy will get HTTP 401 "
                    "until the values are reconciled"
                ),
                "clients": sorted(m["clientId"] for m in mismatches),
                "fingerprints": mismatches,
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def _live_client_requires_secret(existing: dict[str, Any]) -> bool:
    """Return whether the live Keycloak client shape is expected to hold a secret.

    Classified from the LIVE representation rather than the desired spec,
    because a roster entry is partial: it declares only the fields the
    reconciler owns, so a spec that omits ``publicClient`` says nothing about
    whether the live client is public.

    Only ``publicClient`` exempts a client. **``bearerOnly`` deliberately does
    not**, even though Keycloak's textbook model says a bearer-only client is
    a pure resource server that never authenticates outbound. In this realm
    that model does not hold: ``onex-api`` is bearer-only AND is the client
    the runtime presents as HTTP Basic auth on every token-introspection POST,
    so it must hold a secret. Exempting bearer-only clients here would make
    this guard skip the exact client whose emptied secret it exists to catch
    -- the deploy would stay green while introspection returns 401, which is
    the silent failure OMN-16504 is about.

    The architectural mismatch is real and is worth fixing separately, by
    moving introspection onto a non-bearer-only client. Until that happens the
    guard reports what is true of this realm today.
    """
    return existing.get("publicClient") is not True


def _read_client_secret_value(
    kc_url: str, realm: str, token: str, existing: dict[str, Any]
) -> str | None:
    """Return the live client's secret value, or None if absent/empty.

    Same dual-read strategy as the presence check below (dedicated endpoint,
    falling back to the client representation), because the dedicated
    endpoint 404s for some client shapes. The value returned here is used
    ONLY to compute a fingerprint (OMN-18391) -- callers must never log,
    print, or otherwise surface it directly.
    """
    status, body = _request(
        "GET",
        f"{kc_url}/admin/realms/{realm}/clients/{existing['id']}/client-secret",
        token=token,
    )
    if status == 200 and isinstance(body, dict) and body.get("value"):
        return str(body["value"])

    status, body = _request(
        "GET", f"{kc_url}/admin/realms/{realm}/clients/{existing['id']}", token=token
    )
    if status == 200 and isinstance(body, dict) and body.get("secret"):
        return str(body["secret"])
    return None


def _read_client_secret_is_present(
    kc_url: str, realm: str, token: str, existing: dict[str, Any]
) -> bool:
    """Return whether the live client currently holds a non-empty secret.

    Delegates to _read_client_secret_value(); this wrapper exists because
    most callers only need presence, and returning a bool rather than the
    value itself keeps them from ever handling the raw secret.

    Fails CLOSED: any status the read cannot interpret is reported as absent,
    so an unreadable realm fails the Job rather than passing it.
    """
    return _read_client_secret_value(kc_url, realm, token, existing) is not None


def _assert_confidential_clients_have_secrets(
    kc_url: str,
    realm: str,
    token: str,
    clients: list[dict[str, Any]],
) -> None:
    """Fail the Job red if any confidential client ends the reconcile empty.

    OMN-16504. A client secret that Keycloak has emptied is indistinguishable,
    from the outside, from a healthy realm: every reconcile line still reads
    ``op=unchanged``, the Job still exits 0, the deploy still goes green, and
    the only symptom is that one authentication path starts returning 401.
    That is how the onex-api destruction described in _reconcile_client() ran
    unnoticed for three days.

    The guard classifies the LIVE client representation, not the partial
    desired spec -- see _live_client_requires_secret(). Public clients are
    supposed to hold no secret and are skipped; they are the positive control
    for this check. Bearer-only clients are NOT skipped, because this realm
    uses one as its introspection caller.

    Every offending client is collected before failing, so one run names the
    whole set rather than making an operator re-run the Job per client.
    """
    offenders: list[str] = []
    for spec in clients:
        client_id = spec["clientId"]
        existing = _get_existing_client(kc_url, realm, token, client_id)
        if existing is None:
            offenders.append(client_id)
            continue
        if not _live_client_requires_secret(existing):
            continue
        if not _read_client_secret_is_present(kc_url, realm, token, existing):
            offenders.append(client_id)

    if offenders:
        _die_naming_clients(
            "confidential_client_secret_present",
            (
                "one or more confidential clients hold no client secret after "
                "reconcile; every caller authenticating as them will get HTTP "
                "401 until the secret is restored"
            ),
            offenders,
        )


def _assert_client_secrets_match_consumers(
    kc_url: str,
    realm: str,
    token: str,
    clients: list[dict[str, Any]],
) -> None:
    """Fail the Job red if a client's live secret does not match its consumer's copy.

    OMN-18391. _assert_confidential_clients_have_secrets() (OMN-16504) only
    checks that a confidential client ends the reconcile non-empty. When the
    roster flips a client to confidential with no ``secretEnv`` declared (as
    ``onex-api`` was), Keycloak mints its own secret -- non-empty, and no
    consumer holds it. The presence guard reports green while every
    consumer's introspection returns 401.

    Only clients that declare ``consumerSecretEnv`` are checked here; a
    client with a plain ``secretEnv`` already has this reconciler as the
    single source of truth for its Keycloak secret and needs no separate
    consumer comparison. Compares fingerprints only (sha256-12 + length) --
    the secret values themselves are never read into a log line or an error
    message.

    Deliberately does not treat an absent or empty live secret as a
    mismatch: those are the _assert_confidential_clients_have_secrets()
    check's job, so the two failure classes stay distinguishable in the
    Job's log rather than being conflated into one reason string.
    """
    mismatches: list[dict[str, str]] = []
    for spec in clients:
        client_id = spec["clientId"]
        consumer_secret = _resolve_consumer_secret(spec)
        if consumer_secret is None:
            continue
        existing = _get_existing_client(kc_url, realm, token, client_id)
        if existing is None:
            continue  # absence is _assert_confidential_clients_have_secrets' job
        live_secret = _read_client_secret_value(kc_url, realm, token, existing)
        if not live_secret:
            continue  # emptiness is _assert_confidential_clients_have_secrets' job
        live_fp = _secret_fingerprint(live_secret)
        consumer_fp = _secret_fingerprint(consumer_secret)
        if live_fp != consumer_fp:
            mismatches.append(
                {"clientId": client_id, "live": live_fp, "consumer": consumer_fp}
            )

    if mismatches:
        _die_fingerprint_mismatch(mismatches)


# ---------------------------------------------------------------------------
# Read-only client preflight (OMN-18170)
# ---------------------------------------------------------------------------

# How a client's Keycloak secret is supposed to get there. The distinction
# matters to an operator reading a refusal, because it is the difference
# between "seed this value" and "nothing to do".
_SECRET_SOURCE_ROSTER_PUSHED = "roster_pushed"
_SECRET_SOURCE_CONSUMER_OWNED = "consumer_owned"
_SECRET_SOURCE_KEYCLOAK_MINTED = "keycloak_minted"
_SECRET_SOURCE_PUBLIC = "public"


def _classify_secret_source(
    spec: dict[str, Any], existing: dict[str, Any] | None
) -> str:
    """Classify where this client's secret is supposed to come from.

    ``secretEnv`` means the ROSTER pushes the value: this reconciler writes
    the env var's contents into Keycloak, so an absent env var is a
    provisioning gap a human has to close. ``consumerSecretEnv`` means the
    value lives in a consuming Secret and this reconciler only compares
    fingerprints. A confidential client declaring neither is one Keycloak
    MINTS for itself -- there is nothing to seed and nothing absent.

    Public-ness is read from the LIVE representation when there is one, for
    the same reason ``_live_client_requires_secret`` does: a roster entry is
    partial and says nothing about fields it does not declare.
    """
    if spec.get("secretEnv"):
        return _SECRET_SOURCE_ROSTER_PUSHED
    if spec.get("consumerSecretEnv"):
        return _SECRET_SOURCE_CONSUMER_OWNED
    source = existing if existing is not None else spec
    if source.get("publicClient") is True:
        return _SECRET_SOURCE_PUBLIC
    return _SECRET_SOURCE_KEYCLOAK_MINTED


def _preflight_client(
    kc_url: str,
    realm: str,
    token: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Survey one roster entry read-only; never writes, never raises.

    Returns a record whose ``findings`` list holds every REFUSAL for this
    client. Informational facts (what the client's secret source is, whether
    it exists yet, which fields have drifted) are record fields rather than
    findings, so "this run will refuse" and "here is what the run saw" stay
    separable in the output.
    """
    client_id = spec["clientId"]
    existing = _get_existing_client(kc_url, realm, token, client_id)
    secret_source = _classify_secret_source(spec, existing)
    secret_env = spec.get("secretEnv") or spec.get("consumerSecretEnv")
    findings: list[dict[str, str]] = []

    env_present: bool | None = None
    if secret_env:
        env_present = bool(os.environ.get(secret_env))
        if not env_present:
            if secret_source == _SECRET_SOURCE_ROSTER_PUSHED:
                findings.append(
                    {
                        "code": "roster_secret_absent",
                        "key": secret_env,
                        "detail": "absent, roster-pushed, seed required",
                    }
                )
            else:
                findings.append(
                    {
                        "code": "consumer_secret_absent",
                        "key": secret_env,
                        "detail": (
                            "absent, consumer-owned; Keycloak mints this "
                            "client's secret and the consuming Secret's copy "
                            "is what must match it, so the comparison cannot "
                            "run until this value is readable"
                        ),
                    }
                )

    live_secret_present: bool | None = None
    drift_fields: list[str] = []
    if existing is not None:
        drift_fields = sorted(
            field
            for field in BASE_FIELDS - {"defaultClientScopes"}
            if field in spec and existing.get(field) != spec[field]
        )
        if _live_client_requires_secret(existing):
            live_secret = _read_client_secret_value(kc_url, realm, token, existing)
            live_secret_present = live_secret is not None
            if live_secret is None:
                findings.append(
                    {
                        "code": "live_secret_empty",
                        "detail": (
                            "this client already holds NO secret in Keycloak; "
                            "every caller authenticating as it gets HTTP 401. "
                            "Reconciling it would send its declared drift, "
                            "which is what empties a secret in the first "
                            "place, so the run refuses instead"
                        ),
                    }
                )
            elif (
                secret_source == _SECRET_SOURCE_CONSUMER_OWNED
                and env_present
                and secret_env is not None
            ):
                live_fp = _secret_fingerprint(live_secret)
                consumer_fp = _secret_fingerprint(os.environ[secret_env])
                if live_fp != consumer_fp:
                    findings.append(
                        {
                            "code": "consumer_secret_fingerprint_mismatch",
                            "key": secret_env,
                            "live": live_fp,
                            "consumer": consumer_fp,
                            "detail": (
                                "the live Keycloak secret does not match the "
                                "consuming Secret's copy; consumers holding "
                                "the stale copy get HTTP 401"
                            ),
                        }
                    )

    return {
        "op": "preflight",
        "clientId": client_id,
        "present": existing is not None,
        "secret_source": secret_source,
        "secret_env": secret_env,
        "env_present": env_present,
        "live_secret_present": live_secret_present,
        "drift_fields": drift_fields,
        "findings": findings,
    }


def _preflight_clients(
    kc_url: str,
    realm: str,
    token: str,
    clients: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Survey EVERY roster entry read-only, before the first client write.

    OMN-18170. Two ordering defects made real breakage unreportable:

    * ``_resolve_client_secret`` ``_die()``s inside the client loop on the
      first entry whose ``secretEnv`` is unset, so a run named one absent
      variable however many were absent. On the `.201` dev lane that meant an
      operator learned about ten absent values one re-run at a time.
    * Both secret guards run AFTER the loop, so a run that died mid-loop
      reached neither. The live ``onex-api`` client on that lane already holds
      an empty secret and carries declared drift on ``bearerOnly`` -- and
      sending that drift is exactly what makes Keycloak clear a secret. The
      loop would have mutated it and then exited on a later entry, before the
      guard that reports it.

    Surveying first turns both into one refusal that names everything, and
    removes the partial-application hazard rather than reporting it after the
    fact: nothing is written at all when the survey refuses. That is a
    stronger property than letting the loop run to completion and collecting
    refusals on the way, which would still leave earlier clients mutated.

    A client that does not exist yet is NOT a refusal -- it is work the loop
    is about to do. That boundary is the one thing this survey deliberately
    reads differently from ``_assert_confidential_clients_have_secrets``,
    which runs after the loop and is right to treat a still-absent client as
    an offender.
    """
    return [_preflight_client(kc_url, realm, token, spec) for spec in clients]


def _preflight_refusals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every refusal across the survey, each tagged with its client."""
    return [
        {"clientId": record["clientId"], **finding}
        for record in records
        for finding in record["findings"]
    ]


def _die_preflight(records: list[dict[str, Any]]) -> NoReturn:
    """Fail the run once, naming every refusing client and every key.

    Like ``_die_naming_clients`` and unlike ``_die``, this prints clientIds
    and env var NAMES in the clear: both are already emitted by ``_log`` on
    every ordinary reconcile line, and a refusal that will not say which
    client or which variable leaves an operator in the debugging position
    this check exists to remove. No secret value, and no property of one
    beyond a fingerprint, is printed.
    """
    refusals = _preflight_refusals(records)
    print(
        json.dumps(
            {
                "op": "error",
                "check": "client_preflight",
                "reason": (
                    "the read-only client survey refused before any client was "
                    "written; every refusing client and every env var that must "
                    "be seeded is named below, and the realm is unchanged"
                ),
                "clients": sorted({r["clientId"] for r in refusals}),
                "keys": sorted({r["key"] for r in refusals if "key" in r}),
                "findings": refusals,
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Realm-level settings (verify-email + SMTP + action-token lifespan)
# ---------------------------------------------------------------------------

# Top-level realm representation fields this reconciler owns (drift-detected
# and, if any drifts, carried forward verbatim on the full-representation PUT
# — Keycloak's realm update contract replaces the whole object, same as the
# existing per-client PUT above).
_REALM_BASE_FIELDS = {"verifyEmail", "registrationAllowed"}


def _reconcile_realm_settings(
    kc_url: str,
    realm: str,
    token: str,
    spec: dict[str, Any],
) -> None:
    """Reconcile realm-level verify-email / SMTP / action-token-lifespan
    settings against a running realm. Full-representation GET/diff/PUT,
    matching the per-client reconciler's update shape."""
    smtp_server = _resolve_realm_smtp_settings(spec.get("smtpServer", {}))

    # OMN-18170: say which of the two SMTP branches was taken, in the run's own
    # output. A skip that left no trace is indistinguishable from an SMTP block
    # that was reconciled and happened not to drift -- and on a lane with no
    # mail provider the skip is the interesting fact, not the absence of drift.
    _log_realm_smtp(realm, "configured" if smtp_server else "not_configured")

    desired_attributes = dict(spec.get("attributes", {}))

    url = f"{kc_url}/admin/realms/{realm}"
    status, existing = _request("GET", url, token=token)
    if status != 200 or existing is None:
        _die(f"Failed to fetch realm '{realm}': HTTP {status}")

    drift_fields: list[str] = []
    for field in _REALM_BASE_FIELDS:
        if field in spec and existing.get(field) != spec[field]:
            drift_fields.append(field)

    merged_attributes = {**existing.get("attributes", {}), **desired_attributes}
    if merged_attributes != existing.get("attributes", {}):
        drift_fields.append("attributes")

    if smtp_server and existing.get("smtpServer", {}) != smtp_server:
        drift_fields.append("smtpServer")

    if not drift_fields:
        _log("unchanged", f"realm:{realm}")
        return

    update_payload = {**existing}
    for field in _REALM_BASE_FIELDS:
        if field in spec:
            update_payload[field] = spec[field]
    update_payload["attributes"] = merged_attributes
    if smtp_server:
        update_payload["smtpServer"] = smtp_server

    status, _ = _request("PUT", url, token=token, payload=update_payload)
    if status not in (200, 201, 204):
        _die(f"Failed to update realm '{realm}' settings: HTTP {status}")

    _log("updated", f"realm:{realm}", drift_fields)


# ---------------------------------------------------------------------------
# Declarative User Profile (upConfig)
# ---------------------------------------------------------------------------


def _canonical_profile(profile: dict[str, Any]) -> str:
    """Stable string form of a user-profile document, for drift detection.

    Drops ``_``-prefixed documentation keys (they are not part of Keycloak's
    user-profile schema and are never sent) and sorts every object's keys, so a
    round-tripped GET compares equal to the desired document whenever nothing
    has actually changed. Attribute ORDER is significant -- it is the field
    order Keycloak renders on user-facing forms -- so lists are not sorted.
    """
    stripped = {k: v for k, v in profile.items() if not k.startswith("_")}
    return json.dumps(stripped, sort_keys=True)


def _reconcile_user_profile(
    kc_url: str,
    realm: str,
    token: str,
    desired: dict[str, Any],
) -> None:
    """Reconcile the realm's declarative User Profile against desired state.

    OMN-16195: before this, ``desired-user-profile.json`` was validated desired
    state with no applier -- the live upConfig was hand-mutated and could drift
    from the reviewed file without any signal. Same GET/diff/PUT shape as
    :func:`_reconcile_realm_settings`, and the same merge posture: sections the
    desired document does not declare are preserved from the live document
    rather than dropped.
    """
    url = f"{kc_url}/admin/realms/{realm}/users/profile"
    status, existing = _request("GET", url, token=token)
    if status != 200 or existing is None:
        _die(f"Failed to fetch user profile for realm '{realm}': HTTP {status}")

    payload = {k: v for k, v in desired.items() if not k.startswith("_")}
    if not payload:
        _die("User profile config declares no sections")

    merged = {**existing, **payload}
    if _canonical_profile(merged) == _canonical_profile(existing):
        _log("unchanged", f"realm:{realm}:users-profile")
        return

    drift_fields = [
        key
        for key in payload
        if _canonical_profile({"v": existing.get(key)})
        != _canonical_profile({"v": payload[key]})
    ]

    status, _ = _request("PUT", url, token=token, payload=merged)
    if status not in (200, 201, 204):
        _die(f"Failed to update user profile for realm '{realm}': HTTP {status}")

    _log("updated", f"realm:{realm}:users-profile", drift_fields)


# ---------------------------------------------------------------------------
# Bootstrap admin reset
# ---------------------------------------------------------------------------


def _reset_bootstrap_admin(kc_url: str) -> None:
    if "localhost" not in kc_url and "127.0.0.1" not in kc_url:
        return
    result = subprocess.run(
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
        check=False,
        text=True,
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(
            json.dumps(
                {
                    "op": "warning",
                    "message": f"bootstrap-admin: {result.stderr.strip()}",
                }
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
        "--user-profile-config",
        default=os.environ.get("KC_USER_PROFILE_CONFIG", ""),
        help=(
            "Path to desired-user-profile.json. When set, the realm's "
            "declarative User Profile is reconciled too (OMN-16195). Optional: "
            "omitting it leaves the live upConfig untouched, so a deployment "
            "that has not yet mounted the file behaves exactly as before."
        ),
    )
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

    config_path = Path(args.config)
    if not config_path.is_file():
        _die(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = json.load(f)

    clients = config.get("clients", [])
    if not clients:
        _die("No clients found in config file")

    token = _get_token(args.kc_url, args.admin_username, args.admin_password)

    if config.get("realmSettings"):
        _reconcile_realm_settings(
            args.kc_url, args.realm, token, config["realmSettings"]
        )

    if args.user_profile_config:
        profile_path = Path(args.user_profile_config)
        if not profile_path.is_file():
            _die(f"User profile config file not found: {profile_path}")
        with profile_path.open() as f:
            _reconcile_user_profile(args.kc_url, args.realm, token, json.load(f))

    # OMN-18170: survey every roster entry read-only BEFORE the first client
    # write, print what it saw, and refuse once if anything refuses. Placed
    # here rather than before the realm block deliberately: realm settings are
    # not a client mutation, and gating them behind a client-credential gap
    # would re-break the lane this ordering exists to unblock -- a lane with no
    # mail provider and an unseeded client secret can still reconcile its realm.
    #
    # Every record prints whether or not a refusal follows, so a refusal on one
    # client never hides what the survey learned about another.
    preflight = _preflight_clients(args.kc_url, args.realm, token, clients)
    for record in preflight:
        print(json.dumps(record), flush=True)
    if _preflight_refusals(preflight):
        _die_preflight(preflight)

    for client_spec in clients:
        _reconcile_client(args.kc_url, args.realm, token, client_spec)

    # OMN-16504: last thing the Job does, so it sees the realm as the reconcile
    # actually left it rather than as any single client's branch believed it to
    # be. Exits non-zero, which surfaces as a failed Job.
    _assert_confidential_clients_have_secrets(args.kc_url, args.realm, token, clients)

    # OMN-18391: a non-empty secret is not the same as the RIGHT secret. Runs
    # after the presence guard so the two failure classes stay distinguishable.
    _assert_client_secrets_match_consumers(args.kc_url, args.realm, token, clients)


if __name__ == "__main__":
    main()
