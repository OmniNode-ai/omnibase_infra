# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Seed only the approved disposable user's tenant using Gateway primitives.

Run as a one-shot in the Gateway image on the isolated clone network, never as
an HTTP route. This is test setup, not evidence of the production signup flow.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import stat
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

AUTH = "http://auth.localhost:28080"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _reconcile_identity_profile(
    session: object, base: str, profile: dict[str, object]
) -> None:
    """Add canonical tenant identity fields while preserving realm profile data."""
    desired_attributes = profile.get("attributes")
    desired_groups = profile.get("groups")
    if not isinstance(desired_attributes, list) or not isinstance(desired_groups, list):
        raise ValueError("canonical user profile is malformed")
    identity_names = ("tenant_id", "tenant_slug", "principal_id")
    expected_identity: dict[str, dict[str, object]] = {}
    for name in identity_names:
        definitions = [
            item
            for item in desired_attributes
            if isinstance(item, dict) and item.get("name") == name
        ]
        if len(definitions) != 1:
            raise ValueError(f"canonical profile must define {name} exactly once")
        definition = definitions[0]
        if (
            definition.get("permissions")
            != {
                "view": ["admin"],
                "edit": ["admin"],
            }
            or definition.get("multivalued") is not False
        ):
            raise ValueError(f"{name} profile must be single-valued and admin-only")
        expected_identity[name] = definition
    metadata_groups = [
        item
        for item in desired_groups
        if isinstance(item, dict) and item.get("name") == "user-metadata"
    ]
    if len(metadata_groups) != 1:
        raise ValueError("canonical profile must define user-metadata exactly once")
    expected_group = metadata_groups[0]

    response = session.get(  # type: ignore[attr-defined]
        base + "/users/profile", timeout=15, allow_redirects=False
    )
    response.raise_for_status()
    current = response.json()
    current_attributes = current.get("attributes", [])
    current_groups = current.get("groups", [])
    if not isinstance(current_attributes, list) or not isinstance(current_groups, list):
        raise ValueError("existing realm user profile is malformed")
    matches_by_name = {
        name: [
            item
            for item in current_attributes
            if isinstance(item, dict) and item.get("name") == name
        ]
        for name in identity_names
    }
    for name, matches in matches_by_name.items():
        if len(matches) > 1 or (
            matches and _canonical(matches[0]) != _canonical(expected_identity[name])
        ):
            raise ValueError(
                f"existing {name} profile conflicts with canonical definition"
            )
    group_matches = [
        item
        for item in current_groups
        if isinstance(item, dict) and item.get("name") == "user-metadata"
    ]
    if len(group_matches) > 1 or (
        group_matches and _canonical(group_matches[0]) != _canonical(expected_group)
    ):
        raise ValueError("existing user-metadata profile group conflicts")

    updated = copy.deepcopy(current)
    updated.setdefault("attributes", [])
    updated.setdefault("groups", [])
    for name, matches in matches_by_name.items():
        if not matches:
            updated["attributes"].append(expected_identity[name])
    if not group_matches:
        updated["groups"].append(expected_group)
    if _canonical(updated) != _canonical(current):
        response = session.put(  # type: ignore[attr-defined]
            base + "/users/profile",
            json=updated,
            timeout=15,
            allow_redirects=False,
        )
        response.raise_for_status()
    response = session.get(  # type: ignore[attr-defined]
        base + "/users/profile", timeout=15, allow_redirects=False
    )
    response.raise_for_status()
    verified = response.json()
    verified_attributes = verified.get("attributes", [])
    verified_groups = verified.get("groups", [])
    if _canonical(verified_attributes) != _canonical(
        updated["attributes"]
    ) or _canonical(verified_groups) != _canonical(updated["groups"]):
        raise ValueError("Keycloak changed identity or unrelated profile definitions")
    for name, expected in expected_identity.items():
        actual = [
            item
            for item in verified_attributes
            if isinstance(item, dict) and item.get("name") == name
        ]
        if len(actual) != 1 or _canonical(actual[0]) != _canonical(expected):
            raise ValueError(f"Keycloak did not preserve canonical {name} profile")


def seed(private_dir: Path) -> dict[str, bool]:
    import requests
    from db.psycopg_repository import PsycopgRepository
    from routers.provision import _assign_default_plan
    from tenant_creation import insert_tenant_row
    from tenant_identity import derive_principal_id

    logging.disable(logging.CRITICAL)
    if (
        not private_dir.is_absolute()
        or private_dir.is_symlink()
        or stat.S_IMODE(private_dir.stat().st_mode) != 0o700
    ):
        raise ValueError("private directory required")
    identity_file = private_dir / "identity.json"
    if identity_file.exists() or identity_file.is_symlink():
        raise ValueError("identity output must not already exist")
    for filename in (
        "credentials.env",
        "omninode-realm.json",
        "desired-user-profile.json",
    ):
        path = private_dir / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise ValueError("private regular input required")
    env = dict(
        line.split("=", 1)
        for line in (private_dir / "credentials.env").read_text().splitlines()
    )
    realm = json.loads((private_dir / "omninode-realm.json").read_text())
    owner = UUID(realm["users"][0]["attributes"]["tenant_id"][0])
    principal = derive_principal_id(owner)
    desired_profile = json.loads(
        (private_dir / "desired-user-profile.json").read_text()
    )
    dsn = urlsplit(
        os.environ[
            "OMNINODE_CLOUD_DB_URL"
        ]  # url-authority-ok: exact disposable DSN guard below.
    )
    if (dsn.hostname, dsn.port, dsn.path, dsn.username) != (
        "postgres",
        5432,
        "/omninode_cloud",
        "role_omninode",
    ):
        raise ValueError("only the disposable cloud database is supported")
    # Only the approved disposable realm may be used by this helper.
    issuer = os.environ.get("KEYCLOAK_ISSUER_URL")  # url-authority-ok: sim guard
    if issuer != AUTH + "/realms/omninode":
        raise ValueError("only the disposable realm is supported")
    repo = PsycopgRepository()
    tenant_query = (
        "SELECT tenant_id FROM tenants WHERE tenant_id = %s::uuid OR tenant_slug = %s"
    )
    tenant_params = (str(owner), "sim-preflight")
    existing = repo.fetch_one(tenant_query, tenant_params)
    if existing is not None:
        raise ValueError("refusing to overwrite existing tenant identity")
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        AUTH + "/realms/master/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": env["SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME"],
            "password": env["SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD"],
        },
        timeout=15,
        allow_redirects=False,
    )
    response.raise_for_status()
    session.headers["Authorization"] = "Bearer " + response.json()["access_token"]
    base = AUTH + "/admin/realms/omninode"
    _reconcile_identity_profile(session, base, desired_profile)
    response = session.get(
        base + "/users",
        params={"username": env["SIM_PREFLIGHT_DEMO_USERNAME"], "exact": "true"},
        timeout=15,
        allow_redirects=False,
    )
    response.raise_for_status()
    users = response.json()
    if len(users) != 1 or users[0]["username"] != env["SIM_PREFLIGHT_DEMO_USERNAME"]:
        raise ValueError("expected exactly the approved disposable user")
    user = users[0]
    subject = str(UUID(user["id"]))
    attributes = user.get("attributes", {})
    if attributes.get("tenant_id") != [str(owner)] or attributes.get("tenant_slug") != [
        "sim-preflight"
    ]:
        raise ValueError("existing disposable user differs from the selected owner")
    if attributes.get("principal_id") not in (None, [principal]):
        raise ValueError("refusing to replace an existing principal")
    if attributes.get("principal_id") != [principal]:
        raise ValueError("imported disposable user lacks canonical principal identity")
    response = session.get(
        base + "/clients",
        params={"clientId": "omnidash"},
        timeout=15,
        allow_redirects=False,
    )
    response.raise_for_status()
    clients = response.json()
    if len(clients) != 1 or clients[0]["clientId"] != "omnidash":
        raise ValueError("expected approved OmniDash client")
    client = clients[0]
    mapper = {
        "name": "principal-id",
        "protocol": "openid-connect",
        "protocolMapper": "oidc-usermodel-attribute-mapper",
        "consentRequired": False,
        "config": {
            "user.attribute": "principal_id",
            "claim.name": "principal_id",
            "jsonType.label": "String",
            "id.token.claim": "false",
            "access.token.claim": "true",
            "introspection.token.claim": "true",
            "userinfo.token.claim": "false",
        },
    }
    response = session.get(
        base + "/clients/" + client["id"] + "/protocol-mappers/models",
        timeout=15,
        allow_redirects=False,
    )
    response.raise_for_status()
    mappers = response.json()
    matches = [
        m for m in mappers if m.get("config", {}).get("claim.name") == "principal_id"
    ]
    if matches:
        if (
            len(matches) != 1
            or {k: v for k, v in matches[0].items() if k != "id"} != mapper
        ):
            raise ValueError("existing principal mapper differs")
    else:
        raise ValueError("imported OmniDash client lacks canonical principal mapper")
    with repo:
        existing = repo.fetch_one(tenant_query, tenant_params)
        if existing is not None:
            raise ValueError("refusing to overwrite existing tenant identity")
        insert_tenant_row(
            repo,
            tenant_id=owner,
            tenant_slug="sim-preflight",
            name="Disposable replay validation",
            provisioning_state="active",
            created_by_sub=subject,
            owner_email=user["email"],
        )
        _assign_default_plan(repo, str(owner))
    descriptor = os.open(
        identity_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        json.dump(
            {"tenant_id": str(owner), "subject": subject, "principal_id": principal},
            stream,
        )
    return {"disposable_identity_seeded": True, "captured_owner_uuid_preserved": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = seed(args.private_dir)
    except Exception as exc:  # noqa: BLE001 -- private HTTP/DB exceptions may contain secrets
        print(
            json.dumps({"identity_seed_failed": True, "error_type": type(exc).__name__})
        )
        return 65
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
