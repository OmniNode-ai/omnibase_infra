# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correct unused disposable DB-role credentials in a private auth bundle.

This is intentionally limited to the unconsumed database roles and generated
configuration derived from them. OIDC/admin/Postgres/Valkey credentials and
signing keys are copied byte-for-byte. Before any write, changed files are
backed up to a sibling private directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

import yaml

from scripts.runtime_build.bootstrap_sim_preflight_auth_omn19726 import (
    PRINCIPAL_MAPPER,
    ROOT,
    runtime_configs,
)

ROLE_NAMES = (
    "OMNINODE_RUNTIME_PASSWORD",
    "TENANT_PROJECTION_WRITER_PASSWORD",
    "ROLE_OMNIBASE_PASSWORD",
    "ROLE_OMNIINTELLIGENCE_PASSWORD",
    "ROLE_OMNICLAUDE_PASSWORD",
    "ROLE_OMNIMEMORY_PASSWORD",
    "ROLE_OMNINODE_PASSWORD",
    "ROLE_OMNIDASH_PASSWORD",
)
PRESERVED_CREDENTIALS = (
    "POSTGRES_PASSWORD",
    "VALKEY_PASSWORD",
    "SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD",
    "SIM_PREFLIGHT_TENANT_BOOTSTRAP_ADMIN_SECRET",
    "SIM_PREFLIGHT_TENANT_TOPICS_ADMIN_SECRET",
    "SIM_PREFLIGHT_TENANT_CLIENTS_ADMIN_SECRET",
    "SIM_PREFLIGHT_TENANT_OFFBOARD_ADMIN_SECRET",
    "SIM_PREFLIGHT_ALPHA_INVITE_ADMIN_SECRET",
    "SIM_PREFLIGHT_DEMO_PASSWORD",
    "SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET",
    "SESSION_SECRET",
    "SIM_PREFLIGHT_STRIPE_API_KEY",
    "SIM_PREFLIGHT_STRIPE_WEBHOOK_SECRET",
)
PRESERVED_FILES = (
    "gateway-private.pem",
    "gateway-keys.json",
    "terminal-private.pem",
    "terminal-public.pem",
)
RENDERED_FILES = (
    "credentials.env",
    "omnidash.env",
    "omnidash-launch.env",
    "omninode-realm.json",
    "main-runtime-config.yaml",
    "effects-runtime-config.yaml",
)
OPTIONAL_PROFILE = "desired-user-profile.json"


def _private_regular(path: Path) -> bytes:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("bundle inputs must be private regular files")
    return path.read_bytes()


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _private_regular(path).decode().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError("private environment file is malformed")
        values[key] = value
    return values


def _rewrite_analytics_dsn(value: str, password: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 65036
        or parsed.path != "/omnidash_analytics"
        or unquote(parsed.username or "") != "role_omnidash"
        or parsed.fragment
    ):
        raise ValueError("OmniDash analytics DSN is outside the disposable database")
    query_fields = parse_qsl(parsed.query, keep_blank_values=True)
    if query_fields and (len(query_fields) != 1 or query_fields[0][0] != "options"):
        raise ValueError("OmniDash analytics DSN has unsupported query parameters")
    return urlunsplit(
        (
            parsed.scheme,
            f"role_omnidash:{quote(password, safe='')}@127.0.0.1:65036",
            parsed.path,
            parsed.query,
            "",
        )
    )


def _env_bytes(values: dict[str, str]) -> bytes:
    if any("\n" in key or "=" in key or "\n" in value for key, value in values.items()):
        raise ValueError("environment values cannot contain newlines")
    return "".join(f"{key}={value}\n" for key, value in sorted(values.items())).encode()


def _canonical(data: object) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()


def _patched_realm(bundle: Path) -> bytes:
    realm = json.loads(_private_regular(bundle / "omninode-realm.json"))
    users = realm.get("users")
    clients = realm.get("clients")
    if (
        realm.get("realm") != "omninode"
        or not isinstance(users, list)
        or len(users) != 1
    ):
        raise ValueError("private realm must be the unimported disposable user realm")
    if not isinstance(clients, list):
        raise ValueError("private realm clients are malformed")
    user_attributes = users[0].get("attributes")
    if not isinstance(user_attributes, dict):
        raise ValueError("private realm user attributes are malformed")
    tenant_values = user_attributes.get("tenant_id")
    if not isinstance(tenant_values, list) or len(tenant_values) != 1:
        raise ValueError("private realm owner tenant is missing")
    from uuid import UUID

    tenant = UUID(tenant_values[0])
    if str(tenant) != tenant_values[0] or tenant.int == 0:
        raise ValueError("private realm owner tenant is invalid")

    # Use the owning Gateway implementation, just as the primary generator does.
    gateway = (
        Path(os.environ["OMNI_HOME"])
        / "omni_worktrees/OMN-19728/omninode_infra/docker/onex-api"
    )
    from scripts.runtime_build.bootstrap_sim_preflight_auth_omn19726 import (
        owner_principal,
    )

    principal = owner_principal(gateway, str(tenant))
    tenant_slug = user_attributes.get("tenant_slug")
    if tenant_slug not in (None, ["sim-preflight"]):
        raise ValueError("private realm tenant slug conflicts with disposable scope")
    user_attributes["tenant_slug"] = ["sim-preflight"]
    existing = user_attributes.get("principal_id")
    if existing not in (None, [principal]):
        raise ValueError("private realm principal conflicts with selected owner")
    user_attributes["principal_id"] = [principal]

    matching_clients = [
        client for client in clients if client.get("clientId") == "omnidash"
    ]
    if len(matching_clients) != 1:
        raise ValueError("private realm must contain exactly one OmniDash client")
    client = matching_clients[0]
    mappers = client.setdefault("protocolMappers", [])
    if not isinstance(mappers, list):
        raise ValueError("OmniDash protocol mappers are malformed")
    matches = [
        mapper
        for mapper in mappers
        if mapper.get("config", {}).get("claim.name") == "principal_id"
    ]
    if matches and matches != [PRINCIPAL_MAPPER]:
        raise ValueError(
            "private realm principal mapper conflicts with canonical mapper"
        )
    if not matches:
        mappers.append(copy.deepcopy(PRINCIPAL_MAPPER))
    return _canonical(realm)


def _make_backup(bundle: Path, files: tuple[str, ...]) -> Path:
    if stat.S_IMODE(bundle.parent.stat().st_mode) != 0o700:
        raise ValueError("private bundle parent must have mode 0700 for backup")
    backup = Path(
        tempfile.mkdtemp(prefix="sim-preflight-auth-backup-", dir=bundle.parent)
    )
    backup.chmod(0o700)
    try:
        for filename in files:
            contents = _private_regular(bundle / filename)
            target = backup / filename
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(contents)
    except Exception:
        # Do not remove a partial secret backup automatically; the caller gets
        # a private recovery directory and an exception with no secret content.
        raise
    return backup


def correct_bundle(bundle: Path) -> dict[str, Any]:
    """Update a verified unimported bundle; return only non-secret field names."""
    if not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("bundle must be an existing absolute directory")
    if stat.S_IMODE(bundle.stat().st_mode) != 0o700:
        raise ValueError("bundle directory must have mode 0700")
    for filename in (*RENDERED_FILES, *PRESERVED_FILES):
        _private_regular(bundle / filename)
    profile_source = ROOT / "docker/keycloak/desired-user-profile.json"
    profile_bytes = profile_source.read_bytes()
    profile_target = bundle / OPTIONAL_PROFILE
    if profile_target.is_symlink():
        raise ValueError("private profile target must not be a symlink")
    if profile_target.exists():
        if _private_regular(profile_target) != profile_bytes:
            raise ValueError("existing private profile differs from canonical profile")
    env = _read_env(bundle / "credentials.env")
    for name in PRESERVED_CREDENTIALS:
        if not env.get(name):
            raise ValueError("required preserved credential is missing")
    for name in (
        "SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME",
        "SIM_PREFLIGHT_DEMO_USERNAME",
    ):
        if not env.get(name):
            raise ValueError("required disposable identity alias is missing")

    changed: dict[str, bytes] = {}
    updated_env = dict(env)
    for name in ROLE_NAMES:
        updated_env[name] = secrets.token_hex(32)
    updated_env["SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"] = updated_env[
        "ROLE_OMNINODE_PASSWORD"
    ]
    changed["credentials.env"] = _env_bytes(updated_env)

    launch_env = _read_env(bundle / "omnidash-launch.env")
    if "OMNIDASH_ANALYTICS_DB_URL" not in launch_env:
        raise ValueError("OmniDash launch environment lacks its analytics DSN")
    launch_env["OMNIDASH_ANALYTICS_DB_URL"] = _rewrite_analytics_dsn(
        launch_env["OMNIDASH_ANALYTICS_DB_URL"], updated_env["ROLE_OMNIDASH_PASSWORD"]
    )
    changed["omnidash-launch.env"] = _env_bytes(launch_env)

    dash = _read_env(bundle / "omnidash.env")
    expected_dash = {
        "KEYCLOAK_ISSUER": "http://auth.localhost:28080/realms/omninode",
        "KEYCLOAK_CLIENT_ID": "omnidash",
        "KEYCLOAK_CLIENT_SECRET": env["SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET"],
        "SESSION_SECRET": env["SESSION_SECRET"],
    }
    if any(dash.get(key) not in (None, value) for key, value in expected_dash.items()):
        raise ValueError("existing OmniDash auth environment conflicts")
    dash.update(expected_dash)
    changed["omnidash.env"] = _env_bytes(dash)

    profile = json.loads(profile_bytes)
    changed[OPTIONAL_PROFILE] = profile_bytes
    changed["omninode-realm.json"] = _patched_realm(bundle)
    main_config, effects_config = runtime_configs(bundle, updated_env)
    changed["main-runtime-config.yaml"] = yaml.safe_dump(
        main_config, sort_keys=False
    ).encode()
    changed["effects-runtime-config.yaml"] = yaml.safe_dump(
        effects_config, sort_keys=False
    ).encode()

    # Validate role aliases and canonical identity profile before creating backups.
    if (
        updated_env["SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"]
        != updated_env["ROLE_OMNINODE_PASSWORD"]
    ):
        raise ValueError("cloud role alias does not match its role credential")
    if not isinstance(profile.get("attributes"), list):
        raise ValueError("canonical user profile is malformed")

    existing_changed = tuple(
        filename for filename in changed if (bundle / filename).exists()
    )
    if profile_target.exists() and OPTIONAL_PROFILE not in existing_changed:
        existing_changed += (OPTIONAL_PROFILE,)
    backup = _make_backup(bundle, existing_changed)
    if OPTIONAL_PROFILE not in existing_changed:
        (backup / "absent-files.json").write_bytes(
            _canonical({"absent_before_correction": [OPTIONAL_PROFILE]})
        )
        (backup / "absent-files.json").chmod(0o600)
    for filename, data in changed.items():
        target = bundle / filename
        temporary = bundle / f".{filename}.new"
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("temporary correction path already exists")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    return {
        "corrected_fields": [
            *ROLE_NAMES,
            "SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD",
            "OmniDash auth environment",
            "OmniDash analytics DSN",
            "graph DB DSNs",
            "realm principal mapper/profile",
        ],
        "backup_directory": str(backup),
        "preserved_fields": [*PRESERVED_CREDENTIALS, *PRESERVED_FILES],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    result = correct_bundle(parser.parse_args().bundle)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 -- exception bodies can contain paths/secrets
        print(
            json.dumps(
                {"bundle_correction_refused": True, "error_type": type(exc).__name__}
            )
        )
        raise SystemExit(65) from None
