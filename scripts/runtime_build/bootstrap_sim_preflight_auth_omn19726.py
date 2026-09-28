# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generate private, disposable-only auth inputs; never contact a service."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "omnibase-infra-sim-preflight"
GRAPH_WORKFLOW = "delegation-execution-graph-read"
ISSUER = "http://auth.localhost:28080/realms/omninode"
GATEWAY_RUNTIME_ID = "onex-api-sim-preflight"
TERMINAL_RUNTIME_ID = "runtime-effects-sim-preflight"
PRINCIPAL_MAPPER = {
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
SUBJECT_MAPPER = {
    "name": "sub",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-sub-mapper",
    "consentRequired": False,
    "config": {
        "access.token.claim": "true",
        "introspection.token.claim": "true",
    },
}
BASIC_SCOPE = {
    "name": "basic",
    "description": "Disposable Keycloak basic scope with native Subject (sub) mapper.",
    "protocol": "openid-connect",
    "attributes": {
        "include.in.token.scope": "false",
        "display.on.consent.screen": "false",
    },
    "protocolMappers": [SUBJECT_MAPPER],
}


def owner_tenant(metadata: Path) -> str:
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError("owner metadata must be a regular file")
    fields: dict[str, str] = {}
    for line in metadata.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError("owner metadata is malformed")
        fields[key] = value
    if (
        fields.get("source_db") != "omnidash_analytics"
        or fields.get("source_table") != "public.delegation_events"
    ):
        raise ValueError("owner metadata has wrong source scope")
    tenant = fields.get("tenant_id", "")
    parsed = UUID(tenant)
    if str(parsed) != tenant or parsed.int == 0:
        raise ValueError("owner tenant must be a canonical nonempty UUID")
    return tenant


def graph_catalog(gateway: Path) -> dict[str, Any]:
    raw: Any = yaml.safe_load((gateway / "workflow-contracts.yaml").read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("workflows"), list):
        raise ValueError("Gateway catalog is malformed")
    matches = 0
    for entry in raw["workflows"]:
        if not isinstance(entry, dict):
            raise ValueError("Gateway catalog entry is malformed")
        entry["enabled"] = entry.get("workflow_type") == GRAPH_WORKFLOW
        matches += int(entry["enabled"])
    if matches != 1:
        raise ValueError("Gateway graph workflow is not uniquely declared")
    # Validate through the owning Gateway models, not a duplicate schema.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;sys.path.insert(0,sys.argv[1]);from models.model_workflow_contracts import ModelWorkflowContracts;ModelWorkflowContracts.model_validate(json.load(sys.stdin))",
            str(gateway),
        ],
        input=json.dumps(raw),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    if result.returncode:
        raise ValueError("Gateway model refused private workflow catalog")
    return raw


def owner_principal(gateway: Path, tenant_id: str) -> str:
    """Derive the private user principal with the owning Gateway function."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);from uuid import UUID;"
            "from tenant_identity import derive_principal_id;"
            "print(derive_principal_id(UUID(sys.argv[2])))",
            str(gateway),
            tenant_id,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    principal = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"t-[0-9a-f]{32}", principal):
        raise ValueError("Gateway refused disposable owner principal derivation")
    return principal


def _write(directory: Path, name: str, data: bytes) -> Path:
    path = directory / name
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    return path


def _json(data: object) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()


def _keypair() -> tuple[bytes, bytes, bytes]:
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key()
    return (
        private,
        public.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
        public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
    )


def runtime_configs(
    directory: Path, env: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
        ModelExecutionGraphTopologyVersion,
    )
    from omnibase_infra.runtime.execution_graph_topology_registry import (
        PackagedExecutionGraphTopologyContract,
    )
    from omnibase_infra.runtime.models.model_graph_ledger_node_allowlist import (
        GRAPH_LEDGER_NODES,
    )
    from omnibase_infra.runtime.models.model_runtime_config import ModelRuntimeConfig

    snapshots = sorted(
        (ROOT / "src/omnibase_infra/runtime/execution_graph_topologies").glob("*.json")
    )
    if len(snapshots) != 1:
        raise ValueError("pinned graph topology must be uniquely declared")
    snapshot = json.loads(snapshots[0].read_text())
    version = ModelExecutionGraphTopologyVersion.model_validate(
        {
            "contract_version": snapshot["contract_version"],
            "topology_sha256": snapshot["topology_sha256"],
        }
    )
    PackagedExecutionGraphTopologyContract(snapshots[0].parent).resolve(version)
    contract = yaml.safe_load(
        (
            ROOT
            / "src/omnibase_infra/nodes/node_execution_graph_read_effect/contract.yaml"
        ).read_text()
    )
    command_topics = contract["event_bus"]["subscribe_topics"]
    terminal_topics = contract["event_bus"]["publish_topics"]
    if len(command_topics) != 1 or len(terminal_topics) != 1:
        raise ValueError("graph contract routes must be unique")
    main: dict[str, Any] = {
        "name": "sim-preflight-main",
        "group_id": "sim-preflight-main",
        "event_bus": {"type": "kafka", "profile": "lane", "environment": "local"},
        "contract_registry": {"enabled": False},
        "local_ingress": {"enabled": False},
        "pattern_b_broker": {"enabled": False},
        "graph_ledger_node_allowlist": {
            "runtime_lane": "sim-202",
            "nodes": list(GRAPH_LEDGER_NODES),
        },
    }
    effects = copy.deepcopy(main)
    effects.update(
        name="sim-preflight-effects",
        group_id="sim-preflight-effects",
        execution_graph_read_gateway={
            "command_topic": command_topics[0],
            "runtime_id": GATEWAY_RUNTIME_ID,
            "realm": "sim-preflight",
            "bus_id": "sim-preflight",
            "public_key_path": str(directory / "gateway-keys.json"),
        },
        execution_graph_read={
            "databases": {
                "analytics_dsn": f"postgresql://role_omnidash:{env['ROLE_OMNIDASH_PASSWORD']}@postgres:5432/omnidash_analytics",
                "ledger_dsn": f"postgresql://role_omnibase:{env['ROLE_OMNIBASE_PASSWORD']}@postgres:5432/omnibase_infra",
            },
            "topology_version": version.model_dump(mode="json"),
            "terminal_publisher": {
                "terminal_topic": terminal_topics[0],
                "runtime_id": TERMINAL_RUNTIME_ID,
                "realm": "sim-preflight",
                "bus_id": "sim-preflight",
                "workflow_type": GRAPH_WORKFLOW,
            },
            "private_key_path": str(directory / "terminal-private.pem"),
        },
    )
    # Validate with real existing host key files, then set the reviewed mount paths.
    # Final model validation with these container paths is a separate offline image gate.
    ModelRuntimeConfig.model_validate(main)
    ModelRuntimeConfig.model_validate(effects)
    effects["execution_graph_read_gateway"]["public_key_path"] = (
        "/app/config/execution-graph/gateway-keys.json"
    )
    effects["execution_graph_read"]["private_key_path"] = (
        "/app/config/execution-graph/terminal-private.pem"
    )
    return main, effects


def prepare(
    *,
    project: str,
    owner_metadata: Path,
    output_dir: Path,
    gateway: Path,
    omnidash: Path,
    omnidash_origin: str,
) -> dict[str, str]:
    if project != PROJECT:
        raise ValueError("only the disposable sim-preflight project is supported")
    parsed = urlparse(omnidash_origin)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "localhost"
        or parsed.port != 3000
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only the approved localhost:3000 browser origin is supported")
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output directory must be a new absolute private path")
    output_dir = output_dir.resolve()
    if any((parent / ".git").exists() for parent in (output_dir, *output_dir.parents)):
        raise ValueError("credentials must not be written inside a Git checkout")
    tenant = owner_tenant(owner_metadata)
    principal_id = owner_principal(gateway, tenant)
    catalog = graph_catalog(gateway)
    realm: dict[str, Any] = json.loads(
        (ROOT / "docker/keycloak/omninode-realm.json").read_text()
    )
    if realm.get("realm") != "omninode" or realm.get("clients") or realm.get("users"):
        raise ValueError("canonical disposable realm seed changed")
    client = copy.deepcopy(
        json.loads((omnidash / "deploy/keycloak/omnidash-client.json").read_text())
    )
    tenant_scope = json.loads(
        (omnidash / "deploy/keycloak/tenant-client-scope.json").read_text()
    )
    if (
        client.get("clientId") != "omnidash"
        or client.get("directAccessGrantsEnabled") is not False
    ):
        raise ValueError("canonical OmniDash auth-code client changed")
    default_scopes = client.setdefault("defaultClientScopes", [])
    if not isinstance(default_scopes, list) or not all(
        isinstance(scope, str) for scope in default_scopes
    ):
        raise ValueError("canonical OmniDash client scopes are malformed")
    # Keycloak 25+ provides the `sub` access-token mapper in its built-in
    # `basic` client scope. Keep it explicitly assigned when the private realm
    # import replaces the realm's client-scope collection.
    if "basic" not in default_scopes:
        default_scopes.append("basic")
    profile_path = ROOT / "docker/keycloak/desired-user-profile.json"
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    profile_attributes = profile.get("attributes")
    principal_definitions = [
        item
        for item in profile_attributes or []
        if isinstance(item, dict) and item.get("name") == "principal_id"
    ]
    if len(principal_definitions) != 1:
        raise ValueError("canonical profile must declare principal_id exactly once")
    principal_definition = principal_definitions[0]
    if (
        principal_definition.get("permissions")
        != {"view": ["admin"], "edit": ["admin"]}
        or principal_definition.get("multivalued") is not False
    ):
        raise ValueError("canonical principal_id profile must remain admin-only")
    principal_mappers = [
        mapper
        for mapper in client.get("protocolMappers", [])
        if mapper.get("config", {}).get("claim.name") == "principal_id"
    ]
    if principal_mappers and principal_mappers != [PRINCIPAL_MAPPER]:
        raise ValueError("OmniDash principal mapper conflicts with test contract")
    if not principal_mappers:
        client.setdefault("protocolMappers", []).append(copy.deepcopy(PRINCIPAL_MAPPER))
    client_secret = secrets.token_urlsafe(32)
    user_password = secrets.token_urlsafe(32)
    session_secret = secrets.token_urlsafe(48)
    client.update(
        description="Disposable OmniDash replay validation client (OMN-19726).",
        secret=client_secret,
        rootUrl=omnidash_origin,
        redirectUris=[omnidash_origin + "/*"],
        webOrigins=[omnidash_origin],
    )
    client["attributes"]["post.logout.redirect.uris"] = omnidash_origin + "/*"
    realm.update(
        registrationAllowed=False,
        resetPasswordAllowed=False,
        clients=[client],
        clientScopes=[tenant_scope, copy.deepcopy(BASIC_SCOPE)],
        users=[
            {
                "username": "sim-preflight-demo",
                "enabled": True,
                "email": "sim-preflight-demo@example.invalid",
                "emailVerified": True,
                "firstName": "Disposable",
                "lastName": "Demo",
                "requiredActions": [],
                "attributes": {
                    "tenant_id": [tenant],
                    "tenant_slug": ["sim-preflight"],
                    "org_id": [tenant],
                    "principal_id": [principal_id],
                },
                "credentials": [
                    {"type": "password", "value": user_password, "temporary": False}
                ],
            }
        ],
    )
    gateway_private, _, gateway_public = _keypair()
    terminal_private, terminal_public, _ = _keypair()
    env = {
        name: secrets.token_urlsafe(32)
        for name in (
            "POSTGRES_PASSWORD",
            "VALKEY_PASSWORD",
            "SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD",
            "SIM_PREFLIGHT_TENANT_BOOTSTRAP_ADMIN_SECRET",
            "SIM_PREFLIGHT_TENANT_TOPICS_ADMIN_SECRET",
            "SIM_PREFLIGHT_TENANT_CLIENTS_ADMIN_SECRET",
            "SIM_PREFLIGHT_TENANT_OFFBOARD_ADMIN_SECRET",
            "SIM_PREFLIGHT_ALPHA_INVITE_ADMIN_SECRET",
        )
    }
    # Canonical forward migrations interpolate these role credentials only
    # after a strict hex check (the documented openssl rand -hex 32 contract).
    env.update(
        {
            name: secrets.token_hex(32)
            for name in (
                "OMNINODE_RUNTIME_PASSWORD",
                "TENANT_PROJECTION_WRITER_PASSWORD",
                "ROLE_OMNIBASE_PASSWORD",
                "ROLE_OMNIINTELLIGENCE_PASSWORD",
                "ROLE_OMNICLAUDE_PASSWORD",
                "ROLE_OMNIMEMORY_PASSWORD",
                "ROLE_OMNINODE_PASSWORD",
                "ROLE_OMNIDASH_PASSWORD",
            )
        }
    )
    env.update(
        {
            "SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD": env["ROLE_OMNINODE_PASSWORD"],
            "SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME": "sim-preflight-admin",
            "SIM_PREFLIGHT_DEMO_USERNAME": "sim-preflight-demo",
            "SIM_PREFLIGHT_DEMO_PASSWORD": user_password,
            "SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET": client_secret,
            "SESSION_SECRET": session_secret,
            "SIM_PREFLIGHT_STRIPE_API_KEY": "sk_test_disposable_"
            + secrets.token_urlsafe(24),
            "SIM_PREFLIGHT_STRIPE_WEBHOOK_SECRET": "whsec_disposable_"
            + secrets.token_urlsafe(24),
            "SIM_PREFLIGHT_STRIPE_CHECKOUT_SUCCESS_URL": omnidash_origin
            + "/disposable-success",
            "SIM_PREFLIGHT_STRIPE_CHECKOUT_CANCEL_URL": omnidash_origin
            + "/disposable-cancel",
            "SIM_PREFLIGHT_STRIPE_PORTAL_RETURN_URL": omnidash_origin,
            "SIM_PREFLIGHT_SIGNER_REALM": "sim-preflight",
            "SIM_PREFLIGHT_SIGNER_BUS_ID": "sim-preflight",
            "SIM_PREFLIGHT_GATEWAY_RUNTIME_ID": GATEWAY_RUNTIME_ID,
            "SIM_PREFLIGHT_TERMINAL_RUNTIME_ID": TERMINAL_RUNTIME_ID,
            "SIM_PREFLIGHT_TERMINAL_CONSUMER_GROUP_ID": "sim-preflight-graph-terminal",
        }
    )
    files = {
        "omninode-realm.json": _json(realm),
        "desired-user-profile.json": profile_bytes,
        "workflow-contracts.yaml": yaml.safe_dump(catalog, sort_keys=False).encode(),
        "gateway-private.pem": gateway_private,
        "gateway-keys.json": _json(
            {
                "keys": {
                    GATEWAY_RUNTIME_ID: base64.urlsafe_b64encode(
                        gateway_public
                    ).decode()
                }
            }
        ),
        "terminal-private.pem": terminal_private,
        "terminal-public.pem": terminal_public,
    }
    mounts = {
        "SIM_PREFLIGHT_KEYCLOAK_REALM_FILE": "omninode-realm.json",
        "SIM_PREFLIGHT_GATEWAY_CATALOG_FILE": "workflow-contracts.yaml",
        "SIM_PREFLIGHT_GATEWAY_SIGNING_PRIVATE_KEY_FILE": "gateway-private.pem",
        "SIM_PREFLIGHT_GRAPH_GATEWAY_KEYMAP_FILE": "gateway-keys.json",
        "SIM_PREFLIGHT_GRAPH_TERMINAL_PRIVATE_KEY_FILE": "terminal-private.pem",
        "SIM_PREFLIGHT_GRAPH_TERMINAL_PUBLIC_KEY_FILE": "terminal-public.pem",
        "SIM_PREFLIGHT_GRAPH_MAIN_RUNTIME_CONFIG_FILE": "main-runtime-config.yaml",
        "SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE": "effects-runtime-config.yaml",
    }
    env.update({name: str(output_dir / filename) for name, filename in mounts.items()})
    dash = {
        "KEYCLOAK_ISSUER": ISSUER,
        "KEYCLOAK_CLIENT_ID": "omnidash",
        "KEYCLOAK_CLIENT_SECRET": client_secret,
        "SESSION_SECRET": session_secret,
    }
    files["credentials.env"] = "".join(
        f"{key}={value}\n" for key, value in sorted(env.items())
    ).encode()
    files["omnidash.env"] = "".join(
        f"{key}={value}\n" for key, value in sorted(dash.items())
    ).encode()
    output_dir.mkdir(mode=0o700)
    for name, data in files.items():
        _write(output_dir, name, data)
    main_config, effects_config = runtime_configs(output_dir, env)
    _write(
        output_dir,
        "main-runtime-config.yaml",
        yaml.safe_dump(main_config, sort_keys=False).encode(),
    )
    _write(
        output_dir,
        "effects-runtime-config.yaml",
        yaml.safe_dump(effects_config, sort_keys=False).encode(),
    )
    return {
        "directory": str(output_dir),
        "env_file": str(output_dir / "credentials.env"),
        "omnidash_env_file": str(output_dir / "omnidash.env"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--owner-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omnidash-origin", required=True)
    args = parser.parse_args()
    source_root = Path(os.environ["OMNI_HOME"])
    result = prepare(
        project=args.project,
        owner_metadata=args.owner_metadata,
        output_dir=args.output_dir,
        gateway=source_root / "omni_worktrees/OMN-19728/omninode_infra/docker/onex-api",
        omnidash=source_root / "omni_worktrees/OMN-19730/omnidash",
        omnidash_origin=args.omnidash_origin,
    )
    print("disposable auth inputs prepared; private env file=" + result["env_file"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError):
        print("disposable auth bootstrap refused; details redacted", file=sys.stderr)
        raise SystemExit(65) from None
