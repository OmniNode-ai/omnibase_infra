# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No-service tests for private disposable auth generation."""

from __future__ import annotations

import base64
import json
import re
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.runtime_build.bootstrap_sim_preflight_auth_omn19726 import (
    BASIC_SCOPE,
    GATEWAY_RUNTIME_ID,
    PRINCIPAL_MAPPER,
    PROJECT,
    ROOT,
    SUBJECT_MAPPER,
    graph_catalog,
    prepare,
)

pytestmark = pytest.mark.unit
TENANT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, object]:
    owner = tmp_path / "owner.metadata"
    owner.write_text(
        f"tenant_id={TENANT}\nsource_db=omnidash_analytics\nsource_table=public.delegation_events\n"
    )
    gateway = tmp_path / "gateway"
    gateway.mkdir()
    (gateway / "workflow-contracts.yaml").write_text(
        yaml.safe_dump(
            {
                "workflows": [
                    {
                        "workflow_type": "delegation-execution-graph-read",
                        "enabled": False,
                    },
                    {"workflow_type": "delegation-inference"},
                ]
            }
        )
    )
    (gateway / "tenant_identity.py").write_text(
        "def derive_principal_id(tenant_id):\n    return f't-{tenant_id.hex}'\n"
    )
    dash = tmp_path / "dash/deploy/keycloak"
    dash.mkdir(parents=True)
    (dash / "omnidash-client.json").write_text(
        json.dumps(
            {
                "clientId": "omnidash",
                "directAccessGrantsEnabled": False,
                "serviceAccountsEnabled": False,
                "publicClient": False,
                "standardFlowEnabled": True,
                "attributes": {"pkce.code.challenge.method": "S256"},
                "protocolMappers": [
                    {
                        "name": "onex-api-audience",
                        "config": {"included.client.audience": "onex-api"},
                    }
                ],
            }
        )
    )
    (dash / "tenant-client-scope.json").write_text(
        json.dumps(
            {
                "name": "tenant",
                "protocolMappers": [
                    {
                        "name": "tenant-id",
                        "config": {
                            "user.attribute": "tenant_id",
                            "claim.name": "tenant_id",
                        },
                    }
                ],
            }
        )
    )
    return {
        "project": PROJECT,
        "owner_metadata": owner,
        "output_dir": tmp_path / "private",
        "gateway": gateway,
        "omnidash": dash.parents[1],
        "omnidash_origin": "http://localhost:3000",
    }


def _prepare(inputs: dict[str, object]) -> dict[str, str]:
    original_run = subprocess.run

    def gateway_model_run(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        command = args[0]
        if "ModelWorkflowContracts" in command[2]:  # type: ignore[index]
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")
        return original_run(*args, **kwargs)  # type: ignore[arg-type]

    with patch("subprocess.run", side_effect=gateway_model_run):
        return prepare(**inputs)  # type: ignore[arg-type]


def test_private_bundle_real_keys_tenant_and_graph_only_scope(
    inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "inherited-real-secret-must-not-be-used")
    result = _prepare(inputs)
    directory = Path(result["directory"])
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir()
    )
    env = dict(
        line.split("=", 1)
        for line in (directory / "credentials.env").read_text().splitlines()
    )
    assert env["POSTGRES_PASSWORD"] != "inherited-real-secret-must-not-be-used"
    assert len(env["POSTGRES_PASSWORD"]) >= 40
    assert env["ROLE_OMNINODE_PASSWORD"] == env["SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"]
    role_passwords = {
        "OMNINODE_RUNTIME_PASSWORD",
        "TENANT_PROJECTION_WRITER_PASSWORD",
        *(
            name
            for name in env
            if name.startswith("ROLE_") and name.endswith("_PASSWORD")
        ),
    }
    assert len(role_passwords) == 8
    assert all(re.fullmatch(r"[0-9a-f]{64}", env[name]) for name in role_passwords)
    assert len(
        {
            env[name]
            for name in env
            if name.endswith("PASSWORD") and name != "SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"
        }
    ) == len(
        [
            name
            for name in env
            if name.endswith("PASSWORD") and name != "SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"
        ]
    )
    realm = json.loads((directory / "omninode-realm.json").read_text())
    assert realm["realm"] == "omninode"
    assert realm["users"][0]["attributes"]["tenant_id"] == [TENANT]
    assert realm["users"][0]["attributes"]["principal_id"] == [
        f"t-{__import__('uuid').UUID(TENANT).hex}"
    ]
    assert (
        realm["users"][0]["credentials"][0]["value"]
        == env["SIM_PREFLIGHT_DEMO_PASSWORD"]
    )
    client = realm["clients"][0]
    basic_scopes = [
        scope for scope in realm["clientScopes"] if scope["name"] == "basic"
    ]
    assert basic_scopes == [BASIC_SCOPE]
    assert basic_scopes[0]["protocolMappers"] == [SUBJECT_MAPPER]
    assert SUBJECT_MAPPER["protocolMapper"] == "oidc-sub-mapper"
    assert SUBJECT_MAPPER["config"] == {
        "access.token.claim": "true",
        "introspection.token.claim": "true",
    }
    assert client["clientId"] == "omnidash"
    assert (
        client["description"]
        == "Disposable OmniDash replay validation client (OMN-19726)."
    )
    assert len(client["description"]) <= 255
    assert client["secret"] == env["SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET"]
    assert client["redirectUris"] == ["http://localhost:3000/*"]
    assert client["directAccessGrantsEnabled"] is False
    assert client["serviceAccountsEnabled"] is False
    # Keycloak 25+ carries the access-token subject mapper in its built-in
    # basic scope; replacing realm client scopes must not omit that scope.
    assert "basic" in client["defaultClientScopes"]
    assert (
        client["protocolMappers"][0]["config"]["included.client.audience"] == "onex-api"
    )
    principal_mappers = [
        mapper
        for mapper in client["protocolMappers"]
        if mapper.get("config", {}).get("claim.name") == "principal_id"
    ]
    assert principal_mappers == [PRINCIPAL_MAPPER]
    assert principal_mappers[0]["config"]["access.token.claim"] == "true"
    assert principal_mappers[0]["config"]["introspection.token.claim"] == "true"
    canonical_profile = ROOT / "docker/keycloak/desired-user-profile.json"
    assert (
        directory / "desired-user-profile.json"
    ).read_bytes() == canonical_profile.read_bytes()
    profile = json.loads((directory / "desired-user-profile.json").read_text())
    principal_profile = [
        a for a in profile["attributes"] if a["name"] == "principal_id"
    ]
    assert len(principal_profile) == 1
    assert principal_profile[0]["permissions"] == {"view": ["admin"], "edit": ["admin"]}
    entries = yaml.safe_load((directory / "workflow-contracts.yaml").read_text())[
        "workflows"
    ]
    assert [entry["workflow_type"] for entry in entries if entry["enabled"]] == [
        "delegation-execution-graph-read"
    ]
    gateway_key = serialization.load_pem_private_key(
        (directory / "gateway-private.pem").read_bytes(), password=None
    )
    terminal_key = serialization.load_pem_private_key(
        (directory / "terminal-private.pem").read_bytes(), password=None
    )
    assert isinstance(gateway_key, Ed25519PrivateKey)
    assert isinstance(terminal_key, Ed25519PrivateKey)
    raw_public = base64.urlsafe_b64decode(
        json.loads((directory / "gateway-keys.json").read_text())["keys"][
            GATEWAY_RUNTIME_ID
        ]
    )
    assert raw_public == gateway_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    terminal_public = serialization.load_pem_public_key(
        (directory / "terminal-public.pem").read_bytes()
    )
    terminal_public.verify(terminal_key.sign(b"probe"), b"probe")  # type: ignore[union-attr]
    main = yaml.safe_load((directory / "main-runtime-config.yaml").read_text())
    effects = yaml.safe_load((directory / "effects-runtime-config.yaml").read_text())
    for config in (main, effects):
        assert config["graph_ledger_node_allowlist"]["runtime_lane"] == "sim-202"
        assert len(config["graph_ledger_node_allowlist"]["nodes"]) == 4
        assert all(
            config[name]["enabled"] is False
            for name in ("local_ingress", "pattern_b_broker", "contract_registry")
        )
    assert "execution_graph_read" not in main
    assert (
        effects["execution_graph_read_gateway"]["public_key_path"]
        == "/app/config/execution-graph/gateway-keys.json"
    )
    assert (
        effects["execution_graph_read"]["private_key_path"]
        == "/app/config/execution-graph/terminal-private.pem"
    )
    assert (
        "postgresql://role_omnidash:"
        in effects["execution_graph_read"]["databases"]["analytics_dsn"]
    )
    assert (
        "postgresql://role_omnibase:"
        in effects["execution_graph_read"]["databases"]["ledger_dsn"]
    )
    assert capsys.readouterr().out == ""
    assert all(
        value not in json.dumps(result)
        for name, value in env.items()
        if name.endswith("PASSWORD")
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project", "omnibase-infra-sim-202"),
        ("omnidash_origin", "https://dev.dash.omninode.ai"),
        ("omnidash_origin", "http://localhost:3000/escape"),
        ("output_dir", Path("relative-private")),
    ],
)
def test_refuses_nondisposable_scope_before_writing(
    inputs: dict[str, object], field: str, value: object
) -> None:
    inputs[field] = value
    with pytest.raises(ValueError):
        _prepare(inputs)
    assert not Path(inputs["output_dir"]).exists()  # type: ignore[arg-type]


def test_refuses_git_destination_reuse_and_wrong_owner(
    inputs: dict[str, object], tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    inputs["output_dir"] = repo / "private"
    with pytest.raises(ValueError, match="Git"):
        _prepare(inputs)
    inputs["output_dir"] = tmp_path / "private"
    _prepare(inputs)
    with pytest.raises(ValueError, match="new absolute"):
        _prepare(inputs)
    inputs["output_dir"] = tmp_path / "other-private"
    Path(inputs["owner_metadata"]).write_text(
        f"tenant_id={TENANT}\nsource_db=other\nsource_table=public.delegation_events\n"
    )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source scope"):
        _prepare(inputs)


def test_catalog_uses_owning_model_and_refuses_duplicate_graph(
    inputs: dict[str, object],
) -> None:
    gateway = Path(inputs["gateway"])  # type: ignore[arg-type]
    with patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 1)
    ) as run:
        with pytest.raises(ValueError, match="Gateway model"):
            graph_catalog(gateway)
        assert "ModelWorkflowContracts" in run.call_args.args[0][2]
        assert run.call_args.kwargs["env"].keys() == {"PATH"}
    (gateway / "workflow-contracts.yaml").write_text(
        yaml.safe_dump(
            {"workflows": [{"workflow_type": "delegation-execution-graph-read"}] * 2}
        )
    )
    with pytest.raises(ValueError, match="uniquely"):
        graph_catalog(gateway)


def test_refuses_conflicting_principal_mapper_before_private_output(
    inputs: dict[str, object],
) -> None:
    dash = Path(inputs["omnidash"]) / "deploy/keycloak/omnidash-client.json"  # type: ignore[arg-type]
    client = json.loads(dash.read_text())
    client["protocolMappers"].append(
        {
            "name": "wrong-principal",
            "protocol": "openid-connect",
            "protocolMapper": "oidc-usermodel-attribute-mapper",
            "consentRequired": False,
            "config": {
                "user.attribute": "tenant_slug",
                "claim.name": "principal_id",
                "jsonType.label": "String",
                "id.token.claim": "false",
                "access.token.claim": "true",
                "userinfo.token.claim": "false",
            },
        }
    )
    dash.write_text(json.dumps(client))

    with pytest.raises(ValueError, match="principal mapper conflicts"):
        _prepare(inputs)
    assert not Path(inputs["output_dir"]).exists()  # type: ignore[arg-type]
