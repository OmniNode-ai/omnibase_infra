# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Source-only tests for the scoped existing private-bundle correction."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from scripts.runtime_build import bootstrap_sim_preflight_auth_omn19726 as bootstrap
from scripts.runtime_build import correct_sim_preflight_auth_bundle as correction

pytestmark = pytest.mark.unit


@pytest.fixture
def private_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    bundle.chmod(0o700)
    env = {name: f"preserved-{name}" for name in correction.PRESERVED_CREDENTIALS}
    env.update(
        {
            "SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME": "sim-preflight-admin",
            "SIM_PREFLIGHT_DEMO_USERNAME": "sim-preflight-demo",
            "ROLE_OMNINODE_PASSWORD": "stale-cloud-role",
        }
    )
    env.update(dict.fromkeys(correction.ROLE_NAMES, "stale-role"))
    env["SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"] = "stale-cloud-role"
    env.update(
        {
            "SIM_PREFLIGHT_GRAPH_MAIN_RUNTIME_CONFIG_FILE": str(
                bundle / "main-runtime-config.yaml"
            ),
            "SIM_PREFLIGHT_GRAPH_RUNTIME_CONFIG_FILE": str(
                bundle / "effects-runtime-config.yaml"
            ),
        }
    )
    (bundle / "credentials.env").write_text(
        "".join(f"{key}={value}\n" for key, value in env.items())
    )
    (bundle / "omnidash.env").write_text(
        "KEYCLOAK_ISSUER=http://auth.localhost:28080/realms/omninode\n"
        "KEYCLOAK_CLIENT_ID=omnidash\n"
        f"KEYCLOAK_CLIENT_SECRET={env['SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET']}\n"
        f"SESSION_SECRET={env['SESSION_SECRET']}\n"
    )
    (bundle / "omnidash-launch.env").write_text(
        "OMNIDASH_ANALYTICS_DB_URL="
        "postgresql://role_omnidash:stale-password@127.0.0.1:65036/omnidash_analytics?options=-csearch_path%3Dpublic\n"
        "UNCHANGED_SETTING=keep-me\n"
    )
    realm = {
        "realm": "omninode",
        "users": [
            {"attributes": {"tenant_id": ["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"]}}
        ],
        "clients": [{"clientId": "omnidash", "protocolMappers": []}],
    }
    (bundle / "omninode-realm.json").write_text(json.dumps(realm))
    for filename in ("main-runtime-config.yaml", "effects-runtime-config.yaml"):
        (bundle / filename).write_text("old config\n")
    for filename in correction.PRESERVED_FILES:
        (bundle / filename).write_bytes(f"untouched:{filename}".encode())
    for path in bundle.iterdir():
        path.chmod(0o600)
    profile_path = tmp_path / "docker/keycloak/desired-user-profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(
        (bootstrap.ROOT / "docker/keycloak/desired-user-profile.json").read_bytes()
    )
    monkeypatch.setattr(correction, "ROOT", tmp_path)
    monkeypatch.setattr(
        bootstrap,
        "owner_principal",
        lambda gateway, tenant: "t-aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa",
    )
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    monkeypatch.setattr(
        correction,
        "runtime_configs",
        lambda directory, values: (
            {"name": "main"},
            {
                "execution_graph_read": {
                    "databases": {
                        "analytics_dsn": f"role_omnidash:{values['ROLE_OMNIDASH_PASSWORD']}",
                        "ledger_dsn": f"role_omnibase:{values['ROLE_OMNIBASE_PASSWORD']}",
                    }
                }
            },
        ),
    )
    return bundle


def test_correction_preserves_auth_and_signing_material_and_backups_changed_files(
    private_bundle: Path,
) -> None:
    original_env = correction._read_env(private_bundle / "credentials.env")
    original_changed_files = {
        filename: (private_bundle / filename).read_bytes()
        for filename in correction.RENDERED_FILES
    }
    preserved_signers = {
        filename: (private_bundle / filename).read_bytes()
        for filename in correction.PRESERVED_FILES
    }
    result = correction.correct_bundle(private_bundle)

    assert result["corrected_fields"][:8] == list(correction.ROLE_NAMES)
    assert set(result["preserved_fields"]) >= set(correction.PRESERVED_CREDENTIALS)
    updated_env = correction._read_env(private_bundle / "credentials.env")
    assert all(
        updated_env[key] == original_env[key]
        for key in correction.PRESERVED_CREDENTIALS
    )
    assert all(updated_env[name] != "stale-role" for name in correction.ROLE_NAMES)
    assert all(len(updated_env[name]) == 64 for name in correction.ROLE_NAMES)
    assert all(
        all(char in "0123456789abcdef" for char in updated_env[name])
        for name in correction.ROLE_NAMES
    )
    assert (
        updated_env["SIM_PREFLIGHT_CLOUD_ROLE_PASSWORD"]
        == updated_env["ROLE_OMNINODE_PASSWORD"]
    )
    assert all(
        (private_bundle / name).read_bytes() == data
        for name, data in preserved_signers.items()
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 for path in private_bundle.iterdir()
    )
    backup = Path(result["backup_directory"])
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in backup.iterdir())
    assert {
        filename: (backup / filename).read_bytes()
        for filename in correction.RENDERED_FILES
    } == original_changed_files
    assert json.loads((backup / "absent-files.json").read_text()) == {
        "absent_before_correction": [correction.OPTIONAL_PROFILE]
    }

    realm = json.loads((private_bundle / "omninode-realm.json").read_text())
    assert realm["users"][0]["attributes"]["principal_id"] == [
        "t-aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa"
    ]
    assert realm["clients"][0]["protocolMappers"] == [bootstrap.PRINCIPAL_MAPPER]
    dash = correction._read_env(private_bundle / "omnidash.env")
    assert (
        dash["KEYCLOAK_CLIENT_SECRET"]
        == original_env["SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET"]
    )
    assert dash["SESSION_SECRET"] == original_env["SESSION_SECRET"]
    launch = correction._read_env(private_bundle / "omnidash-launch.env")
    assert launch["OMNIDASH_ANALYTICS_DB_URL"].endswith(
        "@127.0.0.1:65036/omnidash_analytics?options=-csearch_path%3Dpublic"
    )
    assert updated_env["ROLE_OMNIDASH_PASSWORD"] in launch["OMNIDASH_ANALYTICS_DB_URL"]
    assert launch["UNCHANGED_SETTING"] == "keep-me"
    effects = (private_bundle / "effects-runtime-config.yaml").read_text()
    assert updated_env["ROLE_OMNIDASH_PASSWORD"] in effects
    assert updated_env["ROLE_OMNIBASE_PASSWORD"] in effects


def test_conflicting_principal_mapper_refuses_before_any_bundle_write(
    private_bundle: Path,
) -> None:
    realm_path = private_bundle / "omninode-realm.json"
    realm = json.loads(realm_path.read_text())
    realm["clients"][0]["protocolMappers"] = [
        {"config": {"claim.name": "principal_id", "user.attribute": "tenant_slug"}}
    ]
    realm_path.write_text(json.dumps(realm))
    realm_path.chmod(0o600)
    before = {path.name: path.read_bytes() for path in private_bundle.iterdir()}

    with pytest.raises(ValueError, match="principal mapper conflicts"):
        correction.correct_bundle(private_bundle)

    assert {path.name: path.read_bytes() for path in private_bundle.iterdir()} == before
    assert not list(private_bundle.parent.glob("sim-preflight-auth-backup-*"))
