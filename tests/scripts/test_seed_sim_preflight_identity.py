# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No-service tests for the disposable, owner-preserving identity seed."""

from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from scripts.runtime_build.bootstrap_sim_preflight_auth_omn19726 import ROOT
from scripts.runtime_build.seed_sim_preflight_identity import seed

pytestmark = pytest.mark.unit

OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SUBJECT = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
PRINCIPAL = "t-aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa"


class FakeResponse:
    def __init__(self, body: Any) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.body


class FakeSession:
    def __init__(self, *, profile: dict[str, Any] | None = None) -> None:
        self.trust_env = True
        self.headers: dict[str, str] = {}
        self.user = {
            "id": SUBJECT,
            "username": "sim-preflight-demo",
            "email": "sim-preflight-demo@example.invalid",
            "enabled": True,
            "attributes": {
                "tenant_id": [OWNER],
                "tenant_slug": ["sim-preflight"],
                "org_id": [OWNER],
                "principal_id": [PRINCIPAL],
            },
        }
        self.client = {
            "id": "client-internal-id",
            "clientId": "omnidash",
        }
        self.mapper = {
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
        self.profile = profile or {
            "attributes": [
                {"name": "email", "permissions": {"view": ["user"]}},
                {"name": "unrelated", "permissions": {"view": ["admin"]}},
            ],
            "groups": [{"name": "unrelated-group", "displayHeader": "Keep me"}],
            "unrelatedSection": {"preserve": True},
        }
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.mutate_readback = False
        self.profile_gets = 0

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/protocol/openid-connect/token"):
            return FakeResponse({"access_token": "opaque-disposable-admin-token"})
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/users"):
            assert kwargs["params"] == {
                "username": "sim-preflight-demo",
                "exact": "true",
            }
            return FakeResponse([self.user])
        if url.endswith("/users/profile"):
            self.profile_gets += 1
            returned_profile = json.loads(json.dumps(self.profile))
            if self.mutate_readback and self.profile_gets > 1:
                returned_profile["attributes"] = [
                    item
                    for item in returned_profile["attributes"]
                    if item.get("name") != "unrelated"
                ]
            return FakeResponse(returned_profile)
        if url.endswith("/clients"):
            assert kwargs["params"] == {"clientId": "omnidash"}
            return FakeResponse([self.client])
        if url.endswith("/protocol-mappers/models"):
            return FakeResponse([self.mapper])
        raise AssertionError(f"unexpected GET {url}")

    def put(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("PUT", url, kwargs))
        assert url.endswith("/users/profile")
        self.puts.append((url, kwargs["json"]))
        self.profile = kwargs["json"]
        return FakeResponse({})


class FakeRepository:
    def __init__(self, existing: tuple[str] | None = None) -> None:
        self.existing = existing
        self.queries: list[tuple[str, tuple[Any, ...]]] = []
        self.in_transaction = False
        self.committed = False
        self.rolled_back = False

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        self.queries.append((query, params))
        return self.existing

    def __enter__(self) -> FakeRepository:
        assert not self.in_transaction
        self.in_transaction = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.in_transaction = False
        self.rolled_back = exc_type is not None
        self.committed = exc_type is None


@pytest.fixture
def seed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]]:
    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    private_dir.chmod(0o700)
    (private_dir / "credentials.env").write_text(
        "SIM_PREFLIGHT_KEYCLOAK_ADMIN_USERNAME=sim-preflight-admin\n"
        "SIM_PREFLIGHT_KEYCLOAK_ADMIN_PASSWORD=disposable-password\n"
        "SIM_PREFLIGHT_DEMO_USERNAME=sim-preflight-demo\n"
    )
    (private_dir / "omninode-realm.json").write_text(
        json.dumps(
            {
                "realm": "omninode",
                "users": [
                    {
                        "attributes": {
                            "tenant_id": [OWNER],
                            "tenant_slug": ["sim-preflight"],
                            "org_id": [OWNER],
                        }
                    }
                ],
            }
        )
    )
    (private_dir / "desired-user-profile.json").write_bytes(
        (ROOT / "docker/keycloak/desired-user-profile.json").read_bytes()
    )
    for path in private_dir.iterdir():
        path.chmod(0o600)
    monkeypatch.setenv(
        "OMNINODE_CLOUD_DB_URL",
        "postgresql://role_omninode:disposable-db-password@postgres:5432/omninode_cloud",
    )
    monkeypatch.setenv(
        "KEYCLOAK_ISSUER_URL", "http://auth.localhost:28080/realms/omninode"
    )

    session = FakeSession()
    repository = FakeRepository()
    inserted: list[dict[str, Any]] = []
    assigned: list[tuple[Any, str]] = []

    requests_module = types.ModuleType("requests")
    requests_module.Session = lambda: session  # type: ignore[attr-defined]
    db_package = types.ModuleType("db")
    db_package.__path__ = []  # type: ignore[attr-defined]
    db_repo_module = types.ModuleType("db.psycopg_repository")
    db_repo_module.PsycopgRepository = lambda: repository  # type: ignore[attr-defined]
    routers_package = types.ModuleType("routers")
    routers_package.__path__ = []  # type: ignore[attr-defined]
    provision_module = types.ModuleType("routers.provision")

    def assign_plan(repo: Any, tenant_id: str) -> str:
        assigned.append((repo, tenant_id))
        return "free"

    provision_module._assign_default_plan = assign_plan  # type: ignore[attr-defined]
    tenant_creation_module = types.ModuleType("tenant_creation")

    def insert_row(repo: Any, **kwargs: Any) -> None:
        assert repo is repository and repo.in_transaction
        inserted.append(kwargs)

    tenant_creation_module.insert_tenant_row = insert_row  # type: ignore[attr-defined]
    tenant_identity_module = types.ModuleType("tenant_identity")
    tenant_identity_module.derive_principal_id = (  # type: ignore[attr-defined]
        lambda tenant_id: "t-" + tenant_id.hex
    )
    for name, module in (
        ("requests", requests_module),
        ("db", db_package),
        ("db.psycopg_repository", db_repo_module),
        ("routers", routers_package),
        ("routers.provision", provision_module),
        ("tenant_creation", tenant_creation_module),
        ("tenant_identity", tenant_identity_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Stash captured expectations on the session for concise assertions.
    session.inserted = inserted  # type: ignore[attr-defined]
    session.assigned = assigned  # type: ignore[attr-defined]
    session.repository = repository  # type: ignore[attr-defined]
    return private_dir, session, repository, inserted


def test_seed_preserves_owner_and_uses_repository_protocol(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    result = seed(private_dir)

    assert result == {
        "disposable_identity_seeded": True,
        "captured_owner_uuid_preserved": True,
    }
    assert repository.committed and not repository.rolled_back
    assert len(repository.queries) == 2
    assert all(
        query
        == "SELECT tenant_id FROM tenants WHERE tenant_id = %s::uuid OR tenant_slug = %s"
        and params == (OWNER, "sim-preflight")
        for query, params in repository.queries
    )
    assert inserted == [
        {
            "tenant_id": UUID(OWNER),
            "tenant_slug": "sim-preflight",
            "name": "Disposable replay validation",
            "provisioning_state": "active",
            "created_by_sub": SUBJECT,
            "owner_email": "sim-preflight-demo@example.invalid",
        }
    ]
    assert session.assigned == [(repository, OWNER)]  # type: ignore[attr-defined]
    assert session.user["attributes"] == {  # type: ignore[attr-defined]
        "tenant_id": [OWNER],
        "tenant_slug": ["sim-preflight"],
        "org_id": [OWNER],
        "principal_id": [PRINCIPAL],
    }
    assert len(session.puts) == 1  # type: ignore[attr-defined]
    assert session.puts[0][0].endswith("/users/profile")  # type: ignore[attr-defined]
    assert not any(  # Verify-only user path: no user or mapper writes.
        method == "PUT" and url.endswith("/users/" + SUBJECT)
        for method, url, _ in session.calls
    )
    assert not any(
        method == "POST" and "protocol-mappers" in url
        for method, url, _ in session.calls
    )
    profile = session.profile  # type: ignore[attr-defined]
    assert {item["name"] for item in profile["attributes"]} >= {
        "email",
        "unrelated",
        "tenant_id",
        "tenant_slug",
        "principal_id",
    }
    assert profile["groups"] == [
        {"name": "unrelated-group", "displayHeader": "Keep me"},
        next(
            item
            for item in json.loads(
                (ROOT / "docker/keycloak/desired-user-profile.json").read_text()
            )["groups"]
            if item["name"] == "user-metadata"
        ),
    ]
    for name in ("tenant_id", "tenant_slug", "principal_id"):
        definition = next(
            item for item in profile["attributes"] if item["name"] == name
        )
        assert definition["permissions"] == {"view": ["admin"], "edit": ["admin"]}
        assert definition["multivalued"] is False
    assert profile["unrelatedSection"] == {"preserve": True}
    identity = json.loads((private_dir / "identity.json").read_text())
    assert identity == {
        "tenant_id": OWNER,
        "subject": SUBJECT,
        "principal_id": PRINCIPAL,
    }
    assert stat.S_IMODE((private_dir / "identity.json").stat().st_mode) == 0o600


def test_existing_tenant_is_refused_before_realm_mutation(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    repository.existing = (OWNER,)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        seed(private_dir)

    assert not session.calls
    assert not inserted
    assert not repository.committed


def test_existing_identity_output_is_refused_before_external_mutation(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    (private_dir / "identity.json").write_text("existing")

    with pytest.raises(ValueError, match="identity output must not already exist"):
        seed(private_dir)

    assert not session.calls
    assert not repository.queries
    assert not inserted


def test_wrong_database_target_is_refused_before_admin_login(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    monkeypatch.setenv(
        "OMNINODE_CLOUD_DB_URL",
        "postgresql://role_omninode:secret@shared-postgres:5432/omninode_cloud",
    )

    with pytest.raises(ValueError, match="only the disposable cloud database"):
        seed(private_dir)

    assert not session.calls
    assert not repository.queries
    assert not inserted


def test_missing_profile_lists_are_initialized_from_canonical_profile(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    session.profile = {"unrelatedSection": {"preserve": True}}

    seed(private_dir)

    assert len(session.profile["attributes"]) == 3
    assert len(session.profile["groups"]) == 1
    assert session.profile["unrelatedSection"] == {"preserve": True}
    assert repository.committed and inserted


def test_profile_readback_must_preserve_unrelated_definitions(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    session.mutate_readback = True

    with pytest.raises(ValueError, match="changed identity or unrelated profile"):
        seed(private_dir)

    assert not repository.committed
    assert not inserted
    assert not (private_dir / "identity.json").exists()


def test_principal_mapper_requires_exact_introspection_claim_setting(
    seed_inputs: tuple[Path, FakeSession, FakeRepository, list[dict[str, Any]]],
) -> None:
    private_dir, session, repository, inserted = seed_inputs
    session.mapper["config"]["introspection.token.claim"] = "false"

    with pytest.raises(ValueError, match="existing principal mapper differs"):
        seed(private_dir)

    assert not repository.committed
    assert not inserted
    assert not (private_dir / "identity.json").exists()
