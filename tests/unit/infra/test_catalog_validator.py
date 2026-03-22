# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the catalog env validator."""

from __future__ import annotations

import pytest

from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest
from omnibase_infra.docker.catalog.validator import (
    validate_env,
    validate_no_localhost_in_container_env,
)


def _make_manifest(
    name: str = "test-svc",
    hardcoded_env: dict[str, str] | None = None,
    operational_defaults: dict[str, str] | None = None,
    catalog_env: dict[str, str] | None = None,
) -> CatalogManifest:
    """Create a minimal manifest for testing."""
    return CatalogManifest(
        name=name,
        description="test",
        image="test:latest",
        layer=EnumInfraLayer.RUNTIME,
        required_env=["POSTGRES_PASSWORD"],
        hardcoded_env=hardcoded_env or {},
        operational_defaults=operational_defaults or {},
        ports=None,
        healthcheck=None,
        volumes=[],
        depends_on=[],
        catalog_env=catalog_env or {},
    )


@pytest.mark.unit
def test_validator_passes_when_all_required_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    monkeypatch.setenv("VALKEY_PASSWORD", "test")
    result = validate_env(required={"POSTGRES_PASSWORD", "VALKEY_PASSWORD"})
    assert result.ok


@pytest.mark.unit
def test_validator_fails_when_required_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALKEY_PASSWORD", raising=False)
    result = validate_env(required={"VALKEY_PASSWORD"})
    assert not result.ok
    assert "VALKEY_PASSWORD" in result.missing


# --- localhost validation tests ---


@pytest.mark.unit
def test_localhost_rejected_in_hardcoded_db_url() -> None:
    """DB URLs with localhost must be rejected in hardcoded_env."""
    manifest = _make_manifest(
        hardcoded_env={
            "OMNIBASE_INFRA_DB_URL": "postgresql://postgres:pw@localhost:5436/omnibase_infra",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert len(violations) == 1
    assert violations[0].var_name == "OMNIBASE_INFRA_DB_URL"
    assert violations[0].env_section == "hardcoded_env"


@pytest.mark.unit
def test_localhost_rejected_in_operational_defaults() -> None:
    """DB URLs with localhost must be rejected in operational_defaults too."""
    manifest = _make_manifest(
        operational_defaults={
            "SOME_DB_URL": "postgresql://user:pass@localhost:5432/mydb",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert len(violations) == 1
    assert violations[0].env_section == "operational_defaults"


@pytest.mark.unit
def test_localhost_rejected_in_catalog_env() -> None:
    """DB URLs with localhost must be rejected in catalog_env too."""
    manifest = _make_manifest(
        catalog_env={
            "DB_URL": "postgresql://u:p@localhost/db",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert len(violations) == 1
    assert violations[0].env_section == "catalog_env"


@pytest.mark.unit
def test_docker_dns_url_passes() -> None:
    """Docker-internal URLs (postgres:5432) must be accepted."""
    manifest = _make_manifest(
        hardcoded_env={
            "OMNIBASE_INFRA_DB_URL": "postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/omnibase_infra",
            "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert violations == []


@pytest.mark.unit
def test_cors_origins_exempt_from_localhost_check() -> None:
    """CORS_ORIGINS legitimately contains localhost (browser URLs)."""
    manifest = _make_manifest(
        operational_defaults={
            "CORS_ORIGINS": "http://localhost:3000,http://localhost:3001",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert violations == []


@pytest.mark.unit
def test_keycloak_issuer_exempt_from_localhost_check() -> None:
    """KEYCLOAK_ISSUER is browser-resolved, localhost is correct."""
    manifest = _make_manifest(
        operational_defaults={
            "KEYCLOAK_ISSUER": "http://localhost:28080/realms/omninode",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert violations == []


@pytest.mark.unit
def test_multiple_violations_reported() -> None:
    """All violations across services and sections should be reported."""
    m1 = _make_manifest(
        name="svc-a",
        hardcoded_env={"DB_URL": "postgresql://u:p@localhost:5436/db"},
    )
    m2 = _make_manifest(
        name="svc-b",
        operational_defaults={"OTHER_URL": "http://localhost:9999/api"},
    )
    violations = validate_no_localhost_in_container_env({"svc-a": m1, "svc-b": m2})
    assert len(violations) == 2
    services_hit = {v.service for v in violations}
    assert services_hit == {"svc-a", "svc-b"}


@pytest.mark.unit
def test_non_url_localhost_not_flagged() -> None:
    """Plain string values that happen to contain 'localhost' without URL
    patterns should not be flagged (no @localhost or //localhost)."""
    manifest = _make_manifest(
        hardcoded_env={
            "SOME_FLAG": "localhost_mode",
        },
    )
    violations = validate_no_localhost_in_container_env({"svc": manifest})
    assert violations == []
