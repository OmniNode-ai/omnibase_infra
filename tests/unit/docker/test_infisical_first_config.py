# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for Infisical-first configuration (OMN-5831).

Validates that:
1. Infisical is in the core bundle (not secrets-only)
2. DB URL vars use Docker-internal addresses in service catalog YAMLs
3. No runtime service catalog YAML has DB URL/DSN vars in required_env
4. The handcrafted compose file does not leak localhost DB URLs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

# Project root: tests/unit/docker/ -> tests/unit/ -> tests/ -> project_root
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CATALOG_DIR = _PROJECT_ROOT / "docker" / "catalog"
_SERVICES_DIR = _CATALOG_DIR / "services"
_BUNDLES_PATH = _CATALOG_DIR / "bundles.yaml"

# DB URL/DSN env var names that must NOT appear in required_env for runtime services.
# These must be hardcoded with Docker-internal addresses instead.
_DB_URL_VARS = frozenset(
    {
        "OMNIBASE_INFRA_DB_URL",
        "OMNIINTELLIGENCE_DB_URL",
        "OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN",
        "OMNIBASE_INFRA_SKILL_LIFECYCLE_POSTGRES_DSN",
        "OMNIBASE_INFRA_CONTEXT_AUDIT_POSTGRES_DSN",
    }
)

# Docker-internal hostname that DB URLs must use (not localhost).
_DOCKER_INTERNAL_HOST = "postgres:5432"


@pytest.fixture(scope="module")
def bundles() -> dict[str, Any]:
    """Load bundle definitions."""
    with open(_BUNDLES_PATH) as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


@pytest.fixture(scope="module")
def service_manifests() -> dict[str, dict[str, Any]]:
    """Load all service catalog YAMLs."""
    manifests: dict[str, dict[str, Any]] = {}
    for yaml_file in _SERVICES_DIR.glob("*.yaml"):
        with open(yaml_file) as f:
            data = cast("dict[str, Any]", yaml.safe_load(f))
        name = data["name"]
        assert isinstance(name, str)
        manifests[name] = data
    return manifests


class TestInfisicalInCoreBundle:
    """Verify Infisical is part of the core bundle."""

    def test_infisical_in_core_services(self, bundles: dict[str, Any]) -> None:
        """Infisical must be listed in the core bundle's services."""
        core = bundles["core"]
        assert isinstance(core, dict)
        services = core.get("services", [])
        assert "infisical" in services, (
            "Infisical must be in the core bundle so it starts with infra-up. "
            "It was moved from the secrets bundle in OMN-5831."
        )

    def test_valkey_in_core_services(self, bundles: dict[str, Any]) -> None:
        """Valkey must be in core because Infisical depends on it."""
        core = bundles["core"]
        assert isinstance(core, dict)
        services = core.get("services", [])
        assert "valkey" in services, (
            "Valkey must be in the core bundle because Infisical depends on it."
        )

    def test_core_injects_infisical_env(self, bundles: dict[str, Any]) -> None:
        """Core bundle must inject INFISICAL_ADDR for runtime services."""
        core = bundles["core"]
        assert isinstance(core, dict)
        inject_env = core.get("inject_env", {})
        assert "INFISICAL_ADDR" in inject_env, (
            "Core bundle must inject INFISICAL_ADDR so runtime services "
            "can discover Infisical."
        )

    def test_core_requires_only_locally_generatable_secrets(
        self, bundles: dict[str, Any]
    ) -> None:
        """Core may require Infisical's own secrets, never a machine identity.

        OMN-16187: the encryption key and auth secret are minted locally with
        openssl and are what the Infisical container needs to boot. The machine
        identity is minted *against* that container afterwards, so requiring it
        to start core was a chicken-and-egg gate; it now sits on the runtime
        bundle, whose ``runtime_host_process`` is its only consumer.
        """
        core = bundles["core"]
        assert isinstance(core, dict)
        required = set(core.get("inject_required_env", []))
        assert {"INFISICAL_ENCRYPTION_KEY", "INFISICAL_AUTH_SECRET"} <= required, (
            "core must still require the secrets Infisical needs to boot"
        )
        identity = {
            "INFISICAL_CLIENT_ID",
            "INFISICAL_CLIENT_SECRET",
            "INFISICAL_PROJECT_ID",
        }
        assert not (identity & required), (
            "core bundle must not require Infisical machine-identity credentials; "
            "they cannot exist before the first bring-up (OMN-16187)."
        )

    def test_infisical_catalog_requires_authenticated_redis_url(
        self, service_manifests: dict[str, dict[str, Any]]
    ) -> None:
        """Infisical must not be generated with an unauthenticated Valkey URL."""
        infisical = service_manifests["infisical"]
        required_env = set(infisical.get("required_env", []))
        hardcoded_env = infisical.get("hardcoded_env", {})

        assert "INFISICAL_REDIS_URL" in required_env
        assert hardcoded_env.get("REDIS_URL") == (
            "${INFISICAL_REDIS_URL:"
            "?INFISICAL_REDIS_URL must be set in ~/.omnibase/.env}"
        )
        assert hardcoded_env.get("REDIS_URL") != "redis://valkey:6379"

    def test_generated_core_compose_uses_infisical_redis_url_source(self) -> None:
        """Generated compose must map Infisical REDIS_URL from INFISICAL_REDIS_URL."""
        resolver = CatalogResolver(catalog_dir=str(_CATALOG_DIR))
        compose = generate_compose(resolver.resolve(["core"]))

        services = compose["services"]
        assert isinstance(services, dict)
        infisical = services["infisical"]
        assert isinstance(infisical, dict)
        environment = infisical["environment"]
        assert isinstance(environment, dict)

        assert environment["REDIS_URL"] == (
            "${INFISICAL_REDIS_URL:"
            "?INFISICAL_REDIS_URL must be set in ~/.omnibase/.env}"
        )
        assert environment["INFISICAL_REDIS_URL"] == (
            "${INFISICAL_REDIS_URL:"
            "?INFISICAL_REDIS_URL must be set in ~/.omnibase/.env}"
        )


class TestNoDbUrlInRequiredEnv:
    """Verify no runtime service has DB URL/DSN vars in required_env."""

    def test_no_db_url_in_required_env(
        self, service_manifests: dict[str, dict[str, Any]]
    ) -> None:
        """DB URL vars must be in hardcoded_env, not required_env."""
        violations: list[str] = []
        for name, manifest in service_manifests.items():
            required = set(manifest.get("required_env", []))
            leaked = required & _DB_URL_VARS
            if leaked:
                violations.append(
                    f"{name}: {', '.join(sorted(leaked))} in required_env"
                )
        assert not violations, (
            "DB URL/DSN vars must not be in required_env (they leak "
            "localhost:5436 from host env). Move to hardcoded_env with "
            "Docker-internal addresses. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestDbUrlsUseDockerInternal:
    """Verify hardcoded DB URLs use Docker-internal addresses."""

    def test_hardcoded_db_urls_use_internal_host(
        self, service_manifests: dict[str, dict[str, Any]]
    ) -> None:
        """All hardcoded DB URL values must reference postgres:5432."""
        violations: list[str] = []
        for name, manifest in service_manifests.items():
            hardcoded = manifest.get("hardcoded_env", {})
            for key, value in hardcoded.items():
                if key in _DB_URL_VARS and isinstance(value, str):
                    if "localhost" in value:
                        violations.append(
                            f"{name}: {key} contains 'localhost' — "
                            f"must use Docker-internal address"
                        )
                    if _DOCKER_INTERNAL_HOST not in value:
                        violations.append(
                            f"{name}: {key} does not contain '{_DOCKER_INTERNAL_HOST}'"
                        )
        assert not violations, (
            "Hardcoded DB URLs must use Docker-internal addresses "
            f"('{_DOCKER_INTERNAL_HOST}'), not localhost. Violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_handcrafted_compose_no_localhost_db_urls(self) -> None:
        """The handcrafted compose must not pass localhost DB URLs."""
        compose_path = _PROJECT_ROOT / "docker" / "docker-compose.infra.yml"
        content = compose_path.read_text()

        # Check that no runtime env block passes localhost DB URLs.
        # The OMNIBASE_INFRA_DB_URL should use postgres:5432, not localhost:5436.
        for var in _DB_URL_VARS:
            # Pattern: VAR: ${VAR:? ... } means it's pulling from host env
            # (which has localhost:5436). This is the bug we're preventing.
            pattern = f"{var}: ${{{var}:"
            assert pattern not in content, (
                f"Handcrafted compose still passes {var} from host env "
                f"(${{{var}:...}}). This leaks localhost:5436 into "
                f"containers. Use a hardcoded Docker-internal address."
            )
