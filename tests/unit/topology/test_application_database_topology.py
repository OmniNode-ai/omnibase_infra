# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Proof for checked-in application-database topology and Docker consumers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.topology import (
    SUPPORTED_TOPOLOGY_PROFILES,
    TOPOLOGY_PROFILE_INSTANCE_MAP,
    load_environment_topology,
    load_topology_profile,
    validate_database_projection,
    validate_docker_catalog_parity,
    validate_docker_topology_profile_injections,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_ROOT = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
PROJECTION_ROOT = REPO_ROOT / "docker" / "catalog" / "database-topology"
PROFILE_CATALOG = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "topology"
    / "application_database_profiles.yaml"
)
EXPECTED_PROFILE_INSTANCE_MAP = {
    "local": "local",
    "test": "local",
    "stability-test": "local",
    "judge": "local",
    "prod": "local",
    "onex-dev": "onex-dev",
    "onex-prod": "onex-prod",
}

pytestmark = pytest.mark.unit


def _copy_topologies(tmp_path: Path) -> Path:
    destination = tmp_path / "instances"
    shutil.copytree(TOPOLOGY_ROOT, destination)
    return destination


def _mutate_topology(
    topology_root: Path,
    environment: str,
    mutation: tuple[str, ...],
    value: object,
) -> None:
    path = topology_root / f"{environment}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = raw
    for key in mutation[:-1]:
        target = target[key]
    target[mutation[-1]] = value
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")


def test_all_environment_instances_are_typed_and_target_one_application_db() -> None:
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        topology = load_topology_profile(profile)
        database = topology.databases["application"]

        assert database.physical_name == "omnidash_analytics"
        assert set(database.schemas) == {
            "tenant",
            "omninode_internal",
            "platform_catalog",
        }
        assert {binding.principal for binding in database.bindings.values()} == {
            "onex_api",
            "tenant_projection_writer",
            "app_dashboard",
            "omninode_runtime",
        }
        assert "omninode_runtime_service" in database.bindings
        assert "omninode_runtime" not in database.bindings


def test_checked_in_docker_projections_exactly_match_typed_instances() -> None:
    for environment in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        validate_database_projection(
            environment,
            PROJECTION_ROOT / f"{environment}.yaml",
        )


def test_topology_profile_map_matches_checked_in_deployment_surfaces() -> None:
    assert frozenset(EXPECTED_PROFILE_INSTANCE_MAP) == SUPPORTED_TOPOLOGY_PROFILES
    assert dict(TOPOLOGY_PROFILE_INSTANCE_MAP) == EXPECTED_PROFILE_INSTANCE_MAP


@pytest.mark.parametrize("profile", ["", "dev", "staging", "production", "test-env"])
def test_unknown_topology_profile_fails_closed(profile: str) -> None:
    with pytest.raises(ValueError, match="Unsupported database topology profile"):
        load_topology_profile(profile)


def test_seeded_profile_instance_mapping_drift_fails_closed(tmp_path: Path) -> None:
    profile_catalog = tmp_path / "application_database_profiles.yaml"
    shutil.copyfile(PROFILE_CATALOG, profile_catalog)
    raw = yaml.safe_load(profile_catalog.read_text(encoding="utf-8"))
    raw["profiles"]["test"]["instance"] = "onex-dev"
    profile_catalog.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="profile/instance drift"):
        load_topology_profile("test", profile_catalog_path=profile_catalog)


def test_seeded_unknown_profile_catalog_entry_fails_closed(tmp_path: Path) -> None:
    profile_catalog = tmp_path / "application_database_profiles.yaml"
    shutil.copyfile(PROFILE_CATALOG, profile_catalog)
    raw = yaml.safe_load(profile_catalog.read_text(encoding="utf-8"))
    raw["profiles"]["test-env"] = raw["profiles"]["test"].copy()
    profile_catalog.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="profile set drift"):
        load_topology_profile("local", profile_catalog_path=profile_catalog)


def test_topologies_and_projections_contain_no_secret_values_or_dsns() -> None:
    files = [
        *TOPOLOGY_ROOT.glob("*.yaml"),
        *PROJECTION_ROOT.glob("*.yaml"),
    ]
    for path in files:
        content = path.read_text(encoding="utf-8").lower()
        assert "postgresql://" not in content
        assert "postgres://" not in content
        assert "password:" not in content
        assert "token:" not in content


def test_host_local_topology_cannot_override_checked_in_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_topology = tmp_path / ".omnibase" / "topology.yaml"
    host_topology.parent.mkdir(parents=True)
    host_topology.write_text(
        "schema_version: '2.0'\ndatabases: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    topology = load_environment_topology("local")

    assert topology.databases["application"].physical_name == "omnidash_analytics"


@pytest.mark.parametrize(
    ("mutation", "value", "message"),
    [
        (
            ("databases", "application", "physical_name"),
            "omninode_cloud",
            "omnidash_analytics",
        ),
        (
            ("databases", "application", "schemas", "tenant", "domain"),
            "OMNINODE_INTERNAL",
            "schema/domain drift",
        ),
        (
            (
                "databases",
                "application",
                "bindings",
                "onex_api",
                "principal",
            ),
            "app_dashboard",
            "principal drift",
        ),
        (
            (
                "databases",
                "application",
                "bindings",
                "tenant_projection",
                "dsn_env",
            ),
            "WRONG_DB_URL",
            "dsn_env drift",
        ),
    ],
)
def test_seeded_database_user_schema_and_dsn_drift_fail_closed(
    tmp_path: Path,
    mutation: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    topology_root = _copy_topologies(tmp_path)
    _mutate_topology(topology_root, "local", mutation, value)

    with pytest.raises(ValueError, match=message):
        load_environment_topology("local", topology_root)


def test_secret_material_is_rejected_by_the_typed_topology(tmp_path: Path) -> None:
    topology_root = _copy_topologies(tmp_path)
    _mutate_topology(
        topology_root,
        "local",
        ("databases", "application", "principals", "onex_api", "password"),
        "not-allowed",
    )

    with pytest.raises(ValidationError, match="password"):
        load_environment_topology("local", topology_root)


def test_docker_catalog_database_and_dsn_consumers_match_topology() -> None:
    validate_docker_catalog_parity()


def test_docker_surfaces_inject_exact_database_topology_profiles() -> None:
    validate_docker_topology_profile_injections()


def test_seeded_docker_topology_profile_injection_drift_fails_closed(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    docker_root = repo_root / "docker"
    docker_root.mkdir(parents=True)
    runtime_policy_path = (
        repo_root / "contracts" / "services" / "runtime_policy.contract.yaml"
    )
    runtime_policy_path.parent.mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml",
        runtime_policy_path,
    )
    for name in (
        "docker-compose.infra.yml",
        "docker-compose.e2e.yml",
        "docker-compose.stability-test.yml",
        "docker-compose.judge.yml",
        "docker-compose.prod.yml",
    ):
        shutil.copyfile(REPO_ROOT / "docker" / name, docker_root / name)

    e2e_path = docker_root / "docker-compose.e2e.yml"
    content = e2e_path.read_text(encoding="utf-8").replace(
        "ONEX_DATABASE_TOPOLOGY_PROFILE: test",
        "ONEX_DATABASE_TOPOLOGY_PROFILE: test-env",
    )
    e2e_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="topology profile injection drift"):
        validate_docker_topology_profile_injections(repo_root=repo_root)


@pytest.mark.parametrize(
    ("service", "mutation", "message"),
    [
        (
            "migration-gate",
            ("hardcoded_env", "NODE_POSTGRES_DB"),
            "NODE_POSTGRES_DB",
        ),
        (
            "omnidash",
            ("required_env",),
            "inventory drift",
        ),
    ],
)
def test_seeded_docker_database_or_dsn_drift_fails_closed(
    tmp_path: Path,
    service: str,
    mutation: tuple[str, ...],
    message: str,
) -> None:
    repo_root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "docker" / "catalog", repo_root / "docker" / "catalog")
    manifest_path = repo_root / "docker" / "catalog" / "services" / f"{service}.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if mutation == ("required_env",):
        raw["required_env"].remove("OMNIDASH_ANALYTICS_DB_URL")
    else:
        raw[mutation[0]][mutation[1]] = "omninode_cloud"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_docker_catalog_parity(repo_root=repo_root)
