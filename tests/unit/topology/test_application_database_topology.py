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
    SUPPORTED_ENVIRONMENTS,
    load_environment_topology,
    validate_database_projection,
    validate_docker_catalog_parity,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_ROOT = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
PROJECTION_ROOT = REPO_ROOT / "docker" / "catalog" / "database-topology"

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
    for environment in sorted(SUPPORTED_ENVIRONMENTS):
        topology = load_environment_topology(environment)
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
    for environment in sorted(SUPPORTED_ENVIRONMENTS):
        validate_database_projection(
            environment,
            PROJECTION_ROOT / f"{environment}.yaml",
        )


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
