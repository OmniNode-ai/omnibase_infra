# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the local-market run-trace writer catalog entry (OMN-19972).

Failure modes guarded:
 1 the writer is in no bundle, so the run-trace contract (runtime profile
   projection-writer-run-trace) is consumed by nothing;
 2 it lands in `local`, whose infra image carries no omnimarket;
 3 it runs the infra runtime image instead of the omnimarket-runtime image;
 4 it has no healthcheck, so the composed boot gate's health step never looks
   at it;
 5 it declares only the tenant_projection binding, so its
   tenant_registry_mirror read has no DSN;
 6 it starts before migration-gate or before redpanda-local-sasl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.topology.application_database import (
    validate_docker_catalog_parity,
)

_REPO = Path(__file__).resolve().parents[3]
_CATALOG_DIR = str(_REPO / "docker" / "catalog")
_PROJECT = "omnibase-infra-local"
_WRITER = "omnimarket-projection-run-trace"


def _services(bundle: str) -> dict[str, Any]:
    services = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
    )["services"]
    assert isinstance(services, dict)
    return services


pytestmark = pytest.mark.unit


def test_local_market_runs_the_run_trace_writer_on_the_market_image() -> None:
    services = _services("local-market")
    assert _WRITER in services
    writer = services[_WRITER]
    assert writer["image"] == f"{_PROJECT}-omnimarket-runtime:latest"
    assert "build" not in writer
    assert writer["command"] == [
        "python",
        "-m",
        "omnimarket.nodes.node_projection_run_trace.handlers.handler_run_trace_writer",
    ]


def test_local_does_not_carry_the_run_trace_writer() -> None:
    services = _services("local")
    assert _WRITER not in services


def test_run_trace_writer_is_health_checked_on_its_runner_port() -> None:
    services = _services("local-market")
    writer = services[_WRITER]
    assert writer["environment"]["PROJECTION_RUNNER_HEALTH_PORT"] == "8099"
    assert writer["healthcheck"]["test"] == [
        "CMD-SHELL",
        "curl -sf http://localhost:8099/ready",
    ]
    assert writer["restart"] != "no"


def test_run_trace_writer_waits_for_migrations_and_sasl() -> None:
    services = _services("local-market")
    writer = services[_WRITER]
    depends_on = writer["depends_on"]
    assert depends_on["migration-gate"] == {"condition": "service_healthy"}
    assert depends_on["redpanda-local-sasl"] == {
        "condition": "service_completed_successfully"
    }


def test_run_trace_writer_declares_both_bindings_and_gets_both_dsns() -> None:
    consumers_path = Path(_CATALOG_DIR) / "database-consumers.yaml"
    with consumers_path.open(encoding="utf-8") as f:
        consumers = yaml.safe_load(f)
    assert set(consumers["consumers"][_WRITER]["bindings"]) == {
        "tenant_projection",
        "omninode_runtime_service",
    }

    services = _services("local-market")
    env = services[_WRITER]["environment"]
    assert env["ONEX_TENANT_DB_URL"].startswith(
        "postgresql://tenant_projection_writer:${TENANT_PROJECTION_WRITER_PASSWORD:?"
    )
    assert env["OMNINODE_INTERNAL_DB_URL"].startswith(
        "postgresql://omninode_runtime:${OMNINODE_RUNTIME_PASSWORD:?"
    )
    assert env["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "local"


def test_docker_catalog_dsn_inventory_still_matches_the_topology() -> None:
    validate_docker_catalog_parity()
