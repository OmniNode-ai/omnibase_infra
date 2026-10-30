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

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def _readiness_probe_code(port: int) -> str:
    probe = _services("local-market")[_WRITER]["healthcheck"]["test"]
    assert probe[:3] == ["CMD", "python", "-c"]
    code = probe[3]
    assert isinstance(code, str)
    # Fail before executing if the catalog address changed or is ambiguous.
    assert code.count("localhost:8099") == 1
    return code.replace("localhost:8099", f"127.0.0.1:{port}")


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
        "CMD",
        "python",
        "-c",
        "import urllib.request; urllib.request.urlopen('http://localhost:8099/ready', timeout=5).close()",
    ]
    assert writer["restart"] != "no"


@pytest.mark.parametrize("status", [200, 503])
def test_run_trace_readiness_probe_observes_http_status(status: int) -> None:
    class ReadinessHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status if self.path == "/ready" else 404)
            self.end_headers()

    with HTTPServer(("127.0.0.1", 0), ReadinessHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Only replace the address; execute the rendered probe's actual code.
            code = _readiness_probe_code(server.server_port)
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=10,
                check=False,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert (result.returncode == 0) is (status == 200), result.stderr.decode()


def test_run_trace_readiness_probe_refuses_a_stopped_server() -> None:
    # Reserve a port without listening so no other test can take the address.
    with HTTPServer(
        ("127.0.0.1", 0), BaseHTTPRequestHandler, bind_and_activate=False
    ) as server:
        server.server_bind()
        code = _readiness_probe_code(server.server_port)
        # A dedicated exit code distinguishes a transport failure from syntax
        # errors or other failures, without inspecting exception formatting.
        transport_check = (
            "import urllib.error\n"
            "try:\n"
            f"    exec({code!r})\n"
            "except (urllib.error.URLError, ConnectionError, TimeoutError):\n"
            "    raise SystemExit(42)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", transport_check],
            capture_output=True,
            timeout=10,
            check=False,
        )
    assert result.returncode == 42, result.stderr.decode()


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
    assert env["ONEX_TENANT_DB_URL"] == (
        "postgresql://tenant_projection_writer:"
        "${TENANT_PROJECTION_WRITER_PASSWORD:?TENANT_PROJECTION_WRITER_PASSWORD required}"
        "@postgres:5432/omnidash_analytics"
    )
    assert env["OMNINODE_INTERNAL_DB_URL"] == (
        "postgresql://omninode_runtime:"
        "${OMNINODE_RUNTIME_PASSWORD:?OMNINODE_RUNTIME_PASSWORD required}"
        "@postgres:5432/omnidash_analytics"
    )
    assert env["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "local"


def test_docker_catalog_dsn_inventory_still_matches_the_topology() -> None:
    validate_docker_catalog_parity()
