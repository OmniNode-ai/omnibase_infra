# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability refresh's real host reads /health over a real socket (OMN-26715)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_stability_integration_refresh import (
    DockerStabilityIntegrationHost,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "src/omnibase_infra/nodes/node_stability_integration_refresh_effect/contract.yaml"
)


def _serve(status: int) -> Iterator[str]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()


def _host(tmp_path: Path) -> DockerStabilityIntegrationHost:
    return DockerStabilityIntegrationHost(
        registry_root=tmp_path / "registry",
        infra_checkout=tmp_path / "infra",
        work_dir=tmp_path / "work",
        operator_env_file=tmp_path / "operator.env",
        health_timeout_seconds=5.0,
        lane_declaration_dir=tmp_path / "config" / "lab",
        lane_overlay_dir=tmp_path / "config" / "lab" / "lane-overlays",
    )


def test_health_status_reads_200_and_503_from_a_live_socket(tmp_path: Path) -> None:
    host = _host(tmp_path)
    for status in (200, 503):
        url = next(gen := _serve(status))
        try:
            assert host.health_status(url) == status
        finally:
            gen.close()


def test_health_status_is_not_200_when_nothing_listens(tmp_path: Path) -> None:
    assert _host(tmp_path).health_status("http://127.0.0.1:1/health") != 200


def test_the_contract_on_disk_declares_the_effect_node() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert contract["node_type"] == "EFFECT_GENERIC"
