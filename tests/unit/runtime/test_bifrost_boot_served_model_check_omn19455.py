# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19455: boot compares each backend with what its endpoint serves.

The recorded-probe fixture pins the overlays in CI only. These tests boot the
renderer against a real local HTTP server, with the default probe, so the
refusal and the dark marking are proven on the wire and not through a stub
callable.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.unit

_DECLARED = "Qwen3.8-27B"


def _serve(model_ids: list[str]) -> tuple[HTTPServer, threading.Thread]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps(
                {"data": [{"id": model_id} for model_id in model_ids]}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def stub_serving_other_model() -> Iterator[int]:
    server, thread = _serve(["some-other-model"])
    yield server.server_address[1]
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


@pytest.fixture
def stub_serving_declared_model() -> Iterator[int]:
    server, thread = _serve([_DECLARED])
    yield server.server_address[1]
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _binding(backend_id: str, port: int) -> dict[str, object]:
    return {
        "backend_id": backend_id,
        "endpoint_url": f"http://127.0.0.1:{port}/v1/chat/completions",
        "served_model_id": _DECLARED,
        "parameter_count": "27B",
        "context_window": 131072,
        "max_tokens": 65536,
        "timeout_ms": 300000,
    }


def _files(
    tmp_path: Path, port: int, *, live_sibling_port: int | None = None
) -> tuple[Path, Path, Path]:
    source = tmp_path / "base.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {"backend_id": "local-coder", "model_name": _DECLARED},
                    {"backend_id": "local-sibling", "model_name": _DECLARED},
                ][: 2 if live_sibling_port is not None else 1]
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": [
                    _binding("local-coder", port),
                    *(
                        [_binding("local-sibling", live_sibling_port)]
                        if live_sibling_port is not None
                        else []
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, overlay, tmp_path / "rendered.yaml"


def test_boot_refuses_a_backend_whose_served_model_differs(
    tmp_path: Path, stub_serving_other_model: int
) -> None:
    source, overlay, target = _files(tmp_path, stub_serving_other_model)

    with pytest.raises(ProtocolConfigurationError, match="local-coder") as raised:
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=overlay,
            target_path=target,
            verify_endpoints=True,
        )

    assert _DECLARED in str(raised.value)
    assert not target.exists()


def test_boot_accepts_a_backend_serving_its_declared_model(
    tmp_path: Path, stub_serving_declared_model: int
) -> None:
    source, overlay, target = _files(tmp_path, stub_serving_declared_model)

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay,
        target_path=target,
        verify_endpoints=True,
    )

    backend = yaml.safe_load(target.read_text(encoding="utf-8"))["backends"][0]
    assert backend["endpoint_url"].endswith("/v1/chat/completions")


def test_boot_marks_an_unreachable_backend_dark_and_continues(
    tmp_path: Path, stub_serving_declared_model: int
) -> None:
    source, overlay, target = _files(
        tmp_path, _closed_port(), live_sibling_port=stub_serving_declared_model
    )

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay,
        target_path=target,
        verify_endpoints=True,
    )

    assert rendered == target
    by_id = {
        b["backend_id"]: b
        for b in yaml.safe_load(target.read_text(encoding="utf-8"))["backends"]
    }
    assert by_id["local-coder"]["endpoint_url"] is None
    assert by_id["local-coder"]["model_name"] == _DECLARED
    assert by_id["local-sibling"]["endpoint_url"].endswith("/v1/chat/completions")
