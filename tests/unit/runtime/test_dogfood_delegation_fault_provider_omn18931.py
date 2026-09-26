# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deterministic, dogfood-only provider fault controls for OMN-18931."""

from __future__ import annotations

import importlib.util
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import ModuleType

import pytest
import yaml


class _ComposeLoader(yaml.SafeLoader):
    """Compose uses !override, which is structural for this source test."""


def _construct_override(loader: _ComposeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_ComposeLoader.add_constructor("!override", _construct_override)

_REPO = Path(__file__).parents[3]
_COMPOSE = _REPO / "docker" / "docker-compose.dogfood.yml"
_PROVIDER = (
    _REPO / "docker" / "fault-providers" / "dogfood_delegation_fault_provider.py"
)
_EXPECTED = {
    "dogfood-delegation-fault-429": 429,
    "dogfood-delegation-fault-503": 503,
}


def _load_provider() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dogfood_fault_provider", _PROVIDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dogfood_compose_declares_only_internal_fixed_fault_services() -> None:
    """Both controls are dogfood-only, internal, and contain no credentials."""
    document = yaml.load(_COMPOSE.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    services = document["services"]
    for service_name, status in _EXPECTED.items():
        service = services[service_name]
        assert service["profiles"] == ["dogfood"]
        assert "ports" not in service
        assert service.get("environment", {}) == {}
        assert service["command"][-2:] == ["--status", str(status)]
        assert service["networks"] == ["omnibase-infra-dogfood-network"]
        assert service["read_only"] is True


def test_fault_provider_only_supports_the_declared_statuses() -> None:
    provider = _load_provider()
    assert provider.parse_status(["--status", "429"]) == 429
    assert provider.parse_status(["--status", "503"]) == 503
    with pytest.raises(SystemExit):
        provider.parse_status(["--status", "500"])


@pytest.mark.parametrize("status", [429, 503])
def test_fault_provider_emits_openai_compatible_error(status: int) -> None:
    provider = _load_provider()
    payload = provider.error_payload(status)
    assert json.loads(json.dumps(payload)) == {
        "error": {
            "code": status,
            "message": f"dogfood deterministic provider fault: HTTP {status}",
            "status": "RESOURCE_EXHAUSTED" if status == 429 else "UNAVAILABLE",
        }
    }


@pytest.mark.parametrize("status", [429, 503])
def test_fault_provider_serves_the_declared_http_status(status: int) -> None:
    provider = _load_provider()
    server = ThreadingHTTPServer(("127.0.0.1", 0), provider.handler_for(status))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("POST", "/v1/chat/completions", body="{}")
        response = connection.getresponse()
        assert response.status == status
        assert json.loads(response.read()) == provider.error_payload(status)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
