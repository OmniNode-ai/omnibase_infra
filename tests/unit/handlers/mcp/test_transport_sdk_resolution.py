# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that the MCP transport resolves against the installed MCP SDK.

Tests cover:
- Every ``mcp`` module imported under ``src`` exists in the installed SDK
  (a planted nonexistent import is detected).
- ``TransportMCPStreamableHttp.create_app`` builds an application.
- The import guard says "not installed" only when ``mcp`` itself is absent, and
  names the installed version and failing module otherwise.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers.mcp import transport_streamable_http
from omnibase_infra.handlers.mcp.transport_streamable_http import (
    TransportMCPStreamableHttp,
)

pytestmark = pytest.mark.unit

_SRC_ROOT = Path(__file__).resolve().parents[4] / "src"


def _mcp_imports(source: str) -> set[str]:
    """Return the dotted ``mcp`` module names a source text imports."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names if a.name.split(".")[0] == "mcp")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == "mcp"
        ):
            modules.add(node.module)
    return modules


def _unresolvable(modules: set[str]) -> list[str]:
    missing: list[str] = []
    for name in sorted(modules):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def _src_mcp_imports() -> set[str]:
    found: set[str] = set()
    for path in _SRC_ROOT.rglob("*.py"):
        found |= _mcp_imports(path.read_text(encoding="utf-8"))
    return found


def test_every_mcp_module_imported_by_src_resolves() -> None:
    imported = _src_mcp_imports()
    assert imported, "expected src to import the MCP SDK"
    assert _unresolvable(imported) == []


def test_planted_nonexistent_mcp_import_is_detected() -> None:
    planted = _mcp_imports("from mcp.server.nonexistent_module import Thing\n")
    assert _unresolvable(planted) == ["mcp.server.nonexistent_module"]


def test_create_app_builds_application() -> None:
    transport = TransportMCPStreamableHttp()
    app = transport.create_app(tools=[], tool_executor=lambda name, args: None)
    assert app is transport.app


def test_guard_reports_not_installed_only_when_mcp_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)

    def _absent(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError("mcp")

    monkeypatch.setattr(
        transport_streamable_http.importlib.metadata, "version", _absent
    )
    with pytest.raises(ImportError, match="MCP SDK not installed"):
        transport_streamable_http._load_mcp_server_class()


def test_guard_names_version_and_module_when_mcp_lacks_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.setattr(
        transport_streamable_http.importlib.metadata, "version", lambda _: "9.9.9"
    )
    with pytest.raises(ImportError) as excinfo:
        transport_streamable_http._load_mcp_server_class()
    message = str(excinfo.value)
    assert "not installed" not in message
    assert "9.9.9" in message
    assert "mcp.server.mcpserver" in message
