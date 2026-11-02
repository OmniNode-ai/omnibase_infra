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
import tomllib
import types
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from omnibase_infra.handlers.mcp import transport_streamable_http
from omnibase_infra.handlers.mcp.transport_streamable_http import (
    TransportMCPStreamableHttp,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_ROOT = _REPO_ROOT / "src"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _mcp_imports(source: str) -> set[tuple[str, str | None]]:
    """Return ``(module, name)`` pairs for the ``mcp`` imports a source text makes.

    ``import mcp.x`` yields ``("mcp.x", None)``; ``from mcp.x import Y`` yields
    ``("mcp.x", "Y")`` so a symbol removed from an existing module is caught.
    Imports under ``TYPE_CHECKING`` are included on purpose: mypy resolves them
    against the same installed SDK.
    """
    found: set[tuple[str, str | None]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(
                (a.name, None) for a in node.names if a.name.split(".")[0] == "mcp"
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and node.module.split(".")[0] == "mcp"
        ):
            found.update((node.module, a.name) for a in node.names if a.name != "*")
    return found


def _unresolvable(imports: set[tuple[str, str | None]]) -> list[str]:
    """Return the imports the installed SDK cannot satisfy, as dotted names."""
    missing: list[str] = []
    for module, name in sorted(imports, key=lambda pair: (pair[0], pair[1] or "")):
        try:
            imported = importlib.import_module(module)
        except ImportError:
            missing.append(module)
            continue
        if name is None or hasattr(imported, name):
            continue
        # ``from package import submodule`` binds a submodule, not an attribute.
        try:
            importlib.import_module(f"{module}.{name}")
        except ImportError:
            missing.append(f"{module}.{name}")
    return missing


def _src_mcp_imports() -> set[tuple[str, str | None]]:
    found: set[tuple[str, str | None]] = set()
    for path in _SRC_ROOT.rglob("*.py"):
        found |= _mcp_imports(path.read_text(encoding="utf-8"))
    return found


def _declared_mcp_specifier() -> SpecifierSet:
    """Return the ``mcp`` specifier declared in ``pyproject.toml``."""
    pyproject = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    for entry in pyproject["project"]["dependencies"]:
        requirement = Requirement(entry)
        if canonicalize_name(requirement.name) == "mcp":
            return requirement.specifier
    raise AssertionError("pyproject.toml does not declare an mcp dependency")


def test_every_mcp_module_imported_by_src_resolves() -> None:
    assert _SRC_ROOT.is_dir(), f"src root not found at {_SRC_ROOT}"
    imported = _src_mcp_imports()
    assert ("mcp.server.mcpserver", "MCPServer") in imported
    assert _unresolvable(imported) == []


def test_planted_nonexistent_mcp_import_is_detected() -> None:
    planted = _mcp_imports("from mcp.server.nonexistent_module import Thing\n")
    assert _unresolvable(planted) == ["mcp.server.nonexistent_module"]


def test_planted_nonexistent_mcp_symbol_is_detected() -> None:
    planted = _mcp_imports("from mcp.server.mcpserver import NoSuchSymbol\n")
    assert _unresolvable(planted) == ["mcp.server.mcpserver.NoSuchSymbol"]


def test_mcp_submodule_import_resolves() -> None:
    imports = _mcp_imports("from mcp.server import mcpserver\n")
    assert _unresolvable(imports) == []


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


def test_guard_names_failing_submodule_from_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failing module in the message comes from ``ImportError.name``.

    ``mcp.server.mcpserver`` imports, but resolving ``MCPServer`` from it fails
    on a different module, so the message must name that module and not the
    default the guard falls back to when ``e.name`` is unset.
    """

    class _BrokenModule(types.ModuleType):
        def __getattr__(self, attribute: str) -> object:
            raise ImportError(
                f"cannot load {attribute}", name="mcp.server.some_other_dependency"
            )

    broken = _BrokenModule("mcp.server.mcpserver")
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", broken)
    monkeypatch.setattr(
        transport_streamable_http.importlib.metadata, "version", lambda _: "9.9.9"
    )
    with pytest.raises(ImportError) as excinfo:
        transport_streamable_http._load_mcp_server_class()
    message = str(excinfo.value)
    assert "mcp.server.some_other_dependency" in message
    assert "'mcp.server.mcpserver'" not in message


def test_installed_mcp_satisfies_declared_floor() -> None:
    """The resolved SDK lies inside the range ``pyproject.toml`` declares.

    The range carries both the 2.3 floor and the ``<3.0.0`` ceiling, so one
    containment check covers each bound, and ``Version`` parses pre-release and
    local version strings that ``int()`` splitting cannot.
    """
    specifier = _declared_mcp_specifier()
    installed = Version(importlib.metadata.version("mcp"))
    assert specifier.contains(installed, prereleases=True), (
        f"installed mcp {installed} is outside the declared range {specifier}"
    )
