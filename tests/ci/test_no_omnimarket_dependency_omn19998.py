# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnibase_infra must not depend on omnimarket (OMN-19998).

Repo layering puts omnimarket ABOVE omnibase_infra: market depends on infra,
never the other way round (operator ruling 2026-10-15T21:52:44Z). This module
is the guard, run by the ``no-omnimarket-dependency`` pre-commit hook and by
the Lint job of ``ci.yml`` (a strict member of the required CI Summary gate).

It refuses omnimarket on the package dependency surface:

* ``pyproject.toml``: any requirement naming omnimarket in
  ``[build-system] requires``, ``[project] dependencies``,
  ``[project.optional-dependencies]``, ``[dependency-groups]``, or the
  ``[tool.uv]`` dev, override and constraint lists; a ``[tool.uv.sources]``
  key for omnimarket; an entry point whose target is an omnimarket module.
* ``uv.lock``: a locked package named omnimarket, or any locked package
  depending on it.
* ``src/omnibase_infra``: an ``import omnimarket`` or ``from omnimarket ...``
  executed at import time, that is anywhere outside a function body.

There is no allowlist and no suppression marker. The negative controls below
feed the same checkers a manifest, a lock and a module that do depend on
omnimarket and assert each one fires.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "omnibase_infra"
FORBIDDEN = "omnimarket"

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    return _normalize(match.group(1)) if match else ""


def _requirements(pyproject: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    """Yield ``(table, requirement)`` for every requirement string declared."""
    build = pyproject.get("build-system", {})
    assert isinstance(build, dict)
    for req in build.get("requires", []):
        yield "build-system.requires", req
    project = pyproject.get("project", {})
    assert isinstance(project, dict)
    for req in project.get("dependencies", []):
        yield "project.dependencies", req
    for extra, reqs in project.get("optional-dependencies", {}).items():
        for req in reqs:
            yield f"project.optional-dependencies.{extra}", req
    groups = pyproject.get("dependency-groups", {})
    assert isinstance(groups, dict)
    for group, reqs in groups.items():
        for req in reqs:
            if isinstance(req, str):  # {include-group = ...} names no package
                yield f"dependency-groups.{group}", req
    tool = pyproject.get("tool", {})
    assert isinstance(tool, dict)
    uv = tool.get("uv", {})
    for key in ("dev-dependencies", "override-dependencies", "constraint-dependencies"):
        for req in uv.get(key, []):
            yield f"tool.uv.{key}", req


def pyproject_findings(text: str) -> list[str]:
    """Every place a pyproject.toml text makes omnimarket a dependency."""
    data = tomllib.loads(text)
    findings = [
        f"{table}: {req!r}"
        for table, req in _requirements(data)
        if _requirement_name(req) == FORBIDDEN
    ]
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    findings += [
        f"tool.uv.sources: {name!r}"
        for name in sources
        if _normalize(name) == FORBIDDEN
    ]
    project = data.get("project", {})
    entry_tables: dict[str, Mapping[str, str]] = {
        "project.scripts": project.get("scripts", {}),
        "project.gui-scripts": project.get("gui-scripts", {}),
    }
    for group, points in project.get("entry-points", {}).items():
        entry_tables[f"project.entry-points.{group}"] = points
    for table, points in entry_tables.items():
        for name, target in points.items():
            if target.strip().split(".")[0].split(":")[0] == FORBIDDEN:
                findings.append(f"{table}: {name} = {target!r}")
    return findings


def lock_findings(text: str) -> list[str]:
    """Every locked package that is, or depends on, omnimarket."""
    findings: list[str] = []
    for package in tomllib.loads(text).get("package", []):
        name = _normalize(package.get("name", ""))
        if name == FORBIDDEN:
            findings.append(f"uv.lock: package {name} is locked")
        deps = list(package.get("dependencies", []))
        for extra in package.get("optional-dependencies", {}).values():
            deps += extra
        for group in package.get("dev-dependencies", {}).values():
            deps += group
        for dep in deps:
            if _normalize(dep.get("name", "")) == FORBIDDEN:
                findings.append(f"uv.lock: package {name} depends on {FORBIDDEN}")
    return findings


class _ImportTimeImports(ast.NodeVisitor):
    """Collect omnimarket imports that run when the module is imported."""

    def __init__(self) -> None:
        self.depth = 0
        self.lines: list[int] = []

    def _function(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._function(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self.depth == 0 and any(
            alias.name.split(".")[0] == FORBIDDEN for alias in node.names
        ):
            self.lines.append(node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (
            self.depth == 0
            and node.level == 0
            and (node.module or "").split(".")[0] == FORBIDDEN
        ):
            self.lines.append(node.lineno)


def source_findings(source: str, label: str) -> list[str]:
    """``label:line`` for each import-time omnimarket import in ``source``."""
    visitor = _ImportTimeImports()
    visitor.visit(ast.parse(source, filename=label))
    return [
        f"{label}:{line}: import-time import of {FORBIDDEN}" for line in visitor.lines
    ]


def _src_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


# --- the live tree ---------------------------------------------------------


def test_pyproject_declares_no_omnimarket_dependency() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Positive control: the walk reads the real requirement tables.
    assert len(list(_requirements(tomllib.loads(text)))) > 10
    assert pyproject_findings(text) == []


def test_lock_resolves_no_omnimarket() -> None:
    text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    # Positive control: the lock parses into its real package list.
    assert len(tomllib.loads(text)["package"]) > 50
    assert lock_findings(text) == []


def test_src_imports_no_omnimarket_at_import_time() -> None:
    files = _src_files()
    # Positive control: the scan reaches the package's modules.
    assert len(files) > 500
    findings = [
        finding
        for path in files
        for finding in source_findings(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    ]
    assert findings == [], (
        "omnibase_infra must not depend on omnimarket (OMN-19998); move the code "
        "to omnimarket or invert the dependency:\n" + "\n".join(findings)
    )


# --- negative controls: each checker fires on a real dependency ------------

_CLEAN_PYPROJECT = """
[build-system]
requires = ["hatchling"]
[project]
name = "omnibase-infra"
dependencies = ["omnibase-core>=0.47,<0.48"]
"""


@pytest.mark.parametrize(
    "fragment",
    [
        '[project]\nname = "x"\ndependencies = ["omnimarket>=0.1"]\n',
        '[project]\nname = "x"\ndependencies = ["OmniMarket[extra] @ git+https://example.invalid/omnimarket"]\n',
        '[project]\nname = "x"\n[project.optional-dependencies]\nlab = ["omnimarket"]\n',
        '[dependency-groups]\ndev = ["pytest", "omnimarket==1.0"]\n',
        '[tool.uv]\noverride-dependencies = ["omnimarket==1.0"]\n',
        '[tool.uv]\nconstraint-dependencies = ["omnimarket<2"]\n',
        '[tool.uv]\ndev-dependencies = ["omnimarket"]\n',
        '[tool.uv.sources]\nomnimarket = { git = "https://example.invalid/omnimarket" }\n',
        '[build-system]\nrequires = ["omnimarket"]\n',
        '[project]\nname = "x"\n[project.scripts]\nonex-x = "omnimarket.cli:main"\n',
        '[project]\nname = "x"\n[project.entry-points."onex.nodes"]\nn = "omnimarket.nodes.node_x"\n',
    ],
)
def test_pyproject_checker_fires_on_omnimarket(fragment: str) -> None:
    assert pyproject_findings(fragment) != []


def test_pyproject_checker_passes_a_clean_manifest() -> None:
    assert pyproject_findings(_CLEAN_PYPROJECT) == []
    # A package whose name only starts with the forbidden one is not it.
    assert (
        pyproject_findings('[project]\nname = "x"\ndependencies = ["omnimarketing"]\n')
        == []
    )


def test_lock_checker_fires_on_omnimarket() -> None:
    locked = (
        '[[package]]\nname = "omnimarket"\nversion = "1.0"\n'
        '[[package]]\nname = "omnibase-infra"\nversion = "1.0"\n'
        'dependencies = [{ name = "omnimarket" }]\n'
    )
    assert lock_findings(locked) == [
        "uv.lock: package omnimarket is locked",
        "uv.lock: package omnibase-infra depends on omnimarket",
    ]
    assert lock_findings('[[package]]\nname = "omnibase-core"\nversion = "1.0"\n') == []


@pytest.mark.parametrize(
    "source",
    [
        "import omnimarket\n",
        "import omnimarket.nodes as nodes\n",
        "from omnimarket.nodes import node_x\n",
        "try:\n    import omnimarket\nexcept ImportError:\n    pass\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from omnimarket import x\n",
        "class C:\n    from omnimarket import x\n",
    ],
)
def test_source_checker_fires_on_import_time_import(source: str) -> None:
    assert source_findings(source, "m.py") != []


def test_source_checker_ignores_function_scope_and_lookalikes() -> None:
    assert source_findings("def f():\n    import omnimarket\n", "m.py") == []
    assert (
        source_findings("import omnimarketing\nfrom . import omnimarket\n", "m.py")
        == []
    )


# --- recorded contact: the real dependent declares the allowed direction ---


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn19998/omnimarket_dev_pyproject_requirements.json"
)
def test_recorded_omnimarket_manifest_depends_on_infra(
    recorded_response: dict[str, object],
) -> None:
    """omnimarket's own dev manifest names omnibase-infra; the reverse is refused.

    The recording is the ``[project] dependencies`` list of omnimarket's
    ``pyproject.toml`` on dev, read from its canonical clone. The checker parses
    that real requirement list, finds the market-on-infra edge, and fires once
    the same requirement is pointed the other way.
    """
    assert recorded_response["project_name"] == FORBIDDEN
    deps = recorded_response["project_dependencies"]
    assert isinstance(deps, list)
    infra = [req for req in deps if _requirement_name(req) == "omnibase-infra"]
    assert len(infra) == 1
    manifest = '[project]\nname = "omnibase-infra"\ndependencies = {}\n'
    assert pyproject_findings(manifest.format(json.dumps(deps))) == []
    reverse = FORBIDDEN + infra[0][len("omnibase-infra") :]
    assert pyproject_findings(manifest.format(json.dumps([*deps, reverse]))) == [
        f"project.dependencies: {reverse!r}"
    ]
