# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21924: the declared omnibase-core floor admits every module the node imports.

HandlerArchitectureValidation imports ``omnibase_core.validation.checker_stack_shape``
and ``validator_stack_shape`` at module import. uv.lock pins a core dev tag that has
them, so CI passes whatever the published range says, but a consumer who resolves
core from the range alone gets an ImportError wiring this node when the floor is
below the release that first shipped those modules.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[4]
NODE_DIR = ROOT / "src" / "omnibase_infra" / "nodes" / "node_architecture_validator"
FIRST_RELEASE = Path(__file__).parent / "fixtures" / "omnibase_core_first_release.yaml"


def _declared_core_floor(manifest: dict[str, object]) -> Version:
    project = manifest["project"]
    assert isinstance(project, dict)
    (requirement,) = (
        Requirement(item)
        for item in project["dependencies"]
        if Requirement(item).name == "omnibase-core"
    )
    (lower,) = (spec for spec in requirement.specifier if spec.operator == ">=")
    return Version(lower.version)


def _imported_core_modules(source_dir: Path) -> set[str]:
    modules: set[str] = set()
    for path in sorted(source_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module and node.module.split(".")[0] == "omnibase_core":
                    modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".")[0] == "omnibase_core"
                )
    return modules


def _floor_shortfalls(
    floor: Version, imported: set[str], first_release: dict[str, str]
) -> list[str]:
    """Imported modules whose first release is unread or newer than the floor."""
    problems: list[str] = []
    for module in sorted(imported):
        released = first_release.get(module)
        if released is None:
            problems.append(f"{module}: first release not declared")
        elif Version(released) > floor:
            problems.append(f"{module}: first shipped in {released}, floor is {floor}")
    return problems


def _first_release() -> dict[str, str]:
    loaded = yaml.safe_load(FIRST_RELEASE.read_text())
    assert isinstance(loaded, dict)
    return {str(key): str(value) for key, value in loaded.items()}


def test_declared_core_floor_admits_every_module_the_node_imports() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    floor = _declared_core_floor(manifest)
    imported = _imported_core_modules(NODE_DIR)
    assert "omnibase_core.validation.checker_stack_shape" in imported
    assert _floor_shortfalls(floor, imported, _first_release()) == []


def test_a_floor_below_the_first_release_of_an_imported_module_is_found() -> None:
    imported = _imported_core_modules(NODE_DIR)
    found = _floor_shortfalls(Version("0.0.0"), imported, _first_release())
    assert any("checker_stack_shape" in problem for problem in found)
    assert any("validator_stack_shape" in problem for problem in found)


def test_an_import_with_no_declared_first_release_is_found() -> None:
    newest = max(Version(value) for value in _first_release().values())
    found = _floor_shortfalls(
        newest, {"omnibase_core.somewhere.new_module"}, _first_release()
    )
    assert found == ["omnibase_core.somewhere.new_module: first release not declared"]
