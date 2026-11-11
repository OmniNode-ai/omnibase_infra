# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: the ``onex occ`` CLI is gone, and with it the omnibase_compat dependency.

``omnibase_infra.cli.cli_occ`` stamped and validated change-control PR metadata
(``Evidence-Source: OCC#<n>``). Its only caller was a pre-commit hook in another
repository that is removed in parallel. It was also this repository's only
``omnibase_compat`` import, so the dependency, its uv override and the lock entry
leave with it.

Falsifier: restore the module, the ``occ`` console script, the baseline line, the
module's test, an ``omnibase_compat`` import under ``src`` or an ``omnibase-compat``
requirement and the matching test names it.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

_COMPAT_NAME = re.compile(r"^\s*omnibase[-_.]compat\b", re.IGNORECASE)


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _strings(node: object) -> list[str]:
    """Every string value reachable in a parsed TOML table."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    if isinstance(node, dict):
        return [s for value in node.values() for s in _strings(value)]
    return []


def _compat_requirements(pyproject: dict[str, object]) -> list[str]:
    """Requirement strings naming omnibase-compat anywhere dependencies live."""
    found: list[str] = []
    project = pyproject.get("project", {})
    assert isinstance(project, dict)
    found += [
        s
        for s in _strings(project.get("dependencies", []))
        + _strings(project.get("optional-dependencies", {}))
        if _COMPAT_NAME.match(s)
    ]
    found += [
        s
        for s in _strings(pyproject.get("dependency-groups", {}))
        if _COMPAT_NAME.match(s)
    ]
    tool = pyproject.get("tool", {})
    assert isinstance(tool, dict)
    uv = tool.get("uv", {})
    assert isinstance(uv, dict)
    for key in ("override-dependencies", "constraint-dependencies", "dev-dependencies"):
        found += [s for s in _strings(uv.get(key, [])) if _COMPAT_NAME.match(s)]
    sources = uv.get("sources", {})
    assert isinstance(sources, dict)
    found += [name for name in sources if _COMPAT_NAME.match(name)]
    return found


def _compat_imports(root: Path) -> list[str]:
    """``path:line`` of every ``omnibase_compat`` import in the Python files under root."""
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(
                n == "omnibase_compat" or n.startswith("omnibase_compat.")
                for n in names
            ):
                hits.append(f"{path.relative_to(root)}:{node.lineno}")
    return hits


def test_cli_occ_module_is_gone() -> None:
    assert importlib.util.find_spec("omnibase_infra.cli.cli_occ") is None
    assert not (SRC / "omnibase_infra" / "cli" / "cli_occ.py").exists()


def test_cli_occ_test_file_is_gone() -> None:
    assert not (ROOT / "tests" / "unit" / "cli" / "test_cli_occ.py").exists()


def test_occ_console_script_and_cli_entry_point_are_gone() -> None:
    """``occ`` was registered as an ``onex.cli`` entry point, not under ``scripts``."""
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    scripts = project["scripts"]
    assert isinstance(scripts, dict)
    onex_cli = project["entry-points"]["onex.cli"]
    assert isinstance(onex_cli, dict)
    for table in (scripts, onex_cli):
        assert "occ" not in table
        assert not [v for v in table.values() if "cli_occ" in str(v)]


def test_cli_occ_baseline_line_is_gone() -> None:
    baseline = (
        ROOT / ".onex_ratchets" / "canonical_file_shape_baseline.txt"
    ).read_text(encoding="utf-8")
    assert "src/omnibase_infra/cli/cli_occ.py" not in baseline.splitlines()


def test_no_omnibase_compat_requirement_in_pyproject() -> None:
    assert _compat_requirements(_pyproject()) == []


def test_no_omnibase_compat_import_under_src() -> None:
    assert _compat_imports(SRC) == []


def test_lock_names_no_omnibase_compat_package() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {p["name"] for p in lock["package"]}
    assert "omnibase-compat" not in names
    root_pkg = next(p for p in lock["package"] if p["name"] == "omnibase-infra")
    dep_names = {d["name"] for d in root_pkg.get("dependencies", [])}
    assert "omnibase-compat" not in dep_names


class TestScannersSeeTheShapesTheyGuard:
    """Positive controls: each zero above is a scanner that can return a hit."""

    def test_requirement_scanner_hits_each_dependency_table(self) -> None:
        synthetic = {
            "project": {
                "dependencies": ["omnibase_compat==1"],
                "optional-dependencies": {"x": ["Omnibase-Compat>=1"]},
            },
            "dependency-groups": {"dev": ["omnibase-compat"]},
            "tool": {
                "uv": {
                    "override-dependencies": ["omnibase-compat==1"],
                    "sources": {"omnibase-compat": {}},
                }
            },
        }
        assert len(_compat_requirements(synthetic)) == 5

    def test_requirement_scanner_ignores_other_packages(self) -> None:
        synthetic = {
            "project": {"dependencies": ["omnibase-core>=1", "omnibase-spi==1"]},
            "tool": {"uv": {"override-dependencies": ["omnibase-spi==1"]}},
        }
        assert _compat_requirements(synthetic) == []

    def test_import_scanner_hits_both_import_forms(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("import omnibase_compat.x\n", encoding="utf-8")
        (tmp_path / "b.py").write_text(
            "def f():\n    from omnibase_compat.contracts import y\n", encoding="utf-8"
        )
        (tmp_path / "c.py").write_text("import omnibase_core\n", encoding="utf-8")
        assert _compat_imports(tmp_path) == ["a.py:1", "b.py:2"]
