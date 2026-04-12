# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the generate_entry_points.py script."""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers — import the functions under test from the script
# ---------------------------------------------------------------------------


def _import_script() -> object:
    """Import generate_entry_points as a module (scripts/ is not a package)."""
    import importlib.util
    import sys

    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    script_path = scripts_dir / "generate_entry_points.py"
    spec = importlib.util.spec_from_file_location("generate_entry_points", script_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_entry_points"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def script():  # type: ignore[return]
    return _import_script()


# ---------------------------------------------------------------------------
# collect_node_dirs()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_collect_node_dirs_returns_node_dirs(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """collect_node_dirs() should return directories matching node_* pattern."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_alpha").mkdir()
    (nodes_dir / "node_beta").mkdir()
    (nodes_dir / "__pycache__").mkdir()
    (nodes_dir / "helpers").mkdir()

    result = script.collect_node_dirs(tmp_path, "mypkg")
    names = [p.name for p in result]

    assert "node_alpha" in names
    assert "node_beta" in names
    assert "__pycache__" not in names
    assert "helpers" not in names


@pytest.mark.unit
def test_collect_node_dirs_empty_when_no_nodes(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """collect_node_dirs() returns empty list when nodes/ dir has no node_ dirs."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "helpers").mkdir()

    result = script.collect_node_dirs(tmp_path, "mypkg")
    assert result == []


@pytest.mark.unit
def test_collect_node_dirs_sorted(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """collect_node_dirs() should return directories in sorted order."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_z").mkdir()
    (nodes_dir / "node_a").mkdir()
    (nodes_dir / "node_m").mkdir()

    result = script.collect_node_dirs(tmp_path, "mypkg")
    names = [p.name for p in result]
    assert names == sorted(names)


@pytest.mark.unit
def test_collect_node_dirs_exclude_migrated(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """collect_node_dirs() with exclude set should omit those node names."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_alpha").mkdir()
    (nodes_dir / "node_beta").mkdir()
    (nodes_dir / "node_gamma").mkdir()

    result = script.collect_node_dirs(
        tmp_path, "mypkg", exclude={"node_beta", "node_gamma"}
    )
    names = [p.name for p in result]

    assert names == ["node_alpha"]


# ---------------------------------------------------------------------------
# build_entry_point_section()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_entry_point_section_format(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """build_entry_point_section() should produce correct TOML entry-points block."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_alpha").mkdir()
    (nodes_dir / "node_beta").mkdir()

    node_dirs = script.collect_node_dirs(tmp_path, "mypkg")
    section = script.build_entry_point_section(node_dirs, "mypkg")

    assert '[project.entry-points."onex.nodes"]' in section
    assert 'node_alpha = "mypkg.nodes.node_alpha"' in section
    assert 'node_beta = "mypkg.nodes.node_beta"' in section


@pytest.mark.unit
def test_build_entry_point_section_empty(script) -> None:  # type: ignore[no-untyped-def]
    """build_entry_point_section() with empty dirs list produces header-only section."""
    section = script.build_entry_point_section([], "mypkg")
    assert '[project.entry-points."onex.nodes"]' in section
    # No entries after the header
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    assert len(lines) == 1


@pytest.mark.unit
def test_build_entry_point_section_matches_omnimarket_format(
    script, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    """Entry lines must follow the exact omnimarket format: name = "pkg.nodes.name"."""
    nodes_dir = tmp_path / "src" / "omnimarket" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_merge_sweep").mkdir()

    node_dirs = script.collect_node_dirs(tmp_path, "omnimarket")
    section = script.build_entry_point_section(node_dirs, "omnimarket")

    assert 'node_merge_sweep = "omnimarket.nodes.node_merge_sweep"' in section


# ---------------------------------------------------------------------------
# check mode (integration-ish, but uses only tmp dirs)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_mode_exits_0_when_clean(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """check_mode() should return 0 when pyproject.toml entry points are up to date."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_alpha").mkdir()

    node_dirs = script.collect_node_dirs(tmp_path, "mypkg")
    section = script.build_entry_point_section(node_dirs, "mypkg")

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "mypkg"\n\n{section}\n')

    result = script.check_mode(tmp_path, "mypkg", node_dirs)
    assert result == 0


@pytest.mark.unit
def test_check_mode_exits_1_when_drifted(script, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """check_mode() should return 1 when pyproject.toml is missing an entry."""
    nodes_dir = tmp_path / "src" / "mypkg" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "node_alpha").mkdir()
    (nodes_dir / "node_beta").mkdir()

    # pyproject only has node_alpha, missing node_beta
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "mypkg"\n\n'
        '[project.entry-points."onex.nodes"]\n'
        'node_alpha = "mypkg.nodes.node_alpha"\n'
    )

    node_dirs = script.collect_node_dirs(tmp_path, "mypkg")
    result = script.check_mode(tmp_path, "mypkg", node_dirs)
    assert result == 1


# ---------------------------------------------------------------------------
# _resolve_repo() — absolute path mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_repo_absolute_path_uses_src_subdir(
    script, tmp_path: Path
) -> None:
    """_resolve_repo with an absolute path derives package name from src/, not dir name."""
    repo_root = tmp_path / "my-repo-with-hyphens"
    src = repo_root / "src" / "my_package"
    src.mkdir(parents=True)

    resolved_root, pkg = script._resolve_repo(str(repo_root))
    assert resolved_root == repo_root
    assert pkg == "my_package"


@pytest.mark.unit
def test_resolve_repo_absolute_path_no_src_raises(
    script, tmp_path: Path
) -> None:
    """_resolve_repo raises ValueError for absolute path with no src/ package."""
    repo_root = tmp_path / "empty-repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="Cannot determine package name"):
        script._resolve_repo(str(repo_root))


# ---------------------------------------------------------------------------
# omnimarket smoke-test (real repo)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_omnimarket_collect_and_build_smoke(script) -> None:  # type: ignore[no-untyped-def]
    """collect_node_dirs + build_entry_point_section work against the real omnimarket repo."""
    try:
        omnimarket_root, pkg = script._resolve_repo("omnimarket")
    except ValueError:
        pytest.skip("omnimarket not resolvable from this environment")
        return  # unreachable; satisfies static analysis

    if not omnimarket_root.exists():
        pytest.skip("omnimarket not found")
        return  # unreachable; satisfies static analysis

    node_dirs = script.collect_node_dirs(omnimarket_root, pkg)
    # omnimarket has many nodes — ensure we found a reasonable number
    assert len(node_dirs) > 50, f"Expected >50 nodes, found {len(node_dirs)}"

    section = script.build_entry_point_section(node_dirs, pkg)
    assert '[project.entry-points."onex.nodes"]' in section
    # Every found dir must have a corresponding entry line
    for nd in node_dirs:
        assert f'{nd.name} = "{pkg}.nodes.{nd.name}"' in section
