# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure source-shape and byte-preserving cascade regressions (OMN-17427)."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_dep_provenance.py"
_TAG = "dev-20261029143607-342000ec5702"
_GIT = 'git = "https://github.com/OmniNode-ai/omnibase_core.git"'


@pytest.fixture
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dep_provenance_shape", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project(source: str) -> str:
    return (
        '[project]\nname = "downstream"\nversion = "0.0.0"\n'
        'dependencies = ["omnibase-core>=0.47.31,<0.48.0"]\n'
        f"[tool.uv.sources]\n{source}"
    )


def _unexpected_resolver(repo: str, ref: str) -> tuple[str | None, str]:
    pytest.fail(f"dev-merge tag must use core's validator, not resolve {repo}@{ref}")


@pytest.mark.parametrize(
    "package", ["omnibase-core", "omnibase_core", '"Omnibase_Core"']
)
@pytest.mark.parametrize("subtable", [False, True])
def test_dev_tag_shape_is_admitted_and_skips_release_tree(
    mod: ModuleType, package: str, subtable: bool
) -> None:
    source = f'{package} = {{ {_GIT}, tag = "{_TAG}" }}\n'
    if subtable:
        source = f'[tool.uv.sources.{package}]\n{_GIT}\ntag = "{_TAG}"\n'
    text = _project(source)
    assert mod.find_violations(text) == []
    assert mod.find_lineage_violations(text, resolve=_unexpected_resolver) == []
    assert mod.find_unmovable_cascade_targets(text, "omnibase-core") == []
    assert mod.find_unmovable_cascade_targets(text, "omnibase_core", "0.47.58") == []
    # Shape admission must not suppress the independent release constraint check.
    assert mod.find_unmovable_cascade_targets(text, "omnibase-core", "0.48.0")


@pytest.mark.parametrize(
    "refs",
    [
        'rev = "deadbeef"',
        'branch = "dev"',
        'tag = "v0.47.57"',
        f'tag = "{_TAG}", rev = "deadbeef"',
        f'tag = "{_TAG}", branch = "dev"',
        f'tag = "{_TAG}", rev = ""',
        f'tag = "{_TAG}", branch = ""',
        'tag = "dev-20261029143607-342000EC5702"',
        'tag = "dev-2026102914360-342000ec5702"',
        f'tag = "{_TAG}extra"',
    ],
)
def test_other_refs_are_still_refused(mod: ModuleType, refs: str) -> None:
    text = _project(f"omnibase-core = {{ {_GIT}, {refs} }}\n")
    calls = []

    def different_trees(repo: str, ref: str) -> tuple[str, str]:
        calls.append((repo, ref))
        return ("released" if ref == "v0.47.58" else "pinned"), "ok"

    assert mod.find_violations(text)
    assert mod.find_lineage_violations(text, resolve=different_trees)
    # A range has no exact released tree to compare. Keep the release-tree
    # resolver check on a separate exact-release declaration, without a uv override.
    assert calls == []
    release_text = text.replace(
        "omnibase-core>=0.47.31,<0.48.0", "omnibase-core==0.47.58"
    )
    assert mod.find_lineage_violations(release_text, resolve=different_trees)
    assert calls
    assert mod.find_unmovable_cascade_targets(text, "omnibase-core")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "package", ["omnibase-core", "omnibase_core", "'Omnibase_Core'"]
)
def test_drop_preserves_every_other_byte(
    mod: ModuleType, tmp_path: Path, newline: str, package: str
) -> None:
    line = f'  {package} = {{ {_GIT}, tag = "{_TAG}" }}  # temporary\n'
    before = _project('# keep this comment\nother = { path = "../other" }\n')
    after = (
        '# keep the next comment\nneighbor = { index = "private" }\n\n'
        '[tool.other]\nmessage = "untouched café"\n'
        # Identical text in another table must remain intact.
        + line
        + '[tool.uv.sources.neighbor2]\npath = "../neighbor2"\n'
    )
    original = (before + line + after).replace("\n", newline).encode()
    expected = (before + after).replace("\n", newline).encode()
    path = tmp_path / "pyproject.toml"
    path.write_bytes(original)
    parsed = tomllib.loads(original.decode())
    del parsed["tool"]["uv"]["sources"][package.strip("'")]
    assert (
        mod.main(["--pyproject", str(path), "--drop-dev-tag-override", "omnibase_core"])
        == 0
    )
    assert path.read_bytes() == expected
    assert tomllib.loads(expected.decode()) == parsed
    # Repeating the drop is a no-op.
    assert (
        mod.main(["--pyproject", str(path), "--drop-dev-tag-override", "omnibase-core"])
        == 0
    )
    assert path.read_bytes() == expected


@pytest.mark.parametrize(
    "text", ['[project]\nname = "downstream"\n', _project('other = { path = "." }\n')]
)
def test_absent_source_is_a_noop(mod: ModuleType, tmp_path: Path, text: str) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_bytes(text.encode())
    assert (
        mod.main(["--pyproject", str(path), "--drop-dev-tag-override", "omnibase-core"])
        == 0
    )
    assert path.read_bytes() == text.encode()


@pytest.mark.parametrize(
    "source",
    [
        f'[tool.uv.sources.omnibase-core]\n{_GIT}\ntag = "{_TAG}"\n',
        f'omnibase-core = {{ {_GIT}, rev = "deadbeef" }}\n',
        f'omnibase-core = {{ {_GIT}, branch = "dev" }}\n',
        f'omnibase-core = {{ {_GIT}, tag = "v0.47.58" }}\n',
        f'omnibase-core = {{ {_GIT}, tag = "{_TAG}", rev = "deadbeef" }}\n',
        f'omnibase-core = {{ {_GIT}, tag = "{_TAG}", branch = "dev" }}\n',
        'omnibase-core = { path = "../core" }\n',
        f'omnibase-core = [{{ {_GIT}, tag = "{_TAG}" }}]\n',
        f'omnibase-core = {{ {_GIT}, tag = """\n{_TAG}""" }}\n',
        f'omnibase-core = {{ {_GIT}, tag = "{_TAG}" }}\n'
        f'omnibase_core = {{ {_GIT}, tag = "{_TAG}" }}\n',
        f'omnibase-core.git = "https://github.com/OmniNode-ai/omnibase_core.git"\n'
        f'omnibase-core.tag = "{_TAG}"\n',
    ],
)
def test_drop_refuses_other_shapes_without_writing(
    mod: ModuleType, tmp_path: Path, source: str
) -> None:
    text = _project(source).encode()
    path = tmp_path / "pyproject.toml"
    path.write_bytes(text)
    assert (
        mod.main(["--pyproject", str(path), "--drop-dev-tag-override", "omnibase-core"])
        == 1
    )
    assert path.read_bytes() == text


def test_drop_refuses_invalid_toml_without_writing(
    mod: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "pyproject.toml"
    text = _project('omnibase-core = { git = "unfinished"\n').encode()
    path.write_bytes(text)
    assert (
        mod.main(["--pyproject", str(path), "--drop-dev-tag-override", "omnibase-core"])
        == 1
    )
    assert path.read_bytes() == text
