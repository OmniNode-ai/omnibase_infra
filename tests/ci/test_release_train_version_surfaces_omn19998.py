# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A release commit moves a repo's declared version surfaces with the manifest (OMN-19998).

omnibase_spi compares the ``SPI-vX.Y.Z`` badge in 15 ``docs/api-reference`` files
with its ``pyproject.toml`` version. The train's release commit moved the manifest
and the lock and left the badges, so the 0.23.7 release PR went red and sat there.
The fixtures below are spelled in this repo: the real policy file is read, the
badge line is copied from an omnibase_spi doc, and no other clone is needed.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = _ROOT / "config" / "release_train_policy.yaml"
_WORKFLOW = _ROOT / ".github" / "workflows" / "release-train-nightly.yml"

_BADGE_LINE = (
    "![Version](https://img.shields.io/badge/SPI-v{v}-blue) "
    "![Status](https://img.shields.io/badge/status-stable-green) "
    "![Since](https://img.shields.io/badge/since-v0.2.0-lightgrey)\n"
)

_PYPROJECT = """\
[project]
name = "omnibase-spi"
version = "0.23.6"
requires-python = ">=3.12"
dependencies = []
"""

_LOCK = """\
version = 1

[[package]]
name = "omnibase-spi"
version = "0.23.6"
source = { editable = "." }
"""

_DOCS = ("CORE.md", "NODES.md")


def _release_train() -> Any:
    name = "release_train_version_surfaces_under_test"
    if name in sys.modules:
        return sys.modules[name]
    path = _ROOT / "scripts" / "ci" / "release_train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def spi_clone(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(_LOCK, encoding="utf-8")
    docs = tmp_path / "docs" / "api-reference"
    docs.mkdir(parents=True)
    for name in _DOCS:
        (docs / name).write_text(
            "# Heading\n\n" + _BADGE_LINE.format(v="0.23.6") + "\nBody\n",
            encoding="utf-8",
        )
    return tmp_path


def _surfaces() -> tuple[Any, ...]:
    module = _release_train()
    policy = module.policy_for(module.load_policy(_POLICY), "omnibase_spi")
    return tuple(policy.version_surfaces)


def test_omnibase_spi_declares_its_badge_surface() -> None:
    surfaces = _surfaces()
    assert [s.glob for s in surfaces] == ["docs/api-reference/*.md"]
    match = re.search(surfaces[0].pattern, _BADGE_LINE.format(v="0.23.6"))
    assert match is not None
    assert match.group("version") == "0.23.6"


@pytest.mark.live_contact("tests/ci/fixtures/release_train_spi_badges_omn19998.json")
def test_the_real_spi_badge_lines_move_with_the_release_commit(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    """The 15 badge lines recorded from omnibase_spi dev are moved by the bump."""
    module = _release_train()
    recorded = recorded_response["response"]
    assert isinstance(recorded, dict)
    lines = recorded["badge_lines"]
    assert isinstance(lines, dict)
    assert len(lines) == 15
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(_LOCK, encoding="utf-8")
    for relative, line in lines.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Title\n\n" + line + "\n", encoding="utf-8")
    written = module.bump_declared_version(tmp_path, "0.23.7", _surfaces())
    assert sorted(written) == sorted(["pyproject.toml", "uv.lock", *lines])
    assert module.stale_version_surfaces(tmp_path, _surfaces(), "0.23.7") == []
    for relative, line in lines.items():
        moved = (tmp_path / relative).read_text(encoding="utf-8").splitlines()[2]
        assert moved == line.replace("SPI-v0.23.6-", "SPI-v0.23.7-")


def test_the_release_commit_leaves_no_stale_badge(spi_clone: Path) -> None:
    module = _release_train()
    code = module.main(
        [
            "bump-version",
            "--policy",
            str(_POLICY),
            "--repo",
            "omnibase_spi",
            "--clone",
            str(spi_clone),
            "--version",
            "0.23.7",
        ]
    )
    assert code == 0
    assert module.stale_version_surfaces(spi_clone, _surfaces(), "0.23.7") == []
    for name in _DOCS:
        text = (spi_clone / "docs" / "api-reference" / name).read_text(encoding="utf-8")
        assert text == "# Heading\n\n" + _BADGE_LINE.format(v="0.23.7") + "\nBody\n"


def test_the_subcommand_prints_the_badge_files_it_wrote(
    spi_clone: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _release_train()
    module.main(
        [
            "bump-version",
            "--policy",
            str(_POLICY),
            "--repo",
            "omnibase_spi",
            "--clone",
            str(spi_clone),
            "--version",
            "0.23.7",
        ]
    )
    assert capsys.readouterr().out.split() == [
        "pyproject.toml",
        "uv.lock",
        "docs/api-reference/CORE.md",
        "docs/api-reference/NODES.md",
    ]


def test_a_manifest_only_bump_is_detected_as_stale_badges(spi_clone: Path) -> None:
    """Control: the bump this replaces leaves the badges at 0.23.6, and the check says so."""
    module = _release_train()
    module.bump_declared_version(spi_clone, "0.23.7")
    stale = module.stale_version_surfaces(spi_clone, _surfaces(), "0.23.7")
    assert stale == [
        "docs/api-reference/CORE.md: found 0.23.6, want 0.23.7",
        "docs/api-reference/NODES.md: found 0.23.6, want 0.23.7",
    ]


def test_a_surface_that_matches_nothing_refuses_and_writes_nothing(
    spi_clone: Path,
) -> None:
    module = _release_train()
    missing = module.ModelVersionSurface(
        glob="docs/gone/*.md", pattern=r"badge/SPI-v(?P<version>\d+\.\d+\.\d+)-"
    )
    with pytest.raises(module.ReleaseTrainConfigError, match="matches nothing"):
        module.bump_declared_version(spi_clone, "0.23.7", (missing,))
    assert 'version = "0.23.6"' in (spi_clone / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def test_a_repo_without_surfaces_bumps_as_before(spi_clone: Path) -> None:
    module = _release_train()
    assert module.bump_declared_version(spi_clone, "0.23.7") == [
        "pyproject.toml",
        "uv.lock",
    ]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            "- glob: /abs/*.md\n        pattern: 'v(?P<version>1)'",
            "clean repo-relative",
        ),
        ("- glob: docs/*.md\n        pattern: 'v(1)'", "named group 'version'"),
        ("- glob: docs/*.md\n        pattern: 'v(?P<version'", "does not compile"),
        ("- glob: docs/*.md", "has no 'pattern'"),
    ],
)
def test_a_malformed_surface_refuses_at_load(
    tmp_path: Path, entry: str, message: str
) -> None:
    module = _release_train()
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "repos:\n"
        "  demo:\n"
        "    package: demo\n"
        "    default_branch: dev\n"
        "    release_relevant_paths: [src]\n"
        "    lab_evidence: none\n"
        "    lab_evidence_note: no lab lane\n"
        "    mode: cut\n"
        "    version_surfaces:\n"
        f"      {entry}\n",
        encoding="utf-8",
    )
    with pytest.raises(module.ReleaseTrainConfigError, match=message):
        module.load_policy(policy)


def test_the_workflow_passes_the_repo_to_the_bump() -> None:
    body = _WORKFLOW.read_text(encoding="utf-8")
    start = body.index('release_train.py" bump-version')
    assert '--repo "${TARGET_REPO}"' in body[start : start + 400], (
        "the cut job bumps without --repo, so no declared version surface moves "
        "with the manifest"
    )
