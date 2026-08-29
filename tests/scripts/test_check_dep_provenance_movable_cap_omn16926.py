# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the cascade-movability range-cap check (OMN-16926).

Root cause this closes: ``find_unmovable_cascade_targets`` (OMN-15604 AC4)
detects exactly ONE immovability channel -- a ``[tool.uv.sources]`` git
override. It does NOT detect a range cap in ``[project.dependencies]``
(e.g. ``omnibase-core>=0.46.13,<0.47.0``) that excludes the release version a
cascade is trying to move to. ``uv lock --upgrade-package`` cannot cross that
cap any more than it can cross a git override, but the pre-OMN-16926 checker
reported the package movable anyway, so the cascade proceeded, re-resolved to
the SAME already-locked version, and the workflow's SKIP-summary step
misreported "already uses the latest version" -- untrue for a repo that is
capped below the release and cannot reach it.

Live blast radius (evidence in the ticket): on the omnibase_core v0.47.0
cascade, 5 of 6 downstream legs (omniintelligence, omnimemory, omniclaude,
onex_change_control, omninode_infra) carry a ``<0.47.0`` cap and were all
misreported as "already on latest" by the pre-fix checker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dep_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_dep_provenance", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _write_pyproject(tmp_path: Path, *, dependency: str) -> Path:
    """Write a minimal pyproject.toml with no [tool.uv.sources] overrides at
    all -- isolates the range-cap channel from the git-override channel
    covered by test_check_dep_provenance_lineage_omn15604.py's AC4 cases.
    """
    content = (
        "[project]\n"
        'name = "downstream-repo"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        f'    "{dependency}",\n'
        "]\n"
        "\n"
        "[tool.uv.sources]\n"
        "\n"
        "[tool.ruff]\n"
        'target-version = "py312"\n'
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# AC1 (RED-first): a range-capped constraint that excludes the target version
# must FAIL, not report movable.
# ---------------------------------------------------------------------------


def test_check_movable_fails_for_a_range_capped_package(mod, tmp_path: Path) -> None:
    """The exact live shape from the ticket: omniintelligence@dev pins
    `omnibase-core>=0.46.13,<0.47.0`, and the cascade is trying to move to
    0.47.0 -- excluded by the `<0.47.0` cap.
    """
    path = _write_pyproject(tmp_path, dependency="omnibase-core>=0.46.13,<0.47.0")
    violations = mod.find_unmovable_cascade_targets(
        path.read_text(), "omnibase-core", target_version="0.47.0"
    )
    assert len(violations) == 1
    assert "0.47.0" in violations[0]
    assert ">=0.46.13,<0.47.0" in violations[0]

    assert (
        mod.main(
            [
                "--pyproject",
                str(path),
                "--check-movable",
                "omnibase-core",
                "--target-version",
                "0.47.0",
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "dependency",
    [
        "omnibase-core>=0.46.3,<0.47.0",
        "omnibase-core>=0.46.13,<0.47.0",
        "omnibase-core>=0.46.5,<0.47.0",
        "omnibase-core>=0.46.8,<0.47.0",
    ],
)
def test_check_movable_fails_for_every_live_capped_shape(
    mod, tmp_path: Path, dependency: str
) -> None:
    """Reproduces the exact constraint strings read live from
    omniintelligence/omnimemory/omniclaude/onex_change_control/
    omninode_infra's dev pyproject.toml on 2026-08-29 (ticket evidence
    table) -- every one of these must fail, not silently report movable.
    """
    path = _write_pyproject(tmp_path, dependency=dependency)
    violations = mod.find_unmovable_cascade_targets(
        path.read_text(), "omnibase-core", target_version="0.47.0"
    )
    assert len(violations) == 1


def test_check_movable_reports_the_cap_and_the_target_version(
    mod, tmp_path: Path
) -> None:
    """AC2/AC4: the failure must be loud and actionable -- name the actual
    declared cap AND the version the cascade could not reach, not a bare
    boolean."""
    path = _write_pyproject(tmp_path, dependency="omnibase-core>=0.46.13,<0.47.0")
    violations = mod.find_unmovable_cascade_targets(
        path.read_text(), "omnibase-core", target_version="0.47.0"
    )
    assert len(violations) == 1
    message = violations[0]
    assert "omnibase-core" in message
    assert ">=0.46.13,<0.47.0" in message
    assert "0.47.0" in message


# ---------------------------------------------------------------------------
# AC2: covers <, <=, ==, ~=, and != exclusions -- not only the `<` cap.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dependency",
    [
        "omnibase-core<0.47.0",
        "omnibase-core<=0.46.13",
        "omnibase-core==0.46.13",
        "omnibase-core~=0.46.0",
        "omnibase-core!=0.47.0,>=0.46.0",
    ],
)
def test_check_movable_fails_for_every_pep440_exclusion_operator(
    mod, tmp_path: Path, dependency: str
) -> None:
    path = _write_pyproject(tmp_path, dependency=dependency)
    violations = mod.find_unmovable_cascade_targets(
        path.read_text(), "omnibase-core", target_version="0.47.0"
    )
    assert len(violations) == 1, f"{dependency} should be flagged as capped"


# ---------------------------------------------------------------------------
# ALLOW: a constraint that DOES admit the target version is genuinely movable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dependency",
    [
        "omnibase-core>=0.46.13",
        "omnibase-core>=0.46.13,<0.48.0",
        "omnibase-core==0.47.0",
        "omnibase-core",
    ],
)
def test_check_movable_allows_a_constraint_that_admits_the_target(
    mod, tmp_path: Path, dependency: str
) -> None:
    path = _write_pyproject(tmp_path, dependency=dependency)
    violations = mod.find_unmovable_cascade_targets(
        path.read_text(), "omnibase-core", target_version="0.47.0"
    )
    assert violations == [], f"{dependency} should admit 0.47.0: {violations}"


def test_check_movable_still_movable_exact_pin_at_target(mod, tmp_path: Path) -> None:
    """omnibase_infra's own live shape: an exact pin the cascade rewrites in
    place before locking is movable once it already equals the target."""
    path = _write_pyproject(tmp_path, dependency="omnibase-core==0.47.0")
    assert (
        mod.find_unmovable_cascade_targets(
            path.read_text(), "omnibase-core", target_version="0.47.0"
        )
        == []
    )


# ---------------------------------------------------------------------------
# Backward compatibility: omitting target_version keeps checking ONLY the
# [tool.uv.sources] git-override channel (the pre-OMN-16926 behavior) -- a
# caller that has no target version yet must not regress to false-negatives
# OR false-positives on a capped-but-unspecified-target package.
# ---------------------------------------------------------------------------


def test_check_movable_without_target_version_ignores_range_caps(
    mod, tmp_path: Path
) -> None:
    path = _write_pyproject(tmp_path, dependency="omnibase-core>=0.46.13,<0.47.0")
    assert mod.find_unmovable_cascade_targets(path.read_text(), "omnibase-core") == []


def test_check_movable_without_target_version_still_catches_git_override(
    mod, tmp_path: Path
) -> None:
    content = (
        "[project]\n"
        'name = "downstream-repo"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        '    "omnibase-core==0.46.8",\n'
        "]\n"
        "\n"
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        'rev = "deadbeef" }\n'
        "\n"
        "[tool.ruff]\n"
        'target-version = "py312"\n'
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    violations = mod.find_unmovable_cascade_targets(path.read_text(), "omnibase-core")
    assert len(violations) == 1
    assert "cannot move" in violations[0]


# ---------------------------------------------------------------------------
# Precedence: a git override is reported even when a range cap is also
# present -- the git-override message is the more specific/actionable one
# and existing callers/tests depend on its exact wording ("cannot move").
# ---------------------------------------------------------------------------


def test_check_movable_git_override_takes_precedence_over_range_cap(
    mod, tmp_path: Path
) -> None:
    content = (
        "[project]\n"
        'name = "downstream-repo"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        '    "omnibase-core>=0.46.13,<0.47.0",\n'
        "]\n"
        "\n"
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        'rev = "deadbeef" }\n'
        "\n"
        "[tool.ruff]\n"
        'target-version = "py312"\n'
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    violations = mod.find_unmovable_cascade_targets(
        path.read_text(), "omnibase-core", target_version="0.47.0"
    )
    assert len(violations) == 1
    assert "cannot move" in violations[0]
